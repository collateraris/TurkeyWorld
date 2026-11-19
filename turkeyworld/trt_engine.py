"""Static, batch-one TensorRT engines for the optional CUDA gameplay backend."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import threading

import torch
from filelock import FileLock


def _cache_key(name, module, inputs, outputs, trt_version):
    device = inputs[0].device
    gpu = torch.cuda.get_device_properties(device)
    identity = {
        'plan_schema': 1, 'name': name, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'tensorrt': trt_version, 'platform': sys.platform, 'gpu': gpu.name,
        'capability': [gpu.major, gpu.minor], 'module': repr(module),
        'inputs': [(tuple(x.shape), str(x.dtype)) for x in inputs], 'outputs': outputs,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    digest.update(inspect.getsource(EngineCache._build).encode())
    sources = set()
    for child in module.modules():
        if type(child).__module__.startswith('turkeyworld.'):
            sources.add(Path(inspect.getfile(type(child))))
    for source in sorted(sources):
        digest.update(source.read_bytes())
    for key, tensor in (*module.named_parameters(), *module.named_buffers()):
        digest.update(repr((key, tuple(tensor.shape), str(tensor.dtype))).encode())
        data = tensor.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy()
        digest.update(memoryview(data))
    return digest.hexdigest()


class InvalidEngine(RuntimeError):
    """A cached plan could not be deserialized on this device."""


class StaticEngine:
    """Enqueue on one non-default stream and return tensors ordered on the caller stream."""

    def __init__(self, trt, logger, path, samples, expected_outputs):
        self.logger = logger
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise InvalidEngine(f'Cannot deserialize TensorRT engine {path}')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f'Cannot create execution context for {path}')
        self.device = samples[0].device
        self.names = tuple(f'input_{i}' for i in range(len(samples)))
        self.inputs = [(tuple(x.shape), x.dtype) for x in samples]
        self.outputs = []
        dtype_map = {trt.float32: torch.float32, trt.float16: torch.float16,
                     trt.int32: torch.int32, trt.int64: torch.int64, trt.bool: torch.bool}
        names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        if names != set(self.names).union(expected_outputs):
            raise InvalidEngine(f'TensorRT engine bindings differ from the model for {path}')
        for name, (shape, dtype) in zip(self.names, self.inputs):
            if (self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT
                    or tuple(self.engine.get_tensor_shape(name)) != shape
                    or dtype_map.get(self.engine.get_tensor_dtype(name)) != dtype):
                raise InvalidEngine(f'Incompatible TensorRT input {name} in {path}')
        for name, (shape, dtype) in expected_outputs.items():
            if (self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT
                    or tuple(self.engine.get_tensor_shape(name)) != shape
                    or dtype_map.get(self.engine.get_tensor_dtype(name)) != dtype):
                raise InvalidEngine(f'Incompatible TensorRT output {name} in {path}')
            self.outputs.append((name, shape, dtype))
        self.stream = torch.cuda.Stream(device=self.device)
        self.enqueue_lock = threading.Lock()

    def __call__(self, *args):
        if len(args) != len(self.inputs):
            raise ValueError('Incorrect TensorRT input count')
        caller = torch.cuda.current_stream(self.device)
        inputs = []
        for value, (shape, dtype) in zip(args, self.inputs):
            if tuple(value.shape) != shape or value.dtype != dtype or value.device != self.device:
                raise ValueError(f'Expected shape {shape}, dtype {dtype} on {self.device}')
            inputs.append(value.contiguous())
        outputs = [torch.empty(shape, dtype=dtype, device=self.device) for _, shape, dtype in self.outputs]
        # All TRT launches share the context stream. Wait for input writes and allocator
        # reuse on the caller stream before enqueue; return the dependency back to it.
        # The host lock protects the mutable bindings of this single context.
        with self.enqueue_lock, torch.cuda.device(self.device):
            transfer = self.stream != caller
            if transfer:
                self.stream.wait_stream(caller)
            for name, value in zip((*self.names, *(o[0] for o in self.outputs)), (*inputs, *outputs)):
                if transfer:
                    value.record_stream(self.stream)
                if not self.context.set_tensor_address(name, value.data_ptr()):
                    raise RuntimeError(f'TensorRT could not bind {name}')
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError('TensorRT inference failed')
            if transfer:
                caller.wait_stream(self.stream)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


class EngineCache:
    def __init__(self, directory):
        try:
            import tensorrt as trt
        except ImportError as error:
            raise RuntimeError('Install requirements-tensorrt.txt in the CUDA game environment') from error
        self.trt = trt
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engines = {}

    @torch.inference_mode()
    def __call__(self, name, module, samples, outputs):
        if not samples or not all(x.is_cuda and x.device == samples[0].device for x in samples):
            raise ValueError('TensorRT requires CUDA example inputs on a single device')
        module.eval()
        example = module(*samples)
        example = (example,) if torch.is_tensor(example) else tuple(example)
        if len(example) != len(outputs):
            raise ValueError(f'Unexpected output count for {name}')
        expected = {binding: (tuple(tensor.shape), tensor.dtype) for binding, tensor in zip(outputs, example)}
        del example
        key = _cache_key(name, module, samples, outputs, self.trt.__version__)
        path = self.directory / f'{name}-{key}.engine'
        with torch.cuda.device(samples[0].device), FileLock(str(path) + '.lock'):
            runner = None
            if path.is_file():
                try:
                    runner = StaticEngine(self.trt, self.logger, path, samples, expected)
                except InvalidEngine:
                    print(f'Rebuilding unreadable TensorRT engine {name}', file=sys.stderr, flush=True)
            cached = runner is not None
            if runner is None:
                print(f'Building TensorRT engine {name} for {torch.cuda.get_device_name(samples[0].device)}',
                      file=sys.stderr, flush=True)
                with tempfile.TemporaryDirectory(dir=self.directory) as tmp:
                    temporary = Path(tmp) / 'new.engine'
                    self._build(module, samples, outputs, temporary)
                    runner = StaticEngine(self.trt, self.logger, temporary, samples, expected)
                    os.replace(temporary, path)
        self.engines[name] = {'path': str(path), 'cache_hit': cached, 'tensorrt': self.trt.__version__}
        return runner

    def _build(self, module, samples, outputs, destination):
        trt = self.trt
        onnx_path = destination.with_suffix('.onnx')
        torch.onnx.export(module, samples, str(onnx_path), dynamo=False, opset_version=18,
                          input_names=[f'input_{i}' for i in range(len(samples))],
                          output_names=list(outputs))
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(network, self.logger)
        if not parser.parse_from_file(str(onnx_path)):
            messages = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f'TensorRT could not parse the gameplay model:\n{messages}')
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        config.builder_optimization_level = 2
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError('TensorRT could not build the gameplay engine')
        destination.write_bytes(bytes(serialized))
