"""Preserve optimizer moments when previously frozen renderer weights unfreeze.

Parameter IDs in an optimizer state dict are positional, not model names.
Mapping explicitly by name keeps core moments attached to their actual tensors
when newly trainable Flow parameters precede the core in registration order.
No model construction, RNG sampling or optimizer mutation occurs here.
"""
from __future__ import annotations

import copy
import re

import torch


_LEGACY_WORLD_PHASE = re.compile(r"teacher\d*(?:vq)?_world(?:_\d+)?\Z")
_FROZEN_PRETRAIN_PREFIXES = ("flow_renderer.", "vjepa_backend.renderer.")


def optimizer_parameter_names(model, optimizer, *, allow_frozen=False):
    """Return validated names for the supported single optimizer group.

    A staged trainer may retain its full AdamW parameter group while freezing
    an inactive component. ``allow_frozen`` only permits naming those existing
    model parameters; duplicate and foreign parameters remain invalid.
    """
    if len(optimizer.param_groups) != 1:
        raise ValueError("Optimizer migration supports exactly one parameter group")
    names_by_identity = {id(parameter): name for name, parameter in model.named_parameters()}
    names, seen = [], set()
    for parameter in optimizer.param_groups[0]["params"]:
        name = names_by_identity.get(id(parameter))
        if name is None:
            raise ValueError("Current optimizer contains a parameter outside the model")
        if name in seen:
            raise ValueError("Current optimizer contains duplicate parameter names")
        if not parameter.requires_grad and not allow_frozen:
            raise ValueError(f"Current optimizer contains a frozen parameter: {name}")
        seen.add(name)
        names.append(name)
    return [names]


def optimizer_state_for_trainables(checkpoint, model, optimizer):
    """Return (name-mapped optimizer state, report), preserving saved options.

    Explicit optimizer_parameter_names is preferred. A legacy checkpoint may
    omit names only in the known teacher*_world pretraining phase, whose order
    consists of every model parameter except Flow/base-renderer prefixes.
    All old parameters must still exist and be present in the new optimizer.
    Newly trainable parameters intentionally have no moment state until stepped.

    State tensors retain their original storage to avoid duplicating several
    gigabytes of AdamW moments; containers/group options are copied. Loading
    the returned state uses PyTorch's ordinary device/dtype casting behavior.
    """
    if not isinstance(checkpoint, dict):
        raise ValueError("Optimizer migration requires a checkpoint dictionary")
    saved = checkpoint.get("optimizer")
    if not isinstance(saved, dict) or not isinstance(saved.get("state"), dict):
        raise ValueError("Checkpoint is missing a valid optimizer state")
    groups = saved.get("param_groups")
    if not isinstance(groups, (list, tuple)) or len(groups) != 1 or not isinstance(groups[0], dict):
        raise ValueError("Saved optimizer must contain exactly one parameter group")
    old_group = groups[0]
    old_ids = old_group.get("params")
    if (not isinstance(old_ids, (list, tuple))
            or any(not isinstance(identifier, int) or isinstance(identifier, bool) for identifier in old_ids)
            or len(set(old_ids)) != len(old_ids)):
        raise ValueError("Saved optimizer parameter IDs must be unique integers")
    if set(saved["state"]) - set(old_ids):
        raise ValueError("Saved optimizer state references unknown parameter IDs")
    current_names = optimizer_parameter_names(model, optimizer)[0]
    current_state = optimizer.state_dict()
    new_ids = current_state["param_groups"][0]["params"]
    parameters = dict(model.named_parameters())
    explicit = checkpoint.get("optimizer_parameter_names")
    group_names = old_group.get("param_names")
    if group_names is not None and not isinstance(group_names, (list, tuple)):
        raise ValueError("Saved parameter-group names must be a list")
    phase = str(checkpoint.get("pipeline_phase", ""))
    if explicit is not None:
        if (not isinstance(explicit, (list, tuple)) or len(explicit) != 1
                or not isinstance(explicit[0], (list, tuple))):
            raise ValueError("Saved optimizer parameter names must be a single nested group")
        old_names = list(explicit[0])
        strategy = "parameter_names"
        if group_names is not None and list(group_names) != old_names:
            raise ValueError("Checkpoint and parameter-group names disagree")
    elif group_names is not None:
        old_names = list(group_names)
        strategy = "parameter_group_names"
    else:
        if not _LEGACY_WORLD_PHASE.fullmatch(phase):
            raise ValueError("Unnamed optimizer state can only be inferred for legacy teacher world pretraining")
        old_names = [name for name in parameters if not name.startswith(_FROZEN_PRETRAIN_PREFIXES)]
        strategy = "legacy_world_pretrain_order"
    if (len(old_names) != len(old_ids)
            or any(not isinstance(name, str) or not name for name in old_names)
            or len(set(old_names)) != len(old_names)):
        raise ValueError("Saved optimizer parameter names/count are invalid or duplicated")
    missing = set(old_names) - set(parameters)
    removed = set(old_names) - set(current_names)
    if missing or removed:
        raise ValueError(f"Previously optimized parameters are missing or no longer trainable: {sorted(missing | removed)}")
    saved_model = checkpoint.get("model")
    if saved_model is not None and not isinstance(saved_model, dict):
        raise ValueError("Saved model state must be a dictionary")
    for name, identifier in zip(old_names, old_ids):
        parameter = parameters[name]
        if saved_model is not None:
            old_parameter = saved_model.get(name)
            if not torch.is_tensor(old_parameter) or old_parameter.shape != parameter.shape:
                raise ValueError(f"Saved model parameter shape/name differs: {name}")
        entry = saved["state"].get(identifier, {})
        if not isinstance(entry, dict):
            raise ValueError(f"Saved optimizer moment state is invalid: {name}")
        for key, value in entry.items():
            if torch.is_tensor(value):
                if key == "step":
                    if value.numel() != 1:
                        raise ValueError(f"Optimizer step must be scalar: {name}")
                elif value.shape != parameter.shape:
                    raise ValueError(f"Optimizer moment shape differs for {name}: {key}")
    new_identifier = dict(zip(current_names, new_ids))
    migrated_state = {new_identifier[name]: dict(saved["state"][identifier])
                      for name, identifier in zip(old_names, old_ids) if identifier in saved["state"]}
    group = copy.deepcopy({key: value for key, value in old_group.items() if key not in {"params", "param_names"}})
    group["params"] = list(new_ids)
    if group_names is not None:
        group["param_names"] = list(current_names)
    old_name_set = set(old_names)
    added = [name for name in current_names if name not in old_name_set]
    report = {"strategy": strategy, "source_phase": phase, "preserved_parameters": len(old_names),
        "current_parameters": len(current_names), "added_parameters": added,
        "state_entries_preserved": len(migrated_state), "parameter_order_unchanged": old_names == current_names,
        "group_options_preserved": sorted(key for key in old_group if key not in {"params", "param_names"}),
        "learning_rate": group.get("lr"), "state_tensors_reused": True}
    return {"state": migrated_state, "param_groups": [group]}, report
