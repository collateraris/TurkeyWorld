"""Five ~1M private dynamics experts, preserving their pretrained functions."""
from copy import deepcopy

import torch
from torch import nn


PROTOCOL='rssm_five_expanded_private_experts_v1'
PREFIX='vjepa_backend.five_role_fusion.'
SETTINGS=dict(protocol=PROTOCOL,version=1,experts=5,width=256,heads=4,
    legacy_hidden=512,private_hidden=288,legacy_parameters_per_expert=527104,
    added_parameters_per_expert=477984,parameters_per_expert=1005088,
    added_parameters=2389920,added_parameter_tensors=70,
    private_initialization_seeds=[173,174,175,176,177],
    initialization='legacy_plus_zero_output_private_Transformer',
    private_Transformer_blocks_per_expert=2,Source_future_inputs=False,
    new_recurrent_memory=False)


class ExpandedDynamicsExpert(nn.Module):
    def __init__(self,legacy,*,seed):
        super().__init__()
        if (not isinstance(legacy,nn.TransformerEncoderLayer)
                or legacy.self_attn.embed_dim!=256 or legacy.self_attn.num_heads!=4
                or not legacy.self_attn.batch_first or not legacy.norm_first
                or legacy.linear1.out_features!=512 or legacy.linear2.out_features!=256
                or legacy.self_attn.dropout!=0 or legacy.activation_relu_or_gelu!=2
                or any(getattr(legacy,k).p!=0 for k in('dropout','dropout1','dropout2'))):
            raise ValueError('Expanded dynamics experts require the intact d256/FF512 private Transformer')
        self.legacy=legacy
        parameter=next(legacy.parameters());devices=[parameter.device.index]if parameter.is_cuda else[]
        with torch.random.fork_rng(devices=devices):
            if parameter.device.type!='meta':torch.random.default_generator.manual_seed(seed)
            factory=dict(device=parameter.device,dtype=parameter.dtype)
            self.private_transformer=nn.TransformerEncoderLayer(256,4,dim_feedforward=288,
                dropout=0.,activation='gelu',batch_first=True,norm_first=True,**factory)
            self.private_output=nn.Linear(256,256,**factory)
            nn.init.zeros_(self.private_output.weight);nn.init.zeros_(self.private_output.bias)
        self.private_transformer.train(legacy.training);self.private_output.train(legacy.training)
        self.training=legacy.training

    def forward(self,src,src_mask=None,src_key_padding_mask=None,is_causal=False):
        old=self.legacy(src,src_mask=src_mask,src_key_padding_mask=src_key_padding_mask,is_causal=is_causal)
        extra=self.private_transformer(src,src_mask=src_mask,src_key_padding_mask=src_key_padding_mask,is_causal=is_causal)
        return old+self.private_output(extra)


def expanded_five_metadata(module):
    if len(module.experts)!=5 or not all(isinstance(e,ExpandedDynamicsExpert)for e in module.experts):
        raise ValueError('Expanded dynamics metadata requires all five private wrappers')
    rename={};added=[]
    for i,expert in enumerate(module.experts):
        for name,_ in expert.legacy.named_parameters():rename[f'experts.{i}.{name}']=f'experts.{i}.legacy.{name}'
        added.extend(f'experts.{i}.{name}'for name,_ in expert.named_parameters()if not name.startswith('legacy.'))
        if sum(p.numel()for p in expert.parameters())!=SETTINGS['parameters_per_expert']:
            raise ValueError('Expanded dynamics expert capacity changed')
    return {**deepcopy(SETTINGS),'legacy_parameter_name_map':rename,'added_parameter_names':added,
        'total_fusion_parameters':sum(p.numel()for p in module.parameters())}


def expand_five_experts(module):
    if (getattr(module,'width',None)!=256 or getattr(module,'heads',None)!=4
            or getattr(module,'expert_hidden',None)!=512 or len(module.experts)!=5):
        raise ValueError('One-million dynamics experts require the existing five d256/FF512 role blocks')
    wrapped=[isinstance(e,ExpandedDynamicsExpert)for e in module.experts]
    if any(wrapped):
        if not all(wrapped):raise ValueError('Cannot activate a partially expanded dynamics expert bank')
        return expanded_five_metadata(module)
    replacements=[ExpandedDynamicsExpert(e,seed=seed)for e,seed in zip(module.experts,SETTINGS['private_initialization_seeds'])]
    for i,expert in enumerate(replacements):module.experts[i]=expert
    return expanded_five_metadata(module)


def legacy_five_named_parameters(module):
    """The actual old namespace over the preserved legacy Parameter objects."""
    metadata=expanded_five_metadata(module)
    reverse={new:old for old,new in metadata['legacy_parameter_name_map'].items()}
    added=set(metadata['added_parameter_names'])
    return [(reverse.get(name,name),p)for name,p in module.named_parameters()if name not in added]
