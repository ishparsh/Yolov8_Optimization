import torch
from torch import nn
from ultralytics.nn.modules.block import C2f, SPPF, Conv
import copy


all_convs = []
all_bns = []

def get_output_channel_importance(weight):
    out_channels = weight.shape[0]
    importances = []
    # compute the importance for each output channel
    for o_c in range(out_channels):
        channel_weight = weight.detach()[o_c]
        importance = torch.norm(channel_weight, p=2)
        importances.append(importance.view(1))
    return torch.cat(importances)

def extract_and_print_conv_layers(module_name,module, indent=0):
    for child_name, child in module.named_children():
            full_name = f"{module_name}.{child_name}" if module_name else child_name

            if isinstance(child, nn.Conv2d):
                 all_convs.append((full_name,child))
            #print(' ' * indent + f'{child_name} : {child} (Conv2d layer)')
            # print(' ' * (indent + 2) + f'out_channels: {out_channels}')
            elif isinstance(child, nn.BatchNorm2d):
            # print(' ' * indent + f'{child_name} : {child} (Batch layer)')
                all_bns.append((full_name,child))
            else:
        # #     Recursively handle nested structures
                extract_and_print_conv_layers(full_name,child, indent + 2)
        

@torch.no_grad()
def sort_model(model: nn.Module) -> nn.Module:
    
    """Apply channel pruning to each of the conv layer in the backbone
    Note that for prune_ratio, we can either provide a floating-point number,
    indicating that we use a uniform pruning rate for all layers, or a list of
    numbers to indicate per-layer pruning rate.
    """
    # sanity check of provided prune_ratio
    
    pruned_model = copy.deepcopy(model)
    for module_name, module in pruned_model.named_modules():
        if (isinstance(module, C2f) or isinstance(module,SPPF) or isinstance(module,Conv)):
            extract_and_print_conv_layers(module_name,module, indent = 2)

    # iterate through conv layers
    for i_conv in range(len(all_convs) - 1):
        # each channel sorting index, we need to apply it to:
        # - the output dimension of the previous conv
        # - the previous BN layer
        # - the input dimension of the next conv (we compute importance here)
        _,curr_conv = all_convs[i_conv]
        _,curr_bn = all_bns[i_conv]
        _,next_conv = all_convs[i_conv + 1]
        # note that we always compute the importance according to input channels
        # Note that we compute the importance according to output channels of the current conv
        importance = get_output_channel_importance(curr_conv.weight)
        # Sorting from large to small
        sort_idx = torch.argsort(importance, descending=True)
        #sort_idx = torch.argsort(importance, descending=True)
        
        # Apply to current conv and its following bn
        # curr_conv.weight.data = torch.index_select(curr_conv.weight.detach(), 0, sort_idx)

        curr_conv_weight = getattr(curr_conv, 'weight')
        sorted_weight = torch.index_select(curr_conv_weight.detach(), 0, sort_idx)
        sorted_weight_param = nn.Parameter(sorted_weight)
        setattr(curr_conv, 'weight', sorted_weight_param)
        for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
            tensor_to_apply = getattr(curr_bn, tensor_name)
            sorted_weight = torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
            sorted_weight_param = nn.Parameter(sorted_weight)
            setattr(curr_bn, tensor_name, sorted_weight_param)


    return model
