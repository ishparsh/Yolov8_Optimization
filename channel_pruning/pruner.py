from . import transfer_weights as tw
from ultralytics.nn.modules.block import C2f, Conv, SPPF
from ultralytics.nn.modules.head import Detect
from .update_model import Detect_v2
import copy

def get_num_channels_to_keep(channels: int, prune_ratio: float) -> int:
    """A function to calculate the number of layers to PRESERVE after pruning
    Note that preserve_rate = 1. - prune_ratio
    """
    return int(round((1-prune_ratio)*channels))

def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add

def prune_layer(model, p_ratio):
    pruned_model = copy.deepcopy(model)
    prev_channels = None
    sppf_addtional_channel = None
    repeat = 0
    c2f_2_cf2 = None
    head = [18, 21 ]
    for name, child_module in pruned_model.model.model.named_children():
        if isinstance(child_module, Conv):
            repeat = 0
            in_channels = prev_channels or child_module.conv.in_channels
            out_channels =  get_num_channels_to_keep(child_module.conv.out_channels, p_ratio)
            new_conv = Conv(in_channels,out_channels,
                           k=child_module.conv.kernel_size,
                            s=child_module.conv.stride,
                             p=child_module.conv.padding,
                            g=child_module.conv.groups
                            )
            tw.transfer_conv_weights(child_module,new_conv)
            setattr(pruned_model.model.model,name,new_conv)
            prev_channels = out_channels
        elif isinstance(child_module,C2f):
            repeat = repeat + 1
            if repeat>=2:
                in_channels = c2f_2_cf2
                print(c2f_2_cf2)
            else:
                in_channels = prev_channels
            if int(name) == 18:
                in_channels = pruned_model.model.model[15].cv1.conv.in_channels
            elif int(name) == 21:
                in_channels = pruned_model.model.model[12].cv1.conv.in_channels
            out_channels = child_module.cv2.conv.out_channels
            n_keep = get_num_channels_to_keep(out_channels, p_ratio)
            shortcut = infer_shortcut(child_module.m[0])
            n=len(child_module.m)
            e=child_module.c / child_module.cv2.conv.out_channels
            c2f_v2 = C2f(in_channels,n_keep,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            tw.transfer_weights(child_module, c2f_v2,num_bottleneck =n)
            setattr(pruned_model.model.model, name, c2f_v2)
            prev_channels = n_keep
            c2f_2_cf2 = int(n_keep*0.5) * 3
            
        elif isinstance(child_module,SPPF):
            repeat = 0 
            in_channels = prev_channels
            out_channels = child_module.cv2.conv.out_channels
            n_keep = get_num_channels_to_keep(out_channels,p_ratio)
            new_sppf = SPPF(in_channels,n_keep,k=child_module.m.kernel_size)
            tw.transfer_sppf_weights(child_module,new_sppf)
            setattr(pruned_model.model.model,name,new_sppf)
            print(sppf_addtional_channel)
            prev_channels = n_keep +round(n_keep/2)
        elif isinstance(child_module,Detect):
            new_detect = Detect_v2(nc=43,ch = [64, 128, 256], prune_ratio=p_ratio)
            tw.transfer_sequential_weights(child_module,new_detect)
            setattr(pruned_model.model.model,name,new_detect)
    return pruned_model