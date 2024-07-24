import torch
import numpy as np 
import tqdm
from ultralytics.nn.modules.block import C2f
from utils.utils import get_model_sparsity
from fine_grained_prune import fine_grained_prune
from torch import nn



@torch.inference_mode()
def evaluate(
  model: nn.Module,
) -> float:
  
  metrics = model.val(data = 'coco8.yaml', split = 'val')
  acc = metrics.results_dict['metrics/mAP50(B)']

  return acc


@torch.no_grad()
def sensitivity_scan(model, scan_step=0.1, scan_start=0.4, scan_end=1.0, verbose=True):
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    accuracies = []
    named_conv_weights = []
    for module_name, module in model.named_modules():
        if isinstance(module, C2f):
            for param_name, param in module.named_parameters():
                if param.dim() > 1:  # Check if parameter is a weight tensor
                    named_conv_weights.append((f"{module_name}.{param_name}", param))
    for i_layer, (name, param) in enumerate(named_conv_weights):
        param_clone = param.detach().clone()
        accuracy = []
        for sparsity in tqdm(sparsities, desc=f'scanning {i_layer}/{len(named_conv_weights)} weight - {name}'):
            with torch.inference_mode():
                fine_grained_prune(param.detach(), sparsity=sparsity)
            acc = evaluate(model)
            if verbose:
                print(f'\r    sparsity={sparsity:.2f}: accuracy={acc:.2f}%', end='')
                print(f'\r Model sparsity:{get_model_sparsity(model)} ' )
            # restore
            with torch.inference_mode():
                param.copy_(param_clone)
            accuracy.append(acc)
        if verbose:
            print(f'\r    sparsity=[{",".join(["{:.2f}".format(x) for x in sparsities])}]: accuracy=[{", ".join(["{:.2f}%".format(x) for x in accuracy])}]', end='')
        accuracies.append(accuracy)
    return sparsities, accuracies

