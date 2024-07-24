from ultralytics.nn.modules.block import C2f
import torch

def fine_grained_prune(tensor: torch.Tensor, sparsity : float) -> torch.Tensor:
    """
    magnitude-based pruning for single tensor
    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    :return:
        torch.(cuda.)Tensor, mask for zeros
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()
    num_zeros = round(num_elements * sparsity) # calculate ideal #zero element in tensor
    # print('number of zeros', num_zeros)
    importance = tensor.abs() # make negative value becomes positive
    # print('importance',importance)
    #The view(-1) method reshapes the importance tensor into a one-dimensional tensor
    # The kthvalue() method returns a tuple containing the kth smallest value in the input tensor and its index.
    #.     The num_zeros parameter specifies the value of k.

    # The values attribute of the tuple returned by kthvalue() contains the kth(#num_zeros) smallest value in the input tensor.

    threshold = importance.view(-1).kthvalue(num_zeros).values
    # print('threshold',threshold)
    mask = torch.gt(importance, threshold) #if the value > treshold, then set to True
    # print('mask',mask)
    tensor.mul_(mask)
    # print('mul_mask',tensor.mul_(mask))
    return mask


class FineGrainedPruner:
    def __init__(self, model, sparsity_dict):
        self.masks = FineGrainedPruner.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        masks = dict()
        for module_name, module in model.named_modules():
            if isinstance(module, C2f):
                for name, param in model.named_parameters():
                    if param.dim() > 1: # we only prune conv and fc weights
                        if isinstance(sparsity_dict, dict):
                            masks[name] = fine_grained_prune(param, sparsity_dict[name])
                        else:
                            assert(sparsity_dict < 1 and sparsity_dict >= 0) # prune every layer
                            if sparsity_dict > 0:
                                masks[name] = fine_grained_prune(param, sparsity_dict)
        return masks