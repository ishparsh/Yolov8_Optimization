import torch
from ultralytics import YOLO
from torch.nn.utils import prune

# Load the model
model = YOLO('yolov8m.pt')

# Prune the model
for name, module in model.named_modules():
    if 'conv' in name:
        prune.l1_unstructured(module, name='weight', amount=0.5)
        prune.remove(module, 'weight')
        if module.bias is not None:
            prune.l1_unstructured(module, name='bias', amount=0.5)
            prune.remove(module, 'bias')

# Save the pruned model
torch.save(model.state_dict(), 'yolo_pruned_with_pytorch.pth')

# Load the pruned model
pruned_model = YOLO('yolov8m.pt')
pruned_model.load_state_dict(torch.load('yolo_pruned_with_pytorch.pth'))

# Export to ONNX
pruned_model.export(format='onnx', save_path='pruned_model.onnx')