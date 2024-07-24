# Yolov8 Optimization

This repo consist ot K_means qunatization, two different kind of pruning which are Channel Pruning and Fine Grained Pruning.
The YOLOv8 model itself is a huge model and running in the resource constraint device is quite challenging to overcome this problem. We came up implementing the different kind optimization technique. Right now, only k_menas, channel pruning and fine grained pruning are implemented. Other optimzation techniques are in the process.


## Installation

```
python3 -m venv venv

```

```
source venv/bin/activate
```

```
pip install -r requirements.txt

```

## How to use

It is pretty simple, the source code are in their respective directory. Meanwhile, the implemenation can seen in Jupyter Notebook, which is quite easy to understand. And for the details, the code itself is pretty well documented.


