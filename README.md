# ITU AI/ML in 5G Challenge 2020: Team ATARI

ITU AI/ML in 5G Challenge 2020 submission of Team ATARI. 

## Challenge

Challenge [website](https://www.upf.edu/web/wnrg/ai_challenge).

## Team ATARI

* Paola Soto-Arenas
* Miguel Camelo
* David Goez
* Natalia Gaviria
* Kevin Mets (*)

## Installation
Several packages need to be installed to run this repo locally. Some of those packages are included in the `requirements.txt`, but others have to be installed as it follows. Such packages are related to [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html), which is the selected framework for implementing our GNN model.

```
pip install -r requirements.txt
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-geometric
```

where `${CUDA}` and `${TORCH}` by your specific CUDA version (`cpu`, `cu92`, `cu101`, `cu102`, `cu110`) and PyTorch version (`1.4.0`, `1.5.0`, `1.6.0`, `1.7.0`), respectively. Check [Pytorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for more details on the installation. 

Additionally, [wandb](https://www.wandb.com/) is highly recommended but it is not necesary to run the repo. Wandb allows you experiment tracking, hyperparameter optimization, model and dataset versioning. However, if you are not using wandb, we recommend to modify the `train.py` to remove the lines that include it. 

## Run
```
python train.py [ARGS]

[ARGS]
--epochs, default=1000, Number of training epochs.
--batch-size, default=32, Training Batch size.
--learning-rate, default=0.01, Learning rate of Adam Optimizer.
--weight-decay, default=5e-4, Weight decay.
--log-interval, default=100, Logging interval.
--checkpoint-interval', default=100, Checkpoint interval.
--checkpoint-dir, default='checkpoints', Checkpoint directory.
```