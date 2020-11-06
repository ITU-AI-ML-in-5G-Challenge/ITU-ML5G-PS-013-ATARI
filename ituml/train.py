import argparse
import os

from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.data import DataLoader

from ituml.dataset import ITUML5G
from ituml.models import MetaNet
from ituml.evaluation import scores

########################################################################################################################
# Weights-and-Biases logging.
########################################################################################################################

import wandb
wandb.init(project="ituml")

########################################################################################################################
# Command line arguments.
########################################################################################################################

parser = argparse.ArgumentParser(description='Pre-process and split the raw graphs dataset.')

parser.add_argument('--epochs', default=1000, type=int, help='Number of training epochs.')
parser.add_argument('--batch-size', default=32, type=int, help='Batch size.')
parser.add_argument('--learning-rate', default=0.01, type=float, help='Learning rate.')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay.')
parser.add_argument('--log-interval', default=100, type=int, help='Logging interval.')
parser.add_argument('--checkpoint-interval', default=100, type=int, help='Checkpoint interval.')
parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory.')

args = parser.parse_args()


########################################################################################################################
# Log parameters.
########################################################################################################################

wandb.config.epochs = args.epochs
wandb.config.batch_size = args.batch_size
wandb.config.learning_rate = args.learning_rate
wandb.config.weight_decay = args.weight_decay

########################################################################################################################
# Dataset.
########################################################################################################################

# Load training dataset.
dataset_train = ITUML5G('./datasets/ITUML5G/', split='train')
dataset_valid = ITUML5G('./datasets/ITUML5G/', split='valid')

# Dataset loaders.
train_loader = DataLoader(dataset_train, batch_size=args.batch_size)
valid_loader = DataLoader(dataset_valid, batch_size=1)

########################################################################################################################
# Device setup.
########################################################################################################################

# Compute device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

########################################################################################################################
# Model.
########################################################################################################################

# Network configuration.
num_node_features = dataset_train[0].x.shape[1]
num_edge_features = dataset_train[0].edge_attr.shape[1]
num_hidden = 128
wandb.config.num_hidden = num_hidden

# Create model.
model = MetaNet(num_node_features, num_edge_features, num_hidden).to(device)

# Monitor gradients and record the graph structure (+-).
wandb.watch(model)

########################################################################################################################
# Training utilities.
########################################################################################################################


def train(dataset):
    # Monitor training.
    losses = []

    # Put model in training mode!
    model.train()
    for batch in dataset:
        # Training step.
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = torch.sqrt(F.mse_loss(out.squeeze()[batch.y_mask], batch.y[batch.y_mask]))
        loss.backward()
        optimizer.step()
        # Monitoring
        losses.append(loss.item())

    # Return training metrics.
    return losses


def evaluate(dataset):
    # Monitor evaluation.
    losses = []
    rmse = []

    # Validation (1)
    model.eval()
    for batch in dataset:
        batch = batch.to(device)

        # Calculate validation losses.
        out = model(batch)
        loss = torch.sqrt(F.mse_loss(out.squeeze()[batch.y_mask], batch.y[batch.y_mask]))

        rmse_batch = scores(batch, out)

        # Metric logging.
        losses.append(loss.item())
        rmse.append(rmse_batch.item())

    return losses, rmse

########################################################################################################################
# Training loop.
########################################################################################################################

# Configuration
NUM_EPOCHS = args.epochs
LOG_INTERVAL = args.log_interval
CHECKPOINT_INTERVAL = args.checkpoint_interval
CHECKPOINT_DIR = args.checkpoint_dir

Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

# Configure optimizer.
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# Metrics recorder per epoch.
train_losses = []

valid_losses = []
valid_losses_corrected = []

# Training loop.
model.train()
for epoch in range(NUM_EPOCHS):
    # Train.
    train_epoch_losses = train(train_loader)
    valid_epoch_losses, valid_epoch_losses_corrected = evaluate(valid_loader)

    # Log training metrics.
    train_avg_loss = np.mean(train_epoch_losses)
    train_losses.append(train_avg_loss)

    # Log validation metrics.
    valid_avg_loss = np.mean(valid_epoch_losses)
    valid_losses.append(valid_avg_loss)

    valid_avg_loss_corrected = np.mean(valid_epoch_losses_corrected)
    valid_losses_corrected.append(valid_avg_loss_corrected)

    wandb.log({'epoch': epoch, 'train_loss': train_avg_loss, 'valid_loss': valid_avg_loss, 'score': valid_avg_loss_corrected})

    # Print metrics
    if epoch % LOG_INTERVAL == 0:
        print(f"epoch={epoch}, train_loss={train_avg_loss}, valid_loss={valid_avg_loss}, valid_loss*={valid_avg_loss_corrected}")

    if epoch % CHECKPOINT_INTERVAL == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_avg_loss,
        }

        checkpoint_fn = os.path.join(CHECKPOINT_DIR, f'checkpoint-{epoch}.tar')
        torch.save(checkpoint, checkpoint_fn)
        wandb.save(checkpoint_fn)


