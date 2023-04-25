import os
import jax
import optax
from flax.training.train_state import TrainState
from typing import Dict
import numpy as np
import orbax.checkpoint
from flax.training import orbax_utils
import datetime

from torch.utils.tensorboard import SummaryWriter

def loss_fn(state:TrainState, params:Dict, batch:tuple):
    x, y = batch
    y_pred = state.apply_fn(params, x).squeeze()
    loss = optax.l2_loss(y_pred, y).mean()
    return loss

@jax.jit  # Jit the function for efficiency
def train_step(state:TrainState, batch):
    grad_fn = jax.value_and_grad(loss_fn, argnums=1)
    loss, grads = grad_fn(state, state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit  # Jit the function for efficiency
def eval_step(state:TrainState, batch):
    # Determine the accuracy
    loss = loss_fn(state, state.params, batch)
    return loss

def train_model(state:TrainState, train_loader, val_loader, num_epochs=100, log_dir="./log", ckpt_dir="./checkpoint", hyperparams=None):
    log_dir = os.path.join(log_dir, hyperparam_str(hyperparams))
    writer = SummaryWriter(log_dir=log_dir)
    ckpt_options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=10, create=True)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, orbax_checkpointer, ckpt_options)

    # Training loop
    
    for epoch in range(num_epochs):
        train_losses = []
        val_losses = []
        # training
        for i, batch in enumerate(train_loader):
            state, loss = train_step(state, batch)
            train_losses += [float(loss)]
            print(f'TRAIN: EPOCH {epoch+1}/{num_epochs} | BATCH {i}/{len(train_loader)} | LOSS: {np.mean(train_losses)}')
        
        # validation
        for i, batch in enumerate(val_loader):
            loss = eval_step(state, batch)
            val_losses += [float(loss)]
            print(f'VAL: EPOCH {epoch+1}/{num_epochs} | BATCH {i}/{len(val_loader)} | LOSS: {np.mean(val_losses)}')
        
        # logging and checkpoint saving
        writer.add_scalar('loss/train', np.mean(train_losses), epoch)
        writer.add_scalar('loss/val', np.mean(val_losses), epoch)
        checkpoint_manager.save(epoch, state.params)
    return state

def hyperparam_str(hyperparams:Dict):
    lr = hyperparams['lr']
    layers = "_".join([str(l) for l in hyperparams['layers']])
    batch_size = hyperparams["batch_size"]
    config = f"lr:{lr}, bs:{batch_size}, lyrs:{layers}"
    if "misc" in config:
        config += hyperparams['misc']
    return config

def save(path, params):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    orbax_checkpointer.save(path, params, save_args=save_args)

def load(path):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    return orbax_checkpointer.restore(path)