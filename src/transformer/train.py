import json
import os
from typing import Callable, Any
from tqdm import tqdm
from timeit import default_timer as timer
from dataclasses import dataclass, field

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader
import pandas as pd  # Add this import for smoothing

from transformer.utils import Batch, Model

@dataclass
class Tracker:
  steps: list[int] = field(default_factory=list)
  values: list = field(default_factory=list)
  
  def add(self, step: int, value):
    self.steps.append(step)
    self.values.append(value)
    
  def mean(self, cutoff: int|None = None):
    return np.mean(self.values[-cutoff:] if cutoff is not None else self.values)

@dataclass
class CosineLRScheduler:
  optimizer: torch.optim.Optimizer
  warmup_steps: int
  total_steps: int
  min_lr: float = 1e-7
  max_lr: float = 1e-5

  def step(self, step: int):
    if step < self.warmup_steps:
      lr = self.max_lr * (step / self.warmup_steps)
    elif step > self.total_steps:
      lr = self.min_lr
    else:
      lr = self.min_lr + (self.max_lr - self.min_lr) * \
           0.5 * (1 + np.cos(np.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
      
    for param_group in self.optimizer.param_groups:
      param_group['lr'] = lr
    return lr

@dataclass
class Trainer:
  model: Model
  train_loader: DataLoader[Batch]
  val_loader: DataLoader[Batch]
  device: torch.device
  max_lr: float = 5e-5
  min_lr: float = 1e-6 
  weight_decay: float = 1e-2
  warmup_steps: int = 0
  n_epochs: int = 3
  log_steps: int|None = 5                     # how many batches between logging
  eval_steps: int|None = 1000                 # how often to evaluate on validation set
  save_steps: int|None = None                 # how often to save checkpoint
  checkpoint_dir: str = '_out'                # where to save checkpoints
  max_eval_batches: int|None = None           # maximum number of batches to run eval on
  max_epochs_without_improvement: int = 5     # how many epochs to wait before early stopping if no val loss improvement
  use_mixed_precision: bool = False           # whether to use mixed precision (FP16) for training
  memory_format: torch.memory_format|None = None
  custom_optimizer: torch.optim.Optimizer|None = None
  custom_eval: Callable[[Model], dict[str, float]|None] | None = None

  def __post_init__(self):
    self.train_losses = Tracker()
    self.val_losses = Tracker()
    self.batch_times = Tracker()
    self.metrics = Tracker()
    self.optimizer: Optimizer = self.custom_optimizer or AdamW(self.model.parameters(), lr=0, weight_decay=self.weight_decay)
    self.lr_scheduler = CosineLRScheduler(
      optimizer=self.optimizer,
      warmup_steps=self.warmup_steps,
      total_steps=self.n_epochs * len(self.train_loader),
      min_lr=self.min_lr,
      max_lr=self.max_lr
    )
    self.lr = self.lr_scheduler.step(0) # set initial learning rate
    self.step_count = 0 # number of batches processed so far
    self.best_val_loss = float('inf') # best validation loss so far
    self.n_epochs_without_improvement = 0 # number of epochs (can be float) for which val loss has not decreased
    self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_mixed_precision) # type: ignore
  
  def steps_to_epochs(self, steps: int):
    return steps / len(self.train_loader)
    
  def save_checkpoint(self, cp_dir: str, to_cpu=False):
    os.makedirs(cp_dir, exist_ok=True)
    torch.save(self.model.to('cpu' if to_cpu else self.device).state_dict(), cp_dir + '/model.pth')
    torch.save(self.optimizer.state_dict(), cp_dir + '/optimizer.pth')
    with open (cp_dir + '/train-state.json', 'w') as f:
      json.dump({
        'step_count': self.step_count,
        'train_losses': self.train_losses.__dict__,
        'val_losses': self.val_losses.__dict__,
        'best_val_loss': self.best_val_loss,
        'metrics': self.metrics.__dict__
      }, f)

  def load_checkpoint(self, cp_dir: str):
    self.model.load_state_dict(torch.load(cp_dir + '/model.pth', map_location=self.device))
    self.model.to(self.device)

    if os.path.exists(cp_dir + '/optimizer.pth'):
      self.optimizer.load_state_dict(torch.load(cp_dir + '/optimizer.pth', map_location=self.device, weights_only=False))
    if os.path.exists(cp_dir + '/train-state.json'):
      with open(cp_dir + '/train-state.json') as f:
        state = json.load(f)
        self.step_count = state['step_count']
        self.train_losses = Tracker(**state['train_losses'])
        self.val_losses = Tracker(**state['val_losses'])
        self.metrics = Tracker(**state['metrics'])
        self.best_val_loss = state["best_val_loss"]

  def plot(self, train_loss_smoothing: int|None=None, plot_metrics=False):
    window = train_loss_smoothing or self.log_steps or 1
    smoothed_train_losses = pd.Series(self.train_losses.values).rolling(window).mean()
    plt.plot(self.train_losses.steps, smoothed_train_losses, label="train loss")
    if self.val_losses.steps:
      plt.plot(self.val_losses.steps, self.val_losses.values, label="val loss")
    plt.legend(); plt.xlabel("steps"); plt.show()
    
    if plot_metrics and self.metrics.values:
      names: list[str] = list(self.metrics.values[0].keys())
      for name in names:
        plt.plot(self.metrics.steps, [ms[name] for ms in self.metrics.values], label=name)
      plt.legend(); plt.xlabel("steps"); plt.show()

  @torch.no_grad()
  def eval(self, data_loader: DataLoader|None=None, verbose=True, max_batches:int|None=None):
    data_loader = data_loader or self.val_loader
    self.model.eval()
    n_batches = min(len(data_loader), max_batches or len(data_loader))
    total_loss = 0
    for i, batch in enumerate(tqdm(data_loader, desc="Evaluating", total=n_batches, disable=not verbose)):
      if i >= n_batches: break
      batch = batch.to(self.device)
      if self.memory_format is not None:
        batch.x = batch.x.to(memory_format=self.memory_format)
      with torch.autocast(enabled=self.use_mixed_precision, device_type=self.device.type, dtype=torch.float16):
        loss = self.model.get_output(batch).loss
        total_loss += loss.item()

    loss = total_loss / n_batches
    self.model.train()
    return loss

  def train(self):
    self.model.train()

    for epoch in range(self.n_epochs):
      if self.n_epochs_without_improvement > self.max_epochs_without_improvement:
        print(f"Stopping early because validation loss has not improved for {self.n_epochs_without_improvement} epochs")
        break 
      
      for batch in self.train_loader:
        step_count = self.step_count
        self.lr = self.lr_scheduler.step(step_count)
        t0 = timer()

        self.optimizer.zero_grad()
        batch = batch.to(self.device)
        if self.memory_format is not None:
          batch.x = batch.x.to(memory_format=self.memory_format)
        with torch.autocast(enabled=self.use_mixed_precision, device_type=self.device.type, dtype=torch.float16):
          loss = self.model.get_output(batch).loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.device.type == 'cuda': torch.cuda.synchronize()
        t1 = timer()
        self.batch_times.add(step_count, t1 - t0)
        self.train_losses.add(step_count, loss.item())

        # log
        if step_count > 0 and self.log_steps and step_count % self.log_steps == 0:
          epoch_progress = self.steps_to_epochs(step_count)
          train_loss = self.train_losses.mean(cutoff=self.log_steps)
          steps_per_sec = 1 / self.batch_times.mean(cutoff=self.log_steps)
          mins_per_epoch = len(self.train_loader) / steps_per_sec / 60
          print(f"step: {step_count:6d} ({epoch_progress:<4.2f} epochs) | train loss: {train_loss:4.4f} | lr: {self.lr:.2e} | " + \
                f"steps/s: {steps_per_sec:4.1f} ({mins_per_epoch:.2f} mins/epoch)")

        # eval
        if step_count > 0 and self.eval_steps and step_count % self.eval_steps == 0:
          t0 = timer()
          loss = self.eval(verbose=False, max_batches=self.max_eval_batches)
          metrics = None
          if self.custom_eval:
            metrics = self.custom_eval(self.model)
            if metrics is not None: self.metrics.add(step_count, metrics)
          eval_time = timer() - t0
          self.val_losses.add(step_count, loss)
          print(f"EVAL: {step_count:6d} | val loss: {loss:4.4f} | eval time: {eval_time:4.2f}s" + \
                f"|  {metrics}") if metrics else ""

        # save
        if step_count > 0 and self.save_steps and step_count % self.save_steps == 0:
          last_val_loss = self.val_losses.values[-1] if self.val_losses.values else float('inf')

          if last_val_loss < self.best_val_loss:
            self.best_val_loss = last_val_loss
            print(f"saving checkpoint to {self.checkpoint_dir}")
            self.save_checkpoint(self.checkpoint_dir)
            self.n_epochs_without_improvement = 0
          else:
            self.n_epochs_without_improvement += self.steps_to_epochs(self.save_steps)
            print(f"not saving because last val loss is higher than best checkpoint ({last_val_loss:.4f} > {self.best_val_loss:.4f})" + \
                  f"({self.n_epochs_without_improvement:.2f} / {self.max_epochs_without_improvement:.2f} epochs without improvement)")

        self.step_count += 1