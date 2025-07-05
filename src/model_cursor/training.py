"""
Training components for TransAct model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from .transact import TransActModel
from .data import UserActionDataLoader


class TransActTrainer:
    """
    Trainer class for TransAct model with comprehensive training functionality.
    """
    
    def __init__(self,
                 model: TransActModel,
                 train_loader: UserActionDataLoader,
                 val_loader: Optional[UserActionDataLoader] = None,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 optimizer: str = 'adam',
                 scheduler: str = 'cosine',
                 device: str = 'auto',
                 save_dir: str = './checkpoints'):
        """
        Initialize the trainer.
        
        Args:
            model: TransAct model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            optimizer: Optimizer type ('adam', 'adamw', 'sgd')
            scheduler: Learning rate scheduler ('step', 'cosine', 'none')
            device: Device to train on ('auto', 'cpu', 'cuda')
            save_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer
        self.scheduler_type = scheduler
        self.save_dir = save_dir
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup scheduler
        self._setup_scheduler()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def _setup_optimizer(self):
        """Setup the optimizer."""
        if self.optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")
    
    def _setup_scheduler(self):
        """Setup the learning rate scheduler."""
        if self.scheduler_type.lower() == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif self.scheduler_type.lower() == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        elif self.scheduler_type.lower() == 'none':
            self.scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_type}")
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Metrics tracking
        predictions = []
        targets = []
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            action_sequence = batch['action_sequence'].to(self.device)
            user_ids = batch['user_id'].to(self.device)
            item_ids = batch['item_id'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(action_sequence, user_ids, item_ids, attention_mask)
            
            # Compute loss
            loss = self.model.compute_loss(logits, labels, self.device)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Store predictions and targets for metrics
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                predictions.append(probs.cpu().numpy())
                targets.append(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Compute average loss
        avg_loss = total_loss / num_batches
        
        # Compute metrics
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        metrics = self._compute_metrics(predictions, targets)
        
        return avg_loss, metrics
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        if self.val_loader is None:
            return 0.0, {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Metrics tracking
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                action_sequence = batch['action_sequence'].to(self.device)
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                logits = self.model(action_sequence, user_ids, item_ids, attention_mask)
                
                # Compute loss
                loss = self.model.compute_loss(logits, labels, self.device)
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and targets
                probs = torch.sigmoid(logits)
                predictions.append(probs.cpu().numpy())
                targets.append(labels.cpu().numpy())
        
        # Compute average loss
        avg_loss = total_loss / num_batches
        
        # Compute metrics
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        metrics = self._compute_metrics(predictions, targets)
        
        return avg_loss, metrics
    
    def _compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Model predictions (batch_size, num_actions)
            targets: Ground truth labels (batch_size, num_actions)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Convert to binary predictions
        binary_preds = (predictions > 0.5).astype(int)
        
        # Per-action metrics
        action_names = ['click', 'repin', 'hide']
        for i, action_name in enumerate(action_names):
            # Precision, Recall, F1
            tp = np.sum((binary_preds[:, i] == 1) & (targets[:, i] == 1))
            fp = np.sum((binary_preds[:, i] == 1) & (targets[:, i] == 0))
            fn = np.sum((binary_preds[:, i] == 0) & (targets[:, i] == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[f'{action_name}_precision'] = precision
            metrics[f'{action_name}_recall'] = recall
            metrics[f'{action_name}_f1'] = f1
            
            # AUC (simplified)
            pos_preds = predictions[targets[:, i] == 1, i]
            neg_preds = predictions[targets[:, i] == 0, i]
            
            if len(pos_preds) > 0 and len(neg_preds) > 0:
                auc = self._compute_auc(pos_preds, neg_preds)
                metrics[f'{action_name}_auc'] = auc
        
        # Overall metrics
        metrics['overall_accuracy'] = np.mean(binary_preds == targets)
        metrics['overall_f1'] = np.mean([metrics[f'{action}_f1'] for action in action_names])
        
        return metrics
    
    def _compute_auc(self, pos_preds: np.ndarray, neg_preds: np.ndarray) -> float:
        """Compute AUC using simplified method."""
        # Simplified AUC computation
        pos_mean = np.mean(pos_preds)
        neg_mean = np.mean(neg_preds)
        
        if pos_mean > neg_mean:
            return 0.5 + 0.5 * (pos_mean - neg_mean)
        else:
            return 0.5 - 0.5 * (neg_mean - pos_mean)
    
    def train(self, 
              num_epochs: int,
              save_best: bool = True,
              patience: int = 10,
              min_delta: float = 1e-4) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_best: Whether to save best model
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_metrics = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)
            
            # Validation
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Train F1: {train_metrics.get('overall_f1', 0):.4f}")
            print(f"Val F1: {val_metrics.get('overall_f1', 0):.4f}")
            
            # Save best model
            if save_best and val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pth')
                print("Saved best model!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'epoch': len(self.train_losses)
        }
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_metrics = checkpoint['train_metrics']
        self.val_metrics = checkpoint['val_metrics']
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # F1 score plot
        if self.train_metrics and self.val_metrics:
            train_f1 = [m.get('overall_f1', 0) for m in self.train_metrics]
            val_f1 = [m.get('overall_f1', 0) for m in self.val_metrics]
            
            axes[0, 1].plot(train_f1, label='Train F1')
            axes[0, 1].plot(val_f1, label='Val F1')
            axes[0, 1].set_title('Training and Validation F1 Score')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Per-action metrics
        if self.val_metrics:
            action_names = ['click', 'repin', 'hide']
            for i, action in enumerate(action_names):
                action_f1 = [m.get(f'{action}_f1', 0) for m in self.val_metrics]
                axes[1, i].plot(action_f1, label=f'{action.capitalize()} F1')
                axes[1, i].set_title(f'{action.capitalize()} F1 Score')
                axes[1, i].set_xlabel('Epoch')
                axes[1, i].set_ylabel('F1 Score')
                axes[1, i].legend()
                axes[1, i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show() 