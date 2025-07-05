"""
Data handling components for TransAct model training and evaluation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import random


class UserActionDataset(Dataset):
    """
    Dataset for user action sequences.
    
    This dataset handles the preparation of user action sequences for training
    the TransAct model, including padding, masking, and label generation.
    """
    
    def __init__(self,
                 user_sequences: List[List[int]],
                 user_ids: List[int],
                 item_ids: List[int],
                 labels: List[List[int]],
                 max_seq_len: int = 50,
                 action_mapping: Optional[Dict[str, int]] = None):
        """
        Initialize the dataset.
        
        Args:
            user_sequences: List of user action sequences
            user_ids: List of user IDs
            item_ids: List of item IDs to predict for
            labels: List of label vectors (multi-task)
            max_seq_len: Maximum sequence length
            action_mapping: Mapping from action names to IDs
        """
        self.user_sequences = user_sequences
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.labels = labels
        self.max_seq_len = max_seq_len
        
        # Default action mapping (click, repin, hide)
        if action_mapping is None:
            self.action_mapping = {
                'click': 0,
                'repin': 1, 
                'hide': 2
            }
        else:
            self.action_mapping = action_mapping
        
        # Validate data
        assert len(user_sequences) == len(user_ids) == len(item_ids) == len(labels)
        
        # Process sequences
        self.processed_sequences = []
        self.attention_masks = []
        
        for seq in user_sequences:
            # Pad or truncate sequence
            if len(seq) > max_seq_len:
                seq = seq[-max_seq_len:]  # Keep most recent actions
            elif len(seq) < max_seq_len:
                seq = seq + [0] * (max_seq_len - len(seq))  # Pad with zeros
            
            self.processed_sequences.append(seq)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            mask = [1] * min(len(seq), max_seq_len) + [0] * max(0, max_seq_len - len(seq))
            self.attention_masks.append(mask)
    
    def __len__(self) -> int:
        return len(self.user_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset."""
        return {
            'action_sequence': torch.tensor(self.processed_sequences[idx], dtype=torch.long),
            'user_id': torch.tensor(self.user_ids[idx], dtype=torch.long),
            'item_id': torch.tensor(self.item_ids[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.bool)
        }


class UserActionDataLoader:
    """
    Data loader for user action sequences with batching and shuffling.
    """
    
    def __init__(self,
                 dataset: UserActionDataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 drop_last: bool = False):
        """
        Initialize the data loader.
        
        Args:
            dataset: UserActionDataset instance
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            drop_last: Whether to drop incomplete batches
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        return {
            'action_sequence': torch.stack([item['action_sequence'] for item in batch]),
            'user_id': torch.stack([item['user_id'] for item in batch]),
            'item_id': torch.stack([item['item_id'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch])
        }
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        return len(self.dataloader)


class SyntheticDataGenerator:
    """
    Generate synthetic data for testing and development.
    
    This class creates realistic user action sequences for training and evaluation.
    """
    
    def __init__(self,
                 num_users: int = 10000,
                 num_items: int = 100000,
                 num_actions: int = 3,
                 max_seq_len: int = 50,
                 action_probs: Optional[List[float]] = None):
        """
        Initialize the synthetic data generator.
        
        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            num_actions: Number of action types
            max_seq_len: Maximum sequence length
            action_probs: Probability distribution for actions
        """
        self.num_users = num_users
        self.num_items = num_items
        self.num_actions = num_actions
        self.max_seq_len = max_seq_len
        
        # Default action probabilities (click: 0.7, repin: 0.2, hide: 0.1)
        if action_probs is None:
            self.action_probs = [0.7, 0.2, 0.1]
        else:
            self.action_probs = action_probs
        
        assert len(self.action_probs) == num_actions
        assert abs(sum(self.action_probs) - 1.0) < 1e-6
    
    def generate_user_sequences(self, 
                               num_sequences: int,
                               min_seq_len: int = 5,
                               max_seq_len: int = None) -> List[List[int]]:
        """
        Generate synthetic user action sequences.
        
        Args:
            num_sequences: Number of sequences to generate
            min_seq_len: Minimum sequence length
            max_seq_len: Maximum sequence length (overrides self.max_seq_len)
            
        Returns:
            List of user action sequences
        """
        if max_seq_len is None:
            max_seq_len = self.max_seq_len
        
        sequences = []
        for _ in range(num_sequences):
            # Random sequence length
            seq_len = random.randint(min_seq_len, max_seq_len)
            
            # Generate sequence of actions
            sequence = []
            for _ in range(seq_len):
                action = np.random.choice(self.num_actions, p=self.action_probs)
                sequence.append(action)
            
            sequences.append(sequence)
        
        return sequences
    
    def generate_labels(self, 
                       num_samples: int,
                       label_sparsity: float = 0.8) -> List[List[int]]:
        """
        Generate synthetic labels for multi-task prediction.
        
        Args:
            num_samples: Number of label vectors to generate
            label_sparsity: Probability of a label being 0 (sparse labels)
            
        Returns:
            List of label vectors
        """
        labels = []
        for _ in range(num_samples):
            # Generate sparse labels
            label_vector = []
            for _ in range(self.num_actions):
                if random.random() < label_sparsity:
                    label_vector.append(0)
                else:
                    label_vector.append(1)
            
            # Ensure at least one positive label
            if sum(label_vector) == 0:
                label_vector[random.randint(0, self.num_actions - 1)] = 1
            
            labels.append(label_vector)
        
        return labels
    
    def generate_dataset(self, 
                        num_samples: int,
                        train_split: float = 0.8,
                        val_split: float = 0.1) -> Tuple[UserActionDataset, UserActionDataset, UserActionDataset]:
        """
        Generate a complete dataset with train/val/test splits.
        
        Args:
            num_samples: Total number of samples
            train_split: Fraction for training
            val_split: Fraction for validation
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Generate sequences and labels
        sequences = self.generate_user_sequences(num_samples)
        labels = self.generate_labels(num_samples)
        
        # Generate user and item IDs
        user_ids = [random.randint(0, self.num_users - 1) for _ in range(num_samples)]
        item_ids = [random.randint(0, self.num_items - 1) for _ in range(num_samples)]
        
        # Split data
        train_size = int(num_samples * train_split)
        val_size = int(num_samples * val_split)
        test_size = num_samples - train_size - val_size
        
        # Create datasets
        train_dataset = UserActionDataset(
            sequences[:train_size],
            user_ids[:train_size],
            item_ids[:train_size],
            labels[:train_size],
            self.max_seq_len
        )
        
        val_dataset = UserActionDataset(
            sequences[train_size:train_size + val_size],
            user_ids[train_size:train_size + val_size],
            item_ids[train_size:train_size + val_size],
            labels[train_size:train_size + val_size],
            self.max_seq_len
        )
        
        test_dataset = UserActionDataset(
            sequences[train_size + val_size:],
            user_ids[train_size + val_size:],
            item_ids[train_size + val_size:],
            labels[train_size + val_size:],
            self.max_seq_len
        )
        
        return train_dataset, val_dataset, test_dataset


class BatchUserFeatures:
    """
    Handle batch user features for the hybrid ranking approach.
    
    This class manages the batch-generated user representations that are
    combined with real-time TransAct features.
    """
    
    def __init__(self, user_dim: int = 128):
        """
        Initialize batch user features.
        
        Args:
            user_dim: Dimension of user embeddings
        """
        self.user_dim = user_dim
        self.user_embeddings = {}
    
    def add_user_embedding(self, user_id: int, embedding: np.ndarray):
        """Add a user embedding."""
        assert embedding.shape[0] == self.user_dim
        self.user_embeddings[user_id] = embedding
    
    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """Get user embedding, return zeros if not found."""
        if user_id in self.user_embeddings:
            return self.user_embeddings[user_id]
        else:
            return np.zeros(self.user_dim)
    
    def get_batch_embeddings(self, user_ids: List[int]) -> np.ndarray:
        """Get embeddings for a batch of users."""
        embeddings = []
        for user_id in user_ids:
            embeddings.append(self.get_user_embedding(user_id))
        return np.array(embeddings)
    
    def save_embeddings(self, filepath: str):
        """Save embeddings to file."""
        np.save(filepath, self.user_embeddings)
    
    def load_embeddings(self, filepath: str):
        """Load embeddings from file."""
        self.user_embeddings = np.load(filepath, allow_pickle=True).item() 