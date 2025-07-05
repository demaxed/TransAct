"""
TransAct: Transformer-based Realtime User Action Model

This module implements the core TransAct model architecture as described in the Pinterest paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class ActionEmbedding(nn.Module):
    """Embedding layer for user actions."""
    
    def __init__(self, num_actions: int, action_dim: int, dropout: float = 0.1):
        super().__init__()
        self.action_embedding = nn.Embedding(num_actions, action_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(action_dim)
    
    def forward(self, action_ids: torch.Tensor) -> torch.Tensor:
        # action_ids: (batch_size, seq_len)
        embedded = self.action_embedding(action_ids)  # (batch_size, seq_len, action_dim)
        embedded = self.layer_norm(embedded)
        embedded = self.dropout(embedded)
        return embedded


class TransformerEncoder(nn.Module):
    """Transformer encoder for processing user action sequences."""
    
    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 num_layers: int, 
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu"):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.pos_encoding = PositionalEncoding(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Apply transformer encoder
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        return output


class MultiTaskPredictionHead(nn.Module):
    """Multi-task prediction head for different user actions."""
    
    def __init__(self, 
                 input_dim: int, 
                 num_actions: int, 
                 hidden_dims: List[int] = [512, 256],
                 dropout: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_actions))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TransActModel(nn.Module):
    """
    TransAct: Transformer-based Realtime User Action Model for Recommendation.
    
    This model combines real-time user action sequences with batch user representations
    for improved recommendation performance.
    """
    
    def __init__(self,
                 num_actions: int = 3,  # click, repin, hide
                 action_dim: int = 64,
                 user_dim: int = 128,
                 item_dim: int = 256,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 max_seq_len: int = 50,
                 dropout: float = 0.1,
                 label_weight_matrix: Optional[Dict] = None):
        super().__init__()
        
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Action embedding layer
        self.action_embedding = ActionEmbedding(num_actions, action_dim, dropout)
        
        # Transformer encoder for real-time action sequences
        self.transformer = TransformerEncoder(
            d_model=action_dim,
            nhead=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # User and item embedding layers
        self.user_embedding = nn.Embedding(1000000, user_dim)  # Large user vocabulary
        self.item_embedding = nn.Embedding(10000000, item_dim)  # Large item vocabulary
        
        # Feature fusion layers
        self.feature_fusion = nn.Sequential(
            nn.Linear(action_dim + user_dim + item_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # Multi-task prediction heads
        self.prediction_head = MultiTaskPredictionHead(
            input_dim=hidden_dim,
            num_actions=num_actions,
            dropout=dropout
        )
        
        # Label weight matrix for multi-task learning
        if label_weight_matrix is None:
            # Default weights based on paper
            self.label_weight_matrix = torch.tensor([
                [100, 0, 100],    # click weights
                [0, 100, 100],    # repin weights  
                [1, 5, 10]        # hide weights
            ], dtype=torch.float32)
        else:
            self.label_weight_matrix = torch.tensor([
                [label_weight_matrix['click']['click'], 
                 label_weight_matrix['click']['repin'], 
                 label_weight_matrix['click']['hide']],
                [label_weight_matrix['repin']['click'], 
                 label_weight_matrix['repin']['repin'], 
                 label_weight_matrix['repin']['hide']],
                [label_weight_matrix['hide']['click'], 
                 label_weight_matrix['hide']['repin'], 
                 label_weight_matrix['hide']['hide']]
            ], dtype=torch.float32)
    
    def forward(self, 
                action_sequence: torch.Tensor,
                user_ids: torch.Tensor,
                item_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the TransAct model.
        
        Args:
            action_sequence: (batch_size, seq_len) - User action sequence
            user_ids: (batch_size,) - User IDs
            item_ids: (batch_size,) - Item IDs to predict for
            attention_mask: (batch_size, seq_len) - Mask for padding
            
        Returns:
            predictions: (batch_size, num_actions) - Action probabilities
        """
        batch_size, seq_len = action_sequence.shape
        
        # 1. Embed actions
        action_embeddings = self.action_embedding(action_sequence)  # (batch_size, seq_len, action_dim)
        
        # 2. Process through transformer
        transformer_output = self.transformer(action_embeddings, attention_mask)
        
        # 3. Pool sequence (use mean pooling or last token)
        if attention_mask is not None:
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(transformer_output)
            transformer_output = transformer_output.masked_fill(mask_expanded, 0)
            seq_lengths = (~attention_mask).sum(dim=1, keepdim=True).float()
            pooled_output = transformer_output.sum(dim=1) / seq_lengths
        else:
            # Use last token
            pooled_output = transformer_output[:, -1, :]  # (batch_size, action_dim)
        
        # 4. Get user and item embeddings
        user_embeddings = self.user_embedding(user_ids)  # (batch_size, user_dim)
        item_embeddings = self.item_embedding(item_ids)  # (batch_size, item_dim)
        
        # 5. Concatenate features
        combined_features = torch.cat([
            pooled_output,  # (batch_size, action_dim)
            user_embeddings,  # (batch_size, user_dim)
            item_embeddings   # (batch_size, item_dim)
        ], dim=1)  # (batch_size, action_dim + user_dim + item_dim)
        
        # 6. Feature fusion
        fused_features = self.feature_fusion(combined_features)  # (batch_size, hidden_dim)
        
        # 7. Multi-task prediction
        logits = self.prediction_head(fused_features)  # (batch_size, num_actions)
        
        return logits
    
    def compute_loss(self, 
                    logits: torch.Tensor, 
                    labels: torch.Tensor,
                    device: torch.device) -> torch.Tensor:
        """
        Compute weighted multi-task loss.
        
        Args:
            logits: (batch_size, num_actions) - Model predictions
            labels: (batch_size, num_actions) - Ground truth labels
            device: Device to compute on
            
        Returns:
            loss: Scalar loss value
        """
        # Move weight matrix to device
        weight_matrix = self.label_weight_matrix.to(device)
        
        # Compute cross-entropy loss for each action
        losses = []
        for i in range(self.num_actions):
            action_loss = F.binary_cross_entropy_with_logits(
                logits[:, i], 
                labels[:, i].float(),
                reduction='none'
            )
            
            # Apply head weighting
            weights = torch.zeros_like(labels[:, i])
            for j in range(self.num_actions):
                weights += weight_matrix[i, j] * labels[:, j]
            
            weighted_loss = (action_loss * weights).mean()
            losses.append(weighted_loss)
        
        return torch.stack(losses).mean()
    
    def predict(self, 
               action_sequence: torch.Tensor,
               user_ids: torch.Tensor,
               item_ids: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Make predictions with the model.
        
        Args:
            action_sequence: (batch_size, seq_len) - User action sequence
            user_ids: (batch_size,) - User IDs
            item_ids: (batch_size,) - Item IDs to predict for
            attention_mask: (batch_size, seq_len) - Mask for padding
            
        Returns:
            probabilities: (batch_size, num_actions) - Action probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(action_sequence, user_ids, item_ids, attention_mask)
            probabilities = torch.sigmoid(logits)
        return probabilities


class HybridRankingModel(nn.Module):
    """
    Hybrid ranking model that combines TransAct with batch user representations.
    
    This implements the hybrid approach described in the paper that combines
    real-time user action signals with batch user representations.
    """
    
    def __init__(self,
                 transact_model: TransActModel,
                 batch_user_dim: int = 128,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.transact_model = transact_model
        
        # Batch user representation processing
        self.batch_user_projection = nn.Sequential(
            nn.Linear(batch_user_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # Final ranking layer
        self.ranking_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self,
                action_sequence: torch.Tensor,
                user_ids: torch.Tensor,
                item_ids: torch.Tensor,
                batch_user_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the hybrid ranking model.
        
        Args:
            action_sequence: (batch_size, seq_len) - Real-time user actions
            user_ids: (batch_size,) - User IDs
            item_ids: (batch_size,) - Item IDs
            batch_user_features: (batch_size, batch_user_dim) - Batch user representations
            attention_mask: (batch_size, seq_len) - Mask for padding
            
        Returns:
            ranking_scores: (batch_size, 1) - Final ranking scores
        """
        # Get TransAct features
        transact_logits = self.transact_model(
            action_sequence, user_ids, item_ids, attention_mask
        )
        
        # Process batch user features
        batch_user_processed = self.batch_user_projection(batch_user_features)
        
        # Combine features
        combined_features = torch.cat([
            transact_logits,
            batch_user_processed
        ], dim=1)
        
        # Final ranking score
        ranking_scores = self.ranking_layer(combined_features)
        
        return ranking_scores 