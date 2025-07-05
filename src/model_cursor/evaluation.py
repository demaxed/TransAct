"""
Evaluation components for TransAct model.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
import time

from .transact import TransActModel
from .data import UserActionDataLoader


class TransActEvaluator:
    """
    Comprehensive evaluator for TransAct model performance.
    """
    
    def __init__(self, 
                 model: TransActModel,
                 test_loader: UserActionDataLoader,
                 device: str = 'auto'):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained TransAct model
            test_loader: Test data loader
            device: Device to evaluate on
        """
        self.model = model
        self.test_loader = test_loader
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self) -> Dict[str, float]:
        """
        Comprehensive evaluation of the model.
        
        Returns:
            Dictionary of evaluation metrics
        """
        print("Starting comprehensive evaluation...")
        
        # Get predictions
        predictions, targets, user_ids, item_ids = self._get_predictions()
        
        # Compute metrics
        metrics = {}
        
        # Overall metrics
        metrics.update(self._compute_overall_metrics(predictions, targets))
        
        # Per-action metrics
        metrics.update(self._compute_per_action_metrics(predictions, targets))
        
        # Ranking metrics
        metrics.update(self._compute_ranking_metrics(predictions, targets))
        
        # Efficiency metrics
        metrics.update(self._compute_efficiency_metrics())
        
        return metrics
    
    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions on test set."""
        predictions = []
        targets = []
        user_ids_list = []
        item_ids_list = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                # Move batch to device
                action_sequence = batch['action_sequence'].to(self.device)
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Get predictions
                logits = self.model(action_sequence, user_ids, item_ids, attention_mask)
                probs = torch.sigmoid(logits)
                
                # Store results
                predictions.append(probs.cpu().numpy())
                targets.append(labels.cpu().numpy())
                user_ids_list.append(user_ids.cpu().numpy())
                item_ids_list.append(item_ids.cpu().numpy())
        
        # Concatenate results
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        user_ids = np.concatenate(user_ids_list, axis=0)
        item_ids = np.concatenate(item_ids_list, axis=0)
        
        return predictions, targets, user_ids, item_ids
    
    def _compute_overall_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute overall performance metrics."""
        metrics = {}
        
        # Convert to binary predictions
        binary_preds = (predictions > 0.5).astype(int)
        
        # Overall accuracy
        metrics['overall_accuracy'] = np.mean(binary_preds == targets)
        
        # Overall F1 score
        overall_f1_scores = []
        for i in range(predictions.shape[1]):
            f1 = self._compute_f1_score(binary_preds[:, i], targets[:, i])
            overall_f1_scores.append(f1)
        metrics['overall_f1'] = np.mean(overall_f1_scores)
        
        # Overall AUC
        overall_auc_scores = []
        for i in range(predictions.shape[1]):
            try:
                auc = roc_auc_score(targets[:, i], predictions[:, i])
                overall_auc_scores.append(auc)
            except ValueError:
                overall_auc_scores.append(0.5)
        metrics['overall_auc'] = np.mean(overall_auc_scores)
        
        # Overall Average Precision
        overall_ap_scores = []
        for i in range(predictions.shape[1]):
            try:
                ap = average_precision_score(targets[:, i], predictions[:, i])
                overall_ap_scores.append(ap)
            except ValueError:
                overall_ap_scores.append(0.0)
        metrics['overall_ap'] = np.mean(overall_ap_scores)
        
        return metrics
    
    def _compute_per_action_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute metrics for each action type."""
        metrics = {}
        action_names = ['click', 'repin', 'hide']
        
        for i, action_name in enumerate(action_names):
            # Binary predictions
            binary_preds = (predictions[:, i] > 0.5).astype(int)
            true_labels = targets[:, i]
            
            # Basic metrics
            tp = np.sum((binary_preds == 1) & (true_labels == 1))
            fp = np.sum((binary_preds == 1) & (true_labels == 0))
            tn = np.sum((binary_preds == 0) & (true_labels == 0))
            fn = np.sum((binary_preds == 0) & (true_labels == 1))
            
            # Precision, Recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Specificity
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # AUC
            try:
                auc = roc_auc_score(true_labels, predictions[:, i])
            except ValueError:
                auc = 0.5
            
            # Average Precision
            try:
                ap = average_precision_score(true_labels, predictions[:, i])
            except ValueError:
                ap = 0.0
            
            # Store metrics
            metrics[f'{action_name}_precision'] = precision
            metrics[f'{action_name}_recall'] = recall
            metrics[f'{action_name}_f1'] = f1
            metrics[f'{action_name}_specificity'] = specificity
            metrics[f'{action_name}_auc'] = auc
            metrics[f'{action_name}_ap'] = ap
            metrics[f'{action_name}_tp'] = tp
            metrics[f'{action_name}_fp'] = fp
            metrics[f'{action_name}_tn'] = tn
            metrics[f'{action_name}_fn'] = fn
        
        return metrics
    
    def _compute_ranking_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute ranking-specific metrics."""
        metrics = {}
        
        # HIT@K metrics (simplified version)
        k_values = [1, 3, 5, 10]
        
        for k in k_values:
            hit_at_k_scores = []
            for i in range(predictions.shape[1]):  # For each action
                # Sort predictions and get top-k
                sorted_indices = np.argsort(predictions[:, i])[::-1]
                top_k_indices = sorted_indices[:k]
                
                # Check if any positive target is in top-k
                hit = np.any(targets[top_k_indices, i] == 1)
                hit_at_k_scores.append(float(hit))
            
            metrics[f'hit_at_{k}'] = np.mean(hit_at_k_scores)
        
        # NDCG@K (simplified version)
        for k in k_values:
            ndcg_at_k_scores = []
            for i in range(predictions.shape[1]):
                # Sort predictions and get top-k
                sorted_indices = np.argsort(predictions[:, i])[::-1]
                top_k_indices = sorted_indices[:k]
                
                # Compute DCG
                dcg = 0.0
                for j, idx in enumerate(top_k_indices):
                    dcg += targets[idx, i] / np.log2(j + 2)
                
                # Compute IDCG (ideal DCG)
                ideal_sorted = np.sort(targets[:, i])[::-1]
                idcg = 0.0
                for j in range(min(k, len(ideal_sorted))):
                    idcg += ideal_sorted[j] / np.log2(j + 2)
                
                # Compute NDCG
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcg_at_k_scores.append(ndcg)
            
            metrics[f'ndcg_at_{k}'] = np.mean(ndcg_at_k_scores)
        
        return metrics
    
    def _compute_efficiency_metrics(self) -> Dict[str, float]:
        """Compute model efficiency metrics."""
        metrics = {}
        
        # Model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        metrics['total_parameters'] = total_params
        metrics['trainable_parameters'] = trainable_params
        
        # Inference time
        batch_times = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                # Move batch to device
                action_sequence = batch['action_sequence'].to(self.device)
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Time inference
                start_time = time.time()
                _ = self.model(action_sequence, user_ids, item_ids, attention_mask)
                end_time = time.time()
                
                batch_time = (end_time - start_time) * 1000  # Convert to milliseconds
                batch_times.append(batch_time)
        
        metrics['avg_inference_time_ms'] = np.mean(batch_times)
        metrics['std_inference_time_ms'] = np.std(batch_times)
        metrics['min_inference_time_ms'] = np.min(batch_times)
        metrics['max_inference_time_ms'] = np.max(batch_times)
        
        return metrics
    
    def _compute_f1_score(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute F1 score."""
        tp = np.sum((predictions == 1) & (targets == 1))
        fp = np.sum((predictions == 1) & (targets == 0))
        fn = np.sum((predictions == 0) & (targets == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1
    
    def plot_confusion_matrices(self, predictions: np.ndarray, targets: np.ndarray, save_path: Optional[str] = None):
        """Plot confusion matrices for each action."""
        action_names = ['Click', 'Repin', 'Hide']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, action_name in enumerate(action_names):
            binary_preds = (predictions[:, i] > 0.5).astype(int)
            true_labels = targets[:, i]
            
            cm = confusion_matrix(true_labels, binary_preds)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{action_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curves(self, predictions: np.ndarray, targets: np.ndarray, save_path: Optional[str] = None):
        """Plot ROC curves for each action."""
        from sklearn.metrics import roc_curve
        
        action_names = ['Click', 'Repin', 'Hide']
        colors = ['blue', 'red', 'green']
        
        plt.figure(figsize=(10, 8))
        
        for i, (action_name, color) in enumerate(zip(action_names, colors)):
            try:
                fpr, tpr, _ = roc_curve(targets[:, i], predictions[:, i])
                auc = roc_auc_score(targets[:, i], predictions[:, i])
                
                plt.plot(fpr, tpr, color=color, label=f'{action_name} (AUC = {auc:.3f})')
            except ValueError:
                plt.plot([0, 1], [0, 1], color=color, linestyle='--', label=f'{action_name} (AUC = 0.500)')
        
        plt.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Each Action Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_precision_recall_curves(self, predictions: np.ndarray, targets: np.ndarray, save_path: Optional[str] = None):
        """Plot Precision-Recall curves for each action."""
        action_names = ['Click', 'Repin', 'Hide']
        colors = ['blue', 'red', 'green']
        
        plt.figure(figsize=(10, 8))
        
        for i, (action_name, color) in enumerate(zip(action_names, colors)):
            try:
                precision, recall, _ = precision_recall_curve(targets[:, i], predictions[:, i])
                ap = average_precision_score(targets[:, i], predictions[:, i])
                
                plt.plot(recall, precision, color=color, label=f'{action_name} (AP = {ap:.3f})')
            except ValueError:
                plt.plot([0, 1], [0, 1], color=color, linestyle='--', label=f'{action_name} (AP = 0.000)')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for Each Action Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_evaluation_report(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive evaluation report."""
        # Run evaluation
        metrics = self.evaluate()
        predictions, targets, _, _ = self._get_predictions()
        
        # Create report
        report = []
        report.append("=" * 60)
        report.append("TRANSACT MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL PERFORMANCE METRICS")
        report.append("-" * 30)
        report.append(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        report.append(f"Overall F1 Score: {metrics['overall_f1']:.4f}")
        report.append(f"Overall AUC: {metrics['overall_auc']:.4f}")
        report.append(f"Overall Average Precision: {metrics['overall_ap']:.4f}")
        report.append("")
        
        # Per-action metrics
        report.append("PER-ACTION PERFORMANCE METRICS")
        report.append("-" * 35)
        action_names = ['click', 'repin', 'hide']
        
        for action in action_names:
            report.append(f"\n{action.upper()}:")
            report.append(f"  Precision: {metrics[f'{action}_precision']:.4f}")
            report.append(f"  Recall: {metrics[f'{action}_recall']:.4f}")
            report.append(f"  F1 Score: {metrics[f'{action}_f1']:.4f}")
            report.append(f"  AUC: {metrics[f'{action}_auc']:.4f}")
            report.append(f"  Average Precision: {metrics[f'{action}_ap']:.4f}")
        
        report.append("")
        
        # Ranking metrics
        report.append("RANKING METRICS")
        report.append("-" * 15)
        for k in [1, 3, 5, 10]:
            report.append(f"HIT@{k}: {metrics[f'hit_at_{k}']:.4f}")
            report.append(f"NDCG@{k}: {metrics[f'ndcg_at_{k}']:.4f}")
        
        report.append("")
        
        # Efficiency metrics
        report.append("EFFICIENCY METRICS")
        report.append("-" * 18)
        report.append(f"Total Parameters: {metrics['total_parameters']:,}")
        report.append(f"Trainable Parameters: {metrics['trainable_parameters']:,}")
        report.append(f"Average Inference Time: {metrics['avg_inference_time_ms']:.2f} ms")
        report.append(f"Inference Time Std: {metrics['std_inference_time_ms']:.2f} ms")
        
        report.append("")
        report.append("=" * 60)
        
        # Convert to string
        report_text = "\n".join(report)
        
        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text 