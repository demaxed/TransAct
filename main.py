"""
TransAct: Transformer-based Realtime User Action Model for Recommendation

This script demonstrates the complete TransAct implementation based on the Pinterest paper.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model_cursor.transact import TransActModel, HybridRankingModel
from src.model_cursor.data import UserActionDataset, UserActionDataLoader, SyntheticDataGenerator
from src.model_cursor.training import TransActTrainer
from src.model_cursor.evaluation import TransActEvaluator


def main():
    """Main function demonstrating TransAct implementation."""
    print("TransAct: Transformer-based Realtime User Action Model for Recommendation")
    print("=" * 70)
    print("Based on: TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest (KDD '23)")
    print()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # 1. Generate synthetic data for demonstration
    print("1. Generating synthetic data...")
    data_generator = SyntheticDataGenerator(
        num_users=10000,
        num_items=100000,
        num_actions=3,  # click, repin, hide
        max_seq_len=50
    )
    
    train_dataset, val_dataset, test_dataset = data_generator.generate_dataset(
        num_samples=10000,
        train_split=0.8,
        val_split=0.1
    )
    
    print(f"Generated {len(train_dataset)} training samples")
    print(f"Generated {len(val_dataset)} validation samples")
    print(f"Generated {len(test_dataset)} test samples")
    print()
    
    # 2. Create data loaders
    print("2. Creating data loaders...")
    train_loader = UserActionDataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True
    )
    
    val_loader = UserActionDataLoader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=False
    )
    
    test_loader = UserActionDataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print()
    
    # 3. Initialize TransAct model
    print("3. Initializing TransAct model...")
    model = TransActModel(
        num_actions=3,  # click, repin, hide
        action_dim=64,
        user_dim=128,
        item_dim=256,
        hidden_dim=512,
        num_heads=8,
        num_layers=4,
        max_seq_len=50,
        dropout=0.1
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model initialized with {total_params:,} total parameters")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # 4. Initialize trainer
    print("4. Initializing trainer...")
    trainer = TransActTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        weight_decay=1e-5,
        optimizer='adam',
        scheduler='cosine',
        device='auto',
        save_dir='./checkpoints'
    )
    
    print("Trainer initialized successfully")
    print()
    
    # 5. Train the model
    print("5. Training TransAct model...")
    print("Note: This is a demonstration with synthetic data. In production, use real user action sequences.")
    print()
    
    # Train for a few epochs (short training for demonstration)
    history = trainer.train(
        num_epochs=5,  # Short training for demonstration
        save_best=True,
        patience=3,
        min_delta=1e-4
    )
    
    print("Training completed!")
    print()
    
    # 6. Evaluate the model
    print("6. Evaluating TransAct model...")
    evaluator = TransActEvaluator(
        model=model,
        test_loader=test_loader,
        device='auto'
    )
    
    # Run comprehensive evaluation
    metrics = evaluator.evaluate()
    
    # Print key metrics
    print("\nKey Performance Metrics:")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Overall F1 Score: {metrics['overall_f1']:.4f}")
    print(f"Overall AUC: {metrics['overall_auc']:.4f}")
    print(f"Click F1: {metrics['click_f1']:.4f}")
    print(f"Repin F1: {metrics['repin_f1']:.4f}")
    print(f"Hide F1: {metrics['hide_f1']:.4f}")
    print(f"Average Inference Time: {metrics['avg_inference_time_ms']:.2f} ms")
    print()
    
    # 7. Generate evaluation report
    print("7. Generating evaluation report...")
    report = evaluator.generate_evaluation_report(save_path='evaluation_report.txt')
    print("Evaluation report saved to 'evaluation_report.txt'")
    print()
    
    # 8. Demonstrate model inference
    print("8. Demonstrating model inference...")
    
    # Get a sample batch
    sample_batch = next(iter(test_loader))
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        action_sequence = sample_batch['action_sequence'].to(device)
        user_ids = sample_batch['user_id'].to(device)
        item_ids = sample_batch['item_id'].to(device)
        attention_mask = sample_batch['attention_mask'].to(device)
        
        # Get predictions
        logits = model(action_sequence, user_ids, item_ids, attention_mask)
        probabilities = torch.sigmoid(logits)
        
        # Get ground truth
        labels = sample_batch['labels']
    
    # Show sample predictions
    print("Sample Predictions (first 5 samples):")
    print("User ID | Item ID | Click Prob | Repin Prob | Hide Prob | True Click | True Repin | True Hide")
    print("-" * 90)
    
    for i in range(min(5, len(probabilities))):
        user_id = user_ids[i].item()
        item_id = item_ids[i].item()
        click_prob = probabilities[i, 0].item()
        repin_prob = probabilities[i, 1].item()
        hide_prob = probabilities[i, 2].item()
        true_click = labels[i, 0].item()
        true_repin = labels[i, 1].item()
        true_hide = labels[i, 2].item()
        
        print(f"{user_id:7d} | {item_id:7d} | {click_prob:9.3f} | {repin_prob:9.3f} | {hide_prob:8.3f} | {true_click:10.0f} | {true_repin:10.0f} | {true_hide:9.0f}")
    
    print()
    
    # 9. Demonstrate hybrid ranking model
    print("9. Demonstrating hybrid ranking model...")
    
    # Create hybrid ranking model
    hybrid_model = HybridRankingModel(
        transact_model=model,
        batch_user_dim=128,
        hidden_dim=256,
        dropout=0.1
    )
    
    # Generate synthetic batch user features
    batch_size = sample_batch['action_sequence'].shape[0]
    batch_user_features = torch.randn(batch_size, 128).to(device)
    
    # Get hybrid ranking scores
    hybrid_model.eval()
    with torch.no_grad():
        ranking_scores = hybrid_model(
            action_sequence,
            user_ids,
            item_ids,
            batch_user_features,
            attention_mask
        )
    
    print("Hybrid Ranking Scores (first 5 samples):")
    print("User ID | Item ID | Ranking Score")
    print("-" * 35)
    
    for i in range(min(5, len(ranking_scores))):
        user_id = user_ids[i].item()
        item_id = item_ids[i].item()
        score = ranking_scores[i, 0].item()
        print(f"{user_id:7d} | {item_id:7d} | {score:13.3f}")
    
    print()
    
    # 10. Plot training history
    print("10. Plotting training history...")
    trainer.plot_training_history(save_path='training_history.png')
    print("Training history plot saved to 'training_history.png'")
    print()
    
    # 11. Summary
    print("=" * 70)
    print("TRANSACT IMPLEMENTATION SUMMARY")
    print("=" * 70)
    print("✅ TransAct model architecture implemented")
    print("✅ Multi-task prediction with head weighting")
    print("✅ Transformer-based sequential modeling")
    print("✅ Hybrid ranking approach")
    print("✅ Comprehensive training pipeline")
    print("✅ Evaluation metrics and visualization")
    print("✅ Production-ready code structure")
    print()
    print("Key Features Implemented:")
    print("- Real-time user action sequence processing")
    print("- Batch user representation integration")
    print("- Multi-task learning for click/repin/hide prediction")
    print("- Configurable head weighting matrix")
    print("- Comprehensive evaluation metrics (AUC, F1, HIT@K, NDCG@K)")
    print("- Training visualization and checkpointing")
    print("- Efficiency metrics and inference timing")
    print()
    print("The implementation follows the architecture described in the Pinterest paper")
    print("and is ready for production deployment with real user data.")
    print()
    print("For production use:")
    print("1. Replace synthetic data with real user action sequences")
    print("2. Tune hyperparameters based on your specific use case")
    print("3. Implement proper data preprocessing and feature engineering")
    print("4. Add monitoring and logging for production deployment")
    print("5. Optimize for your specific latency and throughput requirements")


if __name__ == "__main__":
    main()
