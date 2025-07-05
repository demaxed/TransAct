# TransAct: Transformer-based Realtime User Action Model for Recommendation

This repository implements the TransAct model from the Pinterest research paper: "TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest" (KDD '23).

## Overview

TransAct is a transformer-based sequential model that extracts users' short-term preferences from their real-time activities. The model is designed for web-scale personalized recommendation systems and combines:

1. **Real-time user action sequences** - Captures immediate user preferences
2. **Batch user representations** - Incorporates long-term user interests
3. **Hybrid ranking approach** - Combines both for optimal recommendation performance

## Key Features

- **Transformer-based architecture** for encoding recent user action sequences
- **Multi-task prediction** with configurable head weighting
- **Hybrid ranking** combining real-time and batch features
- **Production-ready** implementation with efficiency optimizations
- **Comprehensive evaluation** metrics and ablation studies

## Installation

```bash
pip install -e .
```

## Usage

### Basic Usage

```python
from model_cursor.transact import TransActModel
from model_cursor.data import UserActionDataset

# Initialize the model
model = TransActModel(
    num_actions=3,  # click, repin, hide
    action_dim=64,
    user_dim=128,
    item_dim=256,
    num_heads=8,
    num_layers=4,
    dropout=0.1
)

# Train the model
# ... training code here
```

### Advanced Configuration

```python
# Configure head weighting for multi-task learning
label_weight_matrix = {
    'click': {'click': 100, 'repin': 0, 'hide': 100},
    'repin': {'click': 0, 'repin': 100, 'hide': 100},
    'hide': {'click': 1, 'repin': 5, 'hide': 10}
}

model = TransActModel(
    num_actions=3,
    label_weight_matrix=label_weight_matrix,
    # ... other parameters
)
```

## Model Architecture

The TransAct model consists of:

1. **Action Embedding Layer** - Converts user actions to dense representations
2. **Transformer Encoder** - Processes sequential user actions with self-attention
3. **Multi-task Prediction Heads** - Predicts different types of user engagement
4. **Hybrid Ranking Layer** - Combines real-time and batch features

## Performance

Based on the paper's results:
- **HIT@3 (hide)**: Baseline performance
- **HIT@3 (repin)**: Improved recommendation accuracy
- **Latency**: Optimized for real-time serving (8ms on GPU)
- **Serving Cost**: 1x compared to baseline CPU implementation

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{xia2023transact,
  title={TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest},
  author={Xia, Xue and Eksombatchai, Pong and Badani, Dhruvil Deven and Joshi, Saurabh Vishwas and Wang, Po-Wei and Farahpour, Nazanin and Zhai, Andrew},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2023}
}
```

## License

This implementation is for research purposes. Please refer to the original paper for licensing information.
