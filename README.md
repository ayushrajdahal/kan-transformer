# Decoder-only Transformer using Kolmogorov-Arnold Networks

This project implements a decoder-only transformer with KAN-powered attention mechanism using PyTorch. The model is designed to generate text based on learned patterns from a given dataset. The main script is inspired by karpathy/minGPT

## Features

- Implements a transformer architecture with multi-head attention
- Uses FastKAN for improved efficiency
- Supports character-level tokenization
- Includes data loading, model training, and text generation capabilities

## Requirements

- Python 3.7+
- PyTorch
- datasets

You can install the required packages using:

```
pip install torch datasets
```

## Configuration

You can adjust the following hyperparameters in the script:

- `batch_size`: Number of sequences processed in parallel
- `n_head`: Number of attention heads in each block
- `n_embd`: Embedding dimension
- `n_layer`: Number of transformer blocks
- `block_size`: Maximum context length
- `max_iters`: Maximum number of training iterations
- `eval_interval`: Frequency of model evaluation
- `eval_iters`: Number of iterations for evaluation
- `learning_rate`: Learning rate for optimizer
- `dropout`: Dropout rate

## Model Architecture

The model consists of the following components:

- Character-level tokenizer
- Embedding layer
- Positional encoding
- Multiple transformer blocks
- Layer normalization
- Linear output layer

## Data

The model uses tiny shakespeare dataset, a collection of approximately 40,000 lines of text from various works by William Shakespeare. You can modify the script to use a different dataset if needed.

## Output

The script will print training and validation losses during the training process. After training, it will generate a sample text based on the trained model.

## License

MIT License