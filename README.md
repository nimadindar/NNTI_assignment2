# Token Dataset Project

This project provides an implementation of a custom PyTorch dataset and dataloader to work with tokenized data. Key components include custom dataset classes, transformations for token length management, and a tailored dataloader. The following concepts are implemented:

## Project Overview

1. **Data Loading**:
   - Loads datasets from JSON and CSV formats, making them easy to integrate with Python-based machine learning workflows.
   - JSON and CSV data are loaded into dictionaries for flexible access and manipulation.

2. **Custom PyTorch Dataset (`TokenDataset`)**:
   - A PyTorch `Dataset` subclass that handles tokenized data.
   - Reads JSON-formatted datasets using a helper function and stores tokenized text data along with associated labels.
   - Provides data access through the `__getitem__` method, which retrieves tokens and labels at a specified index.

3. **Transformations for Token Length (`TransformTokens`)**:
   - Implements padding and truncation of token sequences to a fixed length.
   - This transformation ensures that each token sequence has a uniform length, which is required for efficient batch processing in machine learning models.
   - If a token sequence is shorter than the specified length, it is padded with zeros; if it is longer, it is truncated.

4. **Custom DataLoader (`TokenDataLoader`)**:
   - Wraps PyTorch's `DataLoader` to provide batching, shuffling, and easy access to the custom dataset.
   - Allows for configurable batch size and shuffling to support training processes that rely on randomized batches.

## Folder Structure

- `data/`: Contains data files such as `data.json` and `data.csv`.
- `src/`: Code implementation files:
  - `dataset.py`: Defines the `TokenDataset` class.
  - `transforms.py`: Contains the `TransformTokens` class for padding and truncating tokens.
  - `dataloader.py`: Implements `TokenDataLoader`, a custom loader for managing batches of tokenized data.

## Usage Example

Once the project is set up, you can create an instance of the custom dataset, apply transformations, and load data in batches. Here’s an example usage:

```python
from src.dataset import TokenDataset
from src.transforms import TransformTokens
from src.dataloader import TokenDataLoader

# Define transformation to ensure uniform token length
transform = TransformTokens(output_len=128)

# Load the dataset and apply transformations
dataset = TokenDataset("data/data.json", transform=transform)

# Create a data loader for batch processing
dataloader = TokenDataLoader(dataset, batch_size=32, shuffle=True)

# Iterate over the data loader
for tokens, labels in dataloader:
    print(tokens.shape, labels.shape)
```
This example demonstrates loading and batching tokenized data with length management. The project enables streamlined data handling for token-based models in PyTorch.

Requirements
Python 3.x
PyTorch
JSON and CSV data files formatted according to the expected structure.
