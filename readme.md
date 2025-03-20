# AI Algorithm Implementation Project

This project contains two main parts: A* pathfinding algorithm implementation (prac1) and image recognition model implementation (prac2).

[中文readme](readme-zh.md) | [español readme](readme-es.md)

## Project Structure

```
.
├── prac1/
│   ├── A_star.py        # A* algorithm implementation
│   └── A_star_epsilon.py # A* algorithm ε variant implementation
└── prac2/
    ├── Base.py          # Image recognition model base class
    └── dataset/         # Custom test dataset
```

## prac1: A* Pathfinding Algorithm Implementation

This part implements the A* search algorithm and its variants, used to find the optimal path from a starting point (rabbit) to an endpoint (carrot).

### Main Features

- **A* Algorithm**: Classic A* search algorithm implementation
- **A* Epsilon Variant**: A variant based on the A* algorithm using an ε parameter to control the heuristic function's influence

### Algorithm Characteristics

- Support for different terrain types (rock, water, grass) and their respective movement costs
- Support for multiple heuristic functions:
  - Manhattan distance
  - Euclidean distance
  - Chebyshev distance
  - Octile distance
  - Dijkstra (no heuristic)
- Support for diagonal movement
- Consideration of terrain impact on movement cost

### Usage

```python
from A_star import A_star

# Create an A* algorithm instance (parameters: rabbit coordinates, carrot coordinates, map file)
a_star = A_star(conejo_x, conejo_y, zanahoria_x, zanahoria_y, mapa_file)

# Find and visualize the path
camino = []
a_star.main(camino)

# Get the calories consumed along the path
calorias = a_star.get_calorias()

# Get the movement cost
movimiento = a_star.get_movimiento()

# Get the number of visited nodes
num_nodes = a_star.getNumNodes()
```

### A* Epsilon Variant

```python
from A_star_epsilon import A_star_epsilon

# Create an A* Epsilon algorithm instance (parameters: rabbit coordinates, carrot coordinates, map file, epsilon value)
a_star_e = A_star_epsilon(conejo_x, conejo_y, zanahoria_x, zanahoria_y, mapa_file, epsilon=0.5)

# Usage method is the same as standard A*
camino = []
a_star_e.main(camino)
```

## prac2: Image Recognition Model Implementation

This part implements various neural network models for image recognition tasks, based on the CIFAR-10 dataset and a custom dataset.

### Main Features

- **Model Architectures**:
  - Multilayer Perceptron (MLP)
  - Convolutional Neural Network (CNN)
  - Model variants based on the paper "THE ALL CONVOLUTIONAL NET"

- **Experimental Functions**:
  - Batch size experiments
  - Activation function comparison
  - Model architecture comparison
  - Custom dataset testing

### Model Architectures

#### MLP Model
Multilayer perceptron model with configurable number of hidden layers and neurons per layer.

#### CNN Model
Basic CNN model including convolutional layers, pooling layers, and fully connected layers.

#### ALL-CNN Series Models
Implementation of various model architectures from the paper:
- `model_a`
- `model_b`
- `model_c`
- `strided_cnn_c`
- `convPool_cnn_c`
- `all_cnn_c`

### Usage

#### Basic Usage
```python
from prac2 import ModelExperiment

# Create an experiment instance
experiment = ModelExperiment()

# Set experiment parameters
input_activation = "relu"
model_type = "mlp"  # Options: mlp, cnn, model_a, model_b, model_c, strided_cnn_c, convPool_cnn_c, all_cnn_c
output_activation = "softmax"
batch_size_list = [32, 64, 128]
num_epochs = 50
num_repetitions = 3
hidden_layers = [128, 64, 32]  # Only for MLP

# Run batch size experiment
results = experiment.run_batch_size_experiment(
    batch_sizes=batch_size_list,
    epochs=num_epochs,
    input_activation=input_activation,
    num_repetitions=num_repetitions,
    model_type=model_type,
    output_activation=output_activation,
    n_capas_ocultas=hidden_layers
)

# Analyze results
best_config = experiment.analyze_batch_size_results(results)

# Visualize results
experiment.plot_average_training_history(
    avg_history, 
    best_batch_size, 
    activation=input_activation
)
experiment.plot_confusion_matrix(input_activation, output_activation)
experiment.plot_batch_size_comparison(results)
```

#### Using Custom Dataset
```python
from prac2 import myOwnDataset

# Create an experiment using the custom test dataset
custom_experiment = myOwnDataset()

# Usage method is the same as ModelExperiment
results = custom_experiment.run_batch_size_experiment(...)
```

### Configuration Options

- **Activation Functions**:
  - Input layer: `sigmoid`, `relu`, `tanh`, `softplus`, etc.
  - Output layer: typically `softmax`
  
- **Batch Size**: Different sizes (e.g., 32, 64, 128, 256, 512) can be tested to evaluate their impact

- **Training Epochs**: Controls the number of training rounds

- **Repetition Count**: Performs the experiment multiple times to obtain more reliable results

- **MLP Hidden Layers**: Configuration of the number of neurons in hidden layers

## Datasets

### CIFAR-10
The CIFAR-10 dataset is primarily used for training and validation.

### Custom Dataset
The `prac2/dataset/` directory includes custom images to test the model's generalization capability.

## Running Recommendations

- For complex models (such as `model_a`, `model_b`, `model_c`, `strided_cnn_c`, `convPool_cnn_c`, `all_cnn_c`), GPU usage for training is recommended
- Parameters can be adjusted to balance training time and model performance
- Techniques such as Early Stopping and Learning Rate Reduction are used to improve training efficiency

## Dependencies

- TensorFlow/Keras
- NumPy
- Matplotlib
- scikit-learn
- seaborn

## References

The CNN model implementation in this project is based on the paper "THE ALL CONVOLUTIONAL NET".