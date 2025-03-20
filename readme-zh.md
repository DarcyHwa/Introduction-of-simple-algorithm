# AI算法实现项目

本项目包含两个主要部分：A*寻路算法实现（prac1）和图像识别模型实现（prac2）。

## 项目结构

```
.
├── prac1/
│   ├── A_star.py        # A*算法实现
│   └── A_star_epsilon.py # A*算法的ε变体实现
└── prac2/
    ├── Base.py          # 图像识别模型基础类
    └── dataset/         # 自定义测试数据集
```

## prac1: A*寻路算法实现

这部分实现了A*搜索算法及其变体，用于寻找从起点（兔子）到终点（胡萝卜）的最佳路径。

### 主要功能

* **A*算法** ：经典的A*搜索算法实现
* **A* Epsilon变体* *：基于A*算法的变体，使用ε参数控制启发式函数的影响

### 算法特性

* 支持不同地形类型（岩石、水、草地）及相应的移动代价
* 支持多种启发式函数：
  * 曼哈顿距离
  * 欧几里得距离
  * 切比雪夫距离
  * Octile距离
  * Dijkstra（无启发式）
* 支持对角线移动
* 考虑地形对移动代价的影响

### 使用方法

```python
from A_star import A_star

# 创建A*算法实例（参数：兔子坐标，胡萝卜坐标，地图文件）
a_star = A_star(conejo_x, conejo_y, zanahoria_x, zanahoria_y, mapa_file)

# 寻找并可视化路径
camino = []
a_star.main(camino)

# 获取路径消耗的卡路里
calorias = a_star.get_calorias()

# 获取移动代价
movimiento = a_star.get_movimiento()

# 获取访问的节点数
num_nodes = a_star.getNumNodes()
```

### A* Epsilon变体

```python
from A_star_epsilon import A_star_epsilon

# 创建A* Epsilon算法实例（参数：兔子坐标，胡萝卜坐标，地图文件，epsilon值）
a_star_e = A_star_epsilon(conejo_x, conejo_y, zanahoria_x, zanahoria_y, mapa_file, epsilon=0.5)

# 使用方法与标准A*相同
camino = []
a_star_e.main(camino)
```

## prac2: 图像识别模型实现

这部分实现了多种神经网络模型用于图像识别任务，基于CIFAR-10数据集和自定义数据集。

### 主要功能

* **模型架构** ：
* 多层感知机（MLP）
* 卷积神经网络（CNN）
* 基于论文"THE ALL CONVOLUTIONAL NET"的模型变体
* **实验功能** ：
* 批量大小实验
* 激活函数比较
* 模型架构比较
* 自定义数据集测试

### 模型架构

#### MLP模型

多层感知机模型，可配置隐藏层数量和每层神经元数。

#### CNN模型

基础CNN模型，包含卷积层、池化层和全连接层。

#### ALL-CNN系列模型

实现了论文中的多种模型架构：

* `model_a`
* `model_b`
* `model_c`
* `strided_cnn_c`
* `convPool_cnn_c`
* `all_cnn_c`

### 使用方法

#### 基本使用

```python
from prac2 import ModelExperiment

# 创建实验实例
experiment = ModelExperiment()

# 设置实验参数
input_activation = "relu"
model_type = "mlp"  # 可选: mlp, cnn, model_a, model_b, model_c, strided_cnn_c, convPool_cnn_c, all_cnn_c
output_activation = "softmax"
batch_size_list = [32, 64, 128]
num_epochs = 50
num_repetitions = 3
hidden_layers = [128, 64, 32]  # 仅用于MLP

# 运行批量大小实验
results = experiment.run_batch_size_experiment(
    batch_sizes=batch_size_list,
    epochs=num_epochs,
    input_activation=input_activation,
    num_repetitions=num_repetitions,
    model_type=model_type,
    output_activation=output_activation,
    n_capas_ocultas=hidden_layers
)

# 分析结果
best_config = experiment.analyze_batch_size_results(results)

# 可视化结果
experiment.plot_average_training_history(
    avg_history, 
    best_batch_size, 
    activation=input_activation
)
experiment.plot_confusion_matrix(input_activation, output_activation)
experiment.plot_batch_size_comparison(results)
```

#### 使用自定义数据集

```python
from prac2 import myOwnDataset

# 创建使用自定义测试集的实验
custom_experiment = myOwnDataset()

# 使用方法与ModelExperiment相同
results = custom_experiment.run_batch_size_experiment(...)
```

### 配置选项

* **激活函数** ：
* 输入层：`sigmoid`, `relu`, `tanh`, `softplus`等
* 输出层：通常为 `softmax`
* **批量大小** ：可测试不同大小（如32, 64, 128, 256, 512）的影响
* **训练周期** ：控制训练的轮数
* **重复次数** ：对实验进行多次重复以获得更可靠的结果
* **MLP隐藏层** ：可配置隐藏层神经元数量

## 数据集

### CIFAR-10

主要使用CIFAR-10数据集进行训练和验证。

### 自定义数据集

在 `prac2/dataset/`目录中包含自定义图像数据，用于测试模型的泛化能力。

## 运行建议

* 对于复杂模型（如 `model_a`, `model_b`, `model_c`, `strided_cnn_c`, `convPool_cnn_c`, `all_cnn_c`），推荐使用GPU进行训练
* 可以通过调整参数来平衡训练时间和模型性能
* 使用Early Stopping和Learning Rate Reduction等技术来提高训练效率

## 依赖库

* TensorFlow/Keras
* NumPy
* Matplotlib
* scikit-learn
* seaborn

## 参考资料

本项目中的CNN模型实现部分参考了论文"THE ALL CONVOLUTIONAL NET"。
