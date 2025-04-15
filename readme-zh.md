

# A*寻路算法实现（prac1）
## 项目结构

```
.
├── prac1/
│   ├── A_star.py        # A*算法实现
│   └── A_star_epsilon.py # A*算法的ε变体实现


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

