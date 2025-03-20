# Proyecto de Implementación de Algoritmos de IA

Este proyecto contiene dos partes principales: implementación del algoritmo A* para búsqueda de caminos (prac1) e implementación de modelos de reconocimiento de imágenes (prac2).

## Estructura del Proyecto

```
.
├── prac1/
│   ├── A_star.py        # Implementación del algoritmo A*
│   └── A_star_epsilon.py # Implementación de la variante ε del algoritmo A*
└── prac2/
    ├── Base.py          # Clase base para modelos de reconocimiento de imágenes
    └── dataset/         # Conjunto de datos de prueba personalizado
```

## prac1: Implementación del Algoritmo A*

Esta parte implementa el algoritmo de búsqueda A* y sus variantes, utilizados para encontrar la mejor ruta desde un punto de inicio (conejo) hasta un punto final (zanahoria).

### Funcionalidades Principales

* **Algoritmo A** *: Implementación clásica del algoritmo de búsqueda A*
* **Variante A* Epsilon* *: Implementación basada en A* que utiliza un parámetro ε para controlar la influencia de la función heurística

### Características del Algoritmo

* Soporte para diferentes tipos de terreno (roca, agua, hierba) y sus respectivos costos de movimiento
* Soporte para múltiples funciones heurísticas:
  * Distancia de Manhattan
  * Distancia Euclidiana
  * Distancia de Chebyshev
  * Distancia Octile
  * Dijkstra (sin heurística)
* Soporte para movimiento diagonal
* Consideración del impacto del terreno en el costo del movimiento

### Modo de Uso

```python
from A_star import A_star

# Crear una instancia del algoritmo A* (parámetros: coordenadas del conejo, coordenadas de la zanahoria, archivo del mapa)
a_star = A_star(conejo_x, conejo_y, zanahoria_x, zanahoria_y, mapa_file)

# Buscar y visualizar el camino
camino = []
a_star.main(camino)

# Obtener las calorías consumidas en el camino
calorias = a_star.get_calorias()

# Obtener el costo del movimiento
movimiento = a_star.get_movimiento()

# Obtener el número de nodos visitados
num_nodes = a_star.getNumNodes()
```

### Variante A* Epsilon

```python
from A_star_epsilon import A_star_epsilon

# Crear una instancia del algoritmo A* Epsilon (parámetros: coordenadas del conejo, coordenadas de la zanahoria, archivo del mapa, valor epsilon)
a_star_e = A_star_epsilon(conejo_x, conejo_y, zanahoria_x, zanahoria_y, mapa_file, epsilon=0.5)

# El método de uso es el mismo que el A* estándar
camino = []
a_star_e.main(camino)
```

## prac2: Implementación de Modelos de Reconocimiento de Imágenes

Esta parte implementa varios modelos de redes neuronales para tareas de reconocimiento de imágenes, basados en el conjunto de datos CIFAR-10 y en un conjunto de datos personalizado.

### Funcionalidades Principales

* **Arquitecturas de Modelos** :
* Perceptrón Multicapa (MLP)
* Red Neuronal Convolucional (CNN)
* Variantes de modelos basados en el paper "THE ALL CONVOLUTIONAL NET"
* **Funcionalidades Experimentales** :
* Experimentos de tamaño de lote (batch size)
* Comparación de funciones de activación
* Comparación de arquitecturas de modelos
* Pruebas con conjuntos de datos personalizados

### Arquitecturas de Modelos

#### Modelo MLP

Modelo de perceptrón multicapa con número configurable de capas ocultas y neuronas por capa.

#### Modelo CNN

Modelo CNN básico que incluye capas convolucionales, capas de agrupación y capas completamente conectadas.

#### Serie de Modelos ALL-CNN

Implementación de varias arquitecturas de modelos del paper:

* `model_a`
* `model_b`
* `model_c`
* `strided_cnn_c`
* `convPool_cnn_c`
* `all_cnn_c`

### Modo de Uso

#### Uso Básico

```python
from prac2 import ModelExperiment

# Crear una instancia de experimento
experiment = ModelExperiment()

# Configurar parámetros del experimento
input_activation = "relu"
model_type = "mlp"  # Opciones: mlp, cnn, model_a, model_b, model_c, strided_cnn_c, convPool_cnn_c, all_cnn_c
output_activation = "softmax"
batch_size_list = [32, 64, 128]
num_epochs = 50
num_repetitions = 3
hidden_layers = [128, 64, 32]  # Solo para MLP

# Ejecutar experimento de tamaño de lote
results = experiment.run_batch_size_experiment(
    batch_sizes=batch_size_list,
    epochs=num_epochs,
    input_activation=input_activation,
    num_repetitions=num_repetitions,
    model_type=model_type,
    output_activation=output_activation,
    n_capas_ocultas=hidden_layers
)

# Analizar resultados
best_config = experiment.analyze_batch_size_results(results)

# Visualizar resultados
experiment.plot_average_training_history(
    avg_history, 
    best_batch_size, 
    activation=input_activation
)
experiment.plot_confusion_matrix(input_activation, output_activation)
experiment.plot_batch_size_comparison(results)
```

#### Uso con Conjunto de Datos Personalizado

```python
from prac2 import myOwnDataset

# Crear un experimento utilizando el conjunto de datos personalizado
custom_experiment = myOwnDataset()

# El método de uso es el mismo que ModelExperiment
results = custom_experiment.run_batch_size_experiment(...)
```

### Opciones de Configuración

* **Funciones de Activación** :
* Capa de entrada: `sigmoid`, `relu`, `tanh`, `softplus`, etc.
* Capa de salida: generalmente `softmax`
* **Tamaño de Lote** : Se pueden probar diferentes tamaños (por ejemplo, 32, 64, 128, 256, 512) para evaluar su impacto
* **Épocas de Entrenamiento** : Controla el número de rondas de entrenamiento
* **Número de Repeticiones** : Realiza el experimento varias veces para obtener resultados más confiables
* **Capas Ocultas MLP** : Configuración del número de neuronas en las capas ocultas

## Conjuntos de Datos

### CIFAR-10

Principalmente se utiliza el conjunto de datos CIFAR-10 para entrenamiento y validación.

### Conjunto de Datos Personalizado

En el directorio `prac2/dataset/` se incluyen imágenes personalizadas para probar la capacidad de generalización del modelo.

## Recomendaciones de Ejecución

* Para modelos complejos (como `model_a`, `model_b`, `model_c`, `strided_cnn_c`, `convPool_cnn_c`, `all_cnn_c`), se recomienda utilizar GPU para el entrenamiento
* Los parámetros pueden ajustarse para equilibrar el tiempo de entrenamiento y el rendimiento del modelo
* Se utilizan técnicas como Early Stopping y Learning Rate Reduction para mejorar la eficiencia del entrenamiento

## Bibliotecas Dependientes

* TensorFlow/Keras
* NumPy
* Matplotlib
* scikit-learn
* seaborn

## Referencias

La implementación de los modelos CNN en este proyecto está basada en el paper "THE ALL CONVOLUTIONAL NET".
