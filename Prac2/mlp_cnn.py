import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import time

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split


class Base:
    """
    Clase base para la implementación de experimentos
    """

    def __init__(self):
        """
        Carregar el dataset CIFAR-10 y dividirlo en entrenamiento, validación y prueba
        """
        (x_train, y_train), (self.x_test, self.y_test) = (
            keras.datasets.cifar10.load_data()
        )

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )

        self.x_train = self.x_train.astype("float32") / 255.0
        self.x_val = self.x_val.astype("float32") / 255.0
        self.x_test = self.x_test.astype("float32") / 255.0

    def cnn_model(self, input_activation, output_activation):
        """
        Crear un modelo CNN simple
        ---
        input_activation: str - Función de activación de la capa de entrada
        output_activation: str - Función de activación de la capa de salida
        """
        inputs = keras.Input(shape=(32, 32, 3))
        x = layers.Conv2D(16, (3, 3), padding="same", activation=input_activation)(
            inputs
        )
        x = layers.Conv2D(16, (3, 3), padding="same", activation=input_activation)(x)
        # x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(32, (3, 3), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(32, (3, 3), padding="same", activation=input_activation)(x)
        x = layers.Dropout(0.5)(x)
        x = layers.BatchNormalization()(x)
        # x = layers.MaxPooling2D((2, 2))(x)

        x = layers.GlobalAveragePooling2D()(x)
        output = layers.Dense(10, activation=output_activation)(x)

        self.model = keras.Model(inputs=inputs, outputs=output)

    def mlp_model(
        self, input_activation, output_activation, n_capas_ocultas=[128, 64, 32]
    ):
        """
        Crear un modelo MLP
        ---
        input_activation: str - Función de activación de la capa de entrada
        output_activation: str - Función de activación de la capa de salida
        """
        inputs = keras.Input(shape=(32, 32, 3))
        x = layers.Flatten()(inputs)

        # Capa oculta en la red neuronal
        for _ in n_capas_ocultas:
            x = layers.Dense(_, activation=input_activation)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)

        outputs = layers.Dense(10, activation=output_activation)(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

    def compile_model(self):
        """
        Compilar el modelo, utilizando la funcion de perdida, el optimizador y la metrica.
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.001 
            ),
            loss="sparse_categorical_crossentropy",  # funcion de perdida: sparse categorical crossentropy o CategoricalCrossentropy
            metrics=[
                "accuracy"
            ],  # metrica de evaluacion: accuracy. Normalmente se utiliza accuracy
        )

    def train_multiple_times(
        self,
        n_repetitions=5,
        epochs=10,
        batch_size=32,
        activation="sigmoid",
        model_type="mlp",
        output_activation="softmax",
        n_capas_ocultas=[128, 64, 32],
    ):
        """
        Entrene el modelo varias veces y devuelva los historiales de entrenamiento y las precisiones de prueba
        """
        print(
            "x_train shape:", self.x_train.shape, "y_train shape:", self.y_train.shape
        )
        print("x_val shape:", self.x_val.shape, "y_val shape:", self.y_val.shape)
        print("x_test shape:", self.x_test.shape, "y_test shape:", self.y_test.shape)

        histories = []
        test_accuracies = []
        self.callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                mode="min",
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.1, patience=5, mode="min", verbose=1
            ),
        ]  # API de keras para detener el entrenamiento si no hay mejora en la validación

        for i in range(n_repetitions):
            print(f"\nEntrenamiento {i+1}/{n_repetitions}")
            self.build_model(activation, model_type, output_activation, n_capas_ocultas)
            self.compile_model()
            history = self.model.fit(
                self.x_train,
                self.y_train,
                epochs=epochs,
                validation_data=(self.x_val, self.y_val),
                batch_size=batch_size,
                verbose=1,
                callbacks=self.callbacks,
            )
            histories.append(history.history)

            # Evaluar el modelo en el conjunto de prueba
            test_loss, test_accuracy = self.model.evaluate(
                self.x_test, self.y_test, verbose=0
            )
            print(f"Test accuracy: {test_accuracy:.4f}")
            print(f"Test loss: {test_loss:.4f}")
            test_accuracies.append(test_accuracy)

        return histories, test_accuracies

    def calculate_average_history(self, histories):
        """
        Calcula el promedio de las métricas de todos los entrenamientos
        """
        print("tamano de history:", [len(h["loss"]) for h in histories])

        min_length = min(len(history["loss"]) for history in histories)

        avg_history = {}
        metrics = histories[0].keys()

        for metric in metrics:
            # Trunca cada historial a la longitud mínima
            values = np.array([h[metric][:min_length] for h in histories])
            avg_history[metric] = np.mean(values, axis=0)
            avg_history[f"{metric}_std"] = np.std(values, axis=0)

        return avg_history

    def plot_average_training_history(
        self, avg_history, batch_size, activation="sigmoid", block=False
    ):
        """
        Visualiza la evolución promedio del entrenamiento con bandas de desviación estándar
        Parameters:
            avg_history: historia de entrenamiento
            batch_size: tamaño del lote
            activation: funcion de activacion
            block: mostrar el gráfico
        """
        plt.figure(figsize=(12, 4))

        # Gráfica de accuracy
        plt.subplot(1, 2, 1)
        plt.plot(avg_history["accuracy"], label="Training", color="blue")
        plt.fill_between(
            range(len(avg_history["accuracy"])),
            avg_history["accuracy"] - avg_history["accuracy_std"],
            avg_history["accuracy"] + avg_history["accuracy_std"],
            alpha=0.2,
            color="blue",
        )

        plt.plot(avg_history["val_accuracy"], label="Validation", color="orange")
        plt.fill_between(
            range(len(avg_history["val_accuracy"])),
            avg_history["val_accuracy"] - avg_history["val_accuracy_std"],
            avg_history["val_accuracy"] + avg_history["val_accuracy_std"],
            alpha=0.2,
            color="orange",
        )

        plt.title(
            f"Average Model Accuracy (±std)\nBatch Size: {batch_size}, Activation: {activation}"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        # Gráfica de loss
        plt.subplot(1, 2, 2)
        plt.plot(avg_history["loss"], label="Training", color="blue")
        plt.fill_between(
            range(len(avg_history["loss"])),
            avg_history["loss"] - avg_history["loss_std"],
            avg_history["loss"] + avg_history["loss_std"],
            alpha=0.2,
            color="blue",
        )

        plt.plot(avg_history["val_loss"], label="Validation", color="orange")
        plt.fill_between(
            range(len(avg_history["val_loss"])),
            avg_history["val_loss"] - avg_history["val_loss_std"],
            avg_history["val_loss"] + avg_history["val_loss_std"],
            alpha=0.2,
            color="orange",
        )

        plt.title(
            f"Average Model Loss (±std)\nBatch Size: {batch_size}, Activation: {activation}"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.show(block=block)

    def plot_confusion_matrix(self, input_activation, output_activation):
        """
        Plotea la matriz de confusión y muestra las categorías más confusas
        """

        # Realizar predicciones
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = self.y_test.reshape(-1)

        # Calcular la matriz de confusión
        cm = confusion_matrix(y_true, y_pred_classes)

        # Nombres de las clases
        class_names = [
            "airplane",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
            "vehicle",
        ]

        plt.figure(figsize=(7, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )

        plt.title(f"Confusion Matrix {input_activation} {output_activation}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # calcular la precisión de cada categoría
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        print("\nAcuraccy de cada categoria: ")
        for i, acc in enumerate(class_accuracy):
            print(f"{class_names[i]}: {acc:.2%}")

        n_classes = len(class_names)
        confusion_pairs = []

        for i in range(n_classes):
            for j in range(n_classes):
                if i != j:
                    confusion_pairs.append(
                        (
                            class_names[i],
                            class_names[j],
                            cm[i, j],
                            cm[i, j] / cm[i].sum(),
                        )
                    )

        # ordenar confusion_pairs por frecuencia de error
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        print("\nLas categorias mas confuso son:")
        for true_class, pred_class, count, percentage in confusion_pairs[:5]:
            print(f"{true_class} a {pred_class}: {count} veces ({percentage:.2%})")


class BatchSizeExperiment(Base):
    def __init__(self):
        super().__init__()

    def run_batch_size_experiment(
        self,
        batch_sizes=[32, 64, 128, 256, 512],
        epochs=50,
        input_activation="sigmoid",
        num_repetitions=1,
        model_type="mlp",
        output_activation="softmax",
        n_capas_ocultas=[128, 64, 32],
    ):
        """
        Experimento para comparar diferentes tamaños de batch_sizes
        """
        results = []
        for batch_size in batch_sizes:
            print(f"\nprueba de batch_size = {batch_size}")
            print("model_type:", model_type)
            print("=====================================")
            start_time = time.time()
            histories, test_accuracies = self.train_multiple_times(
                n_repetitions=num_repetitions,
                epochs=epochs,
                batch_size=batch_size,
                activation=input_activation,
                model_type=model_type,
                output_activation=output_activation,
                n_capas_ocultas=n_capas_ocultas,
            )
            training_time = time.time() - start_time
            best_val_accuracies = [max(h["val_accuracy"]) for h in histories]
            avg_best_val_accuracy = sum(best_val_accuracies) / len(best_val_accuracies)
            avg_test_accuracy = sum(test_accuracies) / len(test_accuracies)
            results.append(
                {
                    "batch_size": batch_size,
                    "training_time": training_time,
                    "val_accuracy": avg_best_val_accuracy,
                    "test_accuracy": avg_test_accuracy,
                    "test_accuracies": test_accuracies,
                    "best_val_accuracies": best_val_accuracies,
                    "histories": histories,
                }
            )
        return results

    def plot_batch_size_comparison(self, results, block=True):
        """
        Visualiza los resultados del experimento en un gráfico de barras
        """

        batch_sizes = [r["batch_size"] for r in results]
        accuracies = [r["val_accuracy"] * 100 for r in results]
        times = [r["training_time"] / 60 for r in results]  # convertir a minutos

        fig, ax1 = plt.subplots(figsize=(12, 6))

        x = np.arange(len(batch_sizes))
        width = 0.35
        rects1 = ax1.bar(
            x - width / 2, accuracies, width, label="acurracy (%)", color="skyblue"
        )
        ax1.set_ylabel("acurracy (%)")

        ax2 = ax1.twinx()
        rects2 = ax2.bar(x + width / 2, times, width, label="min", color="lightcoral")
        ax2.set_ylabel("min")

        ax1.set_xticks(x)
        ax1.set_xticklabels(batch_sizes)
        ax1.set_xlabel("Batch Size")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.title("Experimento de Batch Size con diferentes valor (CIFAR-10)")

        # Anotar las etiquetas en las barras
        def autolabel(rects, ax):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        autolabel(rects1, ax1)
        autolabel(rects2, ax2)

        plt.tight_layout()
        plt.show(block=True)

    def analyze_batch_size_results(self, results):
        best_val_accuracy = max(results, key=lambda x: x["val_accuracy"])
        best_test_accuracy = max(results, key=lambda x: x["test_accuracy"])
        fastest = min(results, key=lambda x: x["training_time"])

        print("\n=== Analysis ===")
        print(
            f"Best validation accuracy: {best_val_accuracy['val_accuracy']*100:.2f}% (batch_size = {best_val_accuracy['batch_size']})"
        )
        print(
            f"Best test accuracy: {best_test_accuracy['test_accuracy']*100:.2f}% (batch_size = {best_test_accuracy['batch_size']})"
        )
        print(
            f"Fastest training time: {fastest['training_time']/60:.2f} minutes (batch_size = {fastest['batch_size']})"
        )

        # Calcular la media y la desviación estándar de las métricas
        for r in results:
            r["val_accuracy_mean"] = np.mean(r["best_val_accuracies"])
            r["val_accuracy_std"] = np.std(r["best_val_accuracies"])
            r["test_accuracy_mean"] = np.mean(r["test_accuracies"])
            r["test_accuracy_std"] = np.std(r["test_accuracies"])

        print("\nBatch size performance summary:")
        for r in results:
            print(f"Batch size {r['batch_size']}:")
            print(
                f"  Val accuracy: {r['val_accuracy_mean']*100:.2f}% ± {r['val_accuracy_std']*100:.2f}%"
            )
            print(
                f"  Test accuracy: {r['test_accuracy_mean']*100:.2f}% ± {r['test_accuracy_std']*100:.2f}%"
            )
            print(f"  Training time: {r['training_time']/60:.2f} minutes")

        # Calcular un puntaje balanceado para cada configuración
        for r in results:
            accuracy_weight = 0.4
            test_accuracy_weight = 0.4
            speed_weight = 0.2
            max_val_accuracy = max(r["val_accuracy_mean"] for r in results)
            max_test_accuracy = max(r["test_accuracy_mean"] for r in results)
            min_time = min(r["training_time"] for r in results)

            normalized_val_accuracy = r["val_accuracy_mean"] / max_val_accuracy
            normalized_test_accuracy = r["test_accuracy_mean"] / max_test_accuracy
            normalized_speed = min_time / r["training_time"]

            r["score"] = (
                accuracy_weight * normalized_val_accuracy
                + test_accuracy_weight * normalized_test_accuracy
                + speed_weight * normalized_speed
            )

        best_balanced = max(results, key=lambda x: x["score"])
        print(f"\nRecommended batch_size: {best_balanced['batch_size']}")
        print(
            f"- Validation accuracy: {best_balanced['val_accuracy_mean']*100:.2f}% ± {best_balanced['val_accuracy_std']*100:.2f}%"
        )
        print(
            f"- Test accuracy: {best_balanced['test_accuracy_mean']*100:.2f}% ± {best_balanced['test_accuracy_std']*100:.2f}%"
        )
        print(f"- Training time: {best_balanced['training_time']/60:.2f} min")

        return best_balanced

    def get_best_batch_size(self, results):
        """
        obtener el mejor batch_size
        """
        analized_results = self.analyze_batch_size_results(results)
        return analized_results["batch_size"]


class ModelExperiment(BatchSizeExperiment):
    """
    Este clase es una implementacion de la tesis THE ALL CONVOLUTIONAL NET, que contiene 3 modelos diferentes, a, b, c y la derivada de c.
    """

    def build_model(
        self, input_activation, model_type, output_activation, n_capas_ocultas
    ):
        if model_type == "model_a":
            self.model_a(
                input_activation=input_activation, output_activation=output_activation
            )
        elif model_type == "model_b":
            self.model_b(
                input_activation=input_activation, output_activation=output_activation
            )
        elif model_type == "model_c":
            self.model_c(
                input_activation=input_activation, output_activation=output_activation
            )
        elif model_type == "strided_cnn_c":
            self.strided_cnn_c(
                input_activation=input_activation, output_activation=output_activation
            )
        elif model_type == "convPool_cnn_c":
            self.convPool_cnn_c(
                input_activation=input_activation, output_activation=output_activation
            )
        elif model_type == "all_cnn_c":
            self.all_cnn_c(
                input_activation=input_activation, output_activation=output_activation
            )
        elif model_type == "cnn":
            self.cnn_model(
                input_activation=input_activation, output_activation=output_activation
            )
        elif model_type == "mlp":
            self.mlp_model(
                input_activation=input_activation,
                output_activation=output_activation,
                n_capas_ocultas=n_capas_ocultas,
            )
        else:
            print("Model type not supported")
            exit()

    def model_a(self, input_activation, output_activation):
        print(
            "---------------------------------model_a---------------------------------"
        )
        inputs = keras.Input(shape=(32, 32, 3))

        # first block: 96 5x5 conv + max pooling
        x = layers.Conv2D(96, (5, 5), padding="same", activation=input_activation)(
            inputs
        )
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        x = layers.Dropout(0.5)(x)

        # second block: 192 5x5 conv + max pooling
        x = layers.Conv2D(192, (5, 5), padding="same", activation=input_activation)(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        x = layers.Dropout(0.5)(x)

        # comoon block: 192 3x3 conv + 192 1x1 conv + 10 1x1 conv
        x = layers.Conv2D(192, (3, 3), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(192, (1, 1), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(10, (1, 1), padding="same", activation=input_activation)(x)

        x = layers.GlobalAveragePooling2D()(x)

        outputs = layers.Dense(10, activation=output_activation)(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

    def model_b(self, input_activation, output_activation):
        print(
            "---------------------------------model_b---------------------------------"
        )
        inputs = keras.Input(shape=(32, 32, 3))

        # first block: 96 5x5 conv + 96 1x1 conv + max pooling
        x = layers.Conv2D(96, (5, 5), padding="same", activation=input_activation)(
            inputs
        )
        x = layers.Conv2D(96, (1, 1), padding="same", activation=input_activation)(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        x = layers.Dropout(0.5)(x)

        # second block: 192 5x5 conv + 192 1x1 conv + max pooling
        x = layers.Conv2D(192, (5, 5), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(192, (1, 1), padding="same", activation=input_activation)(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        x = layers.Dropout(0.5)(x)

        # comoon block: 192 3x3 conv + 192 1x1 conv + 10 1x1 conv
        x = layers.Conv2D(192, (3, 3), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(192, (1, 1), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(10, (1, 1), padding="same", activation=input_activation)(x)

        x = layers.GlobalAveragePooling2D()(x)

        outputs = layers.Dense(10, activation=output_activation)(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

    def model_c(self, input_activation, output_activation):
        print(
            "---------------------------------model_c---------------------------------"
        )
        inputs = keras.Input(shape=(32, 32, 3))

        # first block: 96 3x3 conv + max pooling
        x = layers.Conv2D(96, (3, 3), padding="same", activation=input_activation)(
            inputs
        )
        x = layers.Conv2D(96, (3, 3), padding="same", activation=input_activation)(x)
        x = layers.MaxPooling2D((3, 3), padding="same", strides=(2, 2))(x)
        x = layers.Dropout(0.5)(x)

        # second block: 192 3x3 conv + max pooling
        x = layers.Conv2D(192, (3, 3), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(192, (3, 3), padding="same", activation=input_activation)(x)
        x = layers.MaxPooling2D((3, 3), padding="same", strides=(2, 2))(x)
        x = layers.Dropout(0.5)(x)

        # comoon block: 192 3x3 conv + 192 1x1 conv + 10 1x1 conv
        x = layers.Conv2D(192, (3, 3), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(192, (1, 1), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(10, (1, 1), padding="same", activation=input_activation)(x)

        x = layers.GlobalAveragePooling2D()(x)

        outputs = layers.Dense(10, activation=output_activation)(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

    def strided_cnn_c(self, input_activation, output_activation):
        print(
            "---------------------------------strided_cnn_c---------------------------------"
        )
        inputs = keras.Input(shape=(32, 32, 3))

        # first block: 96 3x3 conv + 96 3x3 conv
        x = layers.Conv2D(96, (3, 3), padding="same", activation=input_activation)(
            inputs
        )
        x = layers.Conv2D(
            96, (3, 3), strides=(2, 2), padding="same", activation=input_activation
        )(x)
        x = layers.Dropout(0.5)(x)

        # second block: 192 3x3 conv + 192 3x3 conv
        x = layers.Conv2D(192, (3, 3), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(
            192, (3, 3), strides=(2, 2), padding="same", activation=input_activation
        )(x)
        x = layers.Dropout(0.5)(x)

        # higger layers derived from model c
        x = layers.Conv2D(192, (3, 3), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(192, (1, 1), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(10, (1, 1), padding="same", activation=input_activation)(x)

        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(10, activation=output_activation)(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

    def convPool_cnn_c(self, input_activation, output_activation):
        print(
            "---------------------------------convpool_cnn_c---------------------------------"
        )
        inputs = keras.Input(shape=(32, 32, 3))

        # first block
        x = layers.Conv2D(96, (3, 3), padding="same", activation=input_activation)(
            inputs
        )
        x = layers.Conv2D(96, (3, 3), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(96, (3, 3), padding="same", activation=input_activation)(x)
        x = layers.MaxPooling2D((3, 3), padding="same", strides=(2, 2))(x)
        x = layers.Dropout(0.5)(x)

        # second block
        x = layers.Conv2D(192, (3, 3), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(192, (3, 3), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(192, (3, 3), padding="same", activation=input_activation)(x)
        x = layers.MaxPooling2D((3, 3), padding="same", strides=(2, 2))(x)
        x = layers.Dropout(0.5)(x)

        # higer layers derived from model c
        x = layers.Conv2D(192, (3, 3), activation=input_activation)(x)
        x = layers.Conv2D(192, (1, 1), activation=input_activation)(x)
        x = layers.Conv2D(10, (1, 1), activation=input_activation)(x)

        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(10, activation=output_activation)(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

    def all_cnn_c(self, input_activation, output_activation):
        print(
            "---------------------------------all_cnn_c---------------------------------"
        )
        inputs = keras.Input(shape=(32, 32, 3))

        # first block
        x = layers.Conv2D(96, (3, 3), padding="same", activation=input_activation)(
            inputs
        )
        x = layers.Conv2D(96, (3, 3), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(
            96, (3, 3), strides=(2, 2), padding="same", activation=input_activation
        )(x)
        x = layers.Dropout(0.5)(x)

        # second block
        x = layers.Conv2D(192, (3, 3), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(192, (3, 3), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(
            192, (3, 3), strides=(2, 2), padding="same", activation=input_activation
        )(x)
        x = layers.Dropout(0.5)(x)

        # higer layers derived from model c
        x = layers.Conv2D(192, (3, 3), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(192, (1, 1), padding="same", activation=input_activation)(x)
        x = layers.Conv2D(10, (1, 1), padding="same", activation=input_activation)(x)

        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(10, activation=output_activation)(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)


class myOwnDataset(ModelExperiment):
    def __init__(self):
        super().__init__()  # cargar el dataset CIFAR-10
        
        dataset = keras.preprocessing.image_dataset_from_directory(
            "dataset",
            labels="inferred",
            label_mode="int", 
            image_size=(32, 32),
            seed=123,
        )
        self.x_test = []
        self.y_test = []
        for images, labels in dataset:
            self.x_test.extend(images.numpy())
            self.y_test.extend(labels.numpy())
            
        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test).reshape(-1, 1)
        
        self.x_test = self.x_test.astype("float32") / 255.0

import logging

if __name__ == "__main__":
    """
    Tarea A, B, C, D, E, F: mlp_model()
        En estos tarea se utiliza el modelo MLP, con diferentes configuraciones mencionado en el documento de la practica, para la ejecución, se debe cambiar el model_type a <mlp> y n_capas_ocultas a [32, 64, 128] si es de 3 capas ocultas, [32, 64] si es de 2 capas ocultas y [32] si es de 1 capa oculta etc. Tambien se puede cambiar el input_activation y output_activation, que es la funcion de activacion de la capa de ocultas y salida respectivamente. El funcion de la capa de salida se recomienda utilizar softmax siempre.

        Tambien se puede cambiar el numero de repeticiones, epochs y batch_sizes para la ejecución en codigos del abajo. 
        Para API como EarlyStopping, se debe de cambiar dentro de la funcion, buscar earlystopping y cambiar el parametro de patience, factor, mode, verbose, etc.

        Si quiere visualizar el resultado de diferentes batch_sizes, se debe descomentar la funcion plot_batch_size_comparison(results, block=True) en el codigo de abajo. Block es un parametro que usa para mostrar el grafico cuando todos a la vez. Si se produce fallo, por ejemplo, no se muestra el grafico y termina la ejecucion, se debe cambiar a block=False para mostrar uno por uno. 
        
        En defecto, al ejecutar el codigo se muestra una grafica del entrenamiento y una matriz de confusion. No es necesario cambiar este parametro.
    Tarea G, H: cnn_model()
        En estos tarea se utiliza el modelo CNN, con diferentes configuraciones mencionado en el documento de la practica.
    Tarea I: model_a(), model_b(), model_c(), strided_cnn_c(), convPool_cnn_c(), all_cnn_c()
        En estos tarea se utiliza el modelo derivado de la tesis THE ALL CONVOLUTIONAL NET, con diferentes configuraciones de las capas. Si se quiere cambiar el modelo, se debe cambiar el model_type a <model_a>, <model_b>, etc. Los otros parametros tambien son configurables como en la tarea A, B, C, D, E, F.

    Hasta aqui se usa la clase ModelExperiment, que contiene las funciones de entrenamiento, analisis y visualizacion de los resultados. 
    ----
    ----
    Tarea J, K, L: myOwnDataset()
        En estos tarea se utiliza el dataset propio, que se encuentra en la carpeta dataset. Todos los parametros son configurables como en las tareas anteriores. 

    Recomentacion: 
        Usar una tarjeta grafica potente para <model_a>, <model_b>, <model_c>, <strided_cnn_c>, <convPool_cnn_c>, <all_cnn_c> y <cnn_model> ya que estos modelos tienen muchas capas y requieren de mucho tiempo de entrenamiento.
        
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    """
    input_activation: str - Función de activación de la capa de entrada. Puede ser <sigmoid>, <relu>, <tanh>, <softplus>, etc.
    model_type: str - Tipo de modelo a utilizar. Puede ser <mlp>, <cnn>, <model_a>, <model_b>, <model_c>, <strided_cnn_c>, <convPool_cnn_c>, <all_cnn_c>.
    output_activation: str - Función de activación de la capa de salida. Puede ser "softmax", "sigmoid", etc.
    batch_sizes: list - Lista de tamaños de lote a probar.
    epochs: int - Número de épocas de entrenamiento.
    num_repetitions: int - Número de repeticiones para cada tamaño de lote.
    n_capas_ocultas: list - Número de neuronas en cada capa oculta para el modelo MLP.
    """
    input_activation = "relu"
    model_type = "strided_cnn_c"
    output_activation = "softmax"
    batch_size_list = [512]
    num_epochs = 100
    num_repetitions = 1
    num_capas_ocultas = [32, 64, 128]

    logging.info("Starting experiment with the following parameters:")
    logging.info(f"Input activation: {input_activation}")
    logging.info(f"Model type: {model_type}")
    logging.info(f"Output activation: {output_activation}")
    logging.info(f"Batch sizes: {batch_size_list}")
    logging.info(f"Epochs: {num_epochs}")
    logging.info(f"Repetitions: {num_repetitions}")
    logging.info(f"Hidden layers: {num_capas_ocultas}")

    try:
        """
        Se puede cambiar el experimento a ejecutar, por ejemplo:
        ModelExperiment(): Utilizar la validacion, test y train de CIFAR-10
        myOwnDataset(): Utilizar la validacion y train de Cifar10 y test de mi dataset
        """
        experiment = ModelExperiment()

        """
        ========================================================================================================
        NO CAMBIAR NADA DE AQUI PARA ABAJO. TODOS LAS CONFIGURACIONES SE DEBEN HACER ARRIBA DE ESTE COMENTARIO
        ========================================================================================================
        """
        results = experiment.run_batch_size_experiment(
            batch_sizes=batch_size_list,
            epochs=num_epochs,
            input_activation=input_activation,
            num_repetitions=num_repetitions,
            model_type=model_type,
            output_activation=output_activation,
            n_capas_ocultas=num_capas_ocultas,
        )

        # Analizar los resultados del experimento
        best_config = experiment.analyze_batch_size_results(results)
        # logging.info(f"Best configuration: {best_config}")

        best_batch_size = best_config["batch_size"]
        best_histories = [
            r["histories"] for r in results if r["batch_size"] == best_batch_size
        ][0]
        avg_history = experiment.calculate_average_history(best_histories)

        # Visualizar los resultados
        experiment.plot_average_training_history(
            avg_history, best_batch_size, activation=input_activation, block=False
        )

        experiment.plot_confusion_matrix(input_activation, output_activation)

        # experiment.plot_batch_size_comparison(results, block=True)
    except Exception as e:
        logging.error(f"An error occurred during the experiment: {str(e)}")
        raise

    logging.info("Experimento completado")
