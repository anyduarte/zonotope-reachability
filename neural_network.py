from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np

# Exemplo 1: Classificação (Iris - 3 classes)
#data = load_iris()
#is_classification = True

# Exemplo 2: Classificação (Digits - 10 classes)
data = load_digits()
is_classification = True

# Exemplo 3: Regressão (Diabetes - valor contínuo)
# data = load_diabetes()
# is_classification = False

# Exemplo 4: Regressão (Boston Housing - deprecated, use California Housing)
# from sklearn.datasets import fetch_california_housing
# data = fetch_california_housing()
# is_classification = False

# Exemplo 5: Classificação (Câncer de mama - 2 classes)
# data = load_breast_cancer()
# is_classification = True

# Exemplo 6: Classificação (MNIST - dígitos manuscritos, 10 classes)
#_mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
#data = _mnist; data.target = data.target.astype(int); data.target_names = np.array([str(i) for i in range(10)])
#is_classification = True

if is_classification:
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, stratify=data.target, random_state=42, test_size=0.25
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, random_state=42, test_size=0.25
    )

# Pipeline - usa MLPRegressor para regressão, MLPClassifier para classificação

if is_classification:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            # hidden_layer_sizes=(5, 10, 20, 35, 20, 10, 5),
            # hidden_layer_sizes=(5, 20, 50, 20, 5),
            hidden_layer_sizes=(16, 12, 8),
            activation='relu',
            max_iter=1000,
            random_state=42
        ))
    ])
else:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(16, 12, 8),
            activation='relu',
            max_iter=3000,
            learning_rate_init=0.01,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            solver='adam',
            alpha=0.001
        ))
    ])

pipe.fit(X_train, y_train)

print("\n------ DADOS DA REDE NEURAL TREINADA ------\n")

if is_classification:
    print(f"Acurácia Treino: {pipe.score(X_train, y_train):.4f}")
    print(f"Acurácia Teste: {pipe.score(X_test, y_test):.4f}")
else:
    print(f"R² Score Treino: {pipe.score(X_train, y_train):.4f}")
    print(f"R² Score Teste: {pipe.score(X_test, y_test):.4f}")

print(f"\nTipo de problema: {'Classificação' if is_classification else 'Regressão'}")
print(f"Features: {X_train.shape[1]}")
print(f"Amostras treino: {X_train.shape[0]}")
print(f"Amostras teste: {X_test.shape[0]}")

if is_classification:
    n_classes = len(np.unique(y_train))
    print(f"Número de classes: {n_classes}")