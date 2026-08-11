from neural_network import *


def get_model():
    """
    Extrai, de cada camada da rede treinada, os pesos, o bias e o nome
    da função de ativação. A última camada é sempre tratada como
    'identity' (saída crua da rede, sem softmax/sigmoide aplicado).
    """
    mlp = pipe.named_steps["mlp"]

    layers = []
    num_layers = len(mlp.coefs_)

    for i in range(num_layers):
        if hasattr(mlp, 'out_activation_') and i == num_layers - 1:
            current_activation_name = 'identity'
        else:
            current_activation_name = mlp.activation

        layers.append({
            "weights": mlp.coefs_[i],
            "bias": mlp.intercepts_[i],
            "activation_name": current_activation_name
        })

    return layers


def get_feature_intervals(X):
    return list(zip(X.min(axis=0), X.max(axis=0)))


def get_data():
    return X_train


def get_scaler():
    return pipe.named_steps["scaler"]