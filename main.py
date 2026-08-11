import numpy as np
from apply_analysis import propagate_through_network
from graphics import sample_random_points, ask_theme
from zonotope import project_zonotope, batch_zonotope_contains


# -----------------------------------------------------------------------
# Intervalos Customizados
# -----------------------------------------------------------------------

custom_intervals = np.array([
    [7.0,   28.0],
    [10.0,  30.0],
    [40.0,  200.0],
    [140.0, 2500.0],
    [0.05,  0.16],
    [0.02,  0.35],
    [0.0,   0.43],
    [0.0,   0.20],
    [0.10,  0.30],
    [0.05,  0.10],
    [0.1,   2.5],
    [0.3,   4.0],
    [0.7,   22.0],
    [6.0,   540.0],
    [0.001, 0.031],
    [0.002, 0.135],
    [0.0,   0.396],
    [0.0,   0.053],
    [0.007, 0.079],
    [0.0,   0.030],
    [7.9,   36.0],
    [12.0,  49.0],
    [50.0,  252.0],
    [185.0, 4254.0],
    [0.07,  0.22],
    [0.027, 1.058],
    [0.0,   1.252],
    [0.0,   0.291],
    [0.15,  0.66],
    [0.055, 0.208],
])


# -----------------------------------------------------------------------
# Saídas brutas
# -----------------------------------------------------------------------

def get_raw_outputs(points):
    from neural_network import pipe
    mlp = pipe.named_steps["mlp"]
    scaler = pipe.named_steps["scaler"]

    X_scaled = scaler.transform(points)
    activation = X_scaled
    for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        z = activation @ W + b
        if i == len(mlp.coefs_) - 1:
            activation = z
        else:
            act = mlp.activation
            if act == 'tanh':
                activation = np.tanh(z)
            elif act == 'relu':
                activation = np.maximum(0, z)
            elif act == 'logistic':
                activation = 1 / (1 + np.exp(-z))
            else:
                activation = z

    return activation


# -----------------------------------------------------------------------
# Soundness
# -----------------------------------------------------------------------

def verify_soundness(points, output_zonos):
    """
    Verifica soundness testando, para cada ponto amostrado, se o vetor de
    saída completo pertence ao zonótopo conjunto (todos os neurônios de
    saída projetados juntos), via LP (zonotope_contains).
    """
    print("\n[SOUNDNESS] Verificando pontos do gráfico...")
    raw_outputs = get_raw_outputs(points)

    indices = list(range(len(output_zonos)))
    centers, G = project_zonotope(output_zonos, indices)

    inside = batch_zonotope_contains(centers, G, raw_outputs)
    violations = int(np.sum(~inside))

    n = len(points)
    if violations == 0:
        print(f"  [OK] Todos os {n} pontos têm saídas dentro do zonótopo calculado.")
    else:
        print(f"  [FALHA] {violations} violação(ões) em {n} pontos.")


# -----------------------------------------------------------------------
# Helpers de input
# -----------------------------------------------------------------------

def _ask_neuron(label, n_outputs, default):
    while True:
        raw = input(f"  Neurônio eixo {label} [0-{n_outputs-1}, Enter={default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if 0 <= val < n_outputs:
                return val
            print(f"  [ERRO] Digite um valor entre 0 e {n_outputs - 1}.")
        except ValueError:
            print("  [ERRO] Digite um número inteiro.")


def _ask_n_points(default=300):
    while True:
        raw = input(f"  Número de pontos [Enter={default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if val > 0:
                return val
            print("  [ERRO] Digite um valor maior que zero.")
        except ValueError:
            print("  [ERRO] Digite um número inteiro.")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    from neural_network import X_train
    n_features_dataset = X_train.shape[1]
    n_features_custom = len(custom_intervals)

    print(f"\n[INFO] Dataset atual: {n_features_dataset} features")
    if n_features_custom != n_features_dataset:
        print(f"[AVISO] custom_intervals tem {n_features_custom} entradas, "
              f"mas o dataset tem {n_features_dataset} features.")
    else:
        print(f"[OK] custom_intervals está correto ({n_features_custom} entradas).")

    print("\n1 - Intervalos customizados")
    print("2 - Intervalos do conjunto de treino\n")

    opcao = input("Opção: ").strip()

    if opcao == "1":
        intervals = [(float(row[0]), float(row[1])) for row in custom_intervals]
        input_intervals = intervals
        plot_intervals = intervals
    elif opcao == "2":
        from data import get_data, get_feature_intervals
        X_train_local = get_data()
        intervals = get_feature_intervals(X_train_local)
        input_intervals = None
        plot_intervals = intervals
    else:
        print("Opção inválida.")
        return

    output_zonos = propagate_through_network(input_intervals=input_intervals, verbose=True)
    n_outputs = len(output_zonos)
    amplitudes = [z.radius() * 2 for z in output_zonos]

    print(f"\n[SAÍDAS] {n_outputs} neurônio(s) de saída:")
    sorted_by_amp = sorted(range(n_outputs), key=lambda i: amplitudes[i], reverse=True)
    for i in range(n_outputs):
        lo, hi = output_zonos[i].interval()
        print(f"  Neurônio {i}: [{lo:.4f}, {hi:.4f}]  (amplitude={amplitudes[i]:.4f}, "
              f"n_gens={output_zonos[i].n_generators})")

    print("\n[CONFIGURAÇÃO DO GRÁFICO]")
    n_points = _ask_n_points(default=300)

    neuron_x = neuron_y = neuron_z = None
    if n_outputs == 1:
        pass
    elif n_outputs == 2:
        print("\n  Escolha os neurônios para os eixos:")
        neuron_x = _ask_neuron("X", n_outputs, default=sorted_by_amp[0])
        neuron_y = _ask_neuron("Y", n_outputs, default=sorted_by_amp[1])
    else:
        print("\n  Escolha os neurônios para os eixos:")
        neuron_x = _ask_neuron("X", n_outputs, default=sorted_by_amp[0])
        neuron_y = _ask_neuron("Y", n_outputs, default=sorted_by_amp[1])
        neuron_z = _ask_neuron("Z", n_outputs, default=sorted_by_amp[2])

    points = sample_random_points(plot_intervals, n_points=n_points)

    from neural_network import pipe
    activation_name = pipe.named_steps["mlp"].activation.capitalize()

    # ---- Tema do gráfico ---------------------------------------------
    ask_theme()

    verify_soundness(points, output_zonos)

    from graphics import plot_graph
    plot_graph(output_zonos, points,
               neuron_x=neuron_x, neuron_y=neuron_y, neuron_z=neuron_z,
               title=f"Análise Zonotópica ({activation_name})")


if __name__ == "__main__":
    main()