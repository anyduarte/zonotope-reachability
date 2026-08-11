import numpy as np
from zonotope import Zonotope
from data import get_model, get_scaler, get_data, get_feature_intervals


def normalize_intervals(intervals, mean, std):
    normalized = []
    for (lo, hi), m, s in zip(intervals, mean, std):
        lo_n = (lo - m) / s
        hi_n = (hi - m) / s
        normalized.append((min(lo_n, hi_n), max(lo_n, hi_n)))
    return normalized


def _interval_from_row(center, g_row):
    radius = np.sum(np.abs(g_row))
    return center - radius, center + radius


def _apply_relu_global(centers, G, verbose=False):
    """
    Aplica ReLU a cada neurônio sobre a representação global de geradores.

    Para cada neurônio j:
      - Inativo  (u <= 0): zera centro e linha de G.
      - Ativo    (l >= 0): mantém inalterado.
      - Misto    (l < 0 < u): linearização DeepZ.
          center[j] <- lam * center[j] + mu
          G[j, :]   <- lam * G[j, :]
          + nova coluna: mu em posição j, 0 no resto.

    Ao adicionar a coluna extra FORA do loop (ao final), garantimos que
    os índices de gerador de neurônios distintos nunca colidam.
    """
    n_neurons = len(centers)

    new_centers = centers.copy()
    new_G       = G.copy()
    extra_cols  = []

    for j in range(n_neurons):
        lo, hi = _interval_from_row(centers[j], G[j, :])

        if hi <= 0:
            new_centers[j] = 0.0
            new_G[j, :]    = 0.0

        elif lo >= 0:
            pass   # identidade

        else:
            lam = hi / (hi - lo)
            mu  = -lam * lo / 2.0

            new_centers[j] = lam * centers[j] + mu
            new_G[j, :]    = lam * G[j, :]

            col    = np.zeros(n_neurons)
            col[j] = mu
            extra_cols.append(col)

    if extra_cols:
        new_G = np.hstack([new_G, np.column_stack(extra_cols)])

    return new_centers, new_G


def _make_output_zonos(centers, G):
    return [
        Zonotope(center=np.float64(centers[j]), generators=G[j, :].copy())
        for j in range(len(centers))
    ]


def propagate_through_network(input_intervals=None, verbose=True) -> list:
    """
    Propaga intervalos de entrada pela rede neural usando aritmética
    zonotópica com ReLU.

    Usa índices globais de geradores: cada coluna de G tem significado
    único ao longo de TODA a rede, garantindo soundness mesmo quando
    ReLU introduz novos geradores de erro.

    Args:
        input_intervals: Lista de (min, max) por feature. None = treino.
        verbose        : Imprime detalhes da propagação.

    Returns:
        Lista de Zonotopes escalares — um por neurônio de saída.
    """
    layers  = get_model()
    scaler  = get_scaler()
    X_train = get_data()

    if input_intervals is None:
        raw_intervals = get_feature_intervals(X_train)
    else:
        raw_intervals = list(input_intervals)

    if verbose:
        print(f"\n[ANÁLISE ZONOTÓPICA — ReLU]")
        print(f"Features de entrada: {len(raw_intervals)}")
        for i, (lo, hi) in enumerate(raw_intervals):
            print(f"  Feature {i:2d}: [{lo:.4f}, {hi:.4f}]")

    norm = normalize_intervals(raw_intervals, scaler.mean_, scaler.scale_)

    # Representação inicial: cada feature = 1 gerador independente (diagonal)
    centers = np.array([(lo + hi) / 2.0 for lo, hi in norm])
    G       = np.diag([(hi - lo) / 2.0 for lo, hi in norm])

    if verbose:
        print(f"  Representação inicial: {len(centers)} neurônios, {G.shape[1]} geradores")

    # As linhas abaixo silenciam RuntimeWarnings espúrios que algumas
    # combinações de numpy/BLAS (comuns no Accelerate do macOS) emitem em
    # multiplicações de matriz com muitos zeros — isso não afeta o
    # resultado (os valores calculados batem com implementações sem esse
    # aviso).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for layer_idx, layer in enumerate(layers):
            W, b, act = layer["weights"], layer["bias"], layer["activation_name"]

            # Transformação afim (W tem shape n_in x n_out no sklearn)
            centers = W.T @ centers + b
            G       = W.T @ G

            if verbose:
                print(f"\nCamada {layer_idx+1}: {W.shape[0]}→{W.shape[1]}  "
                      f"ativação={act}  n_gens={G.shape[1]}")

            if act == 'relu':
                centers, G = _apply_relu_global(centers, G, verbose=verbose)

                if verbose:
                    n = len(centers)
                    idxs = sorted(set(list(range(min(3,n))) + list(range(max(3,n-3),n))))
                    for j in idxs:
                        lo, hi = _interval_from_row(centers[j], G[j])
                        print(f"    N{j:3d}: a=[{lo:.4f}, {hi:.4f}]  n_gens={G.shape[1]}")

    output_zonos = _make_output_zonos(centers, G)

    if verbose:
        print("\n── Saídas finais ──")
        for i, z in enumerate(output_zonos):
            lo, hi = z.interval()
            print(f"  Saída {i}: centro={float(z.center):.6f}  "
                  f"intervalo=[{lo:.6f}, {hi:.6f}]  n_gens={z.n_generators}")

    return output_zonos