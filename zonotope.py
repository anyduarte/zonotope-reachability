"""
zonotope.py
-----------
Define a classe Zonotope, usada por `apply_analysis.py` para representar
cada neurônio de saída da rede como um zonótopo escalar:

    Z = center + sum_i(alpha_i * generators[i]),   alpha_i em [-1, 1]

Também traz funções auxiliares para:
  - projetar vários neurônios (Zonotopes) em um único zonótopo
    multidimensional (mesmo espaço de geradores globais);
  - verificar se um ponto pertence a um zonótopo (soundness), resolvendo
    um problema de viabilidade linear (LP);
  - amostrar vértices do zonótopo para fins de visualização (o zonótopo
    é a envoltória convexa dos pontos obtidos combinando os geradores
    com alpha_i em {-1, 1}).
"""

import numpy as np
from scipy.optimize import linprog


class Zonotope:
    """
    Representa um zonótopo escalar (um neurônio de saída):

        Z = (center; generators) = { center + g . alpha | alpha in [-1,1]^m }

    Atributos
    ---------
    center : float
        Centro do zonótopo (valor escalar).
    generators : np.ndarray, shape (m,)
        Vetor de geradores. m é o número de geradores globais acumulados
        ao longo da propagação pela rede (1 por feature de entrada + 1
        por neurônio ReLU misto encontrado em cada camada).
    """

    def __init__(self, center, generators):
        self.center = np.float64(center)
        self.generators = np.asarray(generators, dtype=np.float64).ravel()

    @property
    def n_generators(self):
        return self.generators.shape[0]

    def radius(self):
        """Raio do intervalo (soma dos valores absolutos dos geradores)."""
        return float(np.sum(np.abs(self.generators)))

    def interval(self):
        """Sobreaproximação intervalar (caixa) do zonótopo escalar."""
        r = self.radius()
        return (float(self.center) - r, float(self.center) + r)

    def __repr__(self):
        lo, hi = self.interval()
        return (f"Zonotope(center={self.center:.4f}, "
                f"interval=[{lo:.4f}, {hi:.4f}], "
                f"n_gens={self.n_generators})")


# -----------------------------------------------------------------------
# Projeção de vários neurônios em um zonótopo conjunto (multidimensional)
# -----------------------------------------------------------------------

def project_zonotope(output_zonos, indices):
    """
    Combina os Zonotopes de `indices` (todos compartilham o mesmo espaço
    de geradores globais) em uma representação conjunta:

        centers : np.ndarray, shape (k,)
        G       : np.ndarray, shape (k, m)

    onde k = len(indices) e m = número de geradores globais.
    """
    centers = np.array([output_zonos[i].center for i in indices], dtype=np.float64)
    G = np.vstack([output_zonos[i].generators for i in indices])
    return centers, G


# -----------------------------------------------------------------------
# Verificação de pertinência (soundness) via programação linear
# -----------------------------------------------------------------------

def zonotope_contains(centers, G, point, tol=1e-7):
    """
    Verifica se `point` pertence ao zonótopo (centers; G), isto é, se
    existe alpha em [-1, 1]^m tal que:

        centers + G @ alpha = point

    Resolvido como um problema de viabilidade linear (LP): minimiza uma
    função objetivo nula sujeita à igualdade acima e aos limites de alpha.
    """
    point = np.asarray(point, dtype=np.float64)
    m = G.shape[1]

    c = np.zeros(m)
    bounds = [(-1.0, 1.0)] * m
    b_eq = point - centers

    res = linprog(c, A_eq=G, b_eq=b_eq, bounds=bounds, method="highs")
    return bool(res.success)


def batch_zonotope_contains(centers, G, points, tol=1e-7):
    """Aplica `zonotope_contains` a um conjunto de pontos (linhas de `points`)."""
    return np.array([zonotope_contains(centers, G, p, tol=tol) for p in points])


# -----------------------------------------------------------------------
# Amostragem de vértices para visualização
# -----------------------------------------------------------------------

def sample_zonotope_vertices(centers, G, n_samples=4000, random_state=None):
    """
    Amostra vértices do zonótopo (centers; G).

    Os vértices exatos de um zonótopo ocorrem sempre em combinações com
    alpha_i em {-1, +1} (vértices do hipercubo). Para um número grande de
    geradores m, enumerar todos os 2^m vértices é inviável; em vez disso,
    amostram-se sinais aleatórios, o que produz uma aproximação por pontos
    da fronteira, suficiente para desenhar a envoltória convexa (convex
    hull) do zonótopo projetado em 2D/3D.
    """
    rng = np.random.default_rng(random_state)
    m = G.shape[1]
    signs = rng.choice([-1.0, 1.0], size=(n_samples, m))
    points = centers + signs @ G.T
    return points