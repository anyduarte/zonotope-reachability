import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import ConvexHull
from neural_network import data as dataset, pipe
from zonotope import project_zonotope, sample_zonotope_vertices, batch_zonotope_contains


# -----------------------------------------------------------------------
# Temas
# -----------------------------------------------------------------------

THEMES = {
    "escuro": {
        "BLUE":        "#4C9BE8",
        "RED":         "#E84C4C",
        "DARK":        "#1A1A2E",
        "BG":          "#0D0D1A",
        "RECT_FC":     "#2EC4B6",
        "RECT_EC":     "#A8FFEF",
        "TEXT":        "white",
        "GRID":        "white",
        "SPINE":       "#444",
        "LEGEND_BG":   "#12122a",
        "LEGEND_EDGE": "#444",
    },
    "claro": {
        "BLUE":        "#1565C0",
        "RED":         "#C62828",
        "DARK":        "#F0F0F0",
        "BG":          "#FFFFFF",
        "RECT_FC":     "#00897B",
        "RECT_EC":     "#004D40",
        "TEXT":        "#1A1A1A",
        "GRID":        "#999999",
        "SPINE":       "#CCCCCC",
        "LEGEND_BG":   "#EEEEEE",
        "LEGEND_EDGE": "#AAAAAA",
    },
}

_theme = THEMES["escuro"]


def set_theme(name: str):
    """Define o tema global. name = 'claro' ou 'escuro'."""
    global _theme
    name = name.strip().lower()
    if name not in THEMES:
        raise ValueError(f"Tema '{name}' inválido. Use 'claro' ou 'escuro'.")
    _theme = THEMES[name]


def ask_theme():
    """Pergunta ao usuário qual tema usar e aplica."""
    print("\n[TEMA DO GRÁFICO]")
    print("  1 — Escuro (padrão)")
    print("  2 — Claro")
    escolha = input("  Opção [Enter=1]: ").strip()
    if escolha == "2":
        set_theme("claro")
        print("  Tema claro aplicado.")
    else:
        set_theme("escuro")
        print("  Tema escuro aplicado.")


# -----------------------------------------------------------------------
# Helpers de estilo
# -----------------------------------------------------------------------

def _style_ax(ax):
    t = _theme
    ax.set_facecolor(t["BG"])
    ax.tick_params(colors=t["TEXT"])
    for spine in ax.spines.values():
        spine.set_edgecolor(t["SPINE"])
    ax.grid(alpha=0.12, color=t["GRID"])


def _legend(ax):
    t = _theme
    ax.legend(fontsize=9, facecolor=t["LEGEND_BG"], labelcolor=t["TEXT"],
              framealpha=0.85, edgecolor=t["LEGEND_EDGE"])


# -----------------------------------------------------------------------
# Amostragem
# -----------------------------------------------------------------------

def sample_random_points(intervals, n_points=10_000):
    n_features = len(intervals)
    points = np.zeros((n_points, n_features))
    for i, (low, high) in enumerate(intervals):
        points[:, i] = np.random.uniform(low, high, n_points)
    return points


# -----------------------------------------------------------------------
# Plot principal — Análise Zonotópica
# -----------------------------------------------------------------------

def plot_graph(output_zonos, points, neuron_x=None, neuron_y=None,
               neuron_z=None, save_path="output_plot.png",
               title="Análise Zonotópica", n_hull_samples=6000,
               verify_soundness=True):
    """
    Desenha a envoltória convexa (convex hull) do zonótopo projetado nos
    neurônios de saída escolhidos, junto com os pontos amostrados no
    domínio de entrada (dentro/fora do conjunto calculado).

    output_zonos: lista de Zonotope, um por neurônio de saída
                  (retorno de propagate_through_network).
    points:       pontos amostrados no domínio de entrada (mesmos usados
                  para verificar soundness).
    """
    from main import get_raw_outputs
    t = _theme

    n_outputs   = len(output_zonos)
    n_total     = len(points)
    raw_outputs = get_raw_outputs(points)

    amplitudes = [z.radius() * 2 for z in output_zonos]
    sorted_neurons = sorted(range(n_outputs), key=lambda i: amplitudes[i], reverse=True)

    # ============================================================== 1D
    if n_outputs == 1:
        x_lo, x_hi = output_zonos[0].interval()
        out_x  = raw_outputs[:, 0] if raw_outputs.ndim > 1 else raw_outputs
        inside = (out_x >= x_lo - 1e-9) & (out_x <= x_hi + 1e-9)
        n_in   = int(inside.sum())

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(t["DARK"])
        _style_ax(ax)

        ax.axvspan(x_lo, x_hi, alpha=0.20, color=t["RECT_FC"],
                   label="Intervalo de saída (zonótopo)")
        ax.axvline(x_lo, color=t["RECT_EC"], linewidth=2.0, linestyle="--")
        ax.axvline(x_hi, color=t["RECT_EC"], linewidth=2.0, linestyle="--")

        margin = max((x_hi - x_lo) * 0.15, 0.3)
        ax.set_xlim(out_x.min() - margin, out_x.max() + margin)

        ax.scatter(out_x[inside],  np.zeros(n_in), s=12,
                   color=t["BLUE"], alpha=0.6, label=f"Dentro ({n_in})")
        ax.scatter(out_x[~inside], np.zeros(n_total - n_in), s=12,
                   color=t["RED"], alpha=0.8, label=f"Fora ({n_total - n_in})")

        ax.set_xlabel("Saída — Neurônio 0", color=t["TEXT"], fontsize=11)
        ax.set_title("Análise Zonotópica vs Monte Carlo", color=t["TEXT"], fontsize=13)
        _legend(ax)
        fig.suptitle(f"{title} — {n_total} pontos", color=t["TEXT"], fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        plt.show()
        print(f"\nGráfico salvo: '{save_path}'")
        print(f"Dentro: {n_in}/{n_total}  ({100*n_in/n_total:.1f}%)")
        return

    # ============================================================== 2D
    if n_outputs == 2:
        neuron_x = neuron_x if neuron_x is not None else sorted_neurons[0]
        neuron_y = neuron_y if neuron_y is not None else sorted_neurons[1]

        indices = [neuron_x, neuron_y]
        centers, G = project_zonotope(output_zonos, indices)
        verts = sample_zonotope_vertices(centers, G, n_samples=n_hull_samples)
        hull = ConvexHull(verts)
        hull_pts = verts[hull.vertices]

        out_x = raw_outputs[:, neuron_x]
        out_y = raw_outputs[:, neuron_y]

        if verify_soundness:
            inside = batch_zonotope_contains(centers, G, np.column_stack([out_x, out_y]))
        else:
            # aproximação rápida via caixa envolvente (não exata)
            lo, hi = centers - np.abs(G).sum(axis=1), centers + np.abs(G).sum(axis=1)
            inside = ((out_x >= lo[0]) & (out_x <= hi[0]) &
                      (out_y >= lo[1]) & (out_y <= hi[1]))
        n_in = int(np.sum(inside))

        def _lims(arr, frac=0.20):
            mn, mx = arr.min(), arr.max()
            m = max((mx - mn) * frac, 0.3)
            return mn - m, mx + m

        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor(t["DARK"])
        _style_ax(ax)

        poly = Polygon(hull_pts, closed=True, linewidth=2.2,
                        edgecolor=t["RECT_EC"], facecolor=t["RECT_FC"],
                        alpha=0.20, label="Zonótopo (convex hull)", zorder=1)
        poly_edge = Polygon(hull_pts, closed=True, linewidth=2.2,
                             edgecolor=t["RECT_EC"], facecolor="none",
                             alpha=0.85, zorder=2)
        ax.add_patch(poly)
        ax.add_patch(poly_edge)

        ax.scatter(out_x[inside],  out_y[inside],  s=10, color=t["BLUE"],
                   alpha=0.45, label=f"Dentro ({n_in})", zorder=3)
        ax.scatter(out_x[~inside], out_y[~inside], s=10, color=t["RED"],
                   alpha=0.75, label=f"Fora ({n_total - n_in})", zorder=4)

        allx = np.concatenate([out_x, hull_pts[:, 0]])
        ally = np.concatenate([out_y, hull_pts[:, 1]])
        zx0, zx1 = _lims(allx); zy0, zy1 = _lims(ally)
        ax.set_xlim(zx0, zx1); ax.set_ylim(zy0, zy1)
        ax.set_xlabel(f"Saída — Neurônio {neuron_x}", color=t["TEXT"], fontsize=11)
        ax.set_ylabel(f"Saída — Neurônio {neuron_y}", color=t["TEXT"], fontsize=11)
        ax.set_title("Zoom nos pontos", color=t["TEXT"], fontsize=12)
        _legend(ax)
        fig.suptitle(f"{title} vs Monte Carlo — {n_total} pontos",
                     color=t["TEXT"], fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        plt.show()
        print(f"\nGráfico salvo: '{save_path}'")
        print(f"Neurônios: X={neuron_x}, Y={neuron_y}  |  Dentro: {n_in}/{n_total}  ({100*n_in/n_total:.1f}%)")
        return

    # ============================================================== 3D
    neuron_x = neuron_x if neuron_x is not None else sorted_neurons[0]
    neuron_y = neuron_y if neuron_y is not None else sorted_neurons[1]
    neuron_z = neuron_z if neuron_z is not None else sorted_neurons[2]

    indices = [neuron_x, neuron_y, neuron_z]
    centers, G = project_zonotope(output_zonos, indices)
    verts = sample_zonotope_vertices(centers, G, n_samples=n_hull_samples)
    hull = ConvexHull(verts)

    out_x = raw_outputs[:, neuron_x]
    out_y = raw_outputs[:, neuron_y]
    out_z = raw_outputs[:, neuron_z]

    if verify_soundness:
        inside = batch_zonotope_contains(centers, G, np.column_stack([out_x, out_y, out_z]))
    else:
        lo, hi = centers - np.abs(G).sum(axis=1), centers + np.abs(G).sum(axis=1)
        inside = ((out_x >= lo[0]) & (out_x <= hi[0]) &
                  (out_y >= lo[1]) & (out_y <= hi[1]) &
                  (out_z >= lo[2]) & (out_z <= hi[2]))
    n_in = int(np.sum(inside))

    def _lims3(arr, verts_1d, frac=0.08):
        mn = min(arr.min(), verts_1d.min()); mx = max(arr.max(), verts_1d.max())
        m = max((mx - mn) * frac, 0.3)
        return mn - m, mx + m

    lx0, lx1 = _lims3(out_x, verts[:, 0])
    ly0, ly1 = _lims3(out_y, verts[:, 1])
    lz0, lz1 = _lims3(out_z, verts[:, 2])

    fig = plt.figure(figsize=(9, 8))
    fig.patch.set_facecolor(t["BG"])
    fig.suptitle(
        f"{title} — 3D  ({n_total} pontos)\n"
        f"Neurônios: X={neuron_x}, Y={neuron_y}, Z={neuron_z}  |  Verde = convex hull do Zonótopo",
        color=t["TEXT"], fontsize=12)

    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_facecolor(t["BG"])

    faces = [verts[simplex] for simplex in hull.simplices]
    poly3d = Poly3DCollection(faces, alpha=0.18, facecolor=t["RECT_FC"],
                               edgecolor=t["RECT_EC"], linewidth=0.4)
    ax.add_collection3d(poly3d)

    ax.scatter(out_x[inside],  out_y[inside],  out_z[inside],
               s=7, color=t["BLUE"], alpha=0.40, label=f"Dentro ({n_in})")
    ax.scatter(out_x[~inside], out_y[~inside], out_z[~inside],
               s=7, color=t["RED"],  alpha=0.65, label=f"Fora ({n_total - n_in})")

    ax.set_xlim(lx0, lx1); ax.set_ylim(ly0, ly1); ax.set_zlim(lz0, lz1)
    ax.set_xlabel(f"Neurônio {neuron_x}", color=t["TEXT"], fontsize=9)
    ax.set_ylabel(f"Neurônio {neuron_y}", color=t["TEXT"], fontsize=9)
    ax.set_zlabel(f"Neurônio {neuron_z}", color=t["TEXT"], fontsize=9)
    ax.set_title("Zoom nos pontos", color=t["TEXT"], fontsize=11)
    ax.tick_params(colors=t["TEXT"])

    hull_proxy = Patch(facecolor=t["RECT_FC"], edgecolor=t["RECT_EC"],
                        alpha=0.6, label="Zonótopo (convex hull)")
    ax.legend(handles=[
        hull_proxy,
        plt.Line2D([0],[0], marker='o', color='w',
                   markerfacecolor=t["BLUE"], markersize=7, label=f"Dentro ({n_in})"),
        plt.Line2D([0],[0], marker='o', color='w',
                   markerfacecolor=t["RED"],  markersize=7, label=f"Fora ({n_total - n_in})"),
    ], fontsize=9, facecolor=t["LEGEND_BG"], labelcolor=t["TEXT"],
       framealpha=0.85, edgecolor=t["LEGEND_EDGE"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.show()
    print(f"\nGráfico salvo: '{save_path}'")
    print(f"Neurônios: X={neuron_x}, Y={neuron_y}, Z={neuron_z}  |  Dentro: {n_in}/{n_total}  ({100*n_in/n_total:.1f}%)")