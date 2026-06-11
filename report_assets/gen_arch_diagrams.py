"""
Generate architecture diagrams for SimpleCNN and LSTM models.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def rounded_box(ax, x, y, w, h, color, label_lines, shape_line=None,
                fontsize=8.5, shape_fontsize=7.5, zorder=3):
    """Draw a rounded-rectangle block with centred label + shape annotation."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.04",
        facecolor=color, edgecolor="#333333", linewidth=1.2,
        zorder=zorder
    )
    ax.add_patch(box)

    # Main label (possibly multi-line)
    cx, cy = x, y
    if shape_line:
        cy += h * 0.12

    ax.text(cx, cy, "\n".join(label_lines),
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color="#1a1a1a", zorder=zorder + 1,
            linespacing=1.35)

    if shape_line:
        ax.text(cx, y - h * 0.30,
                shape_line,
                ha="center", va="center",
                fontsize=shape_fontsize, color="#444444",
                style="italic", zorder=zorder + 1)


def arrow(ax, x1, y1, x2, y2, color="#555555"):
    ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>",
                                color=color,
                                lw=1.5,
                                mutation_scale=14),
                zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# Diagram 1 – SimpleCNN
# ─────────────────────────────────────────────────────────────────────────────

def make_simplecnn():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("#f8f9fa")

    # colour palette
    C_GREY  = "#d9d9d9"
    C_BLUE  = "#aec6e8"
    C_GREEN = "#b7e4b0"

    # --- block definitions (cx, cy, w, h, color, label_lines, shape) --------
    BW, BH = 1.45, 2.8      # block width / height
    GAP    = 1.65            # centre-to-centre horizontal gap
    CY     = 2.5             # vertical centre

    blocks = [
        # (cx, color, label_lines, shape_line)
        (0.80,  C_GREY,
         ["Input"],
         "(1, 128, 431)"),

        (0.80 + 1 * GAP, C_BLUE,
         ["Conv Block 1", "Conv2d(1→32, 3×3)", "BN → ReLU", "MaxPool(2,2)", "Drop2d(0.25)"],
         "(32, 64, 215)"),

        (0.80 + 2 * GAP, C_BLUE,
         ["Conv Block 2", "Conv2d(32→64, 3×3)", "BN → ReLU", "MaxPool(2,2)", "Drop2d(0.25)"],
         "(64, 32, 107)"),

        (0.80 + 3 * GAP, C_BLUE,
         ["Conv Block 3", "Conv2d(64→128, 3×3)", "BN → ReLU", "MaxPool(2,2)", "Drop2d(0.25)"],
         "(128, 16, 53)"),

        (0.80 + 4 * GAP, C_BLUE,
         ["Conv Block 4", "Conv2d(128→256, 3×3)", "BN → ReLU", "AdaptAvgPool(1,1)"],
         "(256, 1, 1)"),

        (0.80 + 5 * GAP, C_GREY,
         ["Flatten"],
         "(256,)"),

        (0.80 + 6 * GAP, C_GREEN,
         ["Dropout(0.5)", "Linear(256→10)"],
         "(10,)"),

        (0.80 + 7 * GAP, C_GREEN,
         ["Softmax", "Genre Prediction"],
         None),
    ]

    # Variable widths for first/last blocks
    widths = [1.0, BW, BW, BW, BW, 1.0, 1.3, 1.3]

    prev_rx = None
    for i, (cx, color, labels, shape) in enumerate(blocks):
        w = widths[i]
        rounded_box(ax, cx, CY, w, BH, color, labels,
                    shape_line=shape, fontsize=8, shape_fontsize=7.5)
        # arrow from previous block
        if prev_rx is not None:
            arrow(ax, prev_rx, CY, cx - w / 2, CY)
        prev_rx = cx + w / 2

    # Title
    ax.set_title("SimpleCNN Architecture  (422 K parameters)",
                 fontsize=13, fontweight="bold", pad=10, color="#1a1a1a")

    # Legend
    patches = [
        mpatches.Patch(facecolor=C_GREY,  edgecolor="#333", label="Input / Reshape"),
        mpatches.Patch(facecolor=C_BLUE,  edgecolor="#333", label="Convolutional Block"),
        mpatches.Patch(facecolor=C_GREEN, edgecolor="#333", label="Classifier Head"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              framealpha=0.85, edgecolor="#cccccc")

    plt.tight_layout(pad=0.4)
    out = "/Users/sanskar/dev/DL-GenAI-P/report_assets/arch_simplecnn.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Diagram 2 – LSTM
# ─────────────────────────────────────────────────────────────────────────────

def make_lstm():
    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor("#f8f9fa")

    # ── left panel: main flow ─────────────────────────────────────────────
    ax = fig.add_axes([0.03, 0.05, 0.60, 0.88])
    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 6)
    ax.axis("off")

    C_GREY   = "#d9d9d9"
    C_ORANGE = "#f7c59f"
    C_YELLOW = "#fff2a8"
    C_GREEN  = "#b7e4b0"

    BW, BH = 2.6, 0.78
    CX     = 3.75

    rows = [
        # (cy,   color,    label_lines,               shape_line)
        (5.45, C_GREY,   ["Input  —  MFCC Frames"],    "(batch, 431, 40)"),
        (4.45, C_ORANGE, ["LSTM Layer 1",
                           "hidden = 128,  dropout = 0.3"],
                          "(batch, 431, 128)"),
        (3.45, C_ORANGE, ["LSTM Layer 2",
                           "hidden = 128"],
                          "(batch, 431, 128)"),
        (2.50, C_YELLOW, ["Take Last Timestep  [t = −1]"],
                          "(batch, 128)"),
        (1.65, C_YELLOW, ["Dropout(0.3)"],             "(batch, 128)"),
        (0.80, C_GREEN,  ["Linear(128 → 10)"],         "(batch, 10)"),
        (0.00, C_GREEN,  ["Softmax  →  Genre Prediction"], None),
    ]

    prev_ty = None
    for cy, color, labels, shape in rows:
        rounded_box(ax, CX, cy, BW, BH, color, labels,
                    shape_line=shape, fontsize=8.5, shape_fontsize=7.5)
        if prev_ty is not None:
            arrow(ax, CX, prev_ty, CX, cy + BH / 2)
        prev_ty = cy - BH / 2

    ax.set_title("LSTM Architecture for Audio Genre Classification",
                 fontsize=12, fontweight="bold", x=0.5, y=1.01,
                 color="#1a1a1a")

    patches = [
        mpatches.Patch(facecolor=C_GREY,   edgecolor="#333", label="Input"),
        mpatches.Patch(facecolor=C_ORANGE, edgecolor="#333", label="LSTM Layer"),
        mpatches.Patch(facecolor=C_YELLOW, edgecolor="#333", label="Aggregation / Dropout"),
        mpatches.Patch(facecolor=C_GREEN,  edgecolor="#333", label="Classifier Head"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=7.5,
              framealpha=0.85, edgecolor="#cccccc",
              bbox_to_anchor=(0.0, 0.00))

    # ── right panel: LSTM cell inset ──────────────────────────────────────
    ax2 = fig.add_axes([0.66, 0.10, 0.32, 0.78])
    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, 5.5)
    ax2.axis("off")

    # Background card
    card = FancyBboxPatch((0.05, 0.05), 3.9, 5.4,
                          boxstyle="round,pad=0.12",
                          facecolor="#ffffff", edgecolor="#aaaaaa",
                          linewidth=1.5, zorder=1)
    ax2.add_patch(card)

    ax2.text(2.0, 5.15, "LSTM Cell Internals",
             ha="center", va="center", fontsize=9, fontweight="bold",
             color="#1a1a1a")

    # ── gate boxes ───────────────────────────────────────────────────────
    GATE_C = {"i": "#aec6e8",   # input gate  – blue
              "f": "#f7c59f",   # forget gate – orange
              "g": "#d4b8e0",   # cell gate   – purple
              "o": "#b7e4b0"}   # output gate – green
    GATE_LABEL = {
        "i": ["i", "Input Gate",  "σ(W_i·[h,x]+b)"],
        "f": ["f", "Forget Gate", "σ(W_f·[h,x]+b)"],
        "g": ["g", "Cell Gate",   "tanh(W_g·[h,x]+b)"],
        "o": ["o", "Output Gate", "σ(W_o·[h,x]+b)"],
    }

    # place gates in a 2×2 grid
    positions = {"f": (0.85, 3.85), "i": (2.55, 3.85),
                 "g": (0.85, 2.35), "o": (2.55, 2.35)}
    gw, gh = 1.30, 1.10

    for key, (gx, gy) in positions.items():
        box = FancyBboxPatch((gx - gw/2, gy - gh/2), gw, gh,
                             boxstyle="round,pad=0.05",
                             facecolor=GATE_C[key], edgecolor="#555555",
                             linewidth=1.0, zorder=3)
        ax2.add_patch(box)
        sym, name, eq = GATE_LABEL[key]
        ax2.text(gx, gy + 0.24, sym,
                 ha="center", va="center",
                 fontsize=14, fontweight="bold", color="#1a1a1a", zorder=4)
        ax2.text(gx, gy - 0.05, name,
                 ha="center", va="center",
                 fontsize=7, fontweight="bold", color="#333333", zorder=4)
        ax2.text(gx, gy - 0.32, eq,
                 ha="center", va="center",
                 fontsize=6, color="#555555", zorder=4, style="italic")

    # ── state / recurrence labels ─────────────────────────────────────────
    # inputs on the left
    for txt, ypos in [("x_t  (input)", 4.65), ("h_{t-1}  (prev hidden)", 3.10)]:
        ax2.annotate("",
                     xy=(0.22, ypos), xytext=(0.22, ypos),
                     arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.2))
        ax2.text(0.08, ypos, txt, ha="left", va="center",
                 fontsize=7, color="#444444", style="italic")

    # cell state bar (top)
    ax2.annotate("",
                 xy=(3.85, 4.65), xytext=(0.22, 4.65),
                 arrowprops=dict(arrowstyle="-|>", color="#777777",
                                 lw=1.5, connectionstyle="arc3,rad=0.0"))
    ax2.text(2.0, 4.90, "c_{t-1}  →  cell state  →  c_t",
             ha="center", va="center", fontsize=7.5,
             color="#555555", style="italic")

    # outputs on the right
    for txt, ypos in [("h_t  (hidden)", 1.55), ("c_t  (cell state)", 1.10)]:
        ax2.text(3.92, ypos, txt, ha="left", va="center",
                 fontsize=7, color="#444444", style="italic")
        ax2.annotate("",
                     xy=(3.85, ypos), xytext=(3.55, ypos),
                     arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.2))

    # separator line between gates and outputs
    ax2.plot([0.20, 3.80], [1.80, 1.80], color="#cccccc", lw=1, ls="--")
    ax2.text(2.0, 1.92, "Output",
             ha="center", va="center", fontsize=7.5,
             color="#888888", style="italic")

    # ── equations at bottom ───────────────────────────────────────────────
    eqs = [
        r"$c_t = f \odot c_{t-1} + i \odot g$",
        r"$h_t = o \odot \tanh(c_t)$",
    ]
    for k, eq in enumerate(eqs):
        ax2.text(2.0, 0.68 - k * 0.38, eq,
                 ha="center", va="center", fontsize=8,
                 color="#222222")

    out = "/Users/sanskar/dev/DL-GenAI-P/report_assets/arch_lstm.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    make_simplecnn()
    make_lstm()
    print("Done.")
