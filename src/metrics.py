"""
metrics.py — Comparativa RF-DETR vs YOLO para PSG (broadcast) y Madrid (cenital).

Genera 4 figuras PNG listas para presentación:
  outputs/metrics/fig1_psg_comparativa.png      — barras 4 detectores en 6 métricas
  outputs/metrics/fig2_detections_timeline.png  — detecciones/frame en el tiempo (2×2)
  outputs/metrics/fig3_resumen_tabla.png        — tabla comparativa completa (4 detectores)
  outputs/metrics/fig4_track_stability.png      — distribución longitud de tracks (2×2)

Uso:
  python src/metrics.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Rutas ─────────────────────────────────────────────────────────────────────
OUT_DIR = Path("outputs/metrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSVS = {
    "PSG\nRF-DETR":    OUT_DIR / "psg_rfdetr_500f_tracks.csv",
    "PSG\nYOLO":       OUT_DIR / "psg_yolo_500f_tracks.csv",
    "Madrid\nYOLO":    OUT_DIR / "FINAL_madrid_yolo_tracks.csv",
    "Madrid\nRF-DETR": OUT_DIR / "madrid_rfdetr_500f_tracks.csv",
}

_PLAYER_CLS = {"player", "goalkeeper"}

# ── Estilo ────────────────────────────────────────────────────────────────────
DARK_BG      = "#0f1117"
PANEL_BG     = "#1a1d27"
C_PSG_RF     = "#4fc3f7"   # azul claro   → PSG RF-DETR
C_PSG_YO     = "#ef5350"   # rojo         → PSG YOLO
C_MAD_YO     = "#66bb6a"   # verde        → Madrid YOLO
C_MAD_RF     = "#ffa726"   # ámbar        → Madrid RF-DETR
COLORS       = [C_PSG_RF, C_PSG_YO, C_MAD_YO, C_MAD_RF]
TEXT_COL     = "#e0e0e0"
GRID_COL     = "#2a2d3a"

DATASET_ORDER = ["PSG\nRF-DETR", "PSG\nYOLO", "Madrid\nYOLO", "Madrid\nRF-DETR"]
DATASET_COLORS = {
    "PSG\nRF-DETR":    C_PSG_RF,
    "PSG\nYOLO":       C_PSG_YO,
    "Madrid\nYOLO":    C_MAD_YO,
    "Madrid\nRF-DETR": C_MAD_RF,
}
DATASET_LABELS = {
    "PSG\nRF-DETR":    "PSG RF-DETR",
    "PSG\nYOLO":       "PSG YOLO",
    "Madrid\nYOLO":    "Madrid YOLO",
    "Madrid\nRF-DETR": "Madrid RF-DETR",
}

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_COL,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "text.color":        TEXT_COL,
    "grid.color":        GRID_COL,
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
})


# ══════════════════════════════════════════════════════════════════════════════
# Cálculo de métricas proxy (sin ground truth)
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(df: pd.DataFrame) -> dict:
    players = df[df["class"].isin(_PLAYER_CLS)]
    total_frames = df["frame"].nunique()

    det_per_frame = players.groupby("frame").size()
    det_mean = det_per_frame.mean()
    det_std  = det_per_frame.std()

    conf_median = df["confidence"].median()
    conf_p25    = df["confidence"].quantile(0.25)
    conf_p75    = df["confidence"].quantile(0.75)

    n_tracks = players["track_id"].nunique()

    track_len = players.groupby("track_id").size()
    avg_track_len = track_len.mean()
    med_track_len = track_len.median()

    consistent = 0
    for _, grp in players.groupby("track_id"):
        counts = grp["team_id"].value_counts()
        if len(counts) == 0:
            continue
        if counts.iloc[0] / len(grp) >= 0.85:
            consistent += 1
    team_consistency = consistent / max(n_tracks, 1) * 100

    team_switches = sum(
        1 for _, grp in players.groupby("track_id") if grp["team_id"].nunique() > 1
    )
    pct_switches = team_switches / max(n_tracks, 1) * 100

    ball = df[df["class"] == "ball"]
    ball_rate = ball["frame"].nunique() / total_frames * 100

    tracks_per_100f = n_tracks / total_frames * 100

    return {
        "det_mean":         round(det_mean, 1),
        "det_std":          round(det_std, 1),
        "conf_median":      round(conf_median, 3),
        "conf_p25":         round(conf_p25, 3),
        "conf_p75":         round(conf_p75, 3),
        "n_tracks":         int(n_tracks),
        "avg_track_len":    round(avg_track_len, 1),
        "med_track_len":    round(med_track_len, 1),
        "team_consistency": round(team_consistency, 1),
        "pct_switches":     round(pct_switches, 1),
        "ball_rate":        round(ball_rate, 1),
        "tracks_per_100f":  round(tracks_per_100f, 1),
        "total_frames":     int(total_frames),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Figura 1 — 6 métricas × 4 detectores
# ══════════════════════════════════════════════════════════════════════════════

def fig1_bar_comparison(metrics: dict) -> None:
    short_labels = ["PSG\nRF-DETR", "PSG\nYOLO", "Mad.\nYOLO", "Mad.\nRF-DETR"]

    metric_specs = [
        ("Jugadores detectados\npor frame (media)", "det_mean",         "",   True),
        ("Confianza media (%)",                      "conf_median",      "%",  True),
        ("Longitud media de track\n(frames)",        "avg_track_len",    "f",  True),
        ("Consistencia de equipo\n(% tracks estables)", "team_consistency", "%", True),
        ("Tasa detección balón\n(% frames)",         "ball_rate",        "%",  True),
        ("Tracks únicos jugador\n(menos = mejor)",   "n_tracks",         "",   False),
    ]

    n = len(metric_specs)
    fig, axes = plt.subplots(1, n, figsize=(20, 5.8), facecolor=DARK_BG)
    fig.suptitle("RF-DETR vs YOLO — PSG (Broadcast) y Madrid (Cenital)  |  500 frames",
                 fontsize=14, fontweight="bold", color=TEXT_COL, y=1.03)

    for ax, (label, key, unit, higher_better) in zip(axes, metric_specs):
        vals = []
        for ds in DATASET_ORDER:
            v = metrics[ds][key]
            if unit == "%":
                v = v * 100 if key == "conf_median" else v
            vals.append(v)

        bar_colors = COLORS
        bars = ax.bar(short_labels, vals, color=bar_colors,
                      width=0.6, zorder=3, edgecolor=DARK_BG, linewidth=1.0)

        winner_idx = (vals.index(max(vals)) if higher_better else vals.index(min(vals)))
        bars[winner_idx].set_edgecolor("white")
        bars[winner_idx].set_linewidth(2.5)

        top = max(vals)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + top * 0.03,
                    f"{val:.1f}{unit}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=TEXT_COL)

        ax.set_title(label, fontsize=9.5, color=TEXT_COL, pad=8)
        ax.set_ylim(0, top * 1.30)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID_COL)

    patches = [mpatches.Patch(color=c, label=DATASET_LABELS[k])
               for k, c in DATASET_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10,
               framealpha=0.3, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    path = OUT_DIR / "fig1_psg_comparativa.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Guardada → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figura 2 — Timeline de detecciones (2×2)
# ══════════════════════════════════════════════════════════════════════════════

def fig2_detections_timeline(dfs: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), facecolor=DARK_BG)
    axes = axes.flatten()
    fig.suptitle("Detecciones de jugador por frame a lo largo del tiempo",
                 fontsize=14, fontweight="bold", color=TEXT_COL, y=1.01)

    configs = [
        ("PSG\nRF-DETR",    C_PSG_RF, "PSG (Broadcast) — RF-DETR"),
        ("PSG\nYOLO",       C_PSG_YO, "PSG (Broadcast) — YOLO"),
        ("Madrid\nYOLO",    C_MAD_YO, "Madrid (Cenital) — YOLO"),
        ("Madrid\nRF-DETR", C_MAD_RF, "Madrid (Cenital) — RF-DETR"),
    ]

    for ax, (key, color, title) in zip(axes, configs):
        df = dfs[key]
        players = df[df["class"].isin(_PLAYER_CLS)]
        det = players.groupby("frame").size().reset_index(name="count")
        det["smooth"] = det["count"].rolling(15, min_periods=1, center=True).mean()

        ax.fill_between(det["frame"], det["count"], alpha=0.18, color=color)
        ax.plot(det["frame"], det["smooth"], color=color, linewidth=2.0)
        ax.scatter(det["frame"], det["count"], s=1.5, color=color, alpha=0.4, zorder=2)

        mean_v = det["count"].mean()
        ax.axhline(mean_v, color="white", linewidth=1.0, linestyle="--", alpha=0.5)
        ax.text(det["frame"].max() + 2, mean_v + 0.2,
                f"μ={mean_v:.1f}", color="white", fontsize=9, va="bottom")

        ax.set_title(title, fontsize=11, color=TEXT_COL, pad=4, loc="left")
        ax.set_ylabel("Jugadores\ndetectados", fontsize=9)
        ax.set_ylim(0, det["count"].max() + 3)
        ax.grid(True, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(GRID_COL)

    axes[2].set_xlabel("Frame", fontsize=10)
    axes[3].set_xlabel("Frame", fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.45, wspace=0.30)
    path = OUT_DIR / "fig2_detections_timeline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Guardada → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figura 3 — Tabla resumen + distribución de confianza (4 detectores)
# ══════════════════════════════════════════════════════════════════════════════

def fig3_summary(metrics: dict, dfs: dict) -> None:
    fig = plt.figure(figsize=(19, 7), facecolor=DARK_BG)
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1], wspace=0.06)

    # ── Panel izquierdo: tabla ────────────────────────────────────────────────
    ax_tab = fig.add_subplot(gs[0])
    ax_tab.set_facecolor(PANEL_BG)
    ax_tab.axis("off")

    col_labels = ["Métrica", "PSG\nRF-DETR", "PSG\nYOLO", "Madrid\nYOLO", "Madrid\nRF-DETR"]
    m = {k: metrics[k] for k in DATASET_ORDER}

    row_specs = [
        # (label, key, row_idx_for_lower_is_better)
        ("Jugadores / frame (μ)",   "det_mean",         False),
        ("Desv. estándar det/f",    "det_std",          True),   # lower = more stable
        ("Confianza mediana",       "conf_median",      False),
        ("Tracks únicos jugador",   "n_tracks",         True),   # lower = better
        ("Long. media track (f)",   "avg_track_len",    False),
        ("Consistencia equipo (%)", "team_consistency", False),
        ("Detección balón (%)",     "ball_rate",        False),
        ("Frames analizados",       "total_frames",     None),   # neutral
    ]

    row_data = []
    for label, key, lower_better in row_specs:
        if key == "conf_median":
            row = [label] + [f"{m[ds][key]:.3f}" for ds in DATASET_ORDER]
        else:
            row = [label] + [m[ds][key] for ds in DATASET_ORDER]
        row_data.append(row)

    n_cols = 5  # 1 label + 4 detectors
    cell_colors = []
    for row, (_, key, lower_better) in zip(row_data, row_specs):
        try:
            fvals = [float(v) for v in row[1:]]
        except (ValueError, TypeError):
            cell_colors.append(["#1a1d27"] * n_cols)
            continue

        if lower_better is None:
            cell_colors.append(["#1a1d27"] * n_cols)
            continue

        best_idx  = fvals.index(min(fvals)) if lower_better else fvals.index(max(fvals))
        worst_idx = fvals.index(max(fvals)) if lower_better else fvals.index(min(fvals))
        row_c = ["#1a1d27"]
        for j in range(4):
            if j == best_idx:
                row_c.append("#1b3a1b")
            elif j == worst_idx:
                row_c.append("#3a1b1b")
            else:
                row_c.append("#1a1d27")
        cell_colors.append(row_c)

    table = ax_tab.table(
        cellText=row_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.05)

    col_header_colors = {1: C_PSG_RF, 2: C_PSG_YO, 3: C_MAD_YO, 4: C_MAD_RF}
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COL)
        cell.set_text_props(color=TEXT_COL)
        if r == 0:
            cell.set_facecolor("#252836")
            col_color = col_header_colors.get(c, TEXT_COL)
            cell.set_text_props(fontweight="bold", color=col_color)

    ax_tab.set_title("Tabla comparativa de métricas proxy\n(verde = mejor, rojo = peor)",
                     fontsize=12, color=TEXT_COL, pad=14)

    # ── Panel derecho: distribución de confianza ──────────────────────────────
    ax_box = fig.add_subplot(gs[1])
    ax_box.set_facecolor(PANEL_BG)

    tick_labels = ["PSG\nRF-DETR", "PSG\nYOLO", "Madrid\nYOLO", "Madrid\nRF-DETR"]
    data_conf   = [dfs[k]["confidence"].values for k in DATASET_ORDER]
    colors_c    = COLORS

    bp = ax_box.boxplot(
        data_conf, tick_labels=tick_labels, patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color=TEXT_COL),
        capprops=dict(color=TEXT_COL),
        flierprops=dict(marker=".", color=TEXT_COL, alpha=0.3, markersize=3),
        widths=0.42,
    )
    for patch, color in zip(bp["boxes"], colors_c):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    for i, (data, color) in enumerate(zip(data_conf, colors_c), start=1):
        vp = ax_box.violinplot([data], positions=[i], widths=0.52,
                               showmeans=False, showmedians=False, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.2)
            body.set_edgecolor("none")

    ax_box.set_ylabel("Confianza de detección", fontsize=10)
    ax_box.set_title("Distribución de confianza\npor detector y vídeo",
                     fontsize=12, color=TEXT_COL, pad=14)
    ax_box.set_ylim(0, 1.05)
    ax_box.axhline(0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.4)
    ax_box.grid(axis="y", zorder=0)
    ax_box.set_axisbelow(True)
    for spine in ax_box.spines.values():
        spine.set_color(GRID_COL)
    ax_box.tick_params(axis="x", labelsize=9)

    path = OUT_DIR / "fig3_resumen_tabla.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Guardada → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figura 4 — Track length distribution (2×2)
# ══════════════════════════════════════════════════════════════════════════════

def fig4_track_stability(dfs: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), facecolor=DARK_BG)
    axes = axes.flatten()
    fig.suptitle("Estabilidad del tracking — Distribución de longitud de tracks",
                 fontsize=13, fontweight="bold", color=TEXT_COL, y=1.02)

    configs = [
        ("PSG\nRF-DETR",    C_PSG_RF, "PSG (Broadcast) — RF-DETR"),
        ("PSG\nYOLO",       C_PSG_YO, "PSG (Broadcast) — YOLO"),
        ("Madrid\nYOLO",    C_MAD_YO, "Madrid (Cenital) — YOLO"),
        ("Madrid\nRF-DETR", C_MAD_RF, "Madrid (Cenital) — RF-DETR"),
    ]

    for ax, (key, color, title) in zip(axes, configs):
        df = dfs[key]
        players = df[df["class"].isin(_PLAYER_CLS)]
        track_lens = players.groupby("track_id").size().values

        bins = np.linspace(1, track_lens.max() + 1, 25)
        ax.hist(track_lens, bins=bins, color=color, alpha=0.8,
                edgecolor=DARK_BG, linewidth=0.6, zorder=3)

        mean_l = track_lens.mean()
        ax.axvline(mean_l, color="white", linewidth=1.5, linestyle="--")
        ax.text(mean_l + track_lens.max() * 0.02, ax.get_ylim()[1] * 0.85,
                f"μ={mean_l:.0f}f", color="white", fontsize=10)

        ax.set_title(title, fontsize=10.5, color=TEXT_COL, pad=6)
        ax.set_xlabel("Longitud del track (frames)", fontsize=9)
        ax.set_ylabel("Nº de tracks", fontsize=9)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(GRID_COL)

        pct_short = (track_lens <= 10).sum() / len(track_lens) * 100
        ax.text(0.97, 0.95, f"{pct_short:.0f}% tracks\n≤10 frames",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=TEXT_COL,
                bbox=dict(facecolor=PANEL_BG, edgecolor=GRID_COL,
                          boxstyle="round,pad=0.4", alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.45, wspace=0.30)
    path = OUT_DIR / "fig4_track_stability.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Guardada → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Cargando CSVs...")
    dfs: dict[str, pd.DataFrame] = {}
    for name, path in CSVS.items():
        if not path.exists():
            print(f"  AVISO: no encontrado {path}")
            continue
        dfs[name] = pd.read_csv(path)
        print(f"  {name.replace(chr(10), ' ')}: {len(dfs[name])} filas, "
              f"{dfs[name]['frame'].nunique()} frames")

    print("\nCalculando métricas...")
    metrics = {name: compute_metrics(df) for name, df in dfs.items()}

    print("\n=== RESUMEN DE MÉTRICAS ===")
    for name, m in metrics.items():
        print(f"\n{name.replace(chr(10), ' ')}:")
        for k, v in m.items():
            print(f"  {k:25s}: {v}")

    print("\nGenerando figuras...")
    fig1_bar_comparison(metrics)
    fig2_detections_timeline(dfs)
    fig3_summary(metrics, dfs)
    fig4_track_stability(dfs)

    print("\nDone — 4 figuras guardadas en outputs/metrics/")


if __name__ == "__main__":
    main()
