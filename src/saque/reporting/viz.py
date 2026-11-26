"""Figuras del reporte (PNG estáticos con matplotlib).

Color: los saques son ordinales -> rampa azul de un solo tono (claro = primer saque, oscuro = último),
para que la progresión de la serie (fatiga) se lea sin leyenda; la leyenda se incluye igual.
Fuentes de marcadores (categórico): paleta fija en orden.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
ORDINAL_BLUE = ["#84b9f3", "#519ced", "#257de0", "#1b64be", "#0f4a96", "#0e3d7e"]
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"

plt.rcParams.update({
    "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "lines.linewidth": 2, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "legend.frameon": False,
})


def serve_colors(n: int) -> list[str]:
    idx = np.linspace(0, len(ORDINAL_BLUE) - 1, max(n, 1)).round().astype(int)
    return [ORDINAL_BLUE[i] for i in idx]


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_serve_detection(kin_by_view: dict[str, pd.DataFrame], serves: pd.DataFrame,
                         wrist: str, path: Path) -> Path:
    """Altura de la muñeca de golpeo en todo el video con los impactos detectados."""
    fig, axes = plt.subplots(len(kin_by_view), 1, figsize=(10, 2.4 * len(kin_by_view)),
                             squeeze=False)
    for ax, (view, kin) in zip(axes[:, 0], kin_by_view.items()):
        ax.plot(kin["time_s"], kin[f"y_{wrist}"], color=SERIES[0], lw=1.2)
        for _, s in serves[serves["vista"] == view].iterrows():
            ax.axvspan(s["t_ini"], s["t_fin"], color=SERIES[0], alpha=0.08, lw=0)
            ax.axvline(s["t_impacto"], color=TEXT, lw=0.8, ls="--")
            ax.annotate(f"S{int(s['saque'])}", (s["t_impacto"], 1), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", color=TEXT, fontsize=8)
        ax.set_ylabel("altura muñeca (m)")
        ax.set_title(f"Vista {view}: saques detectados", loc="left")
    axes[-1, 0].set_xlabel("tiempo del video (s)")
    fig.tight_layout()
    return _save(fig, path)


def plot_marker_sources(sources: pd.DataFrame, path: Path) -> Path:
    """Fracción de fotogramas por fuente de cada marcador (estampita / MediaPipe / hueco)."""
    views = list(sources["vista"].unique())
    fig, axes = plt.subplots(1, len(views), figsize=(5 * len(views), 5), squeeze=False)
    labels = [("frac_estampita", "estampita"), ("frac_pose", "MediaPipe"), ("frac_hueco", "hueco")]
    for ax, view in zip(axes[0], views):
        d = sources[sources["vista"] == view].iloc[::-1]
        left = np.zeros(len(d))
        for (col, lab), color in zip(labels, [SERIES[0], SERIES[2], "#bdbdbd"]):
            ax.barh(d["marcador"], d[col], left=left, color=color, label=lab, height=0.7,
                    edgecolor="white", linewidth=1)
            left += d[col].to_numpy()
        ax.set_xlim(0, 1)
        ax.set_xlabel("fracción de fotogramas")
        ax.set_title(f"Vista {view}", loc="left")
        ax.grid(axis="y", visible=False)
    axes[0, 0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    fig.suptitle("Origen de cada marcador", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_normalized_curves(curves: pd.DataFrame, units: dict, path: Path) -> Path:
    """Curvas inicio->impacto (0-100 %) de cada saque, en small multiples por variable."""
    variables = list(curves["variable"].unique())
    ncol = 3
    nrow = int(np.ceil(len(variables) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.8 * nrow), squeeze=False)
    trials = sorted(curves["trial"].unique(), key=lambda s: int(s.split("_")[-1]))
    colors = dict(zip(trials, serve_colors(len(trials))))
    for ax, var in zip(axes.flat, variables):
        g = curves[curves["variable"] == var]
        for tr in trials:
            d = g[g["trial"] == tr]
            ax.plot(d["percent"], d["value"], color=colors[tr], label=tr.replace("saque_", "S"))
        ax.set_title(var.replace("_", " "), loc="left")
        ax.set_ylabel(units.get(var, ""))
        ax.set_xlabel("% inicio → impacto")
    for ax in list(axes.flat)[len(variables):]:
        ax.set_visible(False)
    axes.flat[0].legend(title="saque", ncol=len(trials), loc="lower left", fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def plot_fatigue(values: pd.DataFrame, trends: pd.DataFrame, path: Path) -> Path:
    """Valor por saque de cada indicador con su recta de tendencia."""
    variables = list(values["variable"].unique())
    ncol = 3
    nrow = int(np.ceil(len(variables) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow), squeeze=False)
    tr = trends.set_index("variable")
    for ax, var in zip(axes.flat, variables):
        g = values[values["variable"] == var].dropna(subset=["valor"]).sort_values("saque")
        x, y = g["saque"].to_numpy(float), g["valor"].to_numpy(float)
        ax.plot(x, y, color=SERIES[0], marker="o", ms=6, lw=1.2,
                markeredgecolor="white", markeredgewidth=1.5)
        title = var.replace("_", " ")
        if var in tr.index and "confianza" in tr and isinstance(tr.loc[var, "confianza"], str):
            title += f"  [confianza {tr.loc[var, 'confianza']}]"
        if var in tr.index and "pendiente_por_saque" in tr and np.isfinite(tr.loc[var, "pendiente_por_saque"]):
            b = tr.loc[var, "pendiente_por_saque"]
            a = y.mean() - b * x.mean()
            ax.plot(x, a + b * x, color=TEXT2, lw=1, ls="--")
            title += f"\nρ={tr.loc[var, 'rho_spearman']:+.2f}  Δ1→n={tr.loc[var, 'cambio_primero_ultimo_pct']:+.1f} %"
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xticks(x)
        ax.set_xlabel("saque")
        ax.set_ylabel(g["unidad"].iloc[0] if len(g) else "")
    for ax in list(axes.flat)[len(variables):]:
        ax.set_visible(False)
    fig.suptitle("Indicadores por saque (posible fatiga; exploratorio, n = 5)", x=0.02,
                 ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_jacobian(jac_by_serve: dict[int, pd.DataFrame], path: Path) -> Path:
    """Rapidez de muñeca directa vs Jacobiano, un panel por saque."""
    n = len(jac_by_serve)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3), sharey=True, squeeze=False)
    for ax, (s, d) in zip(axes[0], sorted(jac_by_serve.items())):
        ax.plot(d["time_s"] - d["time_s"].iloc[0], d["speed_directa"], color=SERIES[0], label="directa")
        ax.plot(d["time_s"] - d["time_s"].iloc[0], d["speed_J"], color=SERIES[1], lw=1.2,
                ls="--", label="J(θ)ω")
        ax.set_title(f"Saque {s}", loc="left")
        ax.set_xlabel("t (s)")
    axes[0, 0].set_ylabel("rapidez muñeca (m/s)")
    axes[0, 0].legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


def plot_inverse_dynamics(id_by_serve: dict[int, pd.DataFrame], joints: dict[str, str],
                          title: str, path: Path) -> Path:
    """Momento articular neto por articulación; un color por saque (rampa ordinal)."""
    fig, axes = plt.subplots(len(joints), 1, figsize=(8, 2.4 * len(joints)), sharex=True,
                             squeeze=False)
    colors = serve_colors(len(id_by_serve))
    for ax, (j, lab) in zip(axes[:, 0], joints.items()):
        for c, (s, d) in zip(colors, sorted(id_by_serve.items())):
            t = d["time_s"] - d["time_s"].iloc[-1]
            ax.plot(t, d[f"T_{j}_Nm"], color=c, lw=1.4, label=f"S{s}")
        ax.axhline(0, color=TEXT2, lw=0.8)
        ax.set_ylabel(f"{lab} (N·m)")
    axes[0, 0].legend(ncol=len(id_by_serve), loc="upper left", fontsize=7)
    axes[-1, 0].set_xlabel("tiempo relativo al final de la ventana (s)")
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


SKELETON = [("hombro", "codo"), ("codo", "muneca"), ("hombro", "cadera"), ("cadera", "rodilla"),
            ("rodilla", "tobillo"), ("tobillo", "punta")]


def plot_impact_montage(frames: list[tuple[int, np.ndarray, pd.Series]], view: str,
                        path: Path) -> Path:
    """Fotograma de impacto de cada saque con el esqueleto y los marcadores usados."""
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 4.8), squeeze=False)
    for ax, (s, img, row) in zip(axes[0], frames):
        ax.imshow(img)
        for side in ("izq", "der"):
            for a, b in SKELETON:
                pa, pb = f"{a}_{side}", f"{b}_{side}"
                if f"x_{pa}" in row and f"x_{pb}" in row:
                    ax.plot([row[f"x_{pa}"], row[f"x_{pb}"]], [row[f"y_{pa}"], row[f"y_{pb}"]],
                            color="white", lw=1.5)
        for c in [c for c in row.index if c.startswith("src_")]:
            m = c[4:]
            if pd.notna(row.get(f"x_{m}")):
                color = SERIES[1] if row[c] == "estampita" else SERIES[0]
                ax.plot(row[f"x_{m}"], row[f"y_{m}"], "o", ms=4, color=color,
                        markeredgecolor="white", markeredgewidth=0.8)
        ax.set_title(f"Saque {s}", loc="left")
        ax.axis("off")
    fig.suptitle(f"Vista {view}: impacto (naranja = estampita, azul = MediaPipe)", x=0.02,
                 ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def make_figures(ctx) -> list[Path]:
    """Genera todas las figuras posibles a partir de las tablas y datos procesados."""
    from ..video.reader import frame_times, probe, read_frame

    tables, fig_dir = ctx.paths.tables, ctx.paths.figures
    figs: list[Path] = []
    serves = pd.read_csv(tables / "saques.csv")
    kin = {v: pd.read_parquet(ctx.view_dir(v) / "kinematics.parquet") for v in ctx.cfg["views"]}
    figs.append(plot_serve_detection(kin, serves, ctx.wrist, fig_dir / "deteccion_saques.png"))
    figs.append(plot_marker_sources(pd.read_csv(tables / "fuente_marcadores.csv"),
                                    fig_dir / "fuente_marcadores.png"))
    units = {k: v.get("unit", "") for k, v in ctx.cfg["fatigue"]["variables"].items()}
    if (tables / "curvas_normalizadas.csv").exists():
        figs.append(plot_normalized_curves(pd.read_csv(tables / "curvas_normalizadas.csv"), units,
                                           fig_dir / "curvas_por_saque.png"))
    figs.append(plot_fatigue(pd.read_csv(tables / "fatiga_valores_por_saque.csv"),
                             pd.read_csv(tables / "fatiga_tendencias.csv"),
                             fig_dir / "fatiga_tendencias.png"))
    for view in ctx.cfg["views"]:
        vd = ctx.view_dir(view)
        jac = {int(p.parent.name.split("_")[1]): pd.read_csv(p) for p in sorted(vd.glob("saque_*/jacobian.csv"))}
        if jac:
            figs.append(plot_jacobian(jac, fig_dir / f"jacobiano_{view}.png"))
        for name, joints in (("tren_superior", {"shoulder": "hombro", "elbow": "codo"}),
                             ("tren_inferior", {"hip": "cadera", "knee": "rodilla", "ankle": "tobillo"})):
            idd = {int(p.parent.name.split("_")[1]): pd.read_csv(p)
                   for p in sorted(vd.glob(f"saque_*/dinamica_inversa_{name}.csv"))}
            if idd:
                figs.append(plot_inverse_dynamics(
                    idd, joints, f"Vista {view}: momentos netos, {name.replace('_', ' ')} (exploratorio)",
                    fig_dir / f"dinamica_inversa_{name}_{view}.png"))
        video = ctx.paths.video(view, ctx.cfg)
        if not video.is_file():
            continue
        px = pd.read_parquet(vd / "markers_px.parquet")
        info = probe(video)
        times = frame_times(info)
        frames = []
        for _, s in serves[serves["vista"] == view].iterrows():
            row = px.iloc[int(np.argmin(np.abs(px["time_s"] - s["t_impacto"])))]
            if pd.notna(row["frame"]):
                frames.append((int(s["saque"]), read_frame(info, int(row["frame"]), times), row))
        if frames:
            figs.append(plot_impact_montage(frames, view, fig_dir / f"impactos_{view}.png"))
    return figs
