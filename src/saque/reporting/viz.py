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
    # Cuerpo completo
    if "y_com" in next(iter(kin.values())):
        figs.append(plot_com_grf(kin, serves, fig_dir / "com_fuerza_suelo.png"))
    if (tables / "posturas.csv").exists():
        post = pd.read_csv(tables / "posturas.csv")
        for view in ctx.cfg["views"]:
            if (post["vista"] == view).any():
                figs.append(plot_postures(post, view, fig_dir / f"posturas_{view}.png"))
    if (tables / "rom.csv").exists():
        figs.append(plot_rom(pd.read_csv(tables / "rom.csv"), fig_dir / "rango_movimiento.png"))
    if "posterior" in kin and "theta_incl_linea_hombros" in kin["posterior"]:
        figs.append(plot_frontal(kin["posterior"], serves, fig_dir / "plano_frontal_posterior.png"))
    if (tables / "picos_dinamica.csv").exists():
        figs.append(plot_id_peaks(pd.read_csv(tables / "picos_dinamica.csv"),
                                  fig_dir / "dinamica_inversa_picos.png"))
    if "lateral" in kin and "x_com" in kin["lateral"]:
        figs.append(plot_trajectories(kin["lateral"], serves, ctx.wrist, fig_dir / "trayectorias.png"))
    figs += _basic_figures(ctx, serves, kin)

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
        bcfg = ctx.cfg["body"]
        if view == bcfg.get("lever_view") and (tables / "events.csv").exists():
            from ..analysis.body import center_of_mass
            ev = pd.read_csv(tables / "events.csv")
            ev = ev[(ev["vista"] == view) & (ev["evento"] == "max_flex_rodilla")]
            lf = []
            for _, e in ev.sort_values("saque").iterrows():
                row = px.iloc[int(np.argmin(np.abs(px["time_s"] - e["t"])))]
                if pd.isna(row["frame"]):
                    continue
                x_com = float(center_of_mass(row.to_frame().T.reset_index(drop=True))["x_com"].iloc[0])
                lf.append((int(e["saque"]), read_frame(info, int(row["frame"]), times), row, x_com))
            if lf:
                figs.append(plot_levers_montage(
                    lf, bcfg["lever_joints"], fig_dir / f"palancas_{view}.png",
                    f"Vista {view}, máxima flexión de rodilla: línea del peso (roja) y brazos de "
                    "palanca (naranja)"))
    return figs


# ---------------------------------------------------------------- cuerpo completo
EVENT_LABELS = {"inicio": "inicio", "inicio_flexion_rodilla": "inicio\ncarga",
                "max_flex_rodilla": "máx. flexión\nrodilla", "impacto": "impacto", "fin": "fin"}
NICE = {"theta_interior_hombro": "hombro (brazo-tronco)", "theta_interior_codo": "codo",
        "theta_interior_cadera": "cadera", "theta_interior_rodilla": "rodilla",
        "theta_interior_tobillo": "tobillo", "theta_incl_tronco": "inclinación tronco",
        "theta_incl_linea_hombros": "inclinación línea hombros",
        "theta_incl_linea_caderas": "inclinación línea caderas",
        "theta_sep_hombros_caderas": "hombros − caderas",
        "theta_interior_rodilla_izq": "rodilla izq."}


def _serve_windows(kin: pd.DataFrame, serves: pd.DataFrame, pre=1.6, post=0.8):
    for _, s in serves.sort_values("saque").iterrows():
        m = (kin["time_s"] >= s["t_impacto"] - pre) & (kin["time_s"] <= s["t_impacto"] + post)
        yield int(s["saque"]), kin.loc[m], kin.loc[m, "time_s"] - s["t_impacto"]


def plot_com_grf(kin_by_view: dict[str, pd.DataFrame], serves: pd.DataFrame, path: Path) -> Path:
    """Elevación del CoM y fuerza vertical estimada (×peso), alineadas en el impacto."""
    views = list(kin_by_view)
    fig, axes = plt.subplots(2, len(views), figsize=(5.2 * len(views), 6.2), sharex=True,
                             squeeze=False)
    for j, view in enumerate(views):
        sv = serves[serves["vista"] == view]
        colors = serve_colors(len(sv))
        for c, (s, k, t) in zip(colors, _serve_windows(kin_by_view[view], sv)):
            base = np.nanmedian(k["y_com"].iloc[:36])
            axes[0, j].plot(t, 100 * (k["y_com"] - base), color=c, label=f"S{s}")
            axes[1, j].plot(t, k["grf_z_bw"], color=c, lw=1.4)
        axes[0, j].set_title(f"Vista {view}", loc="left")
        axes[1, j].axhline(1, color=TEXT2, lw=0.8, ls="--")
        axes[1, j].text(-1.55, 1.05, "peso corporal", color=TEXT2, fontsize=7)
        for ax in axes[:, j]:
            ax.axvline(0, color=TEXT, lw=0.8, ls=":")
        axes[1, j].set_xlabel("tiempo relativo al impacto (s)")
    axes[0, 0].set_ylabel("elevación del CoM (cm)")
    axes[1, 0].set_ylabel("fuerza vertical estimada (× peso)")
    axes[0, 0].legend(ncol=5, fontsize=7, loc="upper left")
    fig.suptitle("Centro de masa y fuerza vertical del suelo estimada F = m(a_CoM + g)",
                 x=0.02, ha="left", fontweight="bold")
    fig.text(0.02, -0.01, "La fuerza cerca del impacto no es fiable (brazo de golpeo con huecos y "
             "alta aceleración); interpretar solo el impulso de piernas (antes de −0.2 s).",
             fontsize=7, color=TEXT2)
    fig.tight_layout()
    return _save(fig, path)


def plot_postures(posturas: pd.DataFrame, view: str, path: Path) -> Path:
    """Ángulo de cada articulación en cada evento, un trazo por saque."""
    d = posturas[(posturas["vista"] == view) & posturas["variable"].str.startswith("theta")]
    variables = [v for v in NICE if v in set(d["variable"])]
    events = [e for e in EVENT_LABELS if e in set(d["evento"])]
    ncol = 3
    nrow = int(np.ceil(len(variables) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.8 * nrow), squeeze=False)
    serves = sorted(d["saque"].unique())
    colors = dict(zip(serves, serve_colors(len(serves))))
    for ax, var in zip(axes.flat, variables):
        g = d[d["variable"] == var]
        for s in serves:
            gs = g[g["saque"] == s].set_index("evento").reindex(events)
            ax.plot(range(len(events)), gs["valor"], color=colors[s], marker="o", ms=5,
                    markeredgecolor="white", label=f"S{s}")
        ax.set_xticks(range(len(events)), [EVENT_LABELS[e] for e in events], fontsize=7)
        ax.set_title(NICE[var], loc="left")
        ax.set_ylabel("°")
    for ax in list(axes.flat)[len(variables):]:
        ax.set_visible(False)
    axes.flat[0].legend(ncol=len(serves), fontsize=7, loc="best")
    fig.suptitle(f"Vista {view}: ángulos en los eventos del saque", x=0.02, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_rom(rom: pd.DataFrame, path: Path) -> Path:
    """Rango de movimiento por articulación: media (barra) y cada saque (puntos)."""
    views = list(rom["vista"].unique())
    fig, axes = plt.subplots(1, len(views), figsize=(5.5 * len(views), 4), squeeze=False)
    for ax, view in zip(axes[0], views):
        d = rom[rom["vista"] == view]
        order = d.groupby("variable")["rom_deg"].mean().sort_values()
        y = np.arange(len(order))
        ax.barh(y, order.values, color=SERIES[0], alpha=0.35, height=0.6)
        for i, var in enumerate(order.index):
            vals = d.loc[d["variable"] == var, "rom_deg"]
            ax.plot(vals, np.full(len(vals), i), "o", color=SERIES[0], ms=5, markeredgecolor="white")
        ax.set_yticks(y, [NICE.get(v, v) for v in order.index], fontsize=8)
        ax.set_xlabel("rango de movimiento inicio→impacto (°)")
        ax.set_title(f"Vista {view}", loc="left")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Rango de movimiento por saque (barra = media, puntos = saques)", x=0.02,
                 ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_frontal(kin: pd.DataFrame, serves: pd.DataFrame, path: Path) -> Path:
    """Vista posterior: inclinación de líneas de hombros/caderas y del tronco alrededor del impacto."""
    cols = ["theta_incl_linea_hombros", "theta_incl_linea_caderas", "theta_incl_tronco"]
    cols = [c for c in cols if c in kin]
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 3.4), sharex=True, squeeze=False)
    sv = serves[serves["vista"] == "posterior"]
    colors = serve_colors(len(sv))
    for ax, col in zip(axes[0], cols):
        for c, (s, k, t) in zip(colors, _serve_windows(kin, sv)):
            ax.plot(t, np.degrees(k[col]), color=c, label=f"S{s}", lw=1.5)
        ax.axvline(0, color=TEXT, lw=0.8, ls=":")
        ax.axhline(0, color=TEXT2, lw=0.6)
        ax.set_title(NICE[col], loc="left")
        ax.set_xlabel("t − impacto (s)")
    axes[0, 0].set_ylabel("°  (0 = horizontal / vertical)")
    axes[0, 0].legend(ncol=5, fontsize=7)
    fig.suptitle("Vista posterior (plano frontal); huecos = cuerpo de perfil a la cámara",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_id_peaks(picos: pd.DataFrame, path: Path) -> Path:
    """Picos de momento y potencia articular por saque (dinámica inversa exploratoria)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    labels = {"T_shoulder_Nm": "hombro", "T_elbow_Nm": "codo", "T_hip_Nm": "cadera",
              "T_knee_Nm": "rodilla", "T_ankle_Nm": "tobillo"}
    for ax, pref, unit in ((axes[0], "T_", "N·m"), (axes[1], "P_", "W")):
        d = picos[picos["variable"].str.startswith(pref)]
        keys = [k.replace("T_", pref).replace("_Nm", "_W" if pref == "P_" else "_Nm") for k in labels]
        keys = [k for k in keys if k in set(d["variable"])]
        serves = sorted(d["saque"].unique())
        colors = dict(zip(serves, serve_colors(len(serves))))
        for i, k in enumerate(keys):
            g = d[d["variable"] == k]
            ax.bar(i, g["pico_abs"].median(), color=SERIES[0], alpha=0.3, width=0.6)
            for _, r in g.iterrows():
                ax.plot(i + 0.12 * (r["saque"] - 3), r["pico_abs"], "o", ms=5,
                        color=colors[r["saque"]], markeredgecolor="white")
        names = [labels[k.replace("P_", "T_").replace("_W", "_Nm")] for k in keys]
        ax.set_xticks(range(len(keys)), names)
        ax.set_ylabel(f"pico |{pref[0]}| ({unit})")
        ax.set_title("Momento neto" if pref == "T_" else "Potencia articular", loc="left")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Dinámica inversa (exploratoria: GRF estimada por el CoM, sin raqueta; barra = mediana)",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_trajectories(kin: pd.DataFrame, serves: pd.DataFrame, wrist: str, path: Path) -> Path:
    """Trayectoria 2D de la muñeca de golpeo y del CoM (vista lateral), inicio→fin."""
    sv = serves[serves["vista"] == "lateral"]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    colors = serve_colors(len(sv))
    for c, (s, k, _) in zip(colors, _serve_windows(kin, sv)):
        x0 = np.nanmedian(k["x_com"].iloc[:36])
        ax.plot(k[f"x_{wrist}"] - x0, k[f"y_{wrist}"], color=c, lw=1.4, label=f"S{s} muñeca")
        ax.plot(k["x_com"] - x0, k["y_com"], color=c, lw=2.4, ls="--")
    ax.plot([], [], color=TEXT2, lw=2.4, ls="--", label="CoM")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x relativo al CoM inicial (m)")
    ax.set_ylabel("altura sobre el suelo (m)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Vista lateral: trayectorias de la muñeca y del CoM", loc="left")
    fig.tight_layout()
    return _save(fig, path)


def plot_levers_montage(frames: list[tuple[int, np.ndarray, pd.Series, float]], joints: list[str],
                        path: Path, title: str) -> Path:
    """Postura con la línea de acción del peso (vertical por el CoM) y brazos de palanca."""
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 4.8), squeeze=False)
    for ax, (s, img, row, x_com) in zip(axes[0], frames):
        ax.imshow(img)
        h = img.shape[0]
        ax.plot([x_com, x_com], [0, h], color=SERIES[7], lw=1.5)
        for j in joints:
            if pd.notna(row.get(f"x_{j}")):
                x, y = row[f"x_{j}"], row[f"y_{j}"]
                ax.plot([x, x_com], [y, y], color=SERIES[1], lw=2)
                ax.plot(x, y, "o", color=SERIES[1], ms=4, markeredgecolor="white")
        ax.set_title(f"Saque {s}", loc="left")
        ax.axis("off")
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------- análisis básicos adicionales
PHASE_COLORS = {"preparacion": "#bdbdbd", "carga": SERIES[0], "aceleracion": SERIES[1],
                "seguimiento": SERIES[2]}
JOINT_NAMES = {"hombro": "hombro (brazo-tronco)", "codo": "codo", "cadera": "cadera",
               "rodilla": "rodilla", "tobillo": "tobillo", "incl_tronco": "inclinación tronco"}


def plot_phases(phases: pd.DataFrame, path: Path) -> Path:
    """Duración de cada fase por saque: barras horizontales apiladas, una fila por vista."""
    views = list(phases["vista"].unique())
    order = [f for f in PHASE_COLORS if f in set(phases["fase"])]
    fig, axes = plt.subplots(len(views), 1, figsize=(8, 1.0 + 0.45 * phases["saque"].nunique() * len(views)),
                             sharex=True, squeeze=False)
    for ax, view in zip(axes[:, 0], views):
        p = phases[phases["vista"] == view].pivot_table(index="saque", columns="fase",
                                                        values="duracion_ms").sort_index(ascending=False)
        left = np.zeros(len(p))
        for f in order:
            v = p[f].fillna(0).to_numpy()
            ax.barh([f"S{s}" for s in p.index], v, left=left, color=PHASE_COLORS[f], label=f,
                    height=0.7, edgecolor="white", linewidth=2)
            for y, (l0, w) in enumerate(zip(left, v)):
                if w > 120:
                    ax.text(l0 + w / 2, y, f"{w:.0f}", ha="center", va="center", color="white", fontsize=8)
            left += v
        ax.set_title(f"Vista {view}", loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
    axes[-1, 0].set_xlabel("ms desde el inicio del saque (impacto = fin de la aceleración)")
    axes[0, 0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=len(order))
    fig.tight_layout()
    return _save(fig, path)


def plot_kinematic_sequence(kin_by_serve: dict[int, pd.DataFrame], impacts: dict[int, float],
                            starts: dict[int, float], sequence: dict[str, str], peaks: pd.DataFrame,
                            path: Path) -> Path:
    """Secuencia cinemática: |ω| (o rapidez) normalizada a su pico, un panel por saque.

    Cada serie se divide entre su máximo en la ventana para comparar el *momento* de los picos;
    los puntos marcan el pico usado en la tabla de secuencia.
    """
    n = len(kin_by_serve)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.9 * n), sharex=True, squeeze=False)
    labels = list(sequence)
    colors = dict(zip(labels, SERIES))
    for ax, (s, k) in zip(axes[:, 0], sorted(kin_by_serve.items())):
        t0, ti = starts[s], impacts[s]
        m = (k["time_s"] >= max(t0, ti - 0.6)) & (k["time_s"] <= ti + 0.15)
        t = 1000 * (k.loc[m, "time_s"] - ti)
        for lab, col in sequence.items():
            if col not in k:
                continue
            y = np.abs(k.loc[m, col].to_numpy(float))
            if not np.isfinite(y).any():
                continue
            ax.plot(t, y / np.nanmax(y), color=colors[lab], lw=1.6, label=lab)
            pk = peaks[(peaks["saque"] == s) & (peaks["serie"] == lab)]
            if len(pk):
                tp = pk["t_rel_impacto_ms"].iloc[0]
                ax.plot([tp], [1.0], "o", color=colors[lab], ms=8, markeredgecolor="white",
                        markeredgewidth=2)
        ax.axvline(0, color=TEXT, lw=0.8, ls="--")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(f"S{s}\n|·| / pico")
    axes[0, 0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=len(labels))
    axes[-1, 0].set_xlabel("ms respecto al impacto (línea discontinua)")
    fig.suptitle("Secuencia cinemática próximo-distal (vista lateral)", x=0.02, ha="left",
                 fontweight="bold", y=1.04)
    fig.tight_layout()
    return _save(fig, path)


def plot_joint_kinematics(curves: pd.DataFrame, bands: pd.DataFrame, path: Path) -> Path:
    """θ, ω y α de cada articulación (inicio -> impacto): saques finos + media ± DE."""
    joints = [j for j in ("hombro", "codo", "cadera", "rodilla", "tobillo")
              if f"theta_{j}" in set(curves["variable"])]
    kinds = [("theta", "θ (°)"), ("omega", "ω (rad/s)"), ("alpha", "α (rad/s²)")]
    fig, axes = plt.subplots(len(joints), 3, figsize=(12, 2.2 * len(joints)), sharex=True,
                             squeeze=False)
    trials = sorted(curves["trial"].unique(), key=lambda s: int(s.split("_")[-1]))
    colors = dict(zip(trials, serve_colors(len(trials))))
    for r, j in enumerate(joints):
        for c, (kind, unit) in enumerate(kinds):
            ax, var = axes[r, c], f"{kind}_{j}"
            g = curves[curves["variable"] == var]
            for tr in trials:
                d = g[g["trial"] == tr]
                ax.plot(d["percent"], d["value"], color=colors[tr], lw=0.9, alpha=0.8,
                        label=tr.replace("saque_", "S"))
            b = bands[bands["variable"] == var]
            if len(b):
                ax.fill_between(b["percent"], b["media"] - b["desviacion_estandar"],
                                b["media"] + b["desviacion_estandar"], color=TEXT2, alpha=0.12, lw=0)
                ax.plot(b["percent"], b["media"], color=TEXT, lw=1.6, label="media")
            if r == 0:
                ax.set_title(unit, loc="left")
            if c == 0:
                ax.set_ylabel(JOINT_NAMES.get(j, j), fontweight="bold")
    for ax in axes[-1]:
        ax.set_xlabel("% inicio → impacto")
    axes[0, 0].legend(ncol=3, fontsize=7, loc="best")
    fig.suptitle("Cinemática angular por articulación (vista lateral; banda = media ± DE de 5 saques)",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_variability(bands: pd.DataFrame, rmsd: pd.DataFrame, path: Path) -> Path:
    """Consistencia entre saques: DE entre saques a lo largo del gesto (θ) y RMSD de cada saque."""
    theta = [v for v in bands["variable"].unique() if v.startswith("theta")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    for i, var in enumerate(theta):
        b = bands[bands["variable"] == var]
        a1.plot(b["percent"], b["desviacion_estandar"], color=SERIES[i % len(SERIES)], lw=1.6,
                label=JOINT_NAMES.get(var.replace("theta_", ""), var))
    a1.set_xlabel("% inicio → impacto")
    a1.set_ylabel("DE entre saques (°)")
    a1.set_title("Variabilidad angular a lo largo del gesto", loc="left")
    a1.legend(fontsize=7, ncol=2)
    r = rmsd[rmsd["variable"].isin(theta)]
    p = r.pivot_table(index="variable", columns="trial", values="rmsd_vs_media")
    trials = sorted(p.columns, key=lambda s: int(s.split("_")[-1]))
    colors = serve_colors(len(trials))
    x = np.arange(len(p))
    w = 0.8 / len(trials)
    for i, tr in enumerate(trials):
        a2.bar(x + (i - (len(trials) - 1) / 2) * w, p[tr], width=w * 0.9, color=colors[i],
               label=tr.replace("saque_", "S"))
    a2.set_xticks(x, [JOINT_NAMES.get(v.replace("theta_", ""), v).split(" (")[0] for v in p.index],
                  rotation=30, ha="right")
    a2.set_ylabel("RMSD vs curva media (°)")
    a2.set_title("Distancia de cada saque a la media", loc="left")
    a2.legend(ncol=len(trials), fontsize=7)
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_correlations(corr: pd.DataFrame, path: Path) -> Path:
    """Matriz de ρ de Spearman entre indicadores por saque (diverging, gris = 0)."""
    from matplotlib.colors import LinearSegmentedColormap
    names = list(dict.fromkeys(list(corr["var_a"]) + list(corr["var_b"])))
    n = len(names)
    M = np.full((n, n), np.nan)
    P = np.full((n, n), np.nan)
    idx = {v: i for i, v in enumerate(names)}
    for _, r in corr.iterrows():
        i, j = idx[r["var_a"]], idx[r["var_b"]]
        M[i, j] = M[j, i] = r["rho"]
        P[i, j] = P[j, i] = r["p"]
    cmap = LinearSegmentedColormap.from_list("div", [SERIES[1], "#eeeeee", SERIES[0]])
    fig, ax = plt.subplots(figsize=(0.55 * n + 3, 0.55 * n + 2))
    mask = np.tril(np.ones_like(M, bool))
    # Solo el triángulo superior: se quitan la última fila y la primera columna (vacías).
    im = ax.imshow(np.where(mask, np.nan, M)[:-1, 1:], cmap=cmap, vmin=-1, vmax=1)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if np.isfinite(M[i, j]):
                ax.text(j - 1, i, f"{M[i, j]:+.1f}" + ("*" if P[i, j] < 0.05 else ""), ha="center",
                        va="center", fontsize=7, color=TEXT)
    lab = [s.replace("_", " ") for s in names]
    ax.set_xticks(range(n - 1), lab[1:], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n - 1), lab[:-1], fontsize=7)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.6, label="ρ de Spearman")
    ax.set_title("Asociaciones entre indicadores por saque (n = 5; * p < 0.05; descriptivo)", loc="left")
    fig.tight_layout()
    return _save(fig, path)


def plot_jacobian_validation(jac_by_serve: dict[int, pd.DataFrame], metrics: pd.DataFrame,
                             path: Path) -> Path:
    """Rapidez de la muñeca por J(θ)ω contra la derivada directa (todas las muestras, por saque)."""
    fig, ax = plt.subplots(figsize=(5, 4.6))
    colors = serve_colors(len(jac_by_serve))
    hi = 0.0
    for c, (s, d) in zip(colors, sorted(jac_by_serve.items())):
        ok = d[["speed_directa", "speed_J"]].dropna()
        ax.scatter(ok["speed_directa"], ok["speed_J"], s=10, color=c, alpha=0.6, lw=0, label=f"S{s}")
        hi = max(hi, float(ok.max().max()) if len(ok) else 0)
    ax.plot([0, hi], [0, hi], color=TEXT2, lw=1, ls="--", label="identidad")
    ax.set_xlabel("rapidez directa d(x,y)/dt (m/s)")
    ax.set_ylabel("rapidez J(θ)ω (m/s)")
    if len(metrics):
        txt = (f"r mediana = {metrics['corr_speed'].median():.2f}\n"
               f"RMSE/pico mediano = {100 * metrics['rmse_rel_pico'].median():.0f} %")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=8, color=TEXT)
    ax.set_title("Validación del Jacobiano (vista lateral)", loc="left")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout()
    return _save(fig, path)


def plot_residuals(resid: pd.DataFrame, filters: pd.DataFrame, path: Path,
                   markers: list[str] | None = None) -> Path:
    """Análisis de residuos de Winter: R(fc) de los marcadores de la vista lateral y fc elegidas."""
    views = list(filters["vista"].unique())
    fig, axes = plt.subplots(1, 1 + len(views), figsize=(4.5 * (1 + len(views)), 3.6), squeeze=False)
    ax = axes[0, 0]
    v0 = views[0]
    r = resid[(resid["vista"] == v0) & (resid["coord"] == "y")]
    mk = [m for m in (markers or []) if m in set(r["marcador"])] or list(r["marcador"].unique())
    mk = mk[:8]
    for i, m in enumerate(mk):
        d = r[r["marcador"] == m]
        ax.plot(d["fc_hz"], 1000 * d["residuo"], color=SERIES[i], lw=1.4, label=m)
        ax.plot([d["fc_elegida"].iloc[0]], [1000 * d["R0"].iloc[0]], "o", color=SERIES[i], ms=6,
                markeredgecolor="white")
    ax.set_xlabel("frecuencia de corte (Hz)")
    ax.set_ylabel("residuo RMS (mm)")
    ax.set_title(f"Residuos de Winter (y, {v0})", loc="left",
                 fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    for ax, view in zip(axes[0, 1:], views):
        f = filters[filters["vista"] == view].sort_values("fc_hz")
        ax.barh(f["marcador"], f["fc_hz"], color=SERIES[0], height=0.7)
        ax.set_xlabel("fc elegida (Hz)")
        ax.set_title(f"Vista {view}", loc="left")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    return _save(fig, path)


def plot_qc_summary(qc: pd.DataFrame, n_frames: dict[str, int], path: Path) -> Path:
    """Muestras anuladas por cada control de calidad, en % de fotogramas de la vista."""
    views = list(qc["vista"].unique())
    fig, axes = plt.subplots(1, len(views), figsize=(5 * len(views), 3.2), squeeze=False)
    for ax, view in zip(axes[0], views):
        d = qc[qc["vista"] == view].sort_values("muestras")
        pct = 100 * d["muestras"] / max(n_frames.get(view, 1), 1)
        ax.barh(d["tipo"].str.replace("_", " "), pct, color=SERIES[0], height=0.7)
        for y, v in enumerate(pct):
            ax.text(v, y, f" {v:.1f} %", va="center", fontsize=7, color=TEXT)
        ax.set_xlabel("% de fotogramas (suma sobre marcadores/segmentos)")
        ax.set_title(f"Vista {view}", loc="left")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Control de calidad: muestras anuladas por tipo", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def _basic_figures(ctx, serves: pd.DataFrame, kin: dict[str, pd.DataFrame]) -> list[Path]:
    """Fases, secuencia, θ/ω/α, variabilidad, correlaciones, Jacobiano, filtrado y QC."""
    t, fd = ctx.paths.tables, ctx.paths.figures
    rd = lambda n: pd.read_csv(t / f"{n}.csv") if (t / f"{n}.csv").exists() else pd.DataFrame()  # noqa: E731
    figs = []
    phases = rd("phases")
    if not phases.empty:
        figs.append(plot_phases(phases, fd / "fases.png"))
    view = ctx.cfg["fatigue"]["view"]
    ev, peaks = rd("events"), rd("peaks")
    seq = ctx.cfg["events"].get("sequence") or {}
    vd = ctx.view_dir(view)
    ks = {int(p.parent.name.split("_")[1]): pd.read_csv(p) for p in sorted(vd.glob("saque_*/kinematics.csv"))}
    if ks and seq and not ev.empty and not peaks.empty:
        e = ev[ev["vista"] == view]
        imp = e[e["evento"] == "impacto"].set_index("saque")["t"].to_dict()
        ini = e[e["evento"] == "inicio"].set_index("saque")["t"].to_dict()
        ks = {s: k for s, k in ks.items() if s in imp and s in ini}
        figs.append(plot_kinematic_sequence(ks, imp, ini, seq, peaks[peaks["vista"] == view],
                                            fd / "secuencia_cinematica.png"))
    jc, jb, jr = rd("curvas_articulares"), rd("variabilidad_banda_articular"), rd("variabilidad_rmsd_articular")
    if not jc.empty:
        figs.append(plot_joint_kinematics(jc, jb, fd / "cinematica_articular.png"))
        figs.append(plot_variability(jb, jr, fd / "variabilidad_saques.png"))
    corr = rd("correlaciones")
    if not corr.empty:
        figs.append(plot_correlations(corr, fd / "correlaciones.png"))
    jac = {int(p.parent.name.split("_")[1]): pd.read_csv(p) for p in sorted(vd.glob("saque_*/jacobian.csv"))}
    jm = rd("jacobian_metrics")
    if jac:
        figs.append(plot_jacobian_validation(jac, jm[jm["vista"] == view] if len(jm) else jm,
                                             fd / "jacobiano_validacion.png"))
    resid, filt = rd("residuos_winter"), rd("filtros")
    if not resid.empty and not filt.empty:
        v0 = filt["vista"].iloc[0]
        figs.append(plot_residuals(resid, filt, fd / "filtrado_residuos.png",
                                   ctx.cfg["views"][v0].get("stickers")))
    qc = rd("qc_resumen")
    if not qc.empty:
        figs.append(plot_qc_summary(qc, {v: len(k) for v, k in kin.items()}, fd / "qc_resumen.png"))
    return figs

