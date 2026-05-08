"""
╔══════════════════════════════════════════════════════════════╗
║        SPHY CORE — VALIDATOR + VISUALIZER                    ║
║  Reads sphy_dataset.parquet, validates all SHA-256 hashes,   ║
║  and generates full visualization in sphy_report.png         ║
╚══════════════════════════════════════════════════════════════╝
"""

import hashlib
import json
import time
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

PHI_GOLD = (1 + np.sqrt(5)) / 2

# ─── Paleta escura ───────────────────────────────────────────────────────────
BG       = "#05070f"
ACCENT   = "#00e5ff"
GREEN    = "#00ff99"
RED      = "#ff3355"
GOLD     = "#ffd700"
TEXT     = "#c8d8e8"
GRID_C   = "#1a2a3a"

PLANET_COLORS = {
    "Mercúrio": "#a0855b",
    "Vênus":    "#e8c97a",
    "Terra":    "#4fa3e0",
    "Marte":    "#c1440e",
    "Ceres":    "#9a9a7a",
    "Júpiter":  "#c88b4a",
    "Saturno":  "#e8d5a0",
    "Urano":    "#7de8e8",
    "Netuno":   "#4060e0",
    "Plutão":   "#b0a0c0",
    "STARSHIP": "#00e5ff",
}
MOON_COLOR = "#888899"

# ─── Re-simular um frame para gerar seu hash de validação ──────────────────

def recompute_sha(frame_idx: int, frame_df: pd.DataFrame) -> str:
    """
    Reconstrói o payload canônico a partir das linhas do DataFrame
    e calcula o SHA-256, para comparar com o registrado.
    """
    bodies = []
    starship_row = None

    for _, row in frame_df.iterrows():
        if row["type"] == "starship":
            starship_row = row
        elif row["type"] in ("planet", "moon"):
            body = {
                "type":  row["type"],
                "name":  row["name"],
                "x":     row["x"],
                "y":     row["y"],
                "z":     row["z"],
                "dist":  row["dist"],
                "size":  row["size"],
                "angle": row["angle_rad"],
            }
            if row["type"] == "moon":
                body["parent"] = row["parent"]
            bodies.append(body)

    frame_data = {
        "frame":     int(frame_idx),
        "timestamp": round(float(frame_df.iloc[0]["timestamp_s"]), 6),
        "warp":      float(frame_df.iloc[0]["warp"]),
        "bodies":    bodies,
        "starship": {
            "type":        "starship",
            "name":        "STARSHIP",
            "x":           float(starship_row["x"]),
            "y":           float(starship_row["y"]),
            "z":           float(starship_row["z"]),
            "from_planet": str(starship_row["parent"]),
            "to_planet":   str(starship_row["name"]).split("|")[1] if "|" in str(starship_row["name"]) else "",
            "progress":    float(starship_row["angle_rad"]),
        } if starship_row is not None else {},
    }

    canonical = json.dumps(frame_data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ─── Validação ───────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame):
    frames     = df["frame"].unique()
    total      = len(frames)
    ok_count   = 0
    fail_count = 0
    fail_frames = []

    print(f"\n  Validando {total} frames ...\n")
    t0 = time.perf_counter()

    for i, fid in enumerate(sorted(frames)):
        sub         = df[df["frame"] == fid]
        stored_hash = sub.iloc[0]["sha256"]
        computed    = recompute_sha(fid, sub)

        if computed == stored_hash:
            ok_count += 1
        else:
            fail_count += 1
            fail_frames.append(int(fid))

        pct = (i + 1) / total * 100
        bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
        status = "✓" if computed == stored_hash else "✗"
        print(f"\r  [{bar}] {pct:5.1f}%  frame {int(fid):>5}  {status}", end="", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\n\n  {'─'*56}")
    print(f"  Frames validados : {total}")
    print(f"  ✓ OK             : {ok_count}  ({ok_count/total*100:.1f}%)")
    print(f"  ✗ FALHOS         : {fail_count}")
    if fail_frames:
        print(f"  Frames corrompidos: {fail_frames[:20]}{'…' if len(fail_frames)>20 else ''}")
    print(f"  Tempo de validação: {elapsed:.2f}s")
    print(f"  {'─'*56}\n")

    return ok_count, fail_count, fail_frames


# ─── Visualização ────────────────────────────────────────────────────────────

def build_visualization(df: pd.DataFrame, ok: int, fail: int, fail_frames: list):
    total_frames = df["frame"].nunique()
    print("  Gerando visualização ...")

    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        3, 4,
        figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.05, right=0.97,
        top=0.92,  bottom=0.06,
    )

    def ax_style(ax, title=""):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID_C)
        ax.grid(color=GRID_C, linewidth=0.5, alpha=0.6)
        if title:
            ax.set_title(title, color=ACCENT, fontsize=10, pad=6, fontweight="bold")

    # ── Título ────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.965,
        "SPHY CORE  ·  RELATÓRIO DE VALIDAÇÃO SHA-256",
        ha="center", va="center",
        color=ACCENT, fontsize=17, fontweight="bold",
        fontfamily="monospace",
    )
    fig.text(
        0.5, 0.945,
        f"Dataset: {len(df):,} linhas  ·  {total_frames} frames  ·  "
        f"✓ {ok} OK  ·  ✗ {fail} FALHOS",
        ha="center", va="center",
        color=GREEN if fail == 0 else RED,
        fontsize=10, fontfamily="monospace",
    )

    # ─────────────────────────────────────────────────────────────────────
    # 1) Mapa orbital XY (quadrante grande)
    # ─────────────────────────────────────────────────────────────────────
    ax_orb = fig.add_subplot(gs[0:2, 0:2])
    ax_style(ax_orb, "Órbitas — Plano XY (todos os frames)")

    planets_df  = df[df["type"] == "planet"]
    moons_df    = df[df["type"] == "moon"]
    ship_df     = df[df["type"] == "starship"]

    # Trails dos planetas
    for name, grp in planets_df.groupby("name"):
        c = PLANET_COLORS.get(name, "#ffffff")
        ax_orb.plot(grp["x"], grp["y"], color=c, alpha=0.18, linewidth=0.6)
        # Última posição
        last = grp.iloc[-1]
        ax_orb.scatter(last["x"], last["y"], color=c, s=max(last["size"]**1.6, 12), zorder=5)
        ax_orb.text(last["x"] + 8, last["y"], name, color=c, fontsize=6.5, va="center")

    # Luas (pontilhado)
    for name, grp in moons_df.groupby("name"):
        ax_orb.plot(grp["x"], grp["y"], color=MOON_COLOR, alpha=0.12, linewidth=0.4, linestyle=":")

    # Starship
    ax_orb.plot(ship_df["x"], ship_df["y"], color=ACCENT, alpha=0.55, linewidth=1.0, linestyle="--", label="STARSHIP")
    ax_orb.scatter(ship_df.iloc[-1]["x"], ship_df.iloc[-1]["y"], color=ACCENT, s=60, marker="D", zorder=6)

    # Sol
    ax_orb.scatter(0, 0, color=GOLD, s=120, zorder=7, marker="*")
    ax_orb.text(5, 5, "SOL", color=GOLD, fontsize=7)

    ax_orb.set_xlabel("X (u)", color=TEXT, fontsize=8)
    ax_orb.set_ylabel("Y (u)", color=TEXT, fontsize=8)
    ax_orb.set_aspect("equal", adjustable="datalim")

    # ─────────────────────────────────────────────────────────────────────
    # 2) Painel de validação SHA-256
    # ─────────────────────────────────────────────────────────────────────
    ax_sha = fig.add_subplot(gs[0, 2])
    ax_style(ax_sha, "Status SHA-256 por Frame")

    frame_ids = sorted(df["frame"].unique())
    sha_status = []
    for fid in frame_ids:
        sub  = df[df["frame"] == fid]
        stored = sub.iloc[0]["sha256"]
        comp   = recompute_sha(fid, sub)
        sha_status.append(1 if comp == stored else 0)

    sha_arr = np.array(sha_status)
    n = len(sha_arr)
    cols = min(100, n)
    rows_v = int(np.ceil(n / cols))
    grid = np.zeros((rows_v, cols))
    for i, v in enumerate(sha_arr):
        grid[i // cols, i % cols] = v

    cmap_sha = LinearSegmentedColormap.from_list("sha", [RED, GREEN])
    ax_sha.imshow(grid, cmap=cmap_sha, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax_sha.set_xlabel("Frame (coluna)", color=TEXT, fontsize=7)
    ax_sha.set_ylabel("Bloco", color=TEXT, fontsize=7)
    ax_sha.set_title("Status SHA-256 por Frame", color=ACCENT, fontsize=9, pad=5, fontweight="bold")

    # ─────────────────────────────────────────────────────────────────────
    # 3) Pizza OK / FALHO
    # ─────────────────────────────────────────────────────────────────────
    ax_pie = fig.add_subplot(gs[0, 3])
    ax_pie.set_facecolor(BG)
    labels  = [f"OK ({ok})", f"FALHOS ({fail})"] if fail > 0 else [f"OK ({ok})"]
    sizes   = [ok, fail] if fail > 0 else [ok]
    colors  = [GREEN, RED] if fail > 0 else [GREEN]
    wedges, texts, autotexts = ax_pie.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"color": TEXT, "fontsize": 8},
        wedgeprops={"linewidth": 0.5, "edgecolor": BG},
    )
    for at in autotexts:
        at.set_color(BG)
        at.set_fontsize(8)
    ax_pie.set_title("Integridade", color=ACCENT, fontsize=10, pad=6, fontweight="bold")

    # ─────────────────────────────────────────────────────────────────────
    # 4) Posição Z ao longo do tempo — planetas principais
    # ─────────────────────────────────────────────────────────────────────
    ax_z = fig.add_subplot(gs[1, 2])
    ax_style(ax_z, "Variação Z (inclinação orbital)")
    for name in ["Terra", "Júpiter", "Saturno", "Netuno"]:
        grp = planets_df[planets_df["name"] == name]
        ax_z.plot(grp["frame"], grp["z"], label=name, color=PLANET_COLORS.get(name, "#fff"), linewidth=1.0)
    ax_z.set_xlabel("Frame", color=TEXT, fontsize=7)
    ax_z.set_ylabel("Z (u)", color=TEXT, fontsize=7)
    ax_z.legend(fontsize=6, facecolor=BG, edgecolor=GRID_C, labelcolor=TEXT)

    # ─────────────────────────────────────────────────────────────────────
    # 5) Velocidade angular (Δangle/frame) dos planetas
    # ─────────────────────────────────────────────────────────────────────
    ax_spd = fig.add_subplot(gs[1, 3])
    ax_style(ax_spd, "Velocidade Angular Média")
    avg_speeds = {}
    for name, grp in planets_df.groupby("name"):
        angles = grp.sort_values("frame")["angle_rad"].values
        if len(angles) > 1:
            diffs = np.abs(np.diff(angles))
            avg_speeds[name] = float(np.mean(diffs))
    names_s  = list(avg_speeds.keys())
    speeds_s = list(avg_speeds.values())
    bar_colors = [PLANET_COLORS.get(n, "#aaa") for n in names_s]
    ax_spd.barh(names_s, speeds_s, color=bar_colors, edgecolor=BG, height=0.6)
    ax_spd.set_xlabel("rad/frame", color=TEXT, fontsize=7)
    ax_spd.invert_yaxis()

    # ─────────────────────────────────────────────────────────────────────
    # 6) Starship: progresso e trajeto XZ
    # ─────────────────────────────────────────────────────────────────────
    ax_ship = fig.add_subplot(gs[2, 0:2])
    ax_style(ax_ship, "Starship — Trajeto XZ (altitude)")
    ax_ship.plot(ship_df["x"], ship_df["z"], color=ACCENT, linewidth=1.2, alpha=0.8)
    ax_ship.scatter(ship_df.iloc[0]["x"],  ship_df.iloc[0]["z"],  color=GREEN, s=60, zorder=5, label="Início")
    ax_ship.scatter(ship_df.iloc[-1]["x"], ship_df.iloc[-1]["z"], color=RED,   s=60, zorder=5, label="Fim")
    ax_ship.set_xlabel("X (u)", color=TEXT, fontsize=8)
    ax_ship.set_ylabel("Z (u)", color=TEXT, fontsize=8)
    ax_ship.legend(fontsize=7, facecolor=BG, edgecolor=GRID_C, labelcolor=TEXT)

    # ─────────────────────────────────────────────────────────────────────
    # 7) Hash fingerprint visual (primeiro e último frame)
    # ─────────────────────────────────────────────────────────────────────
    ax_fp = fig.add_subplot(gs[2, 2])
    ax_style(ax_fp, "Hash Fingerprint (frame 0 vs último)")
    h0  = df[df["frame"] == frame_ids[0]].iloc[0]["sha256"]
    hl  = df[df["frame"] == frame_ids[-1]].iloc[0]["sha256"]
    vals0 = [int(c, 16) for c in h0]
    valsl = [int(c, 16) for c in hl]
    x_fp  = np.arange(64)
    ax_fp.fill_between(x_fp, vals0, alpha=0.55, color=GREEN,  label=f"Frame {frame_ids[0]}")
    ax_fp.fill_between(x_fp, valsl, alpha=0.55, color=ACCENT, label=f"Frame {frame_ids[-1]}")
    ax_fp.set_xlim(0, 63)
    ax_fp.set_ylim(0, 16)
    ax_fp.set_xlabel("Nibble do hash", color=TEXT, fontsize=7)
    ax_fp.set_ylabel("Valor hex", color=TEXT, fontsize=7)
    ax_fp.legend(fontsize=6, facecolor=BG, edgecolor=GRID_C, labelcolor=TEXT)

    # ─────────────────────────────────────────────────────────────────────
    # 8) Estatísticas textuais
    # ─────────────────────────────────────────────────────────────────────
    ax_txt = fig.add_subplot(gs[2, 3])
    ax_txt.set_facecolor(BG)
    ax_txt.axis("off")
    ax_txt.set_title("Estatísticas", color=ACCENT, fontsize=10, pad=6, fontweight="bold")

    sample_hash = df.iloc[0]["sha256"]
    integrity   = "✓ ÍNTEGRO" if fail == 0 else f"✗ {fail} CORROMPIDOS"
    int_color   = GREEN if fail == 0 else RED

    stats = [
        ("Arquivo",        "sphy_dataset.parquet"),
        ("Total linhas",   f"{len(df):,}"),
        ("Frames",         str(total_frames)),
        ("Corpos/frame",   str(len(df[df['frame']==frame_ids[0]]))),
        ("Integridade",    integrity),
        ("SHA-256 ok",     str(ok)),
        ("SHA-256 falhos", str(fail)),
        ("Hash[0][:16]",   sample_hash[:16] + "…"),
    ]

    y = 0.93
    for label, val in stats:
        c = int_color if label == "Integridade" else TEXT
        ax_txt.text(0.02, y, f"{label}:", color=ACCENT, fontsize=8, transform=ax_txt.transAxes, va="top", fontfamily="monospace")
        ax_txt.text(0.45, y, val,         color=c,      fontsize=8, transform=ax_txt.transAxes, va="top", fontfamily="monospace")
        y -= 0.115

    # ─── Salvar ──────────────────────────────────────────────────────────
    out = "sphy_relatorio.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓ Visualização salva → {out}\n")
    return out


# ─── Main ────────────────────────────────────────────────────────────────────

BANNER = r"""
  ╔═══════════════════════════════════════════════════════════╗
  ║   SPHY CORE — VALIDADOR & VISUALIZADOR SHA-256 PARQUET   ║
  ╚═══════════════════════════════════════════════════════════╝
"""

def main():
    print(BANNER)

    parquet_path = "sphy_dataset.parquet"
    try:
        df = pd.read_parquet(parquet_path)
    except FileNotFoundError:
        print(f"  [ERRO] Arquivo '{parquet_path}' não encontrado.")
        print("  Execute primeiro:  python sphy_gerador.py\n")
        sys.exit(1)

    print(f"  Dataset carregado: {len(df):,} linhas  |  {df['frame'].nunique()} frames únicos")
    print(f"  Colunas: {list(df.columns)}\n")

    # Validar
    ok, fail, fail_frames = validate(df)

    # Visualizar
    build_visualization(df, ok, fail, fail_frames)

    print("  ─────────────────────────────────────────────────────────")
    if fail == 0:
        print("  ✅  TODOS OS FRAMES VÁLIDOS — dataset íntegro.")
    else:
        print(f"  ⚠️   {fail} FRAMES CORROMPIDOS detectados!")
    print("  ─────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
