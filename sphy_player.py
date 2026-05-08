"""
╔══════════════════════════════════════════════════════════════════╗
║          SPHY CORE — PLAYER / PARQUET VISUALIZER                 ║
║                                                                  ║
║  Reads sphy_dataset.parquet and replays the animation exactly    ║
║  like the original code, validating each frame's SHA-256 in      ║
║  real time.                                                      ║
║                                                                  ║
║  Controls:                                                       ║
║    Mouse wheel     → zoom                                        ║
║    Right drag      → rotate camera                               ║
║    SPACE           → warp speed (advances frames faster)         ║
║    C               → free camera / follow STARSHIP               ║
║    F               → fullscreen                                  ║
║    ← →             → step backward / forward one frame manually  ║
║    R               → reset to frame 0                            ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sys
import hashlib
import json
import numpy as np
import pandas as pd
import py5

# ─── Metadados visuais dos corpos (igual ao original) ───────────────────────
PLANET_META = {
    "Mercúrio": {"hue": 20,  "size": 5,  "ring_type": None},
    "Vênus":    {"hue": 40,  "size": 8,  "ring_type": "venus"},
    "Terra":    {"hue": 140, "size": 9,  "ring_type": None},
    "Marte":    {"hue": 5,   "size": 7,  "ring_type": None},
    "Ceres":    {"hue": 160, "size": 4,  "ring_type": None},
    "Júpiter":  {"hue": 35,  "size": 22, "ring_type": "ice"},
    "Saturno":  {"hue": 130, "size": 18, "ring_type": "saturn"},
    "Urano":    {"hue": 110, "size": 14, "ring_type": "ice"},
    "Netuno":   {"hue": 160, "size": 13, "ring_type": "ice"},
    "Plutão":   {"hue": 25,  "size": 4,  "ring_type": None},
}

MOON_META = {
    "Lua":       {"hue": 0,   "parent": "Terra"},
    "Fobos":     {"hue": 5,   "parent": "Marte"},
    "Deimos":    {"hue": 10,  "parent": "Marte"},
    "Io":        {"hue": 30,  "parent": "Júpiter"},
    "Europa":    {"hue": 150, "parent": "Júpiter"},
    "Ganimedes": {"hue": 20,  "parent": "Júpiter"},
    "Calisto":   {"hue": 10,  "parent": "Júpiter"},
    "Titan":     {"hue": 30,  "parent": "Saturno"},
    "Titânia":   {"hue": 120, "parent": "Urano"},
    "Tritão":    {"hue": 170, "parent": "Netuno"},
    "Caronte":   {"hue": 20,  "parent": "Plutão"},
}

PHI_GOLD = (1 + np.sqrt(5)) / 2

# ─── Estado global ───────────────────────────────────────────────────────────
df              = None          # DataFrame completo
frames_idx      = []            # lista ordenada de frame ids
total_frames    = 0
current_frame   = 0             # índice na lista frames_idx
warp_mode       = False         # SPACE pressionado
follow_ship     = False         # câmera segue nave

cam_dist        = 1200.0
cam_lat         = py5.PI / 3.5
cam_lon         = 0.0

# Históricos de trail (últimos N pontos por corpo)
TRAIL_LEN       = 100
trails          = {}            # nome -> deque numpy (TRAIL_LEN, 3)

# Validação SHA-256
sha_status      = {}            # frame_id -> True/False
sha_cache       = {}            # frame_id -> hash recalculado

# HUD SHA display (último frame validado)
last_sha_ok     = True
last_sha_hex    = ""
last_frame_id   = 0

# Starship visual state
ship_pos        = np.zeros(3)
ship_to         = ""
ship_from       = ""
ship_trail      = np.zeros((200, 3))
ship_trail_ptr  = 0
ship_rotation   = 0.0

# ─── Funções de validação ────────────────────────────────────────────────────

def recompute_sha(frame_id: int, sub: pd.DataFrame) -> str:
    """Reconstrói o JSON canônico de um frame e retorna seu SHA-256."""
    bodies = []
    sr = None
    for _, row in sub.iterrows():
        t = row["type"]
        if t == "starship":
            sr = row
            continue
        body = {
            "type":  str(row["type"]),
            "name":  str(row["name"]),
            "x":     float(row["x"]),
            "y":     float(row["y"]),
            "z":     float(row["z"]),
            "dist":  float(row["dist"]),
            "size":  float(row["size"]),
            "angle": float(row["angle_rad"]),
        }
        if t == "moon":
            body["parent"] = str(row["parent"])
        bodies.append(body)

    to_planet = str(sr["name"]).split("|")[1] if sr is not None and "|" in str(sr["name"]) else ""
    frame_data = {
        "frame":     int(frame_id),
        "timestamp": round(float(sub.iloc[0]["timestamp_s"]), 6),
        "warp":      float(sub.iloc[0]["warp"]),
        "bodies":    bodies,
        "starship": {
            "type":        "starship",
            "name":        "STARSHIP",
            "x":           float(sr["x"]),
            "y":           float(sr["y"]),
            "z":           float(sr["z"]),
            "from_planet": str(sr["parent"]),
            "to_planet":   to_planet,
            "progress":    float(sr["angle_rad"]),
        } if sr is not None else {},
    }
    canonical = json.dumps(frame_data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_frame(frame_id: int) -> bool:
    """Valida o SHA-256 de um frame. Usa cache para não repetir cálculo."""
    if frame_id in sha_cache:
        return sha_status[frame_id]
    sub      = df[df["frame"] == frame_id]
    stored   = sub.iloc[0]["sha256"]
    computed = recompute_sha(frame_id, sub)
    sha_cache[frame_id]  = computed
    sha_status[frame_id] = (computed == stored)
    return sha_status[frame_id]


# ─── Configuração Gráfica (Adicione esta função) ──────────────────────────────

def settings():
    # O tamanho e o motor P3D devem ser definidos aqui para evitar o RuntimeError
    py5.size(1280, 720, py5.P3D)


# ─── Setup ───────────────────────────────────────────────────────────────────

def setup():
    global df, frames_idx, total_frames, trails

    # ── Carregar parquet ────────────────────────────────────────────────────
    parquet_path = "sphy_dataset.parquet"
    try:
        df = pd.read_parquet(parquet_path)
    except FileNotFoundError:
        print(f"\n[ERRO] '{parquet_path}' não encontrado.")
        print("Execute primeiro:  python sphy_gerador.py\n")
        sys.exit(1)

    frames_idx   = sorted(df["frame"].unique())
    total_frames = len(frames_idx)
    print(f"[SPHY PLAYER] {total_frames} frames carregados — {len(df):,} linhas")

    # Inicializar trails
    for name in list(PLANET_META.keys()) + list(MOON_META.keys()) + ["STARSHIP"]:
        trails[name] = np.zeros((TRAIL_LEN, 3))

    py5.window_resizable(True)
    py5.window_title("Harpia Engine — Sphy Player | lendo parquet + SHA-256")
    py5.color_mode(py5.HSB, 255)


# ─── Draw ────────────────────────────────────────────────────────────────────

def draw():
    global current_frame, ship_pos, ship_to, ship_from
    global ship_trail, ship_trail_ptr, ship_rotation
    global last_sha_ok, last_sha_hex, last_frame_id

    py5.background(0)

    # ── Avançar frame ───────────────────────────────────────────────────────
    step = 3 if warp_mode else 1
    current_frame = (current_frame + step) % total_frames
    frame_id      = frames_idx[current_frame]

    # ── Subconjunto do frame atual ──────────────────────────────────────────
    sub          = df[df["frame"] == frame_id]
    planets_rows = sub[sub["type"] == "planet"]
    moons_rows   = sub[sub["type"] == "moon"]
    ship_rows    = sub[sub["type"] == "starship"]

    # ── Validação SHA-256 (em background, usa cache) ─────────────────────
    ok = validate_frame(frame_id)
    last_sha_ok  = ok
    last_sha_hex = sha_cache.get(frame_id, "")
    last_frame_id = frame_id

    # ── Coletar posições dos planetas ───────────────────────────────────────
    planet_pos = {}
    for _, row in planets_rows.iterrows():
        planet_pos[row["name"]] = (float(row["x"]), float(row["y"]), float(row["z"]))

    # ── Posição da nave ─────────────────────────────────────────────────────
    if len(ship_rows) > 0:
        sr = ship_rows.iloc[0]
        ship_pos  = np.array([float(sr["x"]), float(sr["y"]), float(sr["z"])])
        ship_from = str(sr["parent"])
        ship_to   = str(sr["name"]).split("|")[1] if "|" in str(sr["name"]) else ""
        ship_rotation += 1.5 * (1.0 / 60.0)
    ship_trail[ship_trail_ptr] = ship_pos
    ship_trail_ptr = (ship_trail_ptr + 1) % 200

    # ── Atualizar trails dos planetas e luas ────────────────────────────────
    for _, row in planets_rows.iterrows():
        name = row["name"]
        pos  = planet_pos[name]
        trail = trails[name]
        # deslocar e inserir
        trail = np.roll(trail, 1, axis=0)
        trail[0] = pos
        trails[name] = trail

    for _, row in moons_rows.iterrows():
        name = row["name"]
        trail = trails[name]
        trail = np.roll(trail, 1, axis=0)
        trail[0] = [float(row["x"]), float(row["y"]), float(row["z"])]
        trails[name] = trail

    # ── Câmera ──────────────────────────────────────────────────────────────
    if follow_ship:
        sx, sy, sz = ship_pos
        py5.translate(py5.width / 2 - sx, py5.height / 2 - sy, -cam_dist - sz)
    else:
        py5.translate(py5.width / 2, py5.height / 2, -cam_dist)
    py5.rotate_x(cam_lat)
    py5.rotate_z(cam_lon)

    # ── Sol ─────────────────────────────────────────────────────────────────
    pulse = np.sin(py5.frame_count * 0.05) * 8
    py5.no_fill()
    py5.stroke(25, 200, 255, 180)
    py5.sphere(60 + pulse)

    # ── Cinturão de asteroides (estético, gerado proceduralmente) ───────────
    draw_asteroid_belt()

    # ── Planetas ────────────────────────────────────────────────────────────
    for _, row in planets_rows.iterrows():
        name = row["name"]
        meta = PLANET_META.get(name, {"hue": 100, "size": 6, "ring_type": None})
        pos  = planet_pos[name]
        hue  = meta["hue"]

        # trail
        py5.stroke(hue, 100, 255, 40)
        py5.no_fill()
        draw_trail_pts(trails[name])

        # órbita (círculo tênue)
        draw_orbit_circle(float(row["dist"]), hue)

        # corpo
        draw_node(pos, meta["size"], hue, name)

        # anéis
        if meta["ring_type"]:
            layers = 4 if meta["ring_type"] == "saturn" else 1
            draw_rings(pos, meta["size"] * 2.2, hue, layers, meta["ring_type"] == "venus")

    # ── Luas ────────────────────────────────────────────────────────────────
    for _, row in moons_rows.iterrows():
        name = row["name"]
        hue  = MOON_META.get(name, {}).get("hue", 0)
        pos  = (float(row["x"]), float(row["y"]), float(row["z"]))
        size = float(row["size"])

        py5.stroke(hue, 50, 255, 80)
        py5.no_fill()
        draw_trail_pts(trails[name])
        draw_node(pos, size, hue, name)

    # ── Starship ─────────────────────────────────────────────────────────────
    draw_starship()

    # ── HUD (por cima de tudo) ───────────────────────────────────────────────
    draw_hud(frame_id, ok)


# ─── Funções de desenho ───────────────────────────────────────────────────────

def draw_node(pos, size, hue, name):
    py5.push_matrix()
    py5.translate(pos[0], pos[1], pos[2])
    py5.fill(hue, 180, 255)
    py5.no_stroke()
    py5.sphere(size)
    py5.fill(255)
    py5.text_size(11)
    py5.text(name, size + 6, 0)
    py5.pop_matrix()


def draw_trail_pts(trail: np.ndarray):
    py5.begin_shape()
    for pt in trail:
        if not np.all(pt == 0):
            py5.vertex(float(pt[0]), float(pt[1]), float(pt[2]))
    py5.end_shape()


def draw_orbit_circle(dist: float, hue: int):
    """Desenha o círculo de órbita no plano XY."""
    py5.push_matrix()
    py5.rotate_x(py5.HALF_PI)
    py5.no_fill()
    py5.stroke(hue, 60, 120, 35)
    py5.stroke_weight(0.5)
    py5.ellipse(0, 0, dist * 2, dist * 2)
    py5.stroke_weight(1)
    py5.pop_matrix()


def draw_rings(pos, base_size, hue, layers, unstable):
    py5.push_matrix()
    py5.translate(pos[0], pos[1], pos[2])
    py5.rotate_x(py5.PI / 2)
    py5.no_fill()
    for i in range(layers):
        r_size = base_size + (i * 12)
        if unstable:
            r_size += np.sin(py5.frame_count * 0.1) * 6
        py5.stroke(hue, 120, 255, 100 - (i * 20))
        py5.ellipse(0, 0, r_size, r_size)
    py5.pop_matrix()


def draw_asteroid_belt():
    """Cinturão procedural — mesma seed visual do original."""
    rng = np.random.default_rng(42)
    radii   = rng.uniform(380, 480, 400)
    angles  = (rng.uniform(0, py5.TWO_PI, 400) + py5.frame_count * 0.001) % py5.TWO_PI
    offz    = rng.uniform(-15, 15, 400)
    hues    = rng.uniform(10, 30, 400)
    py5.stroke_weight(1.5)
    for i in range(400):
        x = radii[i] * np.cos(angles[i])
        y = radii[i] * np.sin(angles[i])
        z = offz[i]
        py5.stroke(hues[i], 50, 200, 150)
        py5.point(x, y, z)
    py5.stroke_weight(1)


def draw_starship():
    """Disco voador idêntico ao harpia_starship.py."""
    global ship_rotation
    sx, sy, sz = ship_pos

    # ── Trail ────────────────────────────────────────────────────────────
    py5.no_fill()
    py5.stroke_weight(1)
    py5.begin_shape()
    for i in range(200):
        pt = ship_trail[i]
        if not np.all(pt == 0):
            alpha = int(((i + ship_trail_ptr) % 200) / 200 * 160)
            py5.stroke(90, 200, 255, alpha)
            py5.vertex(float(pt[0]), float(pt[1]), float(pt[2]))
    py5.end_shape()

    # ── Corpo ────────────────────────────────────────────────────────────
    sz_ship = 8.0
    engine_pulse = (np.sin(py5.frame_count * 0.2) + 1) * 0.5

    py5.push_matrix()
    py5.translate(sx, sy, sz)
    py5.rotate_y(ship_rotation)

    # Casco inferior achatado
    py5.no_stroke()
    py5.fill(180, 30, 240, 220)
    py5.push_matrix()
    py5.scale(1.0, 1.0, 0.35)
    py5.sphere(sz_ship)
    py5.pop_matrix()

    # Cúpula superior
    py5.fill(140, 180, 255, 200)
    py5.push_matrix()
    py5.translate(0, 0, sz_ship * 0.28)
    py5.scale(0.45, 0.45, 0.32)
    py5.sphere(sz_ship)
    py5.pop_matrix()

    # Anel exterior
    py5.no_fill()
    py5.stroke(90, 200, 255, 200)
    py5.stroke_weight(2.5)
    py5.push_matrix()
    py5.rotate_x(py5.HALF_PI)
    py5.ellipse(0, 0, sz_ship * 2.6, sz_ship * 2.6)
    py5.pop_matrix()

    # Luzes de posição
    py5.stroke_weight(4)
    for i in range(8):
        ang = (py5.TWO_PI / 8) * i + ship_rotation * 2
        lx  = np.cos(ang) * sz_ship * 1.25
        ly  = np.sin(ang) * sz_ship * 1.25
        h   = (py5.frame_count * 3 + i * 30) % 255
        py5.stroke(h, 255, 255, 200)
        py5.point(lx, ly, 0)

    # Motor
    py5.no_stroke()
    eb = int(180 + engine_pulse * 75)
    py5.fill(140, 255, eb, int(120 + engine_pulse * 80))
    py5.push_matrix()
    py5.translate(0, 0, -sz_ship * 0.3)
    py5.scale(0.6, 0.6, 0.18)
    py5.sphere(sz_ship)
    py5.pop_matrix()

    py5.pop_matrix()

    # Rótulo
    py5.push_matrix()
    py5.translate(sx, sy, sz)
    py5.fill(90, 255, 255)
    py5.text_size(11)
    py5.text("STARSHIP", sz_ship + 10, -sz_ship - 5)
    if ship_to:
        py5.fill(60, 200, 220)
        py5.text_size(9)
        py5.text(f"→ {ship_to}", sz_ship + 10, sz_ship + 6)
    py5.pop_matrix()
    py5.stroke_weight(1)


def draw_hud(frame_id, ok: bool):
    py5.hint(py5.DISABLE_DEPTH_TEST)
    py5.push_matrix()
    py5.reset_matrix()

    # ── Linha 1: resolução + fps ────────────────────────────────────────
    py5.text_size(13)
    py5.fill(255)
    fps_val = py5.get_frame_rate()
    py5.text(
        f"RESOLUÇÃO: {py5.width}x{py5.height}  |  FPS: {fps_val:.1f}  |  "
        f"FRAME: {frame_id}/{frames_idx[-1]}  ({current_frame+1}/{total_frames})",
        25, 25
    )

    # ── Linha 2: modo câmera ────────────────────────────────────────────
    cam_label = "SEGUINDO STARSHIP [C]" if follow_ship else "CÂMERA LIVRE [C]"
    py5.fill(90, 255, 255)
    py5.text(cam_label, 25, 45)

    # ── Linha 3: destino da nave ────────────────────────────────────────
    py5.fill(60, 220, 255)
    py5.text(f"STARSHIP: {ship_from} → {ship_to}  |  [SPACE] warp  [←→] frame  [R] reset", 25, 65)

    # ── SHA-256 HUD ─────────────────────────────────────────────────────
    box_x = 25
    box_y = py5.height - 110

    # fundo semi-transparente
    py5.no_stroke()
    py5.fill(0, 0, 20, 200)
    py5.rect(box_x - 10, box_y - 22, py5.width - 40, 100, 8)

    # título + status
    if ok:
        py5.fill(100, 255, 180)
        status_txt = "✓  SHA-256 VÁLIDO"
    else:
        py5.fill(0, 255, 200)
        status_txt = "✗  SHA-256 INVÁLIDO — FRAME CORROMPIDO"

    py5.text_size(12)
    py5.text(f"[FRAME {frame_id}]  {status_txt}", box_x, box_y)

    # hash em texto — cortado em dois pedaços
    if last_sha_hex:
        py5.fill(180, 200, 255)
        py5.text_size(10)
        py5.text(f"SHA: {last_sha_hex[:32]}", box_x, box_y + 18)
        py5.text(f"     {last_sha_hex[32:]}", box_x, box_y + 32)

    # barra visual do hash (64 nibbles)
    if last_sha_hex:
        bar_w   = (py5.width - 80) / 64
        bar_top = box_y + 46
        for i, ch in enumerate(last_sha_hex):
            val = int(ch, 16) / 15.0
            h   = val * 180 + 60
            a   = 200 if ok else 160
            py5.fill(h, 200, 255, a)
            py5.rect(box_x + i * bar_w, bar_top, max(bar_w - 1, 1), 14)

    py5.pop_matrix()
    py5.hint(py5.ENABLE_DEPTH_TEST)


# ─── Controles ───────────────────────────────────────────────────────────────

def mouse_wheel(event):
    global cam_dist
    cam_dist += event.get_count() * 40
    cam_dist = py5.constrain(cam_dist, 50, 7000)


def mouse_dragged():
    global cam_lat, cam_lon
    if py5.mouse_button == py5.RIGHT:
        cam_lon += (py5.mouse_x - py5.pmouse_x) * 0.01
        cam_lat -= (py5.mouse_y - py5.pmouse_y) * 0.01
        cam_lat  = py5.constrain(cam_lat, 0.05, py5.HALF_PI * 0.98)


def key_pressed():
    global warp_mode, follow_ship, current_frame

    k  = py5.key
    kc = py5.key_code

    if k == " ":
        warp_mode = True
    elif k in ("c", "C"):
        follow_ship = not follow_ship
    elif k in ("f", "F"):
        py5.set_full_screen(not py5.is_full_screen())
    elif k in ("r", "R"):
        current_frame = 0
    elif k == py5.CODED:
        if kc == py5.LEFT:
            current_frame = (current_frame - 2) % total_frames
        elif kc == py5.RIGHT:
            current_frame = (current_frame + 1) % total_frames


def key_released():
    global warp_mode
    if py5.key == " ":
        warp_mode = False


def window_resized():
    pass


# ─── Run ─────────────────────────────────────────────────────────────────────
py5.run_sketch()
