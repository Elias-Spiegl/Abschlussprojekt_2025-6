import io
import json
import time
import base64
from datetime import datetime
from html import escape
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
import streamlit as st

from model import Structure
from optimizer import TopologyOptimizer
from visualizer import Visualizer

st.set_page_config(page_title="Modell Optimieren", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {display: none !important;}
    [data-testid="collapsedControl"] {display: none !important;}
    button[kind="primary"] {
      min-height: 3.2rem !important;
      font-size: 1.1rem !important;
      font-weight: 700 !important;
      border-radius: 0.75rem !important;
      border: 1px solid #5a6273 !important;
      background: #2f3542 !important;
      color: #e8ecf4 !important;
      box-shadow: none !important;
    }
    button[kind="primary"]:hover {
      background: #3a4150 !important;
      border-color: #6a7388 !important;
    }
    button[kind="primary"]:active {
      background: #464f61 !important;
      border-color: #7b859b !important;
    }
    button[kind="primary"]:disabled {
      background: #d9dde5 !important;
      border-color: #aeb6c5 !important;
      color: #1f2937 !important;
      box-shadow: none !important;
      opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATE_DIR = DATA_DIR / "states"
INDEX_FILE = DATA_DIR / "model_index.json"
STATE_DIR.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_ts(value: str | None) -> str:
    if not value:
        return "-"
    return str(value).replace("T", " ")


def reset_optimizer_state() -> None:
    st.session_state.opt_initialized = False
    st.session_state.opt_running = False
    st.session_state.opt_history = []
    st.session_state.opt_view_index = -1
    st.session_state.opt_iteration = 0
    st.session_state.opt_target_count = 0
    st.session_state.opt_start_active = 0
    st.session_state.opt_abs_limit = 0.0
    st.session_state.opt_finished = False
    st.session_state.opt_stop_reason = ""
    st.session_state.opt_status_type = "info"
    st.session_state.opt_status_msg = ""
    st.session_state.smooth_history = []
    st.session_state.smooth_index = -1
    st.session_state.opt_gif_bytes = None
    st.session_state.opt_gif_signature = None
    st.session_state.opt_report_bytes = None
    st.session_state.opt_report_signature = None


def push_opt_snapshot(structure: Structure) -> None:
    history = list(st.session_state.opt_history)
    view_idx = int(st.session_state.opt_view_index)
    if view_idx < len(history) - 1:
        history = history[: view_idx + 1]
    history.append(structure.to_dict())
    st.session_state.opt_history = history
    st.session_state.opt_view_index = len(history) - 1
    st.session_state.opt_iteration = max(0, len(history) - 1)


def restore_opt_snapshot(index: int) -> bool:
    history = st.session_state.opt_history
    if index < 0 or index >= len(history):
        return False
    st.session_state.structure = Structure.from_dict(history[index])
    st.session_state.opt_view_index = index
    return True


def restore_smooth_snapshot(index: int) -> bool:
    history = st.session_state.smooth_history
    if index < 0 or index >= len(history):
        return False
    st.session_state.structure = Structure.from_dict(history[index])
    st.session_state.smooth_index = index
    return True


def build_gif_bytes(
    history: list[dict],
    show_deformation: bool,
    scale_factor: float,
    fem_color_map: bool,
    color_percentile: int,
    show_background_nodes: bool,
    line_width: float,
    color_levels: int,
    fixed_color_vmax,
    metric_mode: str,
    normalize_mode: str,
    element_filter: str,
) -> bytes | None:
    if not history:
        return None

    frames = []
    history_len = len(history)
    step = max(1, history_len // 35)
    indices_to_render = list(range(0, history_len, step))
    if indices_to_render[-1] != history_len - 1:
        indices_to_render.append(history_len - 1)

    for i in indices_to_render:
        temp_struct = Structure.from_dict(history[i])
        fig = Visualizer.plot_structure(
            temp_struct,
            show_deformation=show_deformation,
            scale_factor=scale_factor,
            selected_node_id=None,
            colorize_elements=fem_color_map,
            color_percentile=color_percentile,
            show_background_nodes=show_background_nodes,
            line_width=line_width,
            color_levels=color_levels,
            fixed_color_vmax=fixed_color_vmax,
            metric_mode=metric_mode,
            normalize_mode=normalize_mode,
            element_filter=element_filter,
            show_colorbar=False,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close(fig)
        buf.close()

    if not frames:
        return None

    gif_buf = io.BytesIO()
    frames[0].save(
        gif_buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
    )
    return gif_buf.getvalue()


def _history_series(history: list[dict]) -> tuple[list[float], list[float]]:
    if not history:
        return [], []

    start_nodes = history[0].get("nodes", [])
    total_nodes_count = max(1, len(start_nodes))
    mass_data: list[float] = []
    disp_data: list[float] = []

    for state in history:
        nodes = state.get("nodes", [])
        active_count = sum(1 for n in nodes if n.get("active", True))
        mass_data.append((active_count / total_nodes_count) * 100)

        u_max = 0.0
        for n in nodes:
            dist = (n.get("u_x", 0) ** 2 + n.get("u_z", 0) ** 2) ** 0.5
            if dist > u_max:
                u_max = dist
        disp_data.append(float(u_max))

    return mass_data, disp_data


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    buf.close()
    plt.close(fig)
    return encoded


def build_report_html(
    model_meta: dict | None,
    struct: Structure,
    history: list[dict],
    stop_reason: str,
    opt_initialized: bool,
    opt_finished: bool,
    opt_running: bool,
    opt_iteration: int,
    target_mass_percent: int,
    max_stiffness_loss_percent: int,
    force_rows: list[dict],
    support_rows: list[dict],
    smooth_history_len: int,
    gif_bytes: bytes | None,
) -> bytes:
    report_time = now_iso()
    mass_data, disp_data = _history_series(history)
    current_active = sum(1 for n in struct.nodes if n.active)
    total_nodes = max(1, len(struct.nodes))
    current_mass_percent = (current_active / total_nodes) * 100.0
    smooth_ops = max(0, int(smooth_history_len) - 1)

    chart_mass_b64 = ""
    chart_disp_b64 = ""
    if mass_data:
        fig1, ax1 = plt.subplots(figsize=(8, 3))
        ax1.plot(range(len(mass_data)), mass_data, color="#2563eb", linewidth=2)
        ax1.set_title("Massenabbau über Iterationen (%)")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Masse [%]")
        ax1.grid(alpha=0.25)
        chart_mass_b64 = _fig_to_base64(fig1)

    if disp_data:
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(range(len(disp_data)), disp_data, color="#dc2626", linewidth=2)
        ax2.set_title("Max. Verformung über Iterationen")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Max |u|")
        ax2.grid(alpha=0.25)
        chart_disp_b64 = _fig_to_base64(fig2)

    gif_b64 = base64.b64encode(gif_bytes).decode("ascii") if gif_bytes else ""
    model_name = (model_meta or {}).get("name") or "(ohne Namen)"
    model_id = int((model_meta or {}).get("id", 0))
    created_at = format_ts((model_meta or {}).get("created_at") or "")
    updated_at = format_ts((model_meta or {}).get("updated_at") or "")
    status_text = "Läuft" if opt_running else "Fertig" if opt_finished else "Nicht gestartet"
    stop_text = stop_reason or "-"
    initial_mass = mass_data[0] if mass_data else 100.0
    final_mass = mass_data[-1] if mass_data else current_mass_percent
    max_disp = max(disp_data) if disp_data else 0.0

    html = f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <title>Optimierungsbericht - Modell {escape(str(model_name))}</title>
  <style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 28px; color: #111827; }}
    h1,h2 {{ margin-bottom: 8px; }}
    .meta {{ color: #4b5563; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 18px; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 10px; padding: 14px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    td, th {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    .img {{ width: 100%; max-width: 900px; border: 1px solid #d1d5db; border-radius: 8px; }}
    ul {{ margin-top: 6px; }}
  </style>
</head>
<body>
  <h1>Automatisch generierter Optimierungsbericht</h1>
  <div class="meta">Erstellt am: {escape(report_time)}</div>

  <div class="grid">
    <div class="card">
      <h2>Modell</h2>
      <table>
        <tr><th>Feld</th><th>Wert</th></tr>
        <tr><td>ID</td><td>{model_id}</td></tr>
        <tr><td>Name</td><td>{escape(str(model_name))}</td></tr>
        <tr><td>Erstellt</td><td>{escape(created_at)}</td></tr>
        <tr><td>Zuletzt gespeichert</td><td>{escape(updated_at)}</td></tr>
        <tr><td>Breite x Höhe</td><td>{int(struct.width)} x {int(struct.height)}</td></tr>
        <tr><td>Aktive Knoten aktuell</td><td>{current_active} / {total_nodes} ({current_mass_percent:.1f}%)</td></tr>
        <tr><td>Kräfte</td><td>{len(force_rows)}</td></tr>
        <tr><td>Lager</td><td>{len(support_rows)}</td></tr>
      </table>
    </div>
    <div class="card">
      <h2>Optimierung</h2>
      <table>
        <tr><th>Feld</th><th>Wert</th></tr>
        <tr><td>Status</td><td>{escape(status_text)}</td></tr>
        <tr><td>Iterationsstand</td><td>{int(opt_iteration)}</td></tr>
        <tr><td>Ziel-Masse</td><td>{int(target_mass_percent)}%</td></tr>
        <tr><td>Max. Steifigkeitsverlust</td><td>{int(max_stiffness_loss_percent)}%</td></tr>
        <tr><td>Start-Masse</td><td>{initial_mass:.1f}%</td></tr>
        <tr><td>End-Masse</td><td>{final_mass:.1f}%</td></tr>
        <tr><td>Max. Verformung (über Verlauf)</td><td>{max_disp:.6f}</td></tr>
        <tr><td>Stop-Grund</td><td>{escape(stop_text)}</td></tr>
        <tr><td>Glättungsoperationen</td><td>{smooth_ops}</td></tr>
      </table>
    </div>
  </div>

  <div class="card">
    <h2>Was wurde gemacht</h2>
    <ul>
      <li>Modell wurde angelegt und in der aktuellen Sitzung weiter optimiert.</li>
      <li>Optimierungsverlauf umfasst {max(0, len(history) - 1)} Schritte mit fortlaufender Materialreduktion.</li>
      <li>Aktueller Zustand: {current_active} aktive Knoten von {total_nodes}.</li>
      <li>Falls genutzt, wurden Glättungsschritte auf das Ergebnis angewendet ({smooth_ops} Schritte).</li>
    </ul>
  </div>
"""

    if chart_mass_b64:
        html += f"""
  <h2>Massenabbau über Iterationen (%)</h2>
  <img class="img" src="data:image/png;base64,{chart_mass_b64}" alt="Massenabbau über Iterationen" />
"""
    if chart_disp_b64:
        html += f"""
  <h2>Max. Verformung über Iterationen</h2>
  <img class="img" src="data:image/png;base64,{chart_disp_b64}" alt="Maximale Verformung über Iterationen" />
"""
    if gif_b64:
        html += f"""
  <h2>Animation des Optimierungsverlaufs</h2>
  <img class="img" src="data:image/gif;base64,{gif_b64}" alt="Animation des Optimierungsverlaufs" />
"""

    html += "\n</body>\n</html>\n"
    return html.encode("utf-8")


def reset_visualization_state() -> None:
    st.session_state.show_deformation = False
    st.session_state.fem_colormap = False
    st.session_state.deformation_display_percent = 100
    st.session_state.element_focus_radio = "Alle"
    st.session_state.line_width_slider = 0.8
    st.session_state.fem_metric_mode = "Energie/Länge"


def max_displacement(structure: Structure) -> float:
    u_max = 0.0
    for node in structure.nodes:
        dist = float((node.u_x ** 2 + node.u_z ** 2) ** 0.5)
        if dist > u_max:
            u_max = dist
    return u_max


def load_index() -> list[dict]:
    if INDEX_FILE.exists():
        raw = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return raw
    return []


def save_index(index: list[dict]) -> None:
    INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")


def next_model_id(index: list[dict]) -> int:
    if not index:
        return 1
    return max(int(item["id"]) for item in index) + 1


def model_path(model_id: int) -> Path:
    return STATE_DIR / f"model_{model_id}.json"


def model_label(meta: dict) -> str:
    name = meta.get("name") or "(ohne Namen)"
    return f"{name} | erstellt: {format_ts(meta.get('created_at'))}"


def snapshots_equal(a: dict, b: dict) -> bool:
    try:
        return json.dumps(a, sort_keys=True, separators=(",", ":")) == json.dumps(
            b, sort_keys=True, separators=(",", ":")
        )
    except Exception:
        return False


def merge_history(base_history: list[dict], new_segment: list[dict]) -> list[dict]:
    merged = list(base_history or [])
    if not new_segment:
        return merged

    if not merged:
        return list(new_segment)

    # Wenn letzter bekannter Zustand der Start des neuen Segments ist,
    # nur die neuen Schritte anhängen.
    start_idx = 0
    if snapshots_equal(merged[-1], new_segment[0]):
        start_idx = 1
    elif snapshots_equal(merged[-1], new_segment[-1]):
        # Segment bereits enthalten.
        return merged

    for snap in new_segment[start_idx:]:
        if not merged or not snapshots_equal(merged[-1], snap):
            merged.append(snap)
    return merged


def _existing_history_timeline(model_id: int) -> list[dict]:
    path = model_path(model_id)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            history = payload.get("history", {})
            timeline = history.get("timeline", []) if isinstance(history, dict) else []
            if isinstance(timeline, list):
                return timeline
    except Exception:
        pass
    return []


def make_unique_model_name(index: list[dict], requested_name: str, current_model_id: int | None = None) -> str:
    base = (requested_name or "").strip()
    if not base:
        return ""

    taken = set()
    for m in index:
        mid = int(m.get("id", -1))
        if current_model_id is not None and mid == int(current_model_id):
            continue
        name = (m.get("name") or "").strip()
        if name:
            taken.add(name.casefold())

    if base.casefold() not in taken:
        return base

    i = 1
    while True:
        candidate = f"{base}_{i}"
        if candidate.casefold() not in taken:
            return candidate
        i += 1


def to_local_table_rows(index: list[dict]) -> list[dict]:
    rows = []
    for m in sorted(index, key=lambda x: int(x["id"])):
        rows.append(
            {
                "Name": m.get("name") or "-",
                "Erstellt": format_ts(m.get("created_at")),
                "Geändert": format_ts(m.get("updated_at")),
                "Breite": int(m.get("width", 0)),
                "Höhe": int(m.get("height", 0)),
            }
        )
    return rows


def save_model_snapshot(
    struct: Structure,
    model_id: int,
    name: str,
    created_at: str,
    history_timeline: list[dict] | None = None,
) -> dict:
    metadata = {
        "id": int(model_id),
        "name": (name or "").strip(),
        "created_at": created_at,
        "updated_at": now_iso(),
        "width": int(struct.width),
        "height": int(struct.height),
    }
    timeline = history_timeline if history_timeline is not None else _existing_history_timeline(int(model_id))
    payload = {
        "metadata": metadata,
        "structure": struct.to_dict(),
        "history": {"timeline": timeline},
    }
    model_path(model_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metadata


def load_model_snapshot(model_id: int) -> tuple[Structure, dict, list[dict]]:
    payload = json.loads(model_path(model_id).read_text(encoding="utf-8"))
    # Backward compatibility for old plain-structure files
    if "structure" in payload and "metadata" in payload:
        struct = Structure.from_dict(payload["structure"])
        metadata = payload["metadata"]
        history = payload.get("history", {})
        timeline = history.get("timeline", []) if isinstance(history, dict) else []
        if not isinstance(timeline, list):
            timeline = []
    else:
        struct = Structure.from_dict(payload)
        metadata = {
            "id": int(model_id),
            "name": "",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "width": int(struct.width),
            "height": int(struct.height),
        }
        timeline = []
    return struct, metadata, timeline


if "structure" not in st.session_state:
    st.session_state.structure = None
if "model_id" not in st.session_state:
    st.session_state.model_id = None
if "model_name" not in st.session_state:
    st.session_state.model_name = ""
if "model_created_at" not in st.session_state:
    st.session_state.model_created_at = None
if "opt_initialized" not in st.session_state:
    st.session_state.opt_initialized = False
if "opt_running" not in st.session_state:
    st.session_state.opt_running = False
if "opt_history" not in st.session_state:
    st.session_state.opt_history = []
if "opt_view_index" not in st.session_state:
    st.session_state.opt_view_index = -1
if "opt_iteration" not in st.session_state:
    st.session_state.opt_iteration = 0
if "opt_target_count" not in st.session_state:
    st.session_state.opt_target_count = 0
if "opt_start_active" not in st.session_state:
    st.session_state.opt_start_active = 0
if "opt_abs_limit" not in st.session_state:
    st.session_state.opt_abs_limit = 0.0
if "opt_finished" not in st.session_state:
    st.session_state.opt_finished = False
if "opt_stop_reason" not in st.session_state:
    st.session_state.opt_stop_reason = ""
if "opt_status_type" not in st.session_state:
    st.session_state.opt_status_type = "info"
if "opt_status_msg" not in st.session_state:
    st.session_state.opt_status_msg = ""
if "smooth_history" not in st.session_state:
    st.session_state.smooth_history = []
if "smooth_index" not in st.session_state:
    st.session_state.smooth_index = -1
if "show_deformation" not in st.session_state:
    st.session_state.show_deformation = False
if "fem_colormap" not in st.session_state:
    st.session_state.fem_colormap = False
if "deformation_display_percent" not in st.session_state:
    st.session_state.deformation_display_percent = 100
if "element_focus_radio" not in st.session_state:
    st.session_state.element_focus_radio = "Alle"
if "line_width_slider" not in st.session_state:
    st.session_state.line_width_slider = 0.8
if "fem_metric_mode" not in st.session_state:
    st.session_state.fem_metric_mode = "Energie/Länge"
elif st.session_state.fem_metric_mode == "Strain":
    st.session_state.fem_metric_mode = "Dehnung"
elif st.session_state.fem_metric_mode == "Displacement":
    st.session_state.fem_metric_mode = "Verschiebung"
if "opt_gif_bytes" not in st.session_state:
    st.session_state.opt_gif_bytes = None
if "opt_gif_signature" not in st.session_state:
    st.session_state.opt_gif_signature = None
if "opt_report_bytes" not in st.session_state:
    st.session_state.opt_report_bytes = None
if "opt_report_signature" not in st.session_state:
    st.session_state.opt_report_signature = None
if "model_lifetime_history" not in st.session_state:
    st.session_state.model_lifetime_history = []

nav1, nav2, nav3 = st.columns([1, 1, 1])
with nav1:
    if st.button("Modell Übersicht", use_container_width=True, type="primary"):
        st.switch_page("main.py")
with nav2:
    if st.button("Modell Erstellen / Bearbeiten", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Modell_Erstellen.py")
with nav3:
    st.button("Modell Optimieren", use_container_width=True, disabled=True, type="primary")

index = load_index()

pending_load_id = st.session_state.pop("startup_load_model_id", None)
if pending_load_id is not None:
    try:
        loaded_struct, loaded_meta, loaded_timeline = load_model_snapshot(int(pending_load_id))
        st.session_state.structure = loaded_struct
        st.session_state.model_id = int(loaded_meta["id"])
        st.session_state.model_name = loaded_meta.get("name", "")
        st.session_state.model_created_at = loaded_meta.get("created_at", now_iso())
        st.session_state.model_lifetime_history = (
            list(loaded_timeline) if loaded_timeline else [loaded_struct.to_dict()]
        )
        reset_optimizer_state()
        reset_visualization_state()
    except Exception:
        # Startseite kann eine ungültige ID übergeben haben (gelöscht/defekt).
        pass

if st.session_state.structure is None:
    st.subheader("Kein Modell geladen")
    st.info("Öffne ein Modell in der Modell Übersicht oder erstelle zuerst ein neues Modell.")
    st.stop()

struct = st.session_state.structure
if not st.session_state.model_lifetime_history:
    st.session_state.model_lifetime_history = [struct.to_dict()]
opt = TopologyOptimizer(struct)

# Data snapshots for editor/overview
load_nodes = [n for n in struct.nodes if n.force_x != 0 or n.force_z != 0]
support_nodes = [n for n in struct.nodes if n.fixed_x or n.fixed_z]
force_rows = [
    {
        "X": int(n.x),
        "Z": int(n.z),
        "Fx": round(float(n.force_x), 4),
        "Fz": round(float(n.force_z), 4),
    }
    for n in load_nodes
]
support_rows = [
    {
        "X": int(n.x),
        "Z": int(n.z),
        "Typ": "Festlager" if (n.fixed_x and n.fixed_z) else "Loslager" if ((not n.fixed_x) and n.fixed_z) else "Sonderfall",
    }
    for n in support_nodes
]
fixed_support_count = sum(1 for n in struct.nodes if n.fixed_x and n.fixed_z)
roller_support_count = sum(1 for n in struct.nodes if (not n.fixed_x) and n.fixed_z)
current_active = sum(1 for n in struct.nodes if n.active)
total_nodes = len(struct.nodes)
current_model_meta = next(
    (m for m in index if int(m.get("id", -1)) == int(st.session_state.model_id or -1)),
    None,
)

# Visualization settings from session state (controls live further down).
visualization_physics_ok = opt.solve_step()
show_deformation = bool(st.session_state.show_deformation)
if show_deformation:
    scale = float(st.session_state.deformation_display_percent) / 100.0
else:
    scale = 0.0

fem_color_map = bool(st.session_state.fem_colormap)
line_width = float(st.session_state.line_width_slider)
color_percentile = 95
color_levels = 15
fixed_color_vmax = None
normalize_mode = "orientation"
element_filter = {"Alle": "all", "H+V": "hv", "Diagonal": "diag"}.get(st.session_state.element_focus_radio, "all")
metric_mode = {
    "Energie/Länge": "energy_per_length",
    "Dehnung": "strain",
    "Verschiebung": "displacement",
}.get(st.session_state.fem_metric_mode, "energy_per_length")
show_background_nodes = not fem_color_map

# Top layout: left model info + automation, right status + plot.
row_top_left, row_top_right = st.columns([0.85, 2.15])

with row_top_left:
    with st.container(border=True):
        st.markdown(f"### Modellname: {st.session_state.model_name or '(ohne Namen)'}")
        st.caption(f"Modell erstellt: {format_ts(st.session_state.model_created_at)}")
        st.caption(f"Knoten aktiv: {current_active} / {total_nodes}")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Kräfte", len(force_rows))
        with m2:
            st.metric("Loslager", roller_support_count)
        with m3:
            st.metric("Festlager", fixed_support_count)

        with st.expander("Details Kräfte/Lager", expanded=False):
            d1, d2 = st.columns(2)
            with d1:
                if force_rows:
                    st.dataframe(force_rows, use_container_width=True, hide_index=True, height=140)
                else:
                    st.caption("Keine Kräfte gesetzt")
            with d2:
                if support_rows:
                    st.dataframe(support_rows, use_container_width=True, hide_index=True, height=140)
                else:
                    st.caption("Keine Lager gesetzt")

        if st.button("Modellstand Speichern", use_container_width=True):
            if st.session_state.model_id is None:
                st.error("Kein Modellkontext vorhanden.")
            else:
                metadata = save_model_snapshot(
                    struct,
                    int(st.session_state.model_id),
                    st.session_state.model_name or "",
                    st.session_state.model_created_at or now_iso(),
                    history_timeline=merge_history(
                        st.session_state.model_lifetime_history,
                        [struct.to_dict()],
                    ),
                )
                index = [m for m in index if int(m["id"]) != int(metadata["id"])]
                index.append(metadata)
                save_index(index)
                st.session_state.model_lifetime_history = merge_history(
                    st.session_state.model_lifetime_history,
                    [struct.to_dict()],
                )
                st.success("Modell gespeichert.")
                st.rerun()

    # Optimierungsparameter und Steuerung
    with st.container(border=True):
        st.markdown("**Optimierungsparameter**")
        target_mass_percent = st.slider("Ziel-Masse (%)", 10, 99, 70)
        max_stiffness_loss_percent = st.slider(
            "Max. Steifigkeitsverlust (%)", 0, 100, 60, step=1
        )
        allowed_softening_ratio = 1.0 + (max_stiffness_loss_percent / 100.0)

        if not st.session_state.opt_initialized:
            if st.button("Start Optimierung", use_container_width=True):
                if not opt.solve_step():
                    st.session_state.opt_status_type = "error"
                    st.session_state.opt_status_msg = "Start-Modell ist instabil! Bitte Lager/Kräfte prüfen."
                else:
                    reset_optimizer_state()
                    st.session_state.opt_initialized = True
                    st.session_state.opt_running = True
                    st.session_state.opt_finished = False
                    st.session_state.opt_gif_bytes = None
                    st.session_state.opt_gif_signature = None
                    st.session_state.opt_report_bytes = None
                    st.session_state.opt_report_signature = None
                    st.session_state.opt_target_count = int(total_nodes * (target_mass_percent / 100))
                    st.session_state.opt_start_active = sum(1 for n in struct.nodes if n.active)
                    st.session_state.opt_abs_limit = max_displacement(struct) * allowed_softening_ratio
                    st.session_state.opt_status_type = "info"
                    st.session_state.opt_status_msg = "Optimierung läuft..."
                    st.session_state.opt_stop_reason = ""
                    push_opt_snapshot(struct)
                    st.rerun()
        else:
            ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1, 1])
            with ctrl1:
                play_label = "Stop Optimization" if st.session_state.opt_running else "Run Optimization"
                if st.button(play_label, use_container_width=True):
                    if st.session_state.opt_running:
                        st.session_state.opt_running = False
                        st.session_state.opt_status_type = "info"
                        st.session_state.opt_status_msg = "Optimierung pausiert."
                    else:
                        if not opt.solve_step():
                            st.session_state.opt_status_type = "error"
                            st.session_state.opt_status_msg = "Modell ist instabil. Bitte Lager/Kräfte prüfen."
                        else:
                            st.session_state.opt_target_count = int(total_nodes * (target_mass_percent / 100))
                            st.session_state.opt_abs_limit = max_displacement(struct) * allowed_softening_ratio
                            st.session_state.opt_running = True
                            st.session_state.opt_finished = False
                            st.session_state.opt_gif_bytes = None
                            st.session_state.opt_gif_signature = None
                            st.session_state.opt_report_bytes = None
                            st.session_state.opt_report_signature = None
                            st.session_state.smooth_history = []
                            st.session_state.smooth_index = -1
                            st.session_state.opt_stop_reason = ""
                            st.session_state.opt_status_type = "info"
                            st.session_state.opt_status_msg = "Optimierung läuft..."
                    st.rerun()
            with ctrl2:
                back_disabled = st.session_state.opt_running or int(st.session_state.opt_view_index) <= 0
                if st.button("‹", use_container_width=True, disabled=back_disabled):
                    if restore_opt_snapshot(int(st.session_state.opt_view_index) - 1):
                        st.session_state.opt_status_type = "info"
                        st.session_state.opt_status_msg = "Eine Iteration zurück."
                        st.rerun()
            with ctrl3:
                forward_disabled = st.session_state.opt_running or int(st.session_state.opt_view_index) >= (len(st.session_state.opt_history) - 1)
                if st.button("›", use_container_width=True, disabled=forward_disabled):
                    if restore_opt_snapshot(int(st.session_state.opt_view_index) + 1):
                        st.session_state.opt_status_type = "info"
                        st.session_state.opt_status_msg = "Eine Iteration vor."
                        st.rerun()

with row_top_right:
    if not visualization_physics_ok:
        st.warning("Visualisierung: Modell aktuell instabil (Lager/Kräfte prüfen).")

    status_col, export_col = st.columns([3.2, 1.0])
    with status_col:
        with st.container(border=True):
            st.markdown("**Optimierer Status**")    # Statusinformationen und Fehlermeldungen
            opt_status_placeholder = st.empty()
            st.markdown("### Masse")
            opt_mass_placeholder = st.empty()
            opt_progress_bar = st.progress(0.0)
            opt_message_placeholder = st.empty()
            if str(st.session_state.opt_status_msg).strip():
                if st.button("Meldungen zurücksetzen", key="reset_status_messages_top"):
                    st.session_state.opt_status_msg = ""
                    st.session_state.opt_status_type = "info"
                    st.rerun()

    fig = Visualizer.plot_structure(
        struct,
        show_deformation=show_deformation,
        scale_factor=scale,
        selected_node_id=None,
        colorize_elements=fem_color_map,
        color_percentile=color_percentile,
        show_background_nodes=show_background_nodes,
        line_width=line_width,
        color_levels=color_levels,
        fixed_color_vmax=fixed_color_vmax,
        metric_mode=metric_mode,
        normalize_mode=normalize_mode,
        element_filter=element_filter,
        show_colorbar=False,
    )
    png_buffer = io.BytesIO()
    fig.savefig(png_buffer, format="png", dpi=200, bbox_inches="tight")
    png_buffer.seek(0)
    export_history = merge_history(
        st.session_state.model_lifetime_history,
        [struct.to_dict()],
    )
    with export_col:
        gif_ready = bool(st.session_state.opt_finished and len(export_history) > 1)
        if gif_ready:
            gif_signature = (
                int(st.session_state.model_id or 0),
                len(export_history),
                bool(st.session_state.show_deformation),
                int(st.session_state.deformation_display_percent),
                bool(st.session_state.fem_colormap),
                str(st.session_state.fem_metric_mode),
                str(st.session_state.element_focus_radio),
                float(st.session_state.line_width_slider),
            )
            if st.session_state.opt_gif_signature != gif_signature:
                st.session_state.opt_gif_bytes = None
                st.session_state.opt_gif_signature = None

            if st.session_state.opt_gif_bytes:
                st.download_button(
                    "GIF herunterladen",
                    data=st.session_state.opt_gif_bytes,
                    file_name=f"modell_{st.session_state.model_id}_animation.gif",
                    mime="image/gif",
                    use_container_width=True,
                )
            elif st.button("GIF generieren", use_container_width=True):
                with st.spinner("GIF wird erstellt..."):
                    st.session_state.opt_gif_bytes = build_gif_bytes(
                        history=export_history,
                        show_deformation=bool(st.session_state.show_deformation),
                        scale_factor=(float(st.session_state.deformation_display_percent) / 100.0)
                        if bool(st.session_state.show_deformation)
                        else 0.0,
                        fem_color_map=bool(st.session_state.fem_colormap),
                        color_percentile=color_percentile,
                        show_background_nodes=not bool(st.session_state.fem_colormap),
                        line_width=float(st.session_state.line_width_slider),
                        color_levels=color_levels,
                        fixed_color_vmax=fixed_color_vmax,
                        metric_mode={
                            "Energie/Länge": "energy_per_length",
                            "Dehnung": "strain",
                            "Verschiebung": "displacement",
                            "Strain": "strain",
                            "Displacement": "displacement",
                        }.get(st.session_state.fem_metric_mode, "energy_per_length"),
                        normalize_mode=normalize_mode,
                        element_filter={"Alle": "all", "H+V": "hv", "Diagonal": "diag"}.get(
                            st.session_state.element_focus_radio, "all"
                        ),
                    )
                    st.session_state.opt_gif_signature = gif_signature
                if st.session_state.opt_gif_bytes:
                    st.success("GIF erstellt. Jetzt herunterladen.")
                    st.rerun()
                else:
                    st.error("GIF konnte nicht erstellt werden.")
        st.download_button(
            "PNG Exprotieren",
            data=png_buffer.getvalue(),
            file_name=f"modell_{st.session_state.model_id}_topologie.png",
            mime="image/png",
            use_container_width=True,
        )
        report_ready = bool(st.session_state.opt_finished and len(export_history) > 1)
        if report_ready and st.button("Bericht generieren", use_container_width=True):
            with st.spinner("Bericht wird generiert..."):
                report_signature = (
                    int(st.session_state.model_id or 0),
                    len(export_history),
                    int(st.session_state.opt_iteration),
                    int(st.session_state.smooth_index),
                    bool(st.session_state.show_deformation),
                    int(st.session_state.deformation_display_percent),
                    bool(st.session_state.fem_colormap),
                    str(st.session_state.fem_metric_mode),
                    str(st.session_state.element_focus_radio),
                    float(st.session_state.line_width_slider),
                    int(target_mass_percent),
                    int(max_stiffness_loss_percent),
                    str(st.session_state.opt_stop_reason),
                )
                report_gif_bytes = build_gif_bytes(
                    history=export_history,
                    show_deformation=bool(st.session_state.show_deformation),
                    scale_factor=(float(st.session_state.deformation_display_percent) / 100.0)
                    if bool(st.session_state.show_deformation)
                    else 0.0,
                    fem_color_map=bool(st.session_state.fem_colormap),
                    color_percentile=color_percentile,
                    show_background_nodes=not bool(st.session_state.fem_colormap),
                    line_width=float(st.session_state.line_width_slider),
                    color_levels=color_levels,
                    fixed_color_vmax=fixed_color_vmax,
                    metric_mode={
                        "Energie/Länge": "energy_per_length",
                        "Dehnung": "strain",
                        "Verschiebung": "displacement",
                        "Strain": "strain",
                        "Displacement": "displacement",
                    }.get(st.session_state.fem_metric_mode, "energy_per_length"),
                    normalize_mode=normalize_mode,
                    element_filter={"Alle": "all", "H+V": "hv", "Diagonal": "diag"}.get(
                        st.session_state.element_focus_radio, "all"
                    ),
                )
                st.session_state.opt_report_bytes = build_report_html(
                    model_meta=current_model_meta,
                    struct=struct,
                    history=export_history,
                    stop_reason=str(st.session_state.opt_stop_reason),
                    opt_initialized=bool(st.session_state.opt_initialized),
                    opt_finished=bool(st.session_state.opt_finished),
                    opt_running=bool(st.session_state.opt_running),
                    opt_iteration=int(st.session_state.opt_iteration),
                    target_mass_percent=int(target_mass_percent),
                    max_stiffness_loss_percent=int(max_stiffness_loss_percent),
                    force_rows=force_rows,
                    support_rows=support_rows,
                    smooth_history_len=len(st.session_state.smooth_history),
                    gif_bytes=report_gif_bytes,
                )
                st.session_state.opt_report_signature = report_signature
            st.success("Bericht generiert.")
        if st.session_state.opt_report_bytes:
            st.download_button(
                "Bericht herunterladen (.html)",
                data=st.session_state.opt_report_bytes,
                file_name=f"modell_{st.session_state.model_id}_bericht.html",
                mime="text/html",
                use_container_width=True,
            )
    plot_placeholder = st.empty()
    plot_placeholder.pyplot(fig)
    plt.close(fig)

# Bottom layout: smoothing + visualization (side by side).
panel_smooth, panel_vis = st.columns(2)

with panel_smooth:
    with st.container(border=True):
        st.markdown("**Struktur Glätten**")
        smooth_strength = st.radio(
            "Stärke",
            options=[1, 2, 3],
            horizontal=True,
            key="smooth_strength_selector",
        )
        is_smooth_ready = st.session_state.opt_initialized and st.session_state.opt_finished and (not st.session_state.opt_running)
        if not is_smooth_ready:
            st.caption("Wird nach abgeschlossener Optimierung aktiv.")
        s_col1, s_col2, s_col3 = st.columns([1.4, 1, 1])
        with s_col1:
            if st.button("Glättung anwenden", use_container_width=True, disabled=not is_smooth_ready):
                if int(st.session_state.smooth_index) < len(st.session_state.smooth_history) - 1:
                    st.session_state.smooth_history = st.session_state.smooth_history[: int(st.session_state.smooth_index) + 1]
                pp_stats = opt.beautify_topology(iterations=int(smooth_strength))
                st.session_state.smooth_history.append(struct.to_dict())
                st.session_state.smooth_index = len(st.session_state.smooth_history) - 1
                st.session_state.model_lifetime_history = merge_history(
                    st.session_state.model_lifetime_history,
                    [struct.to_dict()],
                )
                if st.session_state.model_id is not None:
                    metadata = save_model_snapshot(
                        struct,
                        int(st.session_state.model_id),
                        st.session_state.model_name or "",
                        st.session_state.model_created_at or now_iso(),
                        history_timeline=st.session_state.model_lifetime_history,
                    )
                    index = [m for m in index if int(m["id"]) != int(metadata["id"])]
                    index.append(metadata)
                    save_index(index)
                st.session_state.opt_gif_bytes = None
                st.session_state.opt_gif_signature = None
                st.session_state.opt_report_bytes = None
                st.session_state.opt_report_signature = None
                st.session_state.opt_status_type = "success"
                st.session_state.opt_status_msg = (
                    f"Glättung angewendet: +{pp_stats['activated']} / -{pp_stats['deactivated']} Knoten."
                )
                st.rerun()
        with s_col2:
            can_undo = is_smooth_ready and int(st.session_state.smooth_index) > 0
            if st.button("Rückgängig", use_container_width=True, disabled=not can_undo):
                if restore_smooth_snapshot(int(st.session_state.smooth_index) - 1):
                    latest_struct = st.session_state.structure
                    st.session_state.model_lifetime_history = merge_history(
                        st.session_state.model_lifetime_history,
                        [latest_struct.to_dict()],
                    )
                    if st.session_state.model_id is not None:
                        metadata = save_model_snapshot(
                            latest_struct,
                            int(st.session_state.model_id),
                            st.session_state.model_name or "",
                            st.session_state.model_created_at or now_iso(),
                            history_timeline=st.session_state.model_lifetime_history,
                        )
                        index = [m for m in index if int(m["id"]) != int(metadata["id"])]
                        index.append(metadata)
                        save_index(index)
                    st.session_state.opt_gif_bytes = None
                    st.session_state.opt_gif_signature = None
                    st.session_state.opt_report_bytes = None
                    st.session_state.opt_report_signature = None
                    st.session_state.opt_status_type = "info"
                    st.session_state.opt_status_msg = "Glättung rückgängig gemacht."
                    st.rerun()
        with s_col3:
            can_redo = is_smooth_ready and int(st.session_state.smooth_index) < (len(st.session_state.smooth_history) - 1)
            if st.button("Wiederholen", use_container_width=True, disabled=not can_redo):
                if restore_smooth_snapshot(int(st.session_state.smooth_index) + 1):
                    latest_struct = st.session_state.structure
                    st.session_state.model_lifetime_history = merge_history(
                        st.session_state.model_lifetime_history,
                        [latest_struct.to_dict()],
                    )
                    if st.session_state.model_id is not None:
                        metadata = save_model_snapshot(
                            latest_struct,
                            int(st.session_state.model_id),
                            st.session_state.model_name or "",
                            st.session_state.model_created_at or now_iso(),
                            history_timeline=st.session_state.model_lifetime_history,
                        )
                        index = [m for m in index if int(m["id"]) != int(metadata["id"])]
                        index.append(metadata)
                        save_index(index)
                    st.session_state.opt_gif_bytes = None
                    st.session_state.opt_gif_signature = None
                    st.session_state.opt_report_bytes = None
                    st.session_state.opt_report_signature = None
                    st.session_state.opt_status_type = "info"
                    st.session_state.opt_status_msg = "Glättung wiederholt."
                    st.rerun()

with panel_vis:
    with st.container(border=True):
        st.markdown("**Visualisierung**")
        st.slider(
            "Linienstärke",
            min_value=0.1,
            max_value=2.0,
            step=0.1,
            key="line_width_slider",
        )
        st.checkbox("Verformung anzeigen", key="show_deformation")
        if st.session_state.show_deformation:
            # Entkoppelte UI-Widget-Variable, damit der intern genutzte Wert stabil bleibt.
            current_percent = int(st.session_state.deformation_display_percent)
            if current_percent <= 0:
                current_percent = 100
                st.session_state.deformation_display_percent = 100
            deformation_percent_ui = st.slider(
                "Verformungsdarstellung (%)", 0, 300, step=5,
                value=current_percent,
                key="deformation_display_percent_slider",
                help="100% = echte berechnete Verformung, 200% = doppelt dargestellt, 0% = undeformiert.",
            )
            st.session_state.deformation_display_percent = int(deformation_percent_ui)
        st.checkbox("FEM-Farbskala anzeigen", key="fem_colormap")
        if st.session_state.fem_colormap:
            st.radio(
                "Elementfokus",
                options=["Alle", "H+V", "Diagonal"],
                horizontal=True,
                key="element_focus_radio",
            )
            st.selectbox(
                "FEM-Größe",
                options=["Energie/Länge", "Dehnung", "Verschiebung"],
                key="fem_metric_mode",
            )

# Live status + optimization loop
live_active = sum(1 for n in struct.nodes if n.active)
live_percent = (live_active / total_nodes) * 100
start_active_for_progress = int(st.session_state.opt_start_active) if int(st.session_state.opt_start_active) > 0 else total_nodes
mass_fraction = live_active / max(1, start_active_for_progress)
opt_mass_placeholder.caption(f"{live_percent:.1f}%")
opt_progress_bar.progress(min(max(mass_fraction, 0.0), 1.0))

if st.session_state.opt_initialized:
    run_state = "Läuft" if st.session_state.opt_running else "Pausiert"
    current_rate = 0.01 if int(st.session_state.opt_iteration) < 5 else 0.015
    opt_status_placeholder.info(
        f"Status: {run_state} &emsp;|&emsp; "
        f"Iteration: {int(st.session_state.opt_iteration)} &emsp;|&emsp; "
        f"Dynamische Entfernungsrate: {current_rate * 100:.1f}%"
    )

if st.session_state.opt_initialized and st.session_state.opt_running:
    run_started = True
    if int(st.session_state.opt_view_index) < len(st.session_state.opt_history) - 1:
        restore_opt_snapshot(int(st.session_state.opt_view_index))
        st.session_state.opt_history = st.session_state.opt_history[: int(st.session_state.opt_view_index) + 1]

    stop_reason = ""
    while st.session_state.opt_running:
        live_active = sum(1 for n in struct.nodes if n.active)
        if live_active <= int(st.session_state.opt_target_count):
            st.session_state.opt_running = False
            stop_reason = "Ziel-Masse erreicht."
            break
        if int(st.session_state.opt_iteration) >= 500:
            st.session_state.opt_running = False
            stop_reason = "Iterationslimit erreicht"
            break

        current_rate = 0.01 if int(st.session_state.opt_iteration) < 5 else 0.015
        success, message = opt.optimize_step(
            remove_ratio=current_rate,
            max_displacement_limit=float(st.session_state.opt_abs_limit),
        )
        if not success:
            st.session_state.opt_running = False
            stop_reason = str(message)
            break

        push_opt_snapshot(struct)
        live_active = sum(1 for n in struct.nodes if n.active)
        live_percent = (live_active / total_nodes) * 100
        current_rate = 0.01 if int(st.session_state.opt_iteration) < 5 else 0.015
        opt_status_placeholder.info(
            f"Status: {run_state} &emsp;|&emsp; "
            f"Iteration: {int(st.session_state.opt_iteration)} &emsp;|&emsp; "
            f"Dynamische Entfernungsrate: {current_rate * 100:.1f}%"
        )
        start_active_for_progress = int(st.session_state.opt_start_active) if int(st.session_state.opt_start_active) > 0 else total_nodes
        mass_fraction = live_active / max(1, start_active_for_progress)
        opt_mass_placeholder.caption(f"{live_percent:.1f}%")
        opt_progress_bar.progress(min(max(mass_fraction, 0.0), 1.0))

        fig = Visualizer.plot_structure(
            struct,
            show_deformation=bool(st.session_state.show_deformation),
            scale_factor=(float(st.session_state.deformation_display_percent) / 100.0) if bool(st.session_state.show_deformation) else 0.0,
            selected_node_id=None,
            colorize_elements=bool(st.session_state.fem_colormap),
            color_percentile=color_percentile,
            show_background_nodes=not bool(st.session_state.fem_colormap),
            line_width=float(st.session_state.line_width_slider),
            color_levels=color_levels,
            fixed_color_vmax=fixed_color_vmax,
            metric_mode={
                "Energie/Länge": "energy_per_length",
                "Dehnung": "strain",
                "Verschiebung": "displacement",
                "Strain": "strain",
                "Displacement": "displacement",
            }.get(st.session_state.fem_metric_mode, "energy_per_length"),
            normalize_mode=normalize_mode,
            element_filter={"Alle": "all", "H+V": "hv", "Diagonal": "diag"}.get(st.session_state.element_focus_radio, "all"),
            show_colorbar=False,
        )
        plot_placeholder.pyplot(fig)
        plt.close(fig)
        time.sleep(0.01)

    if not st.session_state.opt_running:
        st.session_state.opt_stop_reason = stop_reason
        st.session_state.opt_finished = True
        st.session_state.smooth_history = [struct.to_dict()]
        st.session_state.smooth_index = 0
        st.session_state.model_lifetime_history = merge_history(
            st.session_state.model_lifetime_history,
            st.session_state.opt_history,
        )
        st.session_state.model_lifetime_history = merge_history(
            st.session_state.model_lifetime_history,
            [struct.to_dict()],
        )
        if st.session_state.model_id is not None:
            metadata = save_model_snapshot(
                struct,
                int(st.session_state.model_id),
                st.session_state.model_name or "",
                st.session_state.model_created_at or now_iso(),
                history_timeline=st.session_state.model_lifetime_history,
            )
            index = [m for m in index if int(m["id"]) != int(metadata["id"])]
            index.append(metadata)
            save_index(index)
        st.session_state.opt_gif_bytes = None
        st.session_state.opt_gif_signature = None
        st.session_state.opt_report_bytes = None
        st.session_state.opt_report_signature = None
        if "Limit" in stop_reason or "Zu weich" in stop_reason or "Keine weiteren" in stop_reason:
            st.session_state.opt_status_type = "success"
            st.session_state.opt_status_msg = f"Optimierung fertig: {stop_reason}"
        elif "Ziel-Masse erreicht" in stop_reason:
            st.session_state.opt_status_type = "success"
            st.session_state.opt_status_msg = f"Optimierung fertig: {stop_reason}"
        elif stop_reason:
            st.session_state.opt_status_type = "error"
            st.session_state.opt_status_msg = f"Fehler: {stop_reason}"
        st.session_state.opt_running = False
        if run_started:
            st.rerun()

status_msg = str(st.session_state.opt_status_msg).strip()
if status_msg:
    if st.session_state.opt_status_type == "success":
        opt_message_placeholder.success(status_msg)
    elif st.session_state.opt_status_type == "error":
        opt_message_placeholder.error(status_msg)
    else:
        opt_message_placeholder.info(status_msg)

final_percent = (sum(1 for n in struct.nodes if n.active) / total_nodes) * 100
opt_mass_placeholder.caption(f"{final_percent:.1f}%")
