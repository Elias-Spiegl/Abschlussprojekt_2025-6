import io
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from model import Structure
from optimizer import TopologyOptimizer
from visualizer import Visualizer

st.set_page_config(page_title="Topologieoptimierung", layout="wide")
st.title("2D Topologieoptimierung (Abschlussprojekt)")

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


def save_model_snapshot(struct: Structure, model_id: int, name: str, created_at: str) -> dict:
    metadata = {
        "id": int(model_id),
        "name": (name or "").strip(),
        "created_at": created_at,
        "updated_at": now_iso(),
        "width": int(struct.width),
        "height": int(struct.height),
    }
    payload = {"metadata": metadata, "structure": struct.to_dict()}
    model_path(model_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metadata


def load_model_snapshot(model_id: int) -> tuple[Structure, dict]:
    payload = json.loads(model_path(model_id).read_text(encoding="utf-8"))
    # Backward compatibility for old plain-structure files
    if "structure" in payload and "metadata" in payload:
        struct = Structure.from_dict(payload["structure"])
        metadata = payload["metadata"]
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
    return struct, metadata


if "structure" not in st.session_state:
    st.session_state.structure = None
if "model_id" not in st.session_state:
    st.session_state.model_id = None
if "model_name" not in st.session_state:
    st.session_state.model_name = ""
if "model_created_at" not in st.session_state:
    st.session_state.model_created_at = None
if "selected_x" not in st.session_state:
    st.session_state.selected_x = 0
if "selected_z" not in st.session_state:
    st.session_state.selected_z = 0
if "editor_node_id" not in st.session_state:
    st.session_state.editor_node_id = None
if "existing_load_id" not in st.session_state:
    st.session_state.existing_load_id = -1
if "existing_support_id" not in st.session_state:
    st.session_state.existing_support_id = -1
if "editor_focus_mode" not in st.session_state:
    st.session_state.editor_focus_mode = "Knoten"
if "force_edit_fx" not in st.session_state:
    st.session_state.force_edit_fx = 0.0
if "force_edit_fz" not in st.session_state:
    st.session_state.force_edit_fz = 0.0
if "support_edit_mode" not in st.session_state:
    st.session_state.support_edit_mode = "Frei"
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

index = load_index()

pending_load_id = st.session_state.pop("startup_load_model_id", None)
if pending_load_id is not None:
    try:
        loaded_struct, loaded_meta = load_model_snapshot(int(pending_load_id))
        st.session_state.structure = loaded_struct
        st.session_state.model_id = int(loaded_meta["id"])
        st.session_state.model_name = loaded_meta.get("name", "")
        st.session_state.model_created_at = loaded_meta.get("created_at", now_iso())
        st.session_state.selected_x = 0
        st.session_state.selected_z = 0
        st.session_state.editor_node_id = None
        reset_optimizer_state()
        reset_visualization_state()
    except Exception:
        # Startseite kann eine ungültige ID übergeben haben (gelöscht/defekt).
        pass

if st.session_state.structure is None:
    st.subheader("Kein Modell geladen")
    st.info("Öffne ein Modell im Model Hub oder erstelle zuerst ein neues Modell.")
    c_nav1, c_nav2 = st.columns(2)
    with c_nav1:
        if st.button("Zum Model Hub", use_container_width=True):
            st.switch_page("main.py")
    with c_nav2:
        if st.button("Zum Model Create", use_container_width=True):
            st.switch_page("pages/1_Model_Create.py")
    st.stop()

struct = st.session_state.structure
opt = TopologyOptimizer(struct)

# Current model info + save
st.sidebar.header("Editor")
with st.sidebar.expander("Aktuelles Modell", expanded=True):
    st.caption(
        f"Erstellt: {format_ts(st.session_state.model_created_at)}"
    )
    st.session_state.model_name = st.text_input(
        "Name", value=st.session_state.model_name, key="current_model_name"
    )
    if st.button("Aktuellen Stand speichern", use_container_width=True):
        if st.session_state.model_id is None:
            st.error("Kein Modellkontext vorhanden.")
        else:
            requested_name = (st.session_state.model_name or "").strip()
            if not requested_name:
                st.error("Bitte einen Modellnamen eingeben.")
            else:
                final_name = make_unique_model_name(
                    index,
                    requested_name,
                    current_model_id=int(st.session_state.model_id),
                )
                metadata = save_model_snapshot(
                    struct,
                    int(st.session_state.model_id),
                    final_name,
                    st.session_state.model_created_at or now_iso(),
                )
                index = [m for m in index if int(m["id"]) != int(metadata["id"])]
                index.append(metadata)
                save_index(index)
                st.session_state.model_name = final_name
                if final_name != requested_name:
                    st.info(f"Name bereits vergeben, gespeichert als: {final_name}")
                st.success("Modell gespeichert.")
                st.rerun()

# Sidebar: Knoten/Kräfte/Lager
with st.sidebar.expander("Knoten, Kräfte, Lager", expanded=True):
    load_nodes = [n for n in struct.nodes if n.force_x != 0 or n.force_z != 0]
    support_nodes = [n for n in struct.nodes if n.fixed_x or n.fixed_z]
    st.caption(f"Kräfte: {len(load_nodes)} | Lager: {len(support_nodes)}")

    def _jump_to_node(node_id: int) -> None:
        st.session_state.selected_x = int(node_id % struct.width)
        st.session_state.selected_z = int(node_id // struct.width)
        st.session_state.editor_node_id = None

    def _on_load_select_change() -> None:
        node_id = int(st.session_state.existing_load_id)
        if node_id >= 0:
            _jump_to_node(node_id)
            st.session_state.existing_support_id = -1

    def _on_support_select_change() -> None:
        node_id = int(st.session_state.existing_support_id)
        if node_id >= 0:
            _jump_to_node(node_id)
            st.session_state.existing_load_id = -1

    st.session_state.selected_x = min(max(int(st.session_state.selected_x), 0), struct.width - 1)
    st.session_state.selected_z = min(max(int(st.session_state.selected_z), 0), struct.height - 1)
    mode_cols = st.columns(3)
    with mode_cols[0]:
        if st.button("Knoten", use_container_width=True):
            st.session_state.editor_focus_mode = "Knoten"
    with mode_cols[1]:
        if st.button("Kräfte", use_container_width=True):
            st.session_state.editor_focus_mode = "Kräfte"
    with mode_cols[2]:
        if st.button("Lager", use_container_width=True):
            st.session_state.editor_focus_mode = "Lager"

    mode = st.session_state.editor_focus_mode
    st.caption(f"Aktiver Modus: {mode}")

    # Gemeinsamer Knotenindex (für Plot-Highlight und Moduswechsel).
    selected_id = int(st.session_state.selected_z) * struct.width + int(st.session_state.selected_x)
    selected_node = struct.nodes[selected_id]

    if mode == "Knoten":
        c1, c2 = st.columns(2)
        with c1:
            selected_x = st.number_input("X", min_value=0, max_value=struct.width - 1, step=1, key="selected_x")
        with c2:
            selected_z = st.number_input("Z", min_value=0, max_value=struct.height - 1, step=1, key="selected_z")
        selected_id = int(selected_z) * struct.width + int(selected_x)
        selected_node = struct.nodes[selected_id]

        if selected_node.fixed_x and selected_node.fixed_z:
            support_txt = "Festlager"
        elif (not selected_node.fixed_x) and selected_node.fixed_z:
            support_txt = "Loslager"
        else:
            support_txt = "Frei"

        st.caption(f"Knoten: x={int(selected_x)}, z={int(selected_z)}")
        st.caption(
            f"Kraft: Fx={float(selected_node.force_x):.2f}, Fz={float(selected_node.force_z):.2f} | "
            f"Lager: {support_txt}"
        )
        node_active = st.checkbox("Knoten aktiv", value=bool(selected_node.active))
        if bool(selected_node.active) != bool(node_active):
            selected_node.active = bool(node_active)
            reset_optimizer_state()

    elif mode == "Kräfte":
        if load_nodes:
            sorted_load_nodes = sorted(load_nodes, key=lambda n: n.id)
            load_label_by_id = {
                int(n.id): f"x={int(n.x)}, z={int(n.z)} | Fx={n.force_x:.2f}, Fz={n.force_z:.2f}"
                for n in sorted_load_nodes
            }
            load_option_ids = [-1] + [int(n.id) for n in sorted_load_nodes]
            if int(st.session_state.existing_load_id) not in load_option_ids:
                st.session_state.existing_load_id = -1
            st.selectbox(
                "Vorhandene Kräfte",
                load_option_ids,
                key="existing_load_id",
                format_func=lambda nid: "Keine Auswahl" if nid < 0 else load_label_by_id.get(nid, "Kraft"),
                on_change=_on_load_select_change,
            )
            if int(st.session_state.existing_load_id) >= 0:
                did = int(st.session_state.existing_load_id)
                dx = int(did % struct.width)
                dz = int(did // struct.width)
                if st.button(f"Kraft (x={dx}, z={dz}) löschen", use_container_width=True):
                    target_node = struct.nodes[did]
                    target_node.force_x = 0.0
                    target_node.force_z = 0.0
                    st.session_state.editor_node_id = None
                    st.session_state.force_edit_fx = 0.0
                    st.session_state.force_edit_fz = 0.0
                    reset_optimizer_state()
                    st.success(f"Kraft bei x={dx}, z={dz} gelöscht.")
                    st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            selected_x = st.number_input("X", min_value=0, max_value=struct.width - 1, step=1, key="selected_x")
        with c2:
            selected_z = st.number_input("Z", min_value=0, max_value=struct.height - 1, step=1, key="selected_z")
        selected_id = int(selected_z) * struct.width + int(selected_x)
        selected_node = struct.nodes[selected_id]
        if st.session_state.editor_node_id != selected_id:
            st.session_state.editor_node_id = selected_id
            st.session_state.force_edit_fx = float(selected_node.force_x)
            st.session_state.force_edit_fz = float(selected_node.force_z)

        f1, f2 = st.columns(2)
        with f1:
            new_fx = st.number_input("Kraft X", key="force_edit_fx", step=1.0, format="%.2f")
        with f2:
            new_fz = st.number_input("Kraft Z", key="force_edit_fz", step=1.0, format="%.2f")
        changed_force = (
            float(selected_node.force_x) != float(new_fx)
            or float(selected_node.force_z) != float(new_fz)
        )
        if changed_force:
            selected_node.force_x = float(new_fx)
            selected_node.force_z = float(new_fz)
            reset_optimizer_state()

    else:  # Lager
        if support_nodes:
            support_label_by_id = {}
            sorted_support_nodes = sorted(support_nodes, key=lambda n: n.id)
            for n in sorted_support_nodes:
                if n.fixed_x and n.fixed_z:
                    s_txt = "Festlager"
                elif (not n.fixed_x) and n.fixed_z:
                    s_txt = "Loslager"
                else:
                    s_txt = "Sonderfall"
                support_label_by_id[int(n.id)] = f"x={int(n.x)}, z={int(n.z)} | {s_txt}"

            support_option_ids = [-1] + [int(n.id) for n in sorted_support_nodes]
            if int(st.session_state.existing_support_id) not in support_option_ids:
                st.session_state.existing_support_id = -1
            st.selectbox(
                "Vorhandene Lager",
                support_option_ids,
                key="existing_support_id",
                format_func=lambda nid: "Keine Auswahl" if nid < 0 else support_label_by_id.get(nid, "Lager"),
                on_change=_on_support_select_change,
            )
            if int(st.session_state.existing_support_id) >= 0:
                did = int(st.session_state.existing_support_id)
                dx = int(did % struct.width)
                dz = int(did // struct.width)
                if st.button(f"Lager (x={dx}, z={dz}) löschen", use_container_width=True):
                    target_node = struct.nodes[did]
                    target_node.fixed_x = False
                    target_node.fixed_z = False
                    st.session_state.editor_node_id = None
                    st.session_state.support_edit_mode = "Frei"
                    reset_optimizer_state()
                    st.success(f"Lager bei x={dx}, z={dz} gelöscht.")
                    st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            selected_x = st.number_input("X", min_value=0, max_value=struct.width - 1, step=1, key="selected_x")
        with c2:
            selected_z = st.number_input("Z", min_value=0, max_value=struct.height - 1, step=1, key="selected_z")
        selected_id = int(selected_z) * struct.width + int(selected_x)
        selected_node = struct.nodes[selected_id]
        if st.session_state.editor_node_id != selected_id:
            st.session_state.editor_node_id = selected_id
            if selected_node.fixed_x and selected_node.fixed_z:
                st.session_state.support_edit_mode = "Festlager"
            elif (not selected_node.fixed_x) and selected_node.fixed_z:
                st.session_state.support_edit_mode = "Loslager"
            else:
                st.session_state.support_edit_mode = "Frei"

        support_mode = st.selectbox(
            "Lagerzustand",
            ["Frei", "Loslager", "Festlager"],
            key="support_edit_mode",
        )
        if support_mode == "Frei":
            new_fixed_x = False
            new_fixed_z = False
        elif support_mode == "Loslager":
            new_fixed_x = False
            new_fixed_z = True
        else:
            new_fixed_x = True
            new_fixed_z = True
        changed_support = (
            bool(selected_node.fixed_x) != bool(new_fixed_x)
            or bool(selected_node.fixed_z) != bool(new_fixed_z)
        )
        if changed_support:
            selected_node.fixed_x = bool(new_fixed_x)
            selected_node.fixed_z = bool(new_fixed_z)
            reset_optimizer_state()

# Main area
model_rows = to_local_table_rows(index)
st.subheader("Gespeicherte Modelle")
if model_rows:
    st.dataframe(model_rows, use_container_width=True, hide_index=True)

# Force/support overview
force_rows = [
    {
        "X": int(n.x),
        "Z": int(n.z),
        "Fx": round(float(n.force_x), 4),
        "Fz": round(float(n.force_z), 4),
    }
    for n in struct.nodes
    if n.force_x != 0 or n.force_z != 0
]
support_rows = [
    {
        "X": int(n.x),
        "Z": int(n.z),
        "Typ": "Festlager" if (n.fixed_x and n.fixed_z) else "Loslager" if ((not n.fixed_x) and n.fixed_z) else "Sonderfall",
    }
    for n in struct.nodes
    if n.fixed_x or n.fixed_z
]

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.markdown("**Aktive Kräfte**")
    if force_rows:
        st.dataframe(force_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("Keine Kräfte gesetzt")
with col_info2:
    st.markdown("**Aktive Lager**")
    if support_rows:
        st.dataframe(support_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("Keine Lager gesetzt")

col1, col2 = st.columns([1, 2])
current_active = sum(1 for n in struct.nodes if n.active)
total_nodes = len(struct.nodes)

with col2:
    st.subheader("Visualisierung")

    # Aktualisiert Verschiebungen für eine konsistente Anzeige.
    visualization_physics_ok = opt.solve_step()
    if not visualization_physics_ok:
        st.warning("Visualisierung: Modell aktuell instabil (Lager/Kräfte prüfen).")

    show_deformation = st.checkbox("Verformung anzeigen", key="show_deformation")
    if show_deformation:
        deformation_display_percent = st.slider(
            "Verformungsdarstellung (%)", 0, 1000, 100, step=5,
            help="100% = echte berechnete Verformung, 200% = doppelt dargestellt, 0% = undeformiert."
        )
        scale = deformation_display_percent / 100.0
        st.caption(
            f"Anzeige-Multiplikator: x {scale:.2f} (100% = echt)"
        )
    else:
        scale = 0.0

    fem_color_map = st.checkbox("FEM-Farbskala anzeigen", key="fem_colormap")
    line_width = 0.8
    color_percentile = 95
    color_levels = 15
    fixed_color_vmax = None
    normalize_mode = "orientation"
    element_filter = "all"
    metric_mode = "energy_per_length"
    if fem_color_map:
        c_vis1, c_vis2, c_vis3 = st.columns([1, 1, 1.2])
        with c_vis1:
            element_focus = st.radio(
                "Elementfokus",
                options=["Alle", "H+V", "Diagonal"],
                index=0,
                horizontal=True,
                key="element_focus_radio",
            )
        with c_vis2:
            line_width = st.slider(
                "Linienstärke",
                min_value=0.4,
                max_value=2.0,
                step=0.1,
                key="line_width_slider",
            )
        with c_vis3:
            st.selectbox(
                "FEM-Größe",
                options=["Energie/Länge", "Strain", "Displacement"],
                key="fem_metric_mode",
            )
        element_filter = {"Alle": "all", "H+V": "hv", "Diagonal": "diag"}[element_focus]
        metric_mode = {
            "Energie/Länge": "energy_per_length",
            "Strain": "strain",
            "Displacement": "displacement",
        }[st.session_state.fem_metric_mode]
        st.caption("Backend: pro Richtung normiert, 15 Farbstufen.")
        show_background_nodes = False
    else:
        show_background_nodes = True

    plot_placeholder = st.empty()

    fig = Visualizer.plot_structure(
        struct,
        show_deformation=show_deformation,
        scale_factor=scale,
        selected_node_id=selected_id,
        colorize_elements=fem_color_map,
        color_percentile=color_percentile,
        show_background_nodes=show_background_nodes,
        line_width=line_width,
        color_levels=color_levels,
        fixed_color_vmax=fixed_color_vmax,
        metric_mode=metric_mode,
        normalize_mode=normalize_mode,
        element_filter=element_filter,
    )
    plot_placeholder.pyplot(fig)

    png_buffer = io.BytesIO()
    fig.savefig(png_buffer, format="png", dpi=200, bbox_inches="tight")
    png_buffer.seek(0)
    st.download_button(
        "Optimierte Geometrie als PNG herunterladen",
        data=png_buffer.getvalue(),
        file_name=f"modell_{st.session_state.model_id}_topologie.png",
        mime="image/png",
    )
    plt.close(fig)

with col1:
    st.subheader("Automatik-Optimierung")
    mass_info_placeholder = st.empty()
    current_percent = (current_active / total_nodes) * 100
    mass_info_placeholder.info(f"Aktuelle Masse: {current_percent:.1f}%")

    target_mass_percent = st.slider("Ziel-Masse (%)", 10, 99, 50)
    st.caption("Entfernungsrate: Dynamisch (1% -> 1.5%)")

    max_stiffness_loss_percent = st.slider(
        "Max. Steifigkeitsverlust (%)", 0, 100, 50, step=1
    )
    allowed_softening_ratio = 1.0 + (max_stiffness_loss_percent / 100.0)
    st.caption(
        f"Zulässig bis +{max_stiffness_loss_percent}% Verformung gegenüber der Referenz "
        f"(entspricht Faktor {allowed_softening_ratio:.2f})."
    )

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
                st.session_state.opt_target_count = int(total_nodes * (target_mass_percent / 100))
                st.session_state.opt_start_active = sum(1 for n in struct.nodes if n.active)
                st.session_state.opt_abs_limit = max_displacement(struct) * allowed_softening_ratio
                st.session_state.opt_status_type = "info"
                st.session_state.opt_status_msg = "Optimierung läuft..."
                st.session_state.opt_stop_reason = ""
                push_opt_snapshot(struct)
                st.rerun()
    else:
        total_steps = max(1, len(st.session_state.opt_history))
        current_step = min(max(int(st.session_state.opt_view_index) + 1, 1), total_steps)
        if not st.session_state.opt_running:
            st.caption(f"Iteration: {current_step}/{total_steps}")

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
                        # Aktuelle UI-Parameter beim erneuten Start immer neu übernehmen.
                        st.session_state.opt_target_count = int(total_nodes * (target_mass_percent / 100))
                        st.session_state.opt_abs_limit = max_displacement(struct) * allowed_softening_ratio
                        st.session_state.opt_running = True
                        st.session_state.opt_finished = False
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

    status_container = st.empty()
    progress_bar = st.progress(0.0)
    live_active = sum(1 for n in struct.nodes if n.active)
    live_percent = (live_active / total_nodes) * 100
    mass_info_placeholder.info(f"Aktuelle Masse: {live_percent:.1f}%")
    nodes_to_remove_total = max(int(st.session_state.opt_start_active) - int(st.session_state.opt_target_count), 0)
    removed_so_far = max(int(st.session_state.opt_start_active) - live_active, 0)
    if nodes_to_remove_total > 0:
        progress_bar.progress(min(max(removed_so_far / nodes_to_remove_total, 0.0), 1.0))

    if st.session_state.opt_initialized:
        run_state = "Läuft" if st.session_state.opt_running else "Pausiert"
        current_rate = 0.01 if int(st.session_state.opt_iteration) < 5 else 0.015
        status_container.info(
            f"Status: {run_state} | Iter: {int(st.session_state.opt_iteration)} | "
            f"Remove-Rate: {current_rate * 100:.1f}%"
        )

    # Kontinuierliche Optimierung ohne Zwischen-Reruns:
    # während des Loops nur Plot/Status updaten, UI-Widgets erst wieder nach Interaktion.
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
            # Live-Update pro Iteration (ohne erst auf den nächsten Rerun zu warten)
            live_active = sum(1 for n in struct.nodes if n.active)
            live_percent = (live_active / total_nodes) * 100
            mass_info_placeholder.info(f"Aktuelle Masse: {live_percent:.1f}%")
            current_rate = 0.01 if int(st.session_state.opt_iteration) < 5 else 0.015
            status_container.info(
                f"Status: Läuft | Iter: {int(st.session_state.opt_iteration)} | "
                f"Remove-Rate: {current_rate * 100:.1f}%"
            )
            nodes_to_remove_total = max(int(st.session_state.opt_start_active) - int(st.session_state.opt_target_count), 0)
            removed_so_far = max(int(st.session_state.opt_start_active) - live_active, 0)
            if nodes_to_remove_total > 0:
                progress_bar.progress(min(max(removed_so_far / nodes_to_remove_total, 0.0), 1.0))

            fig = Visualizer.plot_structure(
                struct,
                show_deformation=show_deformation,
                scale_factor=scale,
                selected_node_id=selected_id,
                colorize_elements=fem_color_map,
                color_percentile=color_percentile,
                show_background_nodes=show_background_nodes,
                line_width=line_width,
                color_levels=color_levels,
                fixed_color_vmax=fixed_color_vmax,
                metric_mode=metric_mode,
                normalize_mode=normalize_mode,
                element_filter=element_filter,
            )
            plot_placeholder.pyplot(fig)
            plt.close(fig)
            time.sleep(0.01)

        if not st.session_state.opt_running:
            st.session_state.opt_stop_reason = stop_reason
            st.session_state.opt_finished = True
            st.session_state.smooth_history = [struct.to_dict()]
            st.session_state.smooth_index = 0
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

    if st.session_state.opt_initialized and st.session_state.opt_finished and (not st.session_state.opt_running):
        st.markdown("**Smoothening Structure**")
        smooth_strength = st.select_slider(
            "Strength",
            options=[1, 2, 3],
            value=2,
            key="smooth_strength_selector",
        )

        s_col1, s_col2, s_col3 = st.columns([1.4, 1, 1])
        with s_col1:
            if st.button("Apply Smoothening", use_container_width=True):
                if int(st.session_state.smooth_index) < len(st.session_state.smooth_history) - 1:
                    st.session_state.smooth_history = st.session_state.smooth_history[: int(st.session_state.smooth_index) + 1]
                pp_stats = opt.beautify_topology(iterations=int(smooth_strength))
                st.session_state.smooth_history.append(struct.to_dict())
                st.session_state.smooth_index = len(st.session_state.smooth_history) - 1
                st.session_state.opt_status_type = "success"
                st.session_state.opt_status_msg = (
                    f"Smoothening applied: +{pp_stats['activated']} / -{pp_stats['deactivated']} nodes."
                )
                st.rerun()
        with s_col2:
            can_undo = int(st.session_state.smooth_index) > 0
            if st.button("Undo", use_container_width=True, disabled=not can_undo):
                if restore_smooth_snapshot(int(st.session_state.smooth_index) - 1):
                    st.session_state.opt_status_type = "info"
                    st.session_state.opt_status_msg = "Smoothening undo."
                    st.rerun()
        with s_col3:
            can_redo = int(st.session_state.smooth_index) < (len(st.session_state.smooth_history) - 1)
            if st.button("Redo", use_container_width=True, disabled=not can_redo):
                if restore_smooth_snapshot(int(st.session_state.smooth_index) + 1):
                    st.session_state.opt_status_type = "info"
                    st.session_state.opt_status_msg = "Smoothening redo."
                    st.rerun()

    if st.button("Meldungen zurücksetzen"):
        st.session_state.opt_status_msg = ""
        st.session_state.opt_status_type = "info"
        st.rerun()

    status_msg = str(st.session_state.opt_status_msg).strip()
    if status_msg:
        if st.session_state.opt_status_type == "success":
            st.success(status_msg)
        elif st.session_state.opt_status_type == "error":
            st.error(status_msg)
        else:
            st.info(status_msg)

final_percent = (sum(1 for n in struct.nodes if n.active) / total_nodes) * 100
st.metric("Status", f"{sum(1 for n in struct.nodes if n.active)} / {total_nodes} Knoten aktiv ({final_percent:.1f}%)")
