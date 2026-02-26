import json
from datetime import datetime
from pathlib import Path

import altair as alt
import streamlit as st

from model import Structure

st.set_page_config(page_title="Modell Erstellen / Bearbeiten", layout="wide")

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
    return STATE_DIR / f"model_{int(model_id)}.json"


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


def _build_default_structure(width: int, height: int) -> Structure:
    struct = Structure(width, height)
    # Standard-Setup: links unten Loslager, rechts unten Festlager, mittig oben Last nach unten.
    node_left = struct.nodes[(height - 1) * width]
    node_left.fixed_x = False
    node_left.fixed_z = True

    node_right = struct.nodes[height * width - 1]
    node_right.fixed_x = True
    node_right.fixed_z = True

    mid_x = int(width / 2)
    mid_node = struct.nodes[mid_x]
    mid_node.force_x = 0.0
    mid_node.force_z = 1.0
    return struct


def _node_status(node) -> str:
    if not node.active:
        return "Inaktiv"
    if node.fixed_x and node.fixed_z:
        return "Festlager"
    if (not node.fixed_x) and node.fixed_z:
        return "Loslager"
    if node.force_x != 0 or node.force_z != 0:
        return "Kraft"
    return "Aktiv"


def _selection_node_ids(event, fallback_ids: list[int]) -> list[int]:
    # Robust gegen unterschiedliche Event-Formate in Streamlit-Versionen.
    sel_ids: list[int] = []

    def _to_int(value):
        try:
            return int(value)
        except Exception:
            return None

    def _collect_from_bucket(bucket):
        values: list[int] = []
        if isinstance(bucket, dict):
            for key in ("node_id", "point_indices", "indices"):
                if key in bucket:
                    raw_v = bucket.get(key)
                    if isinstance(raw_v, list):
                        values.extend([v for v in (_to_int(x) for x in raw_v) if v is not None])
                    else:
                        v = _to_int(raw_v)
                        if v is not None:
                            values.append(v)
            for key in ("values", "value", "points", "vlPoint"):
                if key in bucket:
                    values.extend(_collect_from_bucket(bucket.get(key)))
        elif isinstance(bucket, list):
            for item in bucket:
                if isinstance(item, dict):
                    if "node_id" in item:
                        v = _to_int(item.get("node_id"))
                        if v is not None:
                            values.append(v)
                    elif "point_index" in item:
                        v = _to_int(item.get("point_index"))
                        if v is not None:
                            values.append(v)
                    else:
                        values.extend(_collect_from_bucket(item))
                else:
                    v = _to_int(item)
                    if v is not None:
                        values.append(v)
        return values

    if event is None:
        return fallback_ids

    raw = event
    if hasattr(event, "to_dict"):
        try:
            raw = event.to_dict()
        except Exception:
            raw = event

    # Dict-basierte Form
    if isinstance(raw, dict):
        selection = raw.get("selection", {})
        if isinstance(selection, dict):
            knoten = selection.get("knoten")
            sel_ids = _collect_from_bucket(knoten)

    # Objekt-basierte Form
    if not sel_ids and hasattr(event, "selection"):
        s = getattr(event, "selection")
        if hasattr(s, "knoten"):
            k = getattr(s, "knoten")
            sel_ids = _collect_from_bucket(k)

    if sel_ids:
        return sorted(set(sel_ids))
    # Leere/uneindeutige Selection-Events nicht als "Auswahl löschen" interpretieren.
    # Auswahl wird explizit über "Auswahl aufheben" gelöscht.
    return fallback_ids


def _set_support(node, support_mode: str) -> None:
    if support_mode == "Loslager":
        node.fixed_x = False
        node.fixed_z = True
    elif support_mode == "Festlager":
        node.fixed_x = True
        node.fixed_z = True
    else:
        node.fixed_x = False
        node.fixed_z = False


def _history_reset(struct: Structure) -> None:
    st.session_state.create_history = [struct.to_dict()]
    st.session_state.create_history_index = 0


def _history_push(struct: Structure) -> None:
    hist = list(st.session_state.create_history)
    idx = int(st.session_state.create_history_index)
    if idx < len(hist) - 1:
        hist = hist[: idx + 1]
    hist.append(struct.to_dict())
    st.session_state.create_history = hist
    st.session_state.create_history_index = len(hist) - 1


def _history_restore(index: int) -> bool:
    hist = st.session_state.create_history
    if index < 0 or index >= len(hist):
        return False
    st.session_state.create_draft_struct = Structure.from_dict(hist[index])
    st.session_state.create_history_index = index
    return True


if "create_draft_struct" not in st.session_state:
    st.session_state.create_draft_struct = None
if "create_selected_ids" not in st.session_state:
    st.session_state.create_selected_ids = []
if "create_last_dims" not in st.session_state:
    st.session_state.create_last_dims = (80, 20)
if "create_width_input" not in st.session_state:
    st.session_state.create_width_input = int(st.session_state.create_last_dims[0])
if "create_height_input" not in st.session_state:
    st.session_state.create_height_input = int(st.session_state.create_last_dims[1])
if "create_action" not in st.session_state:
    st.session_state.create_action = "Knoten aktivieren"
if "create_force_x" not in st.session_state:
    st.session_state.create_force_x = 0.0
if "create_force_z" not in st.session_state:
    st.session_state.create_force_z = 1.0
if "create_support_mode" not in st.session_state:
    st.session_state.create_support_mode = "Loslager"
if "create_model_name" not in st.session_state:
    st.session_state.create_model_name = ""
if "create_editor_action" not in st.session_state:
    st.session_state.create_editor_action = ""
if "create_prev_selection_sig" not in st.session_state:
    st.session_state.create_prev_selection_sig = ""
if "create_chart_key_version" not in st.session_state:
    st.session_state.create_chart_key_version = 0
if "create_history" not in st.session_state:
    st.session_state.create_history = []
if "create_history_index" not in st.session_state:
    st.session_state.create_history_index = -1
if "create_edit_model_id" not in st.session_state:
    st.session_state.create_edit_model_id = None
if "create_edit_model_created_at" not in st.session_state:
    st.session_state.create_edit_model_created_at = None
if "create_mode" not in st.session_state:
    st.session_state.create_mode = None

nav1, nav2, nav3 = st.columns([1, 1, 1])
with nav1:
    if st.button("Modell Übersicht", use_container_width=True, type="primary"):
        st.switch_page("main.py")
with nav2:
    st.button("Modell Erstellen / Bearbeiten", use_container_width=True, disabled=True, type="primary")
with nav3:
    if st.button("Modell Optimieren", use_container_width=True, type="primary"):
        st.switch_page("pages/2_Modell_Optimierung.py")
st.title("Modell Erstellen / Bearbeiten")
st.caption("Wähle zuerst einen Modus.")

index = load_index()
sorted_models = sorted(index, key=lambda x: int(x.get("id", 0)))

if st.session_state.create_mode is None:
    m1, m2 = st.columns(2)
    with m1:
        if st.button("Neues Modell erstellen", use_container_width=True):
            st.session_state.create_mode = "new"
            st.session_state.create_edit_model_id = None
            st.session_state.create_edit_model_created_at = None
            st.session_state.create_model_name = ""
            st.session_state.create_last_dims = (80, 20)
            st.session_state.create_width_input = 80
            st.session_state.create_height_input = 20
            st.session_state.create_draft_struct = _build_default_structure(80, 20)
            st.session_state.create_selected_ids = []
            st.session_state.create_editor_action = ""
            st.session_state.create_prev_selection_sig = ""
            st.session_state.create_chart_key_version += 1
            _history_reset(st.session_state.create_draft_struct)
            st.rerun()
    with m2:
        if st.button("Modell bearbeiten", use_container_width=True):
            st.session_state.create_mode = "edit"
            st.session_state.create_edit_model_id = None
            st.session_state.create_edit_model_created_at = None
            st.session_state.create_selected_ids = []
            st.session_state.create_editor_action = ""
            st.session_state.create_prev_selection_sig = ""
            st.session_state.create_chart_key_version += 1
            st.rerun()
    st.stop()

back_cols = st.columns([1.2, 3.8])
with back_cols[0]:
    if st.button("Zur Modusauswahl", use_container_width=True):
        st.session_state.create_mode = None
        st.session_state.create_selected_ids = []
        st.session_state.create_editor_action = ""
        st.session_state.create_prev_selection_sig = ""
        st.session_state.create_chart_key_version += 1
        st.rerun()

if st.session_state.create_mode == "edit":
    with st.container(border=True):
        load_col1, load_col2 = st.columns([3, 1])
        with load_col1:
            load_options = {
                f"#{int(m.get('id', 0))} - {(m.get('name') or '(ohne Namen)')}": int(m.get("id", 0))
                for m in sorted_models
            }
            selected_load_label = st.selectbox(
                "Vorhandenes Modell laden",
                options=list(load_options.keys()) if load_options else ["(keine Modelle vorhanden)"],
                disabled=not bool(load_options),
                key="create_load_model_label",
            )
        with load_col2:
            st.markdown("<div style='height: 2.0rem;'></div>", unsafe_allow_html=True)
            if st.button("Modell laden", use_container_width=True, disabled=not bool(load_options)):
                try:
                    load_id = int(load_options[selected_load_label])
                    loaded_struct, loaded_meta = load_model_snapshot(load_id)
                    st.session_state.create_draft_struct = loaded_struct
                    st.session_state.create_last_dims = (int(loaded_struct.width), int(loaded_struct.height))
                    st.session_state.create_width_input = int(loaded_struct.width)
                    st.session_state.create_height_input = int(loaded_struct.height)
                    st.session_state.create_model_name = loaded_meta.get("name", "") or ""
                    st.session_state.create_edit_model_id = int(loaded_meta.get("id", load_id))
                    st.session_state.create_edit_model_created_at = loaded_meta.get("created_at", now_iso())
                    st.session_state.create_selected_ids = []
                    st.session_state.create_editor_action = ""
                    st.session_state.create_prev_selection_sig = ""
                    st.session_state.create_chart_key_version += 1
                    _history_reset(st.session_state.create_draft_struct)
                    st.success("Modell geladen. Änderungen können jetzt gespeichert werden.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Laden fehlgeschlagen: {e}")

if st.session_state.create_mode == "edit" and st.session_state.create_edit_model_id is None:
    st.info("Bitte zuerst ein Modell laden.")
    st.stop()

c_name, c_create = st.columns([2, 1])
with c_name:
    st.text_input("Modellname", key="create_model_name")
with c_create:
    st.markdown("<div style='height: 2.1rem;'></div>", unsafe_allow_html=True)
    create_model_clicked = st.button(
        "Modell speichern" if st.session_state.create_mode == "edit" else "Modell erstellen",
        use_container_width=True,
    )

c_top1, c_top2 = st.columns([1, 1])
with c_top1:
    st.slider("Breite (Knoten)", 5, 200, key="create_width_input")
with c_top2:
    st.slider("Höhe (Knoten)", 5, 70, key="create_height_input")

if st.button("Größe anwenden", use_container_width=True):
    new_width = int(st.session_state.create_width_input)
    new_height = int(st.session_state.create_height_input)
    if (new_width, new_height) != tuple(st.session_state.create_last_dims):
        st.session_state.create_draft_struct = _build_default_structure(new_width, new_height)
        st.session_state.create_last_dims = (new_width, new_height)
        st.session_state.create_selected_ids = []
        st.session_state.create_editor_action = ""
        st.session_state.create_chart_key_version += 1
        _history_reset(st.session_state.create_draft_struct)
        st.success(f"Modellgröße angewendet: {new_width} x {new_height}")
        st.rerun()

width = int(st.session_state.create_last_dims[0])
height = int(st.session_state.create_last_dims[1])

if st.session_state.create_draft_struct is None:
    st.session_state.create_draft_struct = _build_default_structure(width, height)
    st.session_state.create_selected_ids = []
    _history_reset(st.session_state.create_draft_struct)

if not st.session_state.create_history:
    _history_reset(st.session_state.create_draft_struct)

draft: Structure = st.session_state.create_draft_struct

left, right = st.columns([1.15, 2.15])

with right:
    plot_data = []
    for n in draft.nodes:
        plot_data.append(
            {
                "node_id": int(n.id),
                "x": int(n.x),
                "z": int(n.z),
                "status": _node_status(n),
                "fx": float(n.force_x),
                "fz": float(n.force_z),
            }
        )

    color_scale = alt.Scale(
        domain=["Inaktiv", "Aktiv", "Kraft", "Loslager", "Festlager"],
        range=["#F0F0F8", "#0033FF", "#00A63E", "#FF8A00", "#FF0033"],
    )

    node_select = alt.selection_point(
        name="knoten",
        fields=["node_id"],
        toggle="true",
        clear=False,
        on="click",
    )

    chart_width = 900
    chart_height = max(220, min(700, int(chart_width * (height / max(width, 1)))))

    stroke_cond = alt.condition(
        node_select,
        alt.value("#000000"),
        alt.value("rgba(0,0,0,0)"),
        empty=False,
    )

    chart = (
        alt.Chart(alt.Data(values=plot_data))
        .mark_circle(size=78, strokeWidth=3.0)
        .encode(
            x=alt.X("x:Q", title="X", scale=alt.Scale(domain=[0, width - 1], nice=False, padding=0)),
            y=alt.Y("z:Q", title="Z", scale=alt.Scale(domain=[0, height - 1], reverse=True, nice=False, padding=0)),
            color=alt.Color("status:N", scale=color_scale, title="Status"),
            stroke=stroke_cond,
            tooltip=[
                alt.Tooltip("node_id:Q", title="Knoten-ID"),
                alt.Tooltip("x:Q", title="X"),
                alt.Tooltip("z:Q", title="Z"),
                alt.Tooltip("status:N", title="Status"),
                alt.Tooltip("fx:Q", title="Fx", format=".2f"),
                alt.Tooltip("fz:Q", title="Fz", format=".2f"),
            ],
        )
        .properties(width=chart_width, height=chart_height)
        .add_params(node_select)
    )

    event = st.altair_chart(
        chart,
        use_container_width=False,
        key=f"create_node_chart_{st.session_state.create_chart_key_version}",
        on_select="rerun",
        selection_mode=["knoten"],
    )

    st.session_state.create_selected_ids = _selection_node_ids(event, st.session_state.create_selected_ids)

    force_count = sum(1 for n in draft.nodes if n.force_x != 0 or n.force_z != 0)
    loslager_count = sum(1 for n in draft.nodes if (not n.fixed_x) and n.fixed_z)
    festlager_count = sum(1 for n in draft.nodes if n.fixed_x and n.fixed_z)
    aktiv_count = sum(1 for n in draft.nodes if n.active)

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Aktive Knoten", aktiv_count)
    i2.metric("Kräfte", force_count)
    i3.metric("Loslager", loslager_count)
    i4.metric("Festlager", festlager_count)

with left:
    selected_ids = st.session_state.create_selected_ids
    selected_nodes = [draft.nodes[int(nid)] for nid in selected_ids]
    selected_sig = ",".join(str(i) for i in selected_ids)
    if selected_sig != st.session_state.create_prev_selection_sig:
        st.session_state.create_editor_action = ""
        st.session_state.create_prev_selection_sig = selected_sig

    st.caption(f"Ausgewählte Knoten: {len(selected_ids)}")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Auswahl aufheben", use_container_width=True, disabled=not selected_ids):
            st.session_state.create_selected_ids = []
            st.session_state.create_editor_action = ""
            st.session_state.create_prev_selection_sig = ""
            st.session_state.create_chart_key_version += 1
            st.rerun()
    with c2:
        if st.button("Rückgängig", use_container_width=True, disabled=st.session_state.create_history_index <= 0):
            if _history_restore(st.session_state.create_history_index - 1):
                st.session_state.create_selected_ids = []
                st.session_state.create_editor_action = ""
                st.session_state.create_prev_selection_sig = ""
                st.session_state.create_chart_key_version += 1
                st.rerun()
    with c3:
        if st.button(
            "Wiederholen",
            use_container_width=True,
            disabled=st.session_state.create_history_index >= len(st.session_state.create_history) - 1,
        ):
            if _history_restore(st.session_state.create_history_index + 1):
                st.session_state.create_selected_ids = []
                st.session_state.create_editor_action = ""
                st.session_state.create_prev_selection_sig = ""
                st.session_state.create_chart_key_version += 1
                st.rerun()

    if not selected_ids:
        st.info("Bitte Knoten im Plot auswählen.")
    else:
        all_inactive = all((not n.active) for n in selected_nodes)

        if all_inactive:
            if st.button("Knoten aktivieren", use_container_width=True):
                changed = 0
                for nid in selected_ids:
                    node = draft.nodes[int(nid)]
                    before = node.active
                    node.active = True
                    changed += int(before != node.active)
                _history_push(draft)
                st.session_state.create_selected_ids = []
                st.session_state.create_editor_action = ""
                st.session_state.create_prev_selection_sig = ""
                st.session_state.create_chart_key_version += 1
                st.success(f"Knoten aktiviert: {changed}")
                st.rerun()
        else:
            action_cols = st.columns(3)
            with action_cols[0]:
                if st.button("Knoten deaktivieren", use_container_width=True):
                    changed = 0
                    for nid in selected_ids:
                        node = draft.nodes[int(nid)]
                        before = (node.active, node.force_x, node.force_z, node.fixed_x, node.fixed_z)
                        node.active = False
                        node.force_x = 0.0
                        node.force_z = 0.0
                        node.fixed_x = False
                        node.fixed_z = False
                        after = (node.active, node.force_x, node.force_z, node.fixed_x, node.fixed_z)
                        changed += int(before != after)
                    _history_push(draft)
                    st.session_state.create_selected_ids = []
                    st.session_state.create_editor_action = ""
                    st.session_state.create_prev_selection_sig = ""
                    st.session_state.create_chart_key_version += 1
                    st.success(f"Knoten deaktiviert: {changed}")
                    st.rerun()
            with action_cols[1]:
                if st.button("Kraft", use_container_width=True):
                    st.session_state.create_editor_action = "edit_force"
            with action_cols[2]:
                if st.button("Lager", use_container_width=True):
                    st.session_state.create_editor_action = "edit_support"

        current_action = st.session_state.create_editor_action

        if current_action == "edit_force":
            st.caption("Kraft überschreiben/löschen")
            fx, fz = st.columns(2)
            with fx:
                st.number_input("Kraft X", key="create_force_x", step=1.0, format="%.2f")
            with fz:
                st.number_input("Kraft Z", key="create_force_z", step=1.0, format="%.2f")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("Kraft anwenden", use_container_width=True):
                    changed = 0
                    for nid in selected_ids:
                        node = draft.nodes[int(nid)]
                        before = (node.force_x, node.force_z)
                        node.force_x = float(st.session_state.create_force_x)
                        node.force_z = float(st.session_state.create_force_z)
                        node.active = True
                        changed += int(before != (node.force_x, node.force_z))
                    _history_push(draft)
                    st.session_state.create_selected_ids = []
                    st.session_state.create_editor_action = ""
                    st.session_state.create_prev_selection_sig = ""
                    st.session_state.create_chart_key_version += 1
                    st.success(f"Kraft gesetzt: {changed}")
                    st.rerun()
            with b2:
                if st.button("Kraft löschen", use_container_width=True):
                    changed = 0
                    for nid in selected_ids:
                        node = draft.nodes[int(nid)]
                        before = (node.force_x, node.force_z)
                        node.force_x = 0.0
                        node.force_z = 0.0
                        changed += int(before != (node.force_x, node.force_z))
                    _history_push(draft)
                    st.session_state.create_selected_ids = []
                    st.session_state.create_editor_action = ""
                    st.session_state.create_prev_selection_sig = ""
                    st.session_state.create_chart_key_version += 1
                    st.success(f"Kraft gelöscht: {changed}")
                    st.rerun()

        elif current_action == "edit_support":
            st.caption("Lager überschreiben/löschen")
            t1, t2 = st.columns(2)
            with t1:
                is_loslager = st.session_state.create_support_mode == "Loslager"
                if st.button(
                    "Loslager",
                    use_container_width=True,
                    key="support_mode_loslager",
                    disabled=is_loslager,
                ):
                    st.session_state.create_support_mode = "Loslager"
                    st.rerun()
            with t2:
                is_festlager = st.session_state.create_support_mode == "Festlager"
                if st.button(
                    "Festlager",
                    use_container_width=True,
                    key="support_mode_festlager",
                    disabled=is_festlager,
                ):
                    st.session_state.create_support_mode = "Festlager"
                    st.rerun()
            st.caption(f"Aktueller Lagertyp: {st.session_state.create_support_mode}")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("Lager anwenden", use_container_width=True):
                    changed = 0
                    for nid in selected_ids:
                        node = draft.nodes[int(nid)]
                        before = (node.fixed_x, node.fixed_z)
                        _set_support(node, str(st.session_state.create_support_mode))
                        node.active = True
                        changed += int(before != (node.fixed_x, node.fixed_z))
                    _history_push(draft)
                    st.session_state.create_selected_ids = []
                    st.session_state.create_editor_action = ""
                    st.session_state.create_prev_selection_sig = ""
                    st.session_state.create_chart_key_version += 1
                    st.success(f"Lager gesetzt: {changed}")
                    st.rerun()
            with b2:
                if st.button("Lager löschen", use_container_width=True):
                    changed = 0
                    for nid in selected_ids:
                        node = draft.nodes[int(nid)]
                        before = (node.fixed_x, node.fixed_z)
                        node.fixed_x = False
                        node.fixed_z = False
                        changed += int(before != (node.fixed_x, node.fixed_z))
                    _history_push(draft)
                    st.session_state.create_selected_ids = []
                    st.session_state.create_editor_action = ""
                    st.session_state.create_prev_selection_sig = ""
                    st.session_state.create_chart_key_version += 1
                    st.success(f"Lager gelöscht: {changed}")
                    st.rerun()

    if st.button("Modell Ausgangszustand wiederherstellen", use_container_width=True):
        st.session_state.create_draft_struct = _build_default_structure(width, height)
        _history_reset(st.session_state.create_draft_struct)
        st.session_state.create_selected_ids = []
        st.session_state.create_editor_action = ""
        st.session_state.create_prev_selection_sig = ""
        st.session_state.create_chart_key_version += 1
        st.success("Rohmodell wurde zurückgesetzt.")
        st.rerun()

if create_model_clicked:
    requested_name = (st.session_state.create_model_name or "").strip()
    if not requested_name:
        st.error("Bitte einen Modellnamen eingeben.")
    else:
        was_editing = st.session_state.create_mode == "edit"
        if was_editing:
            if st.session_state.create_edit_model_id is None:
                st.error("Bitte zuerst ein Modell laden.")
                st.stop()
            target_id = int(st.session_state.create_edit_model_id)
            created_at = st.session_state.create_edit_model_created_at or now_iso()
            final_name = make_unique_model_name(index, requested_name, current_model_id=target_id)
        else:
            target_id = next_model_id(index)
            created_at = now_iso()
            final_name = make_unique_model_name(index, requested_name)

        metadata = save_model_snapshot(draft, target_id, final_name, created_at)

        index = [m for m in index if int(m.get("id", -1)) != int(target_id)]
        index.append(metadata)
        save_index(index)

        st.session_state.startup_load_model_id = int(target_id)
        st.session_state.create_edit_model_id = int(target_id)
        st.session_state.create_edit_model_created_at = created_at
        if final_name != requested_name:
            st.info(f"Name bereits vergeben, gespeichert als: {final_name}")
        if was_editing:
            st.success("Modell gespeichert.")
        else:
            st.success("Modell erstellt.")

if st.session_state.get("startup_load_model_id"):
    if st.button("Modell in Modell Optimieren öffnen", use_container_width=True):
        st.switch_page("pages/2_Modell_Optimierung.py")
