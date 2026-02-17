import io
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from model import Structure
from optimizer import TopologyOptimizer
from visualizer import Visualizer

st.set_page_config(page_title="Topologieoptimierung", layout="wide")
st.title("2D Topologieoptimierung (Abschlussprojekt)")

BASE_DIR = Path(__file__).resolve().parent
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

index = load_index()

# Sidebar: Step 1 - Modellverwaltung
st.sidebar.header("Workflow")
with st.sidebar.expander("1) Modellverwaltung", expanded=True):
    st.caption("Neues Modell anlegen oder vorhandenes Modell laden.")
    width = st.slider("Breite (Knoten)", 5, 200, 80)
    height = st.slider("Höhe (Knoten)", 5, 70, 20)
    new_model_name = st.text_input("Modellname (optional)", value="", key="new_model_name")

    if st.button("Neues Modell erstellen", use_container_width=True):
        struct = Structure(width, height)

        node_left = struct.nodes[(height - 1) * width]
        node_left.fixed_x = False
        node_left.fixed_z = True

        node_right = struct.nodes[height * width - 1]
        node_right.fixed_x = True
        node_right.fixed_z = True

        mid_node = struct.nodes[int(width / 2)]
        mid_node.force_z = 10.0

        new_id = next_model_id(index)
        created_at = now_iso()
        metadata = save_model_snapshot(struct, new_id, new_model_name, created_at)

        index = [m for m in index if int(m["id"]) != int(new_id)]
        index.append(metadata)
        save_index(index)

        st.session_state.structure = struct
        st.session_state.model_id = int(new_id)
        st.session_state.model_name = metadata["name"]
        st.session_state.model_created_at = metadata["created_at"]
        st.session_state.selected_x = 0
        st.session_state.selected_z = 0
        st.session_state.editor_node_id = None
        st.success("Modell erstellt.")
        st.rerun()

    if index:
        sorted_models = sorted(index, key=lambda x: int(x["id"]))
        model_ids = [int(m["id"]) for m in sorted_models]
        label_by_id = {int(m["id"]): model_label(m) for m in sorted_models}
        selected_model_id = st.selectbox(
            "Vorhandene Modelle",
            model_ids,
            key="model_select_id",
            format_func=lambda mid: label_by_id.get(mid, "Modell"),
        )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if st.button("Laden", use_container_width=True):
                try:
                    loaded_struct, loaded_meta = load_model_snapshot(selected_model_id)
                    st.session_state.structure = loaded_struct
                    st.session_state.model_id = int(loaded_meta["id"])
                    st.session_state.model_name = loaded_meta.get("name", "")
                    st.session_state.model_created_at = loaded_meta.get("created_at", now_iso())
                    st.session_state.selected_x = 0
                    st.session_state.selected_z = 0
                    st.session_state.editor_node_id = None
                    st.success("Modell geladen.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Laden fehlgeschlagen: {e}")
        with col_m2:
            if st.button("Löschen", use_container_width=True):
                try:
                    model_path(selected_model_id).unlink(missing_ok=True)
                    index = [m for m in index if int(m["id"]) != int(selected_model_id)]
                    save_index(index)
                    if st.session_state.model_id == selected_model_id:
                        st.session_state.structure = None
                        st.session_state.model_id = None
                        st.session_state.model_name = ""
                        st.session_state.model_created_at = None
                    st.success("Modell gelöscht.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Löschen fehlgeschlagen: {e}")
    else:
        st.info("Noch keine Modelle gespeichert.")

if st.session_state.structure is None:
    st.subheader("Gespeicherte Modelle")
    rows = to_local_table_rows(index)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Noch keine Modelle vorhanden. Erstelle zuerst ein Modell in der Sidebar.")
    st.stop()

struct = st.session_state.structure
opt = TopologyOptimizer(struct)

# Current model info + save
with st.sidebar.expander("2) Aktuelles Modell", expanded=True):
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
            metadata = save_model_snapshot(
                struct,
                int(st.session_state.model_id),
                st.session_state.model_name,
                st.session_state.model_created_at or now_iso(),
            )
            index = [m for m in index if int(m["id"]) != int(metadata["id"])]
            index.append(metadata)
            save_index(index)
            st.success("Modell gespeichert.")
            st.rerun()

# Sidebar: Step 3 - Knoten/Kräfte/Lager
with st.sidebar.expander("3) Knoten, Kräfte, Lager", expanded=True):
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

    st.session_state.selected_x = min(max(int(st.session_state.selected_x), 0), struct.width - 1)
    st.session_state.selected_z = min(max(int(st.session_state.selected_z), 0), struct.height - 1)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        selected_x = st.number_input("X", min_value=0, max_value=struct.width - 1, step=1, key="selected_x")
    with col_s2:
        selected_z = st.number_input("Z", min_value=0, max_value=struct.height - 1, step=1, key="selected_z")

    selected_id = int(selected_z) * struct.width + int(selected_x)
    selected_node = struct.nodes[selected_id]
    st.caption(f"Ausgewählter Knoten: x={int(selected_x)}, z={int(selected_z)}")

    if st.session_state.editor_node_id != selected_id:
        st.session_state.editor_node_id = selected_id
        st.session_state.edit_fx = float(selected_node.force_x)
        st.session_state.edit_fz = float(selected_node.force_z)
        if selected_node.fixed_x and selected_node.fixed_z:
            st.session_state.edit_support_mode = "Festlager"
        elif (not selected_node.fixed_x) and selected_node.fixed_z:
            st.session_state.edit_support_mode = "Loslager"
        else:
            st.session_state.edit_support_mode = "Frei"

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        new_fx = st.number_input("Kraft X", key="edit_fx")
    with col_f2:
        new_fz = st.number_input("Kraft Z", key="edit_fz")

    support_mode = st.selectbox("Lagerzustand", ["Frei", "Loslager", "Festlager"], key="edit_support_mode")

    if st.button("Knotenänderungen anwenden", use_container_width=True):
        selected_node.force_x = float(new_fx)
        selected_node.force_z = float(new_fz)

        if support_mode == "Frei":
            selected_node.fixed_x = False
            selected_node.fixed_z = False
        elif support_mode == "Loslager":
            selected_node.fixed_x = False
            selected_node.fixed_z = True
        else:
            selected_node.fixed_x = True
            selected_node.fixed_z = True

        st.success(f"Knoten {selected_id} aktualisiert.")
        st.rerun()

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

    show_deformation = st.checkbox("Verformung anzeigen", value=True, key="show_deformation")
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
        st.caption("Verformung ist ausgeblendet (undeformte Geometrie).")

    fem_color_map = st.checkbox("FEM-Farbskala für Federn (axiale Dehnung)", value=True, key="fem_colormap")
    line_width = st.slider("Linienstärke", 0.4, 2.0, 0.8, step=0.1, key="element_line_width")
    color_percentile = 95
    if fem_color_map:
        color_percentile = st.slider(
            "Farbkontrast (Perzentil, höher = mehr Ausreißer sichtbar)",
            70,
            99,
            95,
            step=1,
            key="strain_color_percentile",
        )
        show_background_nodes = False
        st.caption("Hintergrundknoten sind im FEM-Farbmodus ausgeblendet.")
    else:
        show_background_nodes = st.checkbox("Hintergrundknoten anzeigen", value=True, key="show_background_nodes")

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

    if st.button("Starten bis Ziel erreicht"):
        if not opt.solve_step():
            st.error("Start-Modell ist instabil! Bitte Lager/Kräfte prüfen.")
        else:
            base_max_u = max_displacement(struct)
            abs_limit = base_max_u * allowed_softening_ratio
            target_count = int(total_nodes * (target_mass_percent / 100))

            status_container = st.empty()
            progress_bar = st.progress(0)

            start_active = current_active
            nodes_to_remove_total = start_active - target_count
            iteration_count = 0
            max_safety_iterations = 500
            stop_reason = "Iterationslimit erreicht"

            while current_active > target_count and iteration_count < max_safety_iterations:
                current_rate = 0.01 if iteration_count < 5 else 0.015
                status_container.info(
                    f"Iter: {iteration_count + 1} | Remove-Rate: {current_rate * 100:.1f}% | Limit-Check aktiv..."
                )
                time.sleep(0.01)

                success, message = opt.optimize_step(
                    remove_ratio=current_rate,
                    max_displacement_limit=abs_limit,
                )

                if not success:
                    stop_reason = message
                    break

                current_active = sum(1 for n in struct.nodes if n.active)
                new_percent = (current_active / total_nodes) * 100
                mass_info_placeholder.info(f"Aktuelle Masse: {new_percent:.1f}%")

                removed_so_far = start_active - current_active
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
                )
                plot_placeholder.pyplot(fig)
                plt.close(fig)
                iteration_count += 1

            if stop_reason:
                if "Limit" in stop_reason or "Zu weich" in stop_reason or "Keine weiteren" in stop_reason:
                    st.success(f"Optimierung fertig: {stop_reason}")
                    st.info("Das ist das leichteste Design, das dein Steifigkeits-Limit noch einhält.")
                else:
                    st.error(f"Fehler: {stop_reason}")
            else:
                st.success("Ziel-Masse erreicht! (Struktur ist noch steif genug)")

    if st.button("Meldungen zurücksetzen"):
        st.rerun()

final_percent = (sum(1 for n in struct.nodes if n.active) / total_nodes) * 100
st.metric("Status", f"{sum(1 for n in struct.nodes if n.active)} / {total_nodes} Knoten aktiv ({final_percent:.1f}%)")
