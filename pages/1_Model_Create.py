import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from model import Structure

st.set_page_config(page_title="Model Create", layout="wide")

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


def make_unique_model_name(index: list[dict], requested_name: str) -> str:
    base = (requested_name or "").strip()
    if not base:
        return ""
    taken = {(m.get("name") or "").strip().casefold() for m in index if (m.get("name") or "").strip()}
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


st.title("Model Create")
st.caption("Erstelle ein neues Modell. Laden/Löschen erfolgt im Model Hub.")

index = load_index()

with st.container(border=True):
    c1, c2 = st.columns(2)
    with c1:
        width = st.slider("Breite (Knoten)", 5, 200, 80)
        name = st.text_input("Modellname", value="")
    with c2:
        height = st.slider("Höhe (Knoten)", 5, 70, 20)

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Modell erstellen", use_container_width=True):
            requested_name = (name or "").strip()
            if not requested_name:
                st.error("Bitte einen Modellnamen eingeben.")
            else:
                struct = Structure(width, height)

                node_left = struct.nodes[(height - 1) * width]
                node_left.fixed_x = False
                node_left.fixed_z = True

                node_right = struct.nodes[height * width - 1]
                node_right.fixed_x = True
                node_right.fixed_z = True

                mid_node = struct.nodes[int(width / 2)]
                mid_node.force_z = 1.0

                new_id = next_model_id(index)
                final_name = make_unique_model_name(index, requested_name)
                created_at = now_iso()
                metadata = save_model_snapshot(struct, new_id, final_name, created_at)

                index = [m for m in index if int(m.get("id", -1)) != int(new_id)]
                index.append(metadata)
                save_index(index)

                st.session_state.startup_load_model_id = int(new_id)
                if final_name != requested_name:
                    st.info(f"Name bereits vergeben, gespeichert als: {final_name}")
                st.success("Modell erstellt.")

    with b2:
        if st.button("Zum Model Hub", use_container_width=True):
            st.switch_page("main.py")

if st.session_state.get("startup_load_model_id"):
    if st.button("Im Workspace öffnen", use_container_width=True):
        st.switch_page("pages/2_Workspace.py")
