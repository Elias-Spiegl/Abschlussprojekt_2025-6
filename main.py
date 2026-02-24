import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Model Hub", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATE_DIR = DATA_DIR / "states"
INDEX_FILE = DATA_DIR / "model_index.json"
STATE_DIR.mkdir(parents=True, exist_ok=True)


def format_ts(value: str | None) -> str:
    if not value:
        return "-"
    return str(value).replace("T", " ")


def load_index() -> list[dict]:
    if INDEX_FILE.exists():
        raw = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return raw
    return []


def save_index(index: list[dict]) -> None:
    INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")


def model_path(model_id: int) -> Path:
    return STATE_DIR / f"model_{int(model_id)}.json"


def model_label(meta: dict) -> str:
    model_id = int(meta.get("id", 0))
    name = (meta.get("name") or "").strip() or "(ohne Namen)"
    return f"#{model_id} - {name}"


st.title("Model Hub")
st.caption("Wähle ein bestehendes Modell oder öffne den Workspace für ein neues Modell.")

index = load_index()
sorted_models = sorted(index, key=lambda x: int(x.get("id", 0)))

c_top1, c_top2 = st.columns([1, 1])
with c_top1:
    if st.button("Workspace öffnen", use_container_width=True):
        st.switch_page("pages/2_Workspace.py")
with c_top2:
    if st.button("Neues Modell erstellen", use_container_width=True):
        st.session_state.pop("startup_load_model_id", None)
        st.switch_page("pages/1_Model_Create.py")

st.markdown("**Vorhandene Modelle**")
if not sorted_models:
    st.info("Noch keine Modelle vorhanden.")
else:
    model_ids = [int(m["id"]) for m in sorted_models]

    if "home_selected_model_id" not in st.session_state:
        st.session_state.home_selected_model_id = model_ids[0] if model_ids else None
    if st.session_state.home_selected_model_id is not None and st.session_state.home_selected_model_id not in model_ids:
        st.session_state.home_selected_model_id = model_ids[0] if model_ids else None
    if "home_selection_dirty" not in st.session_state:
        st.session_state.home_selection_dirty = True

    if st.session_state.home_selection_dirty:
        for mid in model_ids:
            st.session_state[f"home_pick_{mid}"] = (mid == st.session_state.home_selected_model_id)
        st.session_state.home_selection_dirty = False

    def _on_pick_model(mid: int) -> None:
        key = f"home_pick_{mid}"
        picked = bool(st.session_state.get(key, False))
        if picked:
            st.session_state.home_selected_model_id = mid
        elif st.session_state.home_selected_model_id == mid:
            st.session_state.home_selected_model_id = None
        st.session_state.home_selection_dirty = True

    hcols = st.columns([1, 1, 2, 2, 2, 1, 1])
    hcols[0].markdown("**Auswahl**")
    hcols[1].markdown("**ID**")
    hcols[2].markdown("**Name**")
    hcols[3].markdown("**Erstellt**")
    hcols[4].markdown("**Geändert**")
    hcols[5].markdown("**Breite**")
    hcols[6].markdown("**Höhe**")

    for m in sorted_models:
        mid = int(m.get("id", 0))
        rcols = st.columns([1, 1, 2, 2, 2, 1, 1])
        with rcols[0]:
            st.checkbox(
                " ",
                key=f"home_pick_{mid}",
                on_change=_on_pick_model,
                args=(mid,),
                label_visibility="collapsed",
            )
        rcols[1].write(str(mid))
        rcols[2].write((m.get("name") or "-"))
        rcols[3].write(format_ts(m.get("created_at")))
        rcols[4].write(format_ts(m.get("updated_at")))
        rcols[5].write(str(int(m.get("width", 0))))
        rcols[6].write(str(int(m.get("height", 0))))

    selected_model_id = st.session_state.home_selected_model_id
    act1, act2 = st.columns(2)
    with act1:
        if st.button(
            "Ausgewähltes Modell im Workspace öffnen",
            use_container_width=True,
            disabled=selected_model_id is None,
        ):
            if selected_model_id is None or int(selected_model_id) not in model_ids:
                st.error("Bitte zuerst ein Modell auswählen.")
            else:
                st.session_state.startup_load_model_id = int(selected_model_id)
                st.switch_page("pages/2_Workspace.py")
    with act2:
        if st.button(
            "Ausgewähltes Modell löschen",
            use_container_width=True,
            disabled=selected_model_id is None,
        ):
            if selected_model_id is None or int(selected_model_id) not in model_ids:
                st.error("Bitte zuerst ein Modell auswählen.")
            else:
                try:
                    model_path(int(selected_model_id)).unlink(missing_ok=True)
                    index = [m for m in index if int(m.get("id", -1)) != int(selected_model_id)]
                    save_index(index)
                    st.session_state.home_selected_model_id = None
                    st.success("Modell gelöscht.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Löschen fehlgeschlagen: {e}")
