import json
from datetime import datetime
from pathlib import Path

import streamlit as st

# Startseite konfigurieren (das ist unsere Modell-Uebersicht).
st.set_page_config(page_title="Modell Übersicht", layout="wide")

# Kleines CSS fuer ein einheitliches UI:
# - Sidebar komplett aus
# - Buttons optisch gleich
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


# Datenhaltung:
# - model_index.json: Liste mit Meta-Daten (id, name, zeitstempel, groesse)
# - states/model_X.json: eigentlicher Modellzustand

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATE_DIR = DATA_DIR / "states"
INDEX_FILE = DATA_DIR / "model_index.json"

# Falls Ordner noch nicht da ist, direkt erzeugen.
STATE_DIR.mkdir(parents=True, exist_ok=True)


def format_ts(value: str | None) -> str:
    # Zeitstring fuer Anzeige vorbereiten.
    if not value:
        return "-"
    return str(value).replace("T", " ")


def load_index() -> list[dict]:
    # Modell-Index laden (enthält Meta-Daten aller gespeicherten Modelle).
    # Wenn Datei kaputt oder leer ist, lieber sauber mit [] starten statt Crash.
    if INDEX_FILE.exists():
        raw = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return raw
    return []


def save_index(index: list[dict]) -> None:
    # Index wieder als JSON speichern.
    # Diese Funktion ist unsere zentrale Schreibstelle für Meta-Daten.
    INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")


def now_iso() -> str:
    # Einheitliches Datumsformat in der App.
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def make_unique_model_name(index: list[dict], requested_name: str, current_model_id: int | None = None) -> str:
    # Verhindert doppelte Modellnamen (z.B. test -> test_1 -> test_2 ...)

    # Wir vergleichen case-insensitive, damit "Test" und "test" nicht doppelt durchgehen.
    # current_model_id ist wichtig beim Umbenennen:
    # Das eigene Modell darf den bisherigen Namen natuerlich behalten.
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

    # Falls Name schon vergeben ist, solange hochzaehlen bis ein freier Kandidat da ist.
    i = 1
    while True:
        candidate = f"{base}_{i}"
        if candidate.casefold() not in taken:
            return candidate
        i += 1


def model_path(model_id: int) -> Path:
    # Dateiname für den Zustand eines konkreten Models.
    return STATE_DIR / f"model_{int(model_id)}.json"


def model_label(meta: dict) -> str:
    # Lesbares Label fuer UI-Listen.
    model_id = int(meta.get("id", 0))
    name = (meta.get("name") or "").strip() or "(ohne Namen)"
    return f"#{model_id} - {name}"


def rename_model(index: list[dict], model_id: int, new_name: str) -> tuple[list[dict], str]:
    # Modellname im Index + in der eigentlichen JSON-Datei aktualisieren.
    # Hintergrund: Der Name steht bei uns doppelt (in model_index und im State-File).
    # Beides muss synchron sein, sonst sieht man je nach Seite unterschiedliche Namen.
    final_name = make_unique_model_name(index, new_name, current_model_id=model_id)
    updated_at = now_iso()

    new_index: list[dict] = []
    for m in index:
        if int(m.get("id", -1)) == int(model_id):
            m = dict(m)
            m["name"] = final_name
            m["updated_at"] = updated_at
        new_index.append(m)

    path = model_path(int(model_id))
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "metadata" in payload:
            payload["metadata"]["name"] = final_name
            payload["metadata"]["updated_at"] = updated_at
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return new_index, final_name


# Navigation oben (statt Streamlit-Sidebar).
nav1, nav2, nav3 = st.columns([1, 1, 1])
with nav1:
    st.button("Modell Übersicht", use_container_width=True, disabled=True, type="primary")
with nav2:
    if st.button("Modell Erstellen / Bearbeiten", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Modell_Erstellen.py")
with nav3:
    if st.button("Modell Optimieren", use_container_width=True, type="primary"):
        st.switch_page("pages/2_Modell_Optimierung.py")

st.title("Modell Übersicht")
st.caption("Wähle ein bestehendes Modell oder erstelle/bearbeite ein Modell im Bereich Modell Erstellen / Bearbeiten.")

index = load_index()
sorted_models = sorted(index, key=lambda x: int(x.get("id", 0)))

# Hauptbereich: vorhandene Modelle als Tabelle mit Auswahl-Checkbox.
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
    if "home_name_edit_mode" not in st.session_state:
        st.session_state.home_name_edit_mode = False
    if "home_name_edit_value" not in st.session_state:
        st.session_state.home_name_edit_value = ""

    # Checkbox-Zustand bei jedem Rerun sauber auf genau ein ausgewaehltes Modell synchronisieren.
    # Streamlit rendert die Seite bei jeder Interaktion neu. Ohne diese Sync-Logik
    # kann es passieren, dass visuell mehrere Haken stehen, obwohl intern nur ein Modell
    # aktiv sein soll.

    if st.session_state.home_selection_dirty:
        for mid in model_ids:
            st.session_state[f"home_pick_{mid}"] = (mid == st.session_state.home_selected_model_id)
        st.session_state.home_selection_dirty = False
        selected_meta = next(
            (m for m in sorted_models if int(m.get("id", -1)) == int(st.session_state.home_selected_model_id or -1)),
            None,
        )
        st.session_state.home_name_edit_mode = False
        st.session_state.home_name_edit_value = (selected_meta.get("name") or "") if selected_meta else ""

    def _on_pick_model(mid: int) -> None:
        # Callback fuer Checkbox:
        # - Klick auf Checkbox setzt/entfernt das aktive Modell
        # - danach markieren wir den Zustand als "dirty", damit beim naechsten
        #   Rerun alle Checkboxen wieder konsistent (nur 0 oder 1 aktiv) sind.
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

    # Aktionen fuer das ausgewaehlte Modell: oeffnen oder loeschen.
    # Beide Buttons sind gesperrt, solange nichts ausgewaehlt wurde.
    act1, act2 = st.columns(2)
    with act1:
        if st.button(
            "Ausgewähltes Modell in Modell Optimieren öffnen",
            use_container_width=True,
            disabled=selected_model_id is None,
        ):
            if selected_model_id is None or int(selected_model_id) not in model_ids:
                st.error("Bitte zuerst ein Modell auswählen.")
            else:
                st.session_state.startup_load_model_id = int(selected_model_id)
                st.switch_page("pages/2_Modell_Optimierung.py")
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
                    # Modell-Datei und Index-Eintrag entfernen.
                    model_path(int(selected_model_id)).unlink(missing_ok=True)
                    index = [m for m in index if int(m.get("id", -1)) != int(selected_model_id)]
                    save_index(index)
                    st.session_state.home_selected_model_id = None
                    st.success("Modell gelöscht.")
                    # Direkt neu rendern, damit die geloeschte Zeile sofort aus der Tabelle verschwindet.
                    st.rerun()
                except Exception as e:
                    st.error(f"Löschen fehlgeschlagen: {e}")
