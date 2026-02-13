import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt # Wichtig für plt.close()
from model import Structure
from optimizer import TopologyOptimizer
from visualizer import Visualizer

# 1. Konfiguration
st.set_page_config(page_title="Topologieoptimierung", layout="wide")
st.title("2D Topologieoptimierung (Abschlussprojekt)")

# 2. Session State initialisieren
if 'structure' not in st.session_state:
    st.session_state.structure = None

# --- Sidebar: Globale Einstellungen ---
st.sidebar.header("Modell-Einstellungen")
width = st.sidebar.slider("Breite (Knoten)", 5, 200, 80)
height = st.sidebar.slider("Höhe (Knoten)", 5, 70, 20)

# Button: Neues Modell erstellen
if st.sidebar.button("Neues Modell erstellen (Reset)"):
    st.session_state.structure = Structure(width, height)
    
    # --- Preset: Brücke (Links Loslager, Rechts Festlager) ---
    struct = st.session_state.structure
    
    # Links unten: Loslager (fest in Z, beweglich in X)
    node_left = struct.nodes[(height-1)*width]
    node_left.fixed_x = False 
    node_left.fixed_z = True
    
    # Rechts unten: Festlager (fest in Z und X)
    node_right = struct.nodes[height*width - 1]
    node_right.fixed_x = True
    node_right.fixed_z = True
    
    # Kraft 10N in der Mitte oben
    mid_node = struct.nodes[int(width/2)]
    mid_node.force_z = 10.0 
    
    st.success("Modell initialisiert!")
    st.rerun()

# --- Hauptlogik ---
if st.session_state.structure is not None:
    struct = st.session_state.structure
    opt = TopologyOptimizer(struct)
    
    # --- Sidebar: Kräfte bearbeiten ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Kräfte bearbeiten")
    
    max_id = len(struct.nodes) - 1
    selected_id = st.sidebar.number_input("Knoten-ID auswählen", 0, max_id, 0)      # Auswahl nach Knoten-ID
    selected_node = struct.nodes[selected_id]
    
    col_f1, col_f2 = st.sidebar.columns(2)
    with col_f1:
        new_fx = st.number_input("Kraft X", value=selected_node.force_x, key="fx_input")
    with col_f2:
        new_fz = st.number_input("Kraft Z", value=selected_node.force_z, key="fz_input")
        
    if st.sidebar.button("Kraft anwenden"):
        selected_node.force_x = new_fx
        selected_node.force_z = new_fz
        st.success(f"Kraft an Knoten {selected_id} aktualisiert.")
        st.rerun()

    # --- Hauptbereich ---
    col1, col2 = st.columns([1, 2])
    
    # Initiale Berechnung der Knoten
    current_active = sum(1 for n in struct.nodes if n.active)
    total_nodes = len(struct.nodes)
    
    # Plot Spalte rechts
    with col2:
        st.subheader("Visualisierung")
        scale = st.slider("Verformungsskalierung", 0.0, 0.4, 0.0, step=0.005,format="%.3f")
        plot_placeholder = st.empty()
        
        # Aufruf des Plots aus Visiualizer
        fig = Visualizer.plot_structure(struct, show_deformation=True, scale_factor=scale)
        plot_placeholder.pyplot(fig)
        plt.close(fig)

    # Parameter Spalte links
    with col1:
        st.subheader("Automatik-Optimierung")
        
        # Platzhalter erstellen, für dynamische ändern
        mass_info_placeholder = st.empty()
        
        # Masse Anzeige in %
        current_percent = (current_active / total_nodes) * 100
        mass_info_placeholder.info(f"Aktuelle Masse: {current_percent:.1f}%")

        target_mass_percent = st.slider("Ziel-Masse (%)", 10, 99, 50)
        st.caption("Entfernungsrate: Dynamisch (1% -> 1.5%)")
        
        # Steifigkeits-Limit
        # Erklärung:
        # Faktor 1.0 = So steif wie der volle Block (unmöglich leichter zu machen)
        # Faktor 5.0 = Darf sich 5x mehr durchbiegen als der volle Block
        # sollte noch in UI erklärt werden
        limit_factor = st.slider("Max. erlaubter Durchbiegungs-Faktor", 1.0, 5.0, 3.0, step=0.1,format="%.2f")
        
        st.caption(f"Objekt darf maximal {limit_factor} x weicher werden als der Start-Block.")

        if st.button("Starten bis Ziel erreicht"):
            
            # Referenz-Berechnung (voller Block) für Limit-Faktor
            if not opt.solve_step():
                st.error("Start-Modell ist instabil! Bitte Lager/Kräfte prüfen.")
            else:
                # Maximale Verschiebung bei vollem Block
                base_max_u = 0.0
                for n in struct.nodes:
                    dist = (n.u_x**2 + n.u_z**2)**0.5
                    if dist > base_max_u: base_max_u = dist
                
                # Das absolute Limit für die Verschiebung, basierend auf dem Faktor
                abs_limit = base_max_u * limit_factor
                
                # Info über Limit Verformung, schöner umsetztung in UI wäre besser
                # st.toast(f"Basis-Verformung: {base_max_u:.4f} -> Limit: {abs_limit:.4f}")

                # Ziel-Masse und zu entfernende Knoten berechnen
                target_count = int(total_nodes * (target_mass_percent / 100))
                
                status_container = st.empty()
                progress_bar = st.progress(0)
                
                start_active = current_active
                nodes_to_remove_total = start_active - target_count
                
                iteration_count = 0
                max_safety_iterations = 500 
                stop_reason = "Iterationslimit erreicht"  # falls zu viele Iterationen nötig sind
                
                # Optimierungs-Loop bis Ziel-Masse erreicht oder Abbruch durch Limit/Fehler
                while current_active > target_count and iteration_count < max_safety_iterations:
                    
                    # Remove-Rate bestimmen
                    if iteration_count < 5: current_rate = 0.01
                    else: current_rate = 0.015
                    
                    # Status Update
                    status_container.info(f"Iter: {iteration_count+1} | Remove-Rate: {current_rate*100:.1f}% | Limit-Check aktiv...")
                    time.sleep(0.01)

                    # Optimierung mit Limit-Check
                    try:
                        success, message = opt.optimize_step(
                            remove_ratio=current_rate, 
                            max_displacement_limit=abs_limit  # Steifigkeits Limit
                        )
                    except TypeError:
                        # Fallback, falls du optimizer.py doch nicht gespeichert hast
                        success, message = opt.optimize_step(remove_ratio=current_rate)

                    # Abbruchsgründe auswerten
                    if not success:
                        stop_reason = message
                        break
                    
                    # Daten aktualisieren
                    current_active = sum(1 for n in struct.nodes if n.active)
                    
                    # UI Updates (Masse, Progress, Plot)
                    new_percent = (current_active / total_nodes) * 100
                    mass_info_placeholder.info(f"Aktuelle Masse: {new_percent:.1f}%")
                    
                    removed_so_far = start_active - current_active
                    if nodes_to_remove_total > 0:
                        progress_bar.progress(min(max(removed_so_far / nodes_to_remove_total, 0.0), 1.0))
                    
                    fig = Visualizer.plot_structure(struct, show_deformation=True, scale_factor=scale)
                    plot_placeholder.pyplot(fig)
                    plt.close(fig)
                    
                    iteration_count += 1
                
                # Abschluss-Meldung je nach Stop-Grund
                if stop_reason:
                    # Wenn "Limit erreicht" oder "Zu weich" im Text vorkommt -> Optimum gefunden = Erfolg
                    if "Limit" in stop_reason or "Zu weich" in stop_reason or "Keine weiteren" in stop_reason:
                        st.success(f"Optimierung fertig: {stop_reason}")
                        st.info("Das ist das leichteste Design, das dein Steifigkeits-Limit noch einhält.")
                    elif "Undo" in stop_reason:
                        st.warning(f"Grenze erreicht: {stop_reason}")
                    else:
                        st.error(f"Fehler: {stop_reason}")
                else:
                    st.success("Ziel-Masse erreicht! (Struktur ist noch steif genug)")

        if st.button("Meldungen zurücksetzen"):
            st.rerun()
            
    # Ausgabe ganz unten aktualisieren
    final_percent = (sum(1 for n in struct.nodes if n.active) / total_nodes) * 100
    st.metric("Status", f"{sum(1 for n in struct.nodes if n.active)} / {total_nodes} Knoten aktiv ({final_percent:.1f}%)")

else:
    st.info("Bitte erstelle zuerst ein Modell über die Sidebar.")