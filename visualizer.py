import matplotlib.pyplot as plt
from model import Structure

class Visualizer:
    @staticmethod
    def plot_structure(structure: Structure, show_deformation=False, scale_factor=1.0, selected_node_id=None):
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Koordinatensystem: z geht nach unten, y-Achse invertieren 
        ax.invert_yaxis()
        ax.set_aspect('equal')
        
        # Federn zeichnen
        for elem in structure.elements:
            if elem.node_i.active and elem.node_j.active:
                x_vals = [elem.node_i.x, elem.node_j.x]
                z_vals = [elem.node_i.z, elem.node_j.z]
                
                # Wenn Verformung angezeigt werden soll
                if show_deformation:
                    x_vals[0] += elem.node_i.u_x * scale_factor
                    x_vals[1] += elem.node_j.u_x * scale_factor
                    z_vals[0] += elem.node_i.u_z * scale_factor
                    z_vals[1] += elem.node_j.u_z * scale_factor

                ax.plot(x_vals, z_vals, 'b-', linewidth=0.5, alpha=0.6) # Federn in blau

        # Knoten zeichnen (nur aktive)
        active_nodes_x = []
        active_nodes_z = []
        
        for node in structure.nodes:
            if node.active:
                x = node.x + (node.u_x * scale_factor if show_deformation else 0)
                z = node.z + (node.u_z * scale_factor if show_deformation else 0)

                if selected_node_id is not None and node.id == selected_node_id:
                    ax.plot(x, z, marker='o', markersize=9, markerfacecolor='none', markeredgecolor='orange', markeredgewidth=2)
                
                if node.fixed_x or node.fixed_z:        
                    ax.plot(x, z, 'rs', markersize=5)   # Visualisierung Lager (Rotes Quadrat)
                
                elif node.force_x != 0 or node.force_z != 0:
                    ax.plot(x, z, 'go', markersize=5)   # Visualisierung Kraft (Grüner Kreis)
                    ax.arrow(x, z, node.force_x*0.5, node.force_z*0.5, head_width=0.2, color='g')   # Pfeil für Kraft
                    
                else:
                    active_nodes_x.append(x)
                    active_nodes_z.append(z)
        
        # Restliche Knoten grau
        ax.scatter(active_nodes_x, active_nodes_z, c='gray', s=10, alpha=0.5)

        ax.set_title(f"Struktur (Verformung x {scale_factor})")
        return fig
