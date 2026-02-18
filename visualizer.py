import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from model import Structure

class Visualizer:
    @staticmethod
    def plot_structure(
        structure: Structure,
        show_deformation=False,
        scale_factor=1.0,
        selected_node_id=None,
        colorize_elements=False,
        color_percentile=95,
        show_background_nodes=True,
        line_width=1.0,
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Koordinatensystem: z geht nach unten, y-Achse invertieren 
        ax.invert_yaxis()
        ax.set_aspect('equal')
        
        # Axiale Dehnung als Farbwert vorbereiten (|eps|).
        element_strains = []
        if colorize_elements:
            for elem in structure.elements:
                if not (elem.node_i.active and elem.node_j.active):
                    continue
                vec = elem.node_j.pos - elem.node_i.pos
                length = float(np.linalg.norm(vec))
                if length <= 0.0:
                    element_strains.append(0.0)
                    continue
                e_n = vec / length
                du = np.array([
                    elem.node_j.u_x - elem.node_i.u_x,
                    elem.node_j.u_z - elem.node_i.u_z,
                ])
                axial_strain = float(np.dot(du, e_n) / length)
                element_strains.append(abs(axial_strain))

            if element_strains:
                vmax = float(np.percentile(np.array(element_strains), float(color_percentile)))
            else:
                vmax = 0.0
            vmax = max(vmax, 1e-12)
            # PowerNorm verstärkt niedrige Werte visuell, damit Unterschiede besser sichtbar sind.
            norm = mpl.colors.PowerNorm(gamma=0.6, vmin=0.0, vmax=vmax)
            cmap = mpl.cm.get_cmap("plasma")
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
        else:
            norm = None
            cmap = None
            sm = None

        # Federn zeichnen
        strain_idx = 0
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

                if colorize_elements and cmap is not None and norm is not None:
                    line_color = cmap(norm(element_strains[strain_idx]))
                    strain_idx += 1
                    ax.plot(x_vals, z_vals, color=line_color, linewidth=line_width, alpha=1.0)
                else:
                    ax.plot(x_vals, z_vals, 'b-', linewidth=max(0.3, line_width * 0.7), alpha=0.6) # Federn in blau

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
                    if show_background_nodes:
                        active_nodes_x.append(x)
                        active_nodes_z.append(z)
        
        # Restliche Knoten grau
        if show_background_nodes and active_nodes_x:
            scatter_alpha = 0.15 if colorize_elements else 0.5
            scatter_size = 5 if colorize_elements else 10
            ax.scatter(active_nodes_x, active_nodes_z, c='gray', s=scatter_size, alpha=scatter_alpha)

        if colorize_elements and sm is not None:
            cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("|axiale Dehnung|", rotation=90)

        ax.set_title(f"Struktur (Darstellung {scale_factor * 100:.0f}% | 100%=echt)")
        return fig
