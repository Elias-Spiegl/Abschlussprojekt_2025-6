import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from model import Structure

class Visualizer:
    @staticmethod
    def _element_orientation(elem) -> str:
        dx = abs(float(elem.node_j.x - elem.node_i.x))
        dz = abs(float(elem.node_j.z - elem.node_i.z))
        if dz == 0.0 and dx > 0.0:
            return "horizontal"
        if dx == 0.0 and dz > 0.0:
            return "vertical"
        return "diagonal"

    @staticmethod
    def _orientation_allowed(orientation: str, element_filter: str) -> bool:
        if element_filter == "hv":
            return orientation in ("horizontal", "vertical")
        if element_filter == "diag":
            return orientation == "diagonal"
        return True

    @staticmethod
    def compute_element_values(
        structure: Structure,
        metric_mode: str = "energy",
        element_filter: str = "all",
    ) -> list[float]:
        values: list[float] = []
        for elem in structure.elements:
            if not (elem.node_i.active and elem.node_j.active):
                continue
            orientation = Visualizer._element_orientation(elem)
            if not Visualizer._orientation_allowed(orientation, element_filter):
                continue
            vec = elem.node_j.pos - elem.node_i.pos
            length = float(np.linalg.norm(vec))
            if length <= 0.0:
                values.append(0.0)
                continue
            e_n = vec / length
            du = np.array([
                elem.node_j.u_x - elem.node_i.u_x,
                elem.node_j.u_z - elem.node_i.u_z,
            ])
            delta_l = float(np.dot(du, e_n))
            abs_delta_l = abs(delta_l)
            abs_strain = abs_delta_l / length

            if metric_mode == "strain":
                values.append(abs_strain)
            elif metric_mode == "force":
                values.append(abs(elem.k * delta_l))
            elif metric_mode == "energy_per_length":
                values.append(0.5 * elem.k * (delta_l ** 2) / length)
            else:  # default: energy
                values.append(0.5 * elem.k * (delta_l ** 2))
        return values

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
        color_levels=16,
        fixed_color_vmax=None,
        metric_mode="energy",
        normalize_mode="global",
        element_filter="all",
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Koordinatensystem: z geht nach unten, y-Achse invertieren 
        ax.invert_yaxis()
        ax.set_aspect('equal')
        
        # Axiale Dehnung als Farbwert vorbereiten (|eps|).
        element_records = []
        for elem in structure.elements:
            if not (elem.node_i.active and elem.node_j.active):
                continue
            orientation = Visualizer._element_orientation(elem)
            if not Visualizer._orientation_allowed(orientation, element_filter):
                continue

            vec = elem.node_j.pos - elem.node_i.pos
            length = float(np.linalg.norm(vec))
            if length <= 0.0:
                value = 0.0
            else:
                e_n = vec / length
                du = np.array([
                    elem.node_j.u_x - elem.node_i.u_x,
                    elem.node_j.u_z - elem.node_i.u_z,
                ])
                delta_l = float(np.dot(du, e_n))
                abs_delta_l = abs(delta_l)
                abs_strain = abs_delta_l / length

                if metric_mode == "strain":
                    value = abs_strain
                elif metric_mode == "force":
                    value = abs(elem.k * delta_l)
                elif metric_mode == "energy_per_length":
                    value = 0.5 * elem.k * (delta_l ** 2) / length
                else:  # default: energy
                    value = 0.5 * elem.k * (delta_l ** 2)

            element_records.append((elem, orientation, float(value)))

        color_values = []
        if colorize_elements:
            raw_values = [val for _, _, val in element_records]
            if normalize_mode == "orientation":
                groups = {"horizontal": [], "vertical": [], "diagonal": []}
                for _, ori, val in element_records:
                    groups[ori].append(val)

                group_vmax = {}
                for ori, vals in groups.items():
                    if vals:
                        ori_vmax = float(np.percentile(np.array(vals), float(color_percentile)))
                    else:
                        ori_vmax = 0.0
                    group_vmax[ori] = max(ori_vmax, 1e-12)

                color_values = [min(val / group_vmax[ori], 1.0) for _, ori, val in element_records]
                norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
            elif fixed_color_vmax is not None and float(fixed_color_vmax) > 0:
                vmax = float(fixed_color_vmax)
                vmax = max(vmax, 1e-12)
                color_values = [min(val / vmax, 1.0) for val in raw_values]
                norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
            elif raw_values:
                vmax = float(np.percentile(np.array(raw_values), float(color_percentile)))
                vmax = max(vmax, 1e-12)
                color_values = [min(val / vmax, 1.0) for val in raw_values]
                norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
            else:
                color_values = [0.0 for _ in element_records]
                norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

            if color_levels and int(color_levels) > 1:
                boundaries = np.linspace(0.0, 1.0, int(color_levels) + 1)
                norm = mpl.colors.BoundaryNorm(boundaries, ncolors=256, clip=True)
            cmap = mpl.cm.get_cmap("plasma")
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
        else:
            norm = None
            cmap = None
            sm = None

        # Federn zeichnen
        value_idx = 0
        for elem, _, _ in element_records:
            x_vals = [elem.node_i.x, elem.node_j.x]
            z_vals = [elem.node_i.z, elem.node_j.z]

            # Wenn Verformung angezeigt werden soll
            if show_deformation:
                x_vals[0] += elem.node_i.u_x * scale_factor
                x_vals[1] += elem.node_j.u_x * scale_factor
                z_vals[0] += elem.node_i.u_z * scale_factor
                z_vals[1] += elem.node_j.u_z * scale_factor

            if colorize_elements and cmap is not None and norm is not None:
                line_color = cmap(norm(color_values[value_idx]))
                value_idx += 1
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
            metric_label = {
                "strain": "|axiale Dehnung|",
                "force": "|axiale Kraft|",
                "energy": "elastische Energie",
                "energy_per_length": "Energie / Länge",
            }.get(metric_mode, "Elementwert")
            if normalize_mode == "orientation":
                cbar.set_label(f"{metric_label} (pro Richtung normiert)", rotation=90)
            else:
                cbar.set_label(f"{metric_label} (normiert)", rotation=90)

        ax.set_title(f"Struktur (Darstellung {scale_factor * 100:.0f}% | 100%=echt)")
        return fig
