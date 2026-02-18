import numpy as np
from typing import List, Tuple

class Node:
    def __init__(self, id: int, x: float, z: float):
        self.id = id
        self.x = x
        self.z = z      # z zeigt nach unten
        self.active = True  # Für Topologieoptimierung
        self.u_x = 0.0  # Verschiebung x
        self.u_z = 0.0  # Verschiebung z
        self.force_x = 0.0  # Kräfte
        self.force_z = 0.0
        self.fixed_x = False    # Lagerbedingung: False = frei, True = fest
        self.fixed_z = False

    @property
    def pos(self):
        return np.array([self.x, self.z])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "x": self.x,
            "z": self.z,
            "active": self.active,
            "u_x": self.u_x,
            "u_z": self.u_z,
            "force_x": self.force_x,
            "force_z": self.force_z,
            "fixed_x": self.fixed_x,
            "fixed_z": self.fixed_z,
        }

class Element:  # Federelement zwischen zwei Knoten
    def __init__(self, node_i: Node, node_j: Node, k: float):
        self.node_i = node_i
        self.node_j = node_j
        self.k = k # Federsteifigkeit

class Structure:
    def __init__(self, width: int, height: int):
        self.nodes: List[Node] = []     # Liste aller Knoten
        self.elements: List[Element] = []   # Liste aller Elemente (Federn)
        self.width = width
        self.height = height
        self._initialize_grid()

    def _initialize_grid(self):
        # Erstellt das Gitter aus Knoten und Federn.

        node_id = 0
        grid = {} # Konten Dictionary Mapping (x, z)

        # Knoten erstellen in Höhe und Breite
        for z in range(self.height):
            for x in range(self.width):
                node = Node(node_id, float(x), float(z))
                self.nodes.append(node)
                grid[(x, z)] = node     # Dictionary füllen
                node_id += 1

        # Federn erstellen (Horizontal, Vertikal, Diagonal)
        # Steifigkeiten definiert: k_ortho = 1, k_diag = 1/sqrt(2)
        k_ortho = 1.0
        k_diag = 1.0 / np.sqrt(2.0)

        for z in range(self.height):
            for x in range(self.width):
                node = grid[(x, z)]     
                
                # Nachbarn verbinden (von Links oben nach Rechts unten in Grid)
                neighbors = [
                    (x + 1, z, k_ortho),      # Rechts
                    (x, z + 1, k_ortho),      # Unten
                    (x + 1, z + 1, k_diag),   # Diagonal Rechts-Unten
                    (x - 1, z + 1, k_diag)    # Diagonal Links-Unten
                ]

                for nx, nz, k in neighbors:
                    if (nx, nz) in grid:
                        neighbor = grid[(nx, nz)]
                        self.elements.append(Element(node, neighbor, k))

    def reset_forces(self):    # Kräfte zurücksetzen
        for node in self.nodes:
            node.force_x = 0.0
            node.force_z = 0.0

    def get_dof_indices(self) -> int:       # Freiheitsgrade bestimmen: 2 * Anzahl Knoten
        return len(self.nodes) * 2

    def to_dict(self) -> dict:
        return {
            "format_version": 1,
            "width": self.width,
            "height": self.height,
            "nodes": [node.to_dict() for node in self.nodes],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Structure":
        width = int(data["width"])
        height = int(data["height"])
        structure = cls(width, height)

        raw_nodes = data.get("nodes", [])
        if len(raw_nodes) != len(structure.nodes):
            raise ValueError("Ungültige Dateistruktur: Knotenanzahl passt nicht zum Raster.")

        for raw in raw_nodes:
            node_id = int(raw["id"])
            node = structure.nodes[node_id]
            node.active = bool(raw.get("active", True))
            node.u_x = float(raw.get("u_x", 0.0))
            node.u_z = float(raw.get("u_z", 0.0))
            node.force_x = float(raw.get("force_x", 0.0))
            node.force_z = float(raw.get("force_z", 0.0))
            node.fixed_x = bool(raw.get("fixed_x", False))
            node.fixed_z = bool(raw.get("fixed_z", False))

        return structure
