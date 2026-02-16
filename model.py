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