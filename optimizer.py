import numpy as np
from model import Structure
from solver import PhysicsSolver
from collections import deque

class TopologyOptimizer:
    def __init__(self, structure: Structure):
        self.structure = structure
        # Vorberechnung der Nachbarschaftsliste, für Graph-Checks, spart Zeit pro Schritt
        self._precompute_adjacency()

    def _precompute_adjacency(self):

        # Erstellt eine Liste aller Verbindungen.
        self.full_adj = {n.id: [] for n in self.structure.nodes}
        for elem in self.structure.elements:    # Beide Knoten einer Feder werden als Nachbarn eingetragen 
            self.full_adj[elem.node_i.id].append(elem.node_j.id)
            self.full_adj[elem.node_j.id].append(elem.node_i.id)

    def check_stability_and_get_main_component(self) -> tuple[bool, set[int]]:
        """
        Graph-Check: Startet bei den Lasten und prüft, ob ALLE Lager erreicht werden.
        Zwingt die Struktur dazu, eine Brücke zu bleiben.
        """
        # 1. Listen erstellen
        active_nodes_indices = [n.id for n in self.structure.nodes if n.active]
        
        support_nodes = []  # Lager-IDs
        load_nodes = []     # Last-IDs (Startpunkte)
        
        for n_id in active_nodes_indices:   # Listen befüllen (Lasten und Lager)
            n = self.structure.nodes[n_id]
            if n.fixed_x or n.fixed_z:
                support_nodes.append(n_id)
            if n.force_x != 0 or n.force_z != 0:
                load_nodes.append(n_id)
        
        if not load_nodes: return True, set(active_nodes_indices)   # Keine Lasten -> alles i.O.
        if not support_nodes: return False, set()   # Keine Lager -> Instabil, Abbruch

        # Prüfen ob die Kraft in die Lager abgeleitet wird
        queue = deque(load_nodes)   # Zu pürfende Knoten (Start bei Lasten)
        visited = set(load_nodes)   # Bereits geprüfte Knoten
        
        full_adj = self.full_adj
        nodes = self.structure.nodes
        
        # Breitensuche durch die Nachbarschaftsliste 
        while queue:
            current_id = queue.popleft()    # Nächster Knoten aus der Queue
            
            for neighbor_id in full_adj[current_id]:
                if neighbor_id not in visited:  # Noch nicht geprüft
                    if nodes[neighbor_id].active:   # Nur aktive Knoten berücksichtigen
                        visited.add(neighbor_id)    # geprüft
                        queue.append(neighbor_id)   # wieder in Prüfliste

        # Alle Lager müssen erreicht werden
        for sn in support_nodes:
            if sn not in visited:
                return False, visited   # Alarm wenn ein Lager abgetrennt ist
                
        return True, visited    # Alle Kräfte zu allen Lagern abgeleitet


    def assemble_global_matrix(self):
        # Vektorisierte zusammensetztung der globalen Steifigkeits-Matrix + Übergabe von Kraftvektor und Randbedingungen

        n_dof = self.structure.get_dof_indices()
        K_g = np.zeros((n_dof, n_dof))
        F_g = np.zeros(n_dof)
        fixed_dofs = []

        # Lasten & Randbedingungen
        for node in self.structure.nodes:
            if not node.active: # Inaktive Knoten werden fixiert, es liegen keine Federn mehr an
                fixed_dofs.extend([2*node.id, 2*node.id + 1])   # Jeder Konoten hat 2 Werte in der Liste (x und z, stehen hintereinander)
                continue
            
            if node.force_x != 0: F_g[2*node.id] = node.force_x # Kräfte eintragen
            if node.force_z != 0: F_g[2*node.id + 1] = node.force_z
            if node.fixed_x: fixed_dofs.append(2*node.id)   # Lager eintrgen
            if node.fixed_z: fixed_dofs.append(2*node.id + 1)

        # Matrix zusammensetzen
        for elem in self.structure.elements:
            if elem.node_i.active and elem.node_j.active:       # Wenn beide Knoten aktiv, sonst liegt keine Feder mehr an
                K_el = PhysicsSolver.calculate_element_stiffness(elem.k, elem.node_i.pos, elem.node_j.pos)  # Lokale 4x4 Steifigkeitsmatrix dieser Feder
                idxs = [2*elem.node_i.id, 2*elem.node_i.id+1, 2*elem.node_j.id, 2*elem.node_j.id+1]         # 4 Adressen in der globalen Matrix, die zu diesen Knoten gehören
                K_g[np.ix_(idxs, idxs)] += K_el     # Lokale Matrix an den richtigen Stellen in der globalen Matrix addieren (Superposition)

        return K_g, F_g, fixed_dofs

    def solve_step(self) -> bool:
        # Löst das System und aktualisiert Verschiebungen in den Knoten. Gibt False zurück, wenn das System instabil ist (Singularität).
        
        K_g, F_g, fixed_dofs = self.assemble_global_matrix()
        
        # Regularisierung von Federn ohne Steifigkeit (z.B. bei dünnen Linien)
        indices = np.arange(K_g.shape[0])
        K_g[indices, indices] += 1e-5   # verhindern von Division durch Null
        
        u = PhysicsSolver.solve_system(K_g, F_g, fixed_dofs)
        
        if u is None: 
            return False # Nur bei echter mathematischer Singularität abbrechen
        
        for node in self.structure.nodes:   # Verschiebungen den Konten zuordnen
            node.u_x = u[2*node.id]
            node.u_z = u[2*node.id + 1]
        
        return True


    def optimize_step(self, remove_ratio: float = 0.02, max_displacement_limit: float = None) -> tuple[bool, str]:
        
        # Optimierung der Strucktur, durch Entfernen von Knoten.
        
        # Start-Prüfung & Referenz-Verschiebung holen
        if not self.solve_step():
            return False, "Start-System instabil."
        
        # Aktuelle maximale Verformung merken vor dem Löschen
        current_u_max = 0.0
        for n in self.structure.nodes:
            dist = (n.u_x**2 + n.u_z**2)**0.5
            if dist > current_u_max: current_u_max = dist
            
        # Falls kein Limit übergeben, setzen des 1.5-fachen vom Startwert
        # Verhindert, zu große Verformungen, die zu Instabilität führen
        if max_displacement_limit is None:
            allowed_deflection = current_u_max * 1.5 # maximal 50% mehr Durchbiegung
        else:
            allowed_deflection = max_displacement_limit

        # Energie berechnen
        raw_energies = {n.id: 0.0 for n in self.structure.nodes}
        for elem in self.structure.elements:
            if elem.node_i.active and elem.node_j.active:
                u_vec = np.array([elem.node_i.u_x, elem.node_i.u_z, elem.node_j.u_x, elem.node_j.u_z])      # Verschiebungsvektor der beiden Knoten dieses Elements
                K_el = PhysicsSolver.calculate_element_stiffness(elem.k, elem.node_i.pos, elem.node_j.pos)  # Lokale Steifigkeitsmatrix 
                val = 0.5 * np.dot(u_vec, np.dot(K_el, u_vec))  # Energie in diesem Element
                raw_energies[elem.node_i.id] += val / 2.0       # Energie wird auf beide Knoten verteilt
                raw_energies[elem.node_j.id] += val / 2.0
        
        
        # Sensitivity Filter (Dicke Balken statt dünne Linien)
        active_nodes = [n for n in self.structure.nodes if n.active]    # Nur aktive Knoten betrachten
        neighbors = {n.id: [] for n in active_nodes}    # Nachbarschaftsliste für aktive Knoten, wird gleich gefüllt
        
        # Nachbarn sammeln
        for elem in self.structure.elements:
            if elem.node_i.active and elem.node_j.active:
                neighbors[elem.node_i.id].append(elem.node_j.id)
                neighbors[elem.node_j.id].append(elem.node_i.id)

        filtered_energies = {}
        filter_factor = 3.0 # Knoten mit wenig Energie, aber vielen energiereichen Nachbarn, werden beibehalten.

        for node in active_nodes:
            my_energy = raw_energies[node.id]   # Eigenenergie vom Knoten
            my_nbs = neighbors.get(node.id, [])  # Nachbarn vom Knoten
            
            if my_nbs:  # Wenn Nachbarn vorhanden, Berechnung des Durchschnitts der Nachbar-Energien
                nb_sum = sum(raw_energies[nid] for nid in my_nbs)
                
                avg_energy = nb_sum / len(my_nbs)
                filtered_energies[node.id] = (my_energy + (filter_factor * avg_energy)) / (1 + filter_factor)   
                # Formel zur Glättung: (Eigenenergie + Faktor * Durchschnitt) / (1 + Faktor) -> prozetualer Einfluss der Nachbarn
            else:
                filtered_energies[node.id] = my_energy

        # Extra Bestrafung für dünne Linien
        # Ziel: Zick-Zack-Linien entfernen, massive Dreiecke behalten.
        for node in active_nodes:
            # Überspringe Lasten/Lager
            if node.force_x != 0 or node.force_z != 0 or node.fixed_x or node.fixed_z:
                continue

            my_nbs = neighbors.get(node.id, []) # Liste der Nachbarn
            num_neighbors = len(my_nbs) # Anzahl der Nachbarn
            
            # Logik:
            if num_neighbors >= 3:
                pass # Alles gut, Teil eines Netzes (Dreieck oder mehr)
                
            elif num_neighbors == 2:    # Teil einer Kette/Linie
                filtered_energies[node.id] *= 0.4   # Bestrafung über Energie (Faktor 0.4)
                
            elif num_neighbors <= 1:  # Alleinstehender Knoten oder Endpunkt
                filtered_energies[node.id] *= 0.01  # Starke Bestrafung, da nutzlos

        # Sortieren nach Energie: Wenig Energie = Unwichtig
        active_nodes.sort(key=lambda n: filtered_energies.get(n.id, 0)) # Knoten mit wenig Energie zuerst
        
        target_remove = max(1, int(len(active_nodes) * remove_ratio))
        removed_count = 0
        nodes_removed_this_step = []

        # Löschen mit Graph-Check        
        candidates = [] 
        for node in active_nodes:
            if removed_count >= target_remove: break 
            if node.force_x != 0 or node.force_z != 0 or node.fixed_x or node.fixed_z: continue # Lasten und Lager nicht entfernen
            
            candidates.append(node) # Knoten zu den Entfernungskandidaten
            node.active = False # Vorerst deaktivieren, damit er im Graph-Check nicht mehr zählt
            
            # Graph-Check
            is_connected, _ = self.check_stability_and_get_main_component() 
            if is_connected:
                removed_count += 1  # Knoten bleibt deaktiviert, da er nicht zu einem Kollaps führt
                nodes_removed_this_step.append(node)    # Liste der tatsächlich entfernten Knoten für möglichen Rückgängig-Schritt
            else:
                node.active = True # Graph kaputt -> wieder aktivieren, da er essentiell für die Verbindung ist
        
        if removed_count == 0:
             return False, "Keine weiteren Knoten entfernbar."

        # Steifigkeits-Check
        # Verschiebungen neu berechnen
        physics_ok = self.solve_step()
        
        if not physics_ok:
            # Fall A: Matrix singulär (z.B. durch freistehende Teile)
            undo_needed = True
            reason = "Physikalischer Kollaps (Singularität)"
        else:
            # Fall B: Matrix okay, aber Verformung zu groß?
            new_u_max = 0.0
            for n in self.structure.nodes:  # Suche nach Konten mit maximaler Verschiebung
                dist = (n.u_x**2 + n.u_z**2)**0.5
                if dist > new_u_max: new_u_max = dist
            
            # Vergleich mit Limit
            if new_u_max > allowed_deflection:
                undo_needed = True
                reason = f"Zu weich! Durchbiegung {new_u_max:.2f} > Limit {allowed_deflection:.2f}"
            else:
                undo_needed = False
                reason = "OK"

        if undo_needed:
            for n in nodes_removed_this_step: n.active = True   # Alles rückgängig machen
            self.solve_step()   # Stabiler Zustand wiederherstellen
            return False, reason

        return True, "OK"   # alle Checks bestanden, Optimierungsschritt erfolgreich