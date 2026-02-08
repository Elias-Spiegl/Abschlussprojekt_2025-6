import numpy as np
import numpy.typing as npt

class PhysicsSolver:
    #Klasse für die Berechnungen

    @staticmethod
    def calculate_element_stiffness(k: float, node_i_pos: np.array, node_j_pos: np.array) -> np.ndarray:

        #Berechnet die globale Steifigkeitsmatrix für ein einzelnes Federelement.

        # Richtungsvektor bestimmen
        vec = node_j_pos - node_i_pos
        length = np.linalg.norm(vec)
        
        if length == 0:
            return np.zeros((4, 4))

        e_n = vec / length # Einheitsvektor

        # Lokale Steifigkeitsmatrix (1D)
        K_local = k * np.array([[1.0, -1.0], [-1.0, 1.0]])

        # Transformationsmatrix O = e_n (outer) e_n
        O = np.outer(e_n, e_n)
                            
        # Globale Elementsteifigkeitsmatrix durch Kronecker-Produkt 4x4 = 2x2 (K_local) kron 2x2 (O)
        K_global_element = np.kron(K_local, O)
                                         
        return K_global_element
                                            
    @staticmethod
    def solve_system(K: npt.NDArray[np.float64], F: npt.NDArray[np.float64], fixed_dofs: list[int], eps=1e-9) -> npt.NDArray[np.float64] | None:
  
        # Löst das lineare Gleichungssystem Ku = F unter Berücksichtigung der Randbedingungen.
    
        assert K.shape[0] == K.shape[1],   "Steifigkeitsmatrix K muss quadratisch sein."
        assert K.shape[0] == F.shape[0],   "Kraftvektor F muss gleichgroß wie K sein."
          
        K_calc = K.copy()   # Kopie der Steifigkeitsmatrix, damit wir die Originale nicht verändern
        F_calc = F.copy()   # Kopie des Kraftvektors
                  
        # Randbedingungen der Lager umsetzen
        # Zeilen und Spalten nullen, Diagonale auf 1 setzen, Kraft auf 0
        for d in fixed_dofs:
            K_calc[d, :] = 0.0
            K_calc[:, d] = 0.0
            K_calc[d, d] = 1.0
            F_calc[d] = 0.0 # Verschiebung ist hier 0

        try:
            u = np.linalg.solve(K_calc, F_calc)
            return u
        except np.linalg.LinAlgError:
            # Regularisierung falls singulär
            K_calc += np.eye(K_calc.shape[0]) * eps
            try:
                u = np.linalg.solve(K_calc, F_calc)
                return u
            except np.linalg.LinAlgError:
                return None