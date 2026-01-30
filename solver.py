import numpy as np
import numpy.typing as npt

def solve(K: npt.NDArray[np.float64], F: npt.NDArray[np.float64], u_fixed_idx: list[int], eps=1e-9) -> npt.NDArray[np.float64] | None:
    """Solve the linear system Ku = F with fixed boundary conditions.

    Parameters
    ----------
    K : npt.NDArray[np.float64]
        Stiffness matrix.
    F : npt.NDArray[np.float64]
        Force vector.
    u_fixed_idx : list[int]
        List of indices where the displacement is fixed (Dirichlet boundary conditions).
    eps : float, optional
        Regularization parameter to avoid singular matrix, by default 1e-9

    Returns
    -------
    npt.NDArray[np.float64] | None
        Displacement vector or None if the system is unsolvable.
    """

    assert K.shape[0] == K.shape[1], "Stiffness matrix K must be square."
    assert K.shape[0] == F.shape[0], "Force vector F must have the same size as K."

    for d in u_fixed_idx:
        K[d, :] = 0.0
        K[:, d] = 0.0
        K[d, d] = 1.0

    try:
        u = np.linalg.solve(K, F) # solve the linear system Ku = F
        u[u_fixed_idx] = 0.0

        return u
    
    except np.linalg.LinAlgError:
        # If the stiffness matrix is singular we can try a small regularization to still get a solution
        K += np.eye(K.shape[0]) * eps

        try:
            u = np.linalg.solve(K, F) # solve the linear system Ku = F
            u[u_fixed_idx] = 0.0

            return u
        
        except np.linalg.LinAlgError:
            # If it is still singular we give up
            return None

def test_case_horizontal():
    # Horizontal spring element between two nodes i and j
    e_n = np.array([1.0, 0.0])
    e_n = e_n / np.linalg.norm(e_n)

    k = 1.0
    K = k * np.array([[1.0, -1.0], [-1.0, 1.0]])
    print(f"{K=}")

    O = np.outer(e_n, e_n)
    print(f"{O=}")

    Ko = np.kron(K, O)
    print(f"{Ko=}")

    u_fixed_idx = [0, 1] # fix node i in both directions

    F = np.array([0.0, 0.0, 10.0, 0.0]) # apply force at node j in x-direction

    u = solve(Ko, F, u_fixed_idx)
    print(f"{u=}")

def test_case_diagonal():
    # Diagonal spring element at 45° between two nodes i and j
    e_n = np.array([1.0, 1.0])
    e_n = e_n / np.linalg.norm(e_n)

    k = 1.0 / np.sqrt(2.0) # diagonal spring has less stiffness
    K = k * np.array([[1.0, -1.0], [-1.0, 1.0]])
    print(f"{K=}")

    O = np.outer(e_n, e_n)
    print(f"{O=}")

    Ko = np.kron(K, O)
    print(f"{Ko=}")

    u_fixed_idx = [0, 1] # fix node i in both directions

    F = np.array([0.0, 0.0, 1.0, 1.0]) # apply force at node j in diagonal direction

    u = solve(Ko, F, u_fixed_idx)
    print(f"{u=}")

if __name__ == "__main__":

    test_case_horizontal()

    test_case_diagonal()

