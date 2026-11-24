import numpy as np


def imaginary_time_evolution(K, psi0, dq, n_steps):

    psi = psi0.copy()

    # Normalización inicial
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dq)

    history = [psi.copy()]

    for _ in range(n_steps):


        # Aplicación del propagador
        psi = dq * (K @ psi)

        # Normalización
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dq)
        psi /= norm

        history.append(psi.copy())

    return psi, np.array(history)


def calculate_energy(psi, q, dq, lam):
    
    V = lam**2 * (1 - np.exp(-q))**2

    # Derivada espacial de la función de onda
    dpsi = np.gradient(psi, dq)

    # Energía cinética y potencial
    E_kin = np.sum(dpsi**2) * dq
    E_pot = np.sum(V * psi**2) * dq

    return E_kin + E_pot, E_kin, E_pot