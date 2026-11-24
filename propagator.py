import numpy as np
from config import LAMBDA, DQ
from potential import morse_potential


def build_propagator(q, ds, lam=LAMBDA):
    
    V = morse_potential(q)

    # Diferencias entre todos los pares de puntos
    dq_matrix = q[:, None] - q[None, :]

    
    kinetic = -dq_matrix**2 / (4 * ds)

    potential = -ds * (V[:, None] + V[None, :]) / 2

    # Propagador
    K = (1 / np.sqrt(4 * np.pi * ds)) * np.exp(
        kinetic + potential
    )

    return K