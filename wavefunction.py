import numpy as np
from config import LAMBDA, q, DQ


def trial_wavefunction(q, lam=LAMBDA):
    return (lam / np.pi)**0.25 * np.exp(-lam * q**2 / 2)


def normalize_wavefunction(psi, dq=DQ):
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dq)
    return psi / norm