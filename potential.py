
import numpy as np

from config import LAMBDA


def morse_potential(q,lam = LAMBDA):
    

    return (lam**2)*(1 - np.exp(-q))**2


def harmonic_potential(q):


    return q**2


def morse_energy(n, lam=LAMBDA):
    
    return (2*lam * (n + 1/2)) - (n + 1/2)**2 


def morse_ground_energy(lam=LAMBDA):
    
    return morse_energy(0, lam)