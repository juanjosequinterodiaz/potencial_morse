
import numpy as np

LAMBDA = 8.0

Q_MIN = -2.0
Q_MAX = 4.0

N_POINTS = 1000

q = np.linspace(Q_MIN, Q_MAX, N_POINTS)

DQ = q[1] - q[0]


print(f"Numero de puntos: {N_POINTS}")
print(f"Intervalo: [{Q_MIN}, {Q_MAX}]")
print(f"Espaciamiento: {DQ:.6f}")
 