
import matplotlib.pyplot as plt
import numpy as np
from wavefunction import trial_wavefunction, normalize_wavefunction
from propagator import build_propagator
from config import q, LAMBDA,DQ
from potential import (
    morse_potential,
    harmonic_potential,
    morse_ground_energy
)
from evolution import imaginary_time_evolution
from evolution import calculate_energy




psi_T = trial_wavefunction(q)
psi_T = normalize_wavefunction(psi_T)

print("Normalizacion:", np.sum(psi_T**2) * DQ)

# Calcular potenciales
V_morse = morse_potential(q)
V_harmonic = harmonic_potential(q)

# Energia fundamental exacta
E0 = morse_ground_energy()

# # Grafica
# plt.figure(figsize=(9, 5))

# plt.plot(q, V_morse, label="Potencial de Morse")
# ##plt.plot(q, V_harmonic, "--", label="Oscilador armonico")

# plt.axhline(LAMBDA**2, color="gray", linestyle=":", label="Disociacion")

# plt.xlim(-2, 4)
# plt.ylim(0, 100)

# plt.xlabel(r"$q$")
# plt.ylabel(r"$V(q)/E_*$")

# plt.title(f"Potenciales adimensionales (lambda = {LAMBDA})")

# plt.legend()
# plt.grid(alpha=0.3)

# plt.show()

# plt.figure(figsize=(9, 5))

# plt.plot(q, psi_T**2, label=r"$|\psi_T(q)|^2$")
# plt.xlabel(r"$q$")
# plt.ylabel(r"$|\psi_T(q)|^2$")
# plt.title("Densidad de probabilidad inicial")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()

print(f"Lambda: {LAMBDA}")
print(f"Energia fundamental exacta: {E0:.8f} E")

N_STEPS = 4800
DS = 0.00125

K = build_propagator(q, DS)

# print("Dimensiones del propagador:", K.shape)
# print("Propagador simetrico:", np.allclose(K, K.T))
# print("Todos los elementos positivos:", np.all(K > 0))
# print("Elementos finitos:", np.all(np.isfinite(K)))
# print("Valor minimo:", np.min(K))
# print("Valor maximo:", np.max(K))
# print("Cantidad de ceros:", np.sum(K == 0))
# print("Cantidad de negativos:", np.sum(K < 0))

psi_final, history = imaginary_time_evolution(
    K, psi_T, DQ, N_STEPS
)

# print("Normalizacion final:",
#       np.sum(psi_final**2) * DQ)

# print("Tiempo imaginario total:", DS * N_STEPS)

# plt.figure(figsize=(9, 5))

# plt.plot(q, psi_T**2,
#          label="Densidad inicial")

# plt.plot(q, psi_final**2,
#          label="Densidad después de la evolución")

# plt.xlabel(r"$q$")
# plt.ylabel(r"$|\psi(q)|^2$")
# plt.title("Proyección en tiempo imaginario")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()

# plt.figure(figsize=(9, 5))

# steps_to_plot = [0, 10, 25, 50, 100,200,400]

# for step in steps_to_plot:
#     plt.plot(q, history[step]**2,
#              label=f"Paso {step}")

# plt.xlabel(r"$q$")
# plt.ylabel(r"$|\psi(q)|^2$")
# plt.title("Evolución de la densidad de probabilidad")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()

# Energía en función del tiempo imaginario

tau_history = []
energy_history = []

for step, psi in enumerate(history):

    E, E_kin, E_pot = calculate_energy(
        psi, q, DQ, LAMBDA
    )

    tau = step * DS

    tau_history.append(tau)
    energy_history.append(E)


# Gráfica
plt.figure(figsize=(9, 5))

plt.plot(tau_history, energy_history,
         label="Energía numérica")

plt.axhline(E0, color="red", linestyle="--",
            label=f"Energía exacta = {E0:.4f}")

plt.xlabel(r"Tiempo imaginario ($\tau$)")
plt.ylabel(r"Energía $E(\tau)$")
plt.title("Convergencia de la energía en tiempo imaginario")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


print("Energía analítica:", E0)

# Energía durante la proyección

tau_history = []
energy_history = []

for step, psi in enumerate(history):

    E, E_kin, E_pot = calculate_energy(
        psi, q, DQ, LAMBDA
    )

    tau_history.append(step * DS)
    energy_history.append(E)


plt.figure(figsize=(8, 5))

plt.plot(tau_history, energy_history,
         label="Energía numérica")

plt.axhline(E0, color="red", linestyle="--",
            label=f"Energía analítica = {E0:.4f}")

plt.xlabel(r"Tiempo imaginario ($\tau$)")
plt.ylabel(r"$E(\tau)$")
plt.title("Convergencia de la energía")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Evolución de la densidad de probabilidad

steps_to_plot = [0, 100, 400, 800, 1600, 4800]

plt.figure(figsize=(9, 5))

for step in steps_to_plot:
    plt.plot(
        q,
        history[step]**2,
        label=rf"$\tau={step*DS:.2f}$"
    )

plt.xlabel(r"$q$")
plt.ylabel(r"$|\psi(q,\tau)|^2$")
plt.title("Evolución de la densidad de probabilidad")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Comparación de densidades

psi_analytical = (
    np.exp(-(LAMBDA - 0.5)*q)
    * np.exp(-LAMBDA*np.exp(-q))
)

psi_analytical = normalize_wavefunction(psi_analytical)

plt.figure(figsize=(9, 5))

plt.plot(q, history[-1]**2,
         label="Proyección numérica")

plt.plot(q, psi_analytical**2, "--",
         label="Estado fundamental analítico")

plt.xlabel(r"$q$")
plt.ylabel(r"$|\psi(q)|^2$")
plt.title("Comparación de densidades de probabilidad")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()