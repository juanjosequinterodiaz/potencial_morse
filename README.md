[README.md](https://github.com/user-attachments/files/35535562/README.md)
# Proyecto 02 — Potencial de Morse por proyección de integrales de camino euclidianas

Código del artículo *Reconstrucción espectral y espacial de vibraciones moleculares anarmónicas mediante
proyección de integrales de camino euclidianas*.

La integral de camino euclidiana se evalúa de forma determinista (sin Monte Carlo) como potencias de una
matriz de transferencia, y se proyecta desde una gaussiana que no es el estado fundamental. Los estados
excitados se obtienen con ortogonalización de Gram–Schmidt en cada iteración.

## Requisitos

Python 3 con `numpy`, `scipy` y `matplotlib`:

```bash
pip install numpy scipy matplotlib
```

## Uso

```bash
python morse_proyeccion.py              # calcula todo y genera las figuras (~1 min)
python morse_proyeccion.py --figuras    # solo redibuja desde resultados/datos.npz
```

## Salida (`resultados/`)

| Archivo | Contenido |
|---|---|
| `tablas.txt` | Todas las cifras citadas en el artículo |
| `datos.npz` | Datos mínimos para regenerar las figuras |
| `fig1_proyeccion` | Densidad según la duración τ y convergencia en τ (Fig. 1) |
| `fig2_densidad` | Densidad asimétrica frente al oscilador armónico y a Schrödinger (Fig. 2) |
| `fig3_controles` | Paso temporal, caja, nivel n = 3 y desaparición del nivel más alto (Fig. 3) |

## Estructura del script

1. **Parámetros declarados**: D, ε, h, dominios, duraciones, funciones de prueba y tolerancia.
2. **Método**: matriz de transferencia, proyección con Gram–Schmidt, solucionador de diferencias finitas
   (referencia independiente), momentos y criterio de "resuelto".
3. **Cálculos**: verificación con el oscilador armónico, fundamental en dos dominios y dos duraciones, controles
   de paso temporal, rejilla y caja, estados excitados y desaparición del nivel más alto.
4. **Figuras** del artículo.

## Unidades

ħ = μ = a = 1. Longitudes en 1/a, energías en ħ²a²/μ, tiempo imaginario en μ/(ħa²).
V(x) = D(1 − e^{−x})², λ = √(2D). Con D = 8: λ = 4 y cuatro estados ligados (E = 1.875, 4.875, 6.875, 7.875).

## Resultado principal

E₀ extrapolado a ε → 0: 1.875000 ± 0.000004 (exacto 1.875). La densidad del fundamental está desplazada hacia el
lado de disociación: máximo en x = 0.14, media ⟨x⟩ = 0.207 y asimetría γ₁ = 0.39 (oscilador armónico: 0, 0 y 0).
