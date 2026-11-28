"""
Proyecto 02 - Potencial de Morse y estados ligados moleculares
Reconstruccion espectral y espacial por proyeccion de integrales de camino euclidianas.

Un solo script con todo el trabajo del articulo:
    1. Parametros declarados
    2. Metodo: matriz de transferencia, proyeccion, Gram-Schmidt, referencia por diferencias finitas
    3. Calculos: verificacion, fundamental, controles (tau, eps, caja, rejilla), excitados, desaparicion
    4. Figuras del articulo

Uso:
    python morse_proyeccion.py              # calcula todo y dibuja (~1.5 min)
    python morse_proyeccion.py --figuras    # solo redibuja a partir de resultados/datos.npz

Requisitos: numpy, scipy, matplotlib.
Salida (carpeta resultados/):
    tablas.txt          todas las cifras citadas en el articulo
    datos.npz           datos minimos para regenerar las figuras
    fig1_proyeccion     densidad segun tau y convergencia en tau
    fig2_densidad       figura central: densidad asimetrica frente al oscilador armonico
    fig3_controles      paso temporal, caja, nivel n=3 y desaparicion del nivel mas alto

Unidades: hbar = mu = a = 1. Longitudes en 1/a, energias en hbar^2 a^2/mu, tiempo imaginario en mu/(hbar a^2).
Potencial: V(x) = D (1 - e^{-x})^2, lambda = sqrt(2 D).

Referencias del metodo: Lepage (hep-lat/0506036), Lehner (Lattice QCD I, cap. 1),
Thirumalai, Bruskin y Berne, J. Chem. Phys. 79, 5063 (1983); Berne y Thirumalai, Annu. Rev. Phys. Chem. 37, 401 (1986).
"""
import os
import sys
import numpy as np
from scipy.linalg import eigh_tridiagonal

# =============================================================================================
# 1. PARAMETROS DECLARADOS
# =============================================================================================
D = 8.0                        # profundidad del pozo -> lambda = 4, cuatro estados ligados

EPS = 0.0125                   # ancho de una rebanada de tiempo imaginario
H = 0.02                       # paso de la rejilla espacial
X_MIN = -1.5                   # borde izquierdo (dentro de la pared: V(-1.5) ~ 97 >> D)
DOMINIOS = [12.0, 20.0]        # bordes derechos de los dos dominios declarados
TAUS = [2.0, 5.0]              # las dos duraciones de proyeccion del FUNDAMENTAL
TAU_REF = 10.0                 # duracion de referencia para medir la convergencia en tau

PRUEBA_FUND = (0.6, 0.8)       # funcion de prueba del fundamental: gaussiana (centro, ancho), NO exacta
PRUEBA_EXC = (1.0, 1.5)        # funcion de prueba de los excitados (mas ancha)

TAU_EXC = [10.0, 40.0]         # duraciones en que se evalua el criterio de "resuelto" (excitados)
TOL = 1e-4                     # tolerancia del criterio

EPS_LISTA = [0.05, 0.025, 0.0125, 0.00625]      # control del paso temporal
XMAX_LISTA = [6.0, 8.0, 10.0, 12.0, 16.0, 20.0]  # control de caja
H_LISTA = [0.04, 0.02, 0.01]                     # control de rejilla

D_LISTA = [8.0, 7.0, 6.5, 6.2, 6.0]              # desaparicion del nivel mas alto
DOMINIOS_D = [20.0, 40.0]
H_D = 0.04                     # rejilla mas gruesa para cajas grandes (ver control de rejilla)

CARPETA = "resultados"


# =============================================================================================
# 2. METODO
# =============================================================================================
def potencial(Dv):
    """V(x) = D (1 - e^{-x})^2."""
    return lambda x: Dv * (1 - np.exp(-x))**2


def energias_exactas(Dv):
    """Niveles ligados exactos: E_n = lambda (n+1/2) - (n+1/2)^2/2, con n + 1/2 < lambda = sqrt(2 D)."""
    lam = np.sqrt(2 * Dv)
    n = np.arange(int(np.ceil(lam - 0.5)))
    return lam * (n + 0.5) - (n + 0.5)**2 / 2


def psi0_exacta(x, Dv):
    """Fundamental exacto (sin normalizar): psi_0 ~ xi^{lambda-1/2} e^{-xi/2}, xi = 2 lambda e^{-x}."""
    lam = np.sqrt(2 * Dv)
    xi = 2 * lam * np.exp(-x)
    return np.exp((lam - 0.5) * np.log(xi) - xi / 2)


def rejilla(xmax, xmin=X_MIN, h=H):
    return np.arange(xmin, xmax + h / 2, h)


def gaussiana(x, centro, ancho):
    return np.exp(-(x - centro)**2 / (2 * ancho**2))


def matriz_transferencia(x, V, eps):
    """Una rebanada de la integral de camino sobre la rejilla (regla del trapecio):
        M[i,j] = h K_eps(x_i, x_j),  K_eps(x,y) = (2 pi eps)^(-1/2) exp[-(x-y)^2/(2 eps) - eps (V(x)+V(y))/2].
    Aplicar M N veces = integral de camino con N rebanadas. M es simetrica."""
    h = x[1] - x[0]
    Vx = V(x)
    dx2 = (x[:, None] - x[None, :])**2
    return np.exp(-dx2 / (2 * eps) - 0.5 * eps * (Vx[:, None] + Vx[None, :])) / np.sqrt(2 * np.pi * eps) * h


def proyectar(M, psi0, eps, tau, inferiores=()):
    """Proyeccion en tiempo imaginario con Gram-Schmidt.
    En cada iteracion: (1) phi = M psi; (2) se resta a phi su componente sobre cada estado 'inferior'
    ya obtenido; (3) psi = phi/||phi||. La operacion (2) tambien se aplica a la funcion de prueba.
    Para el fundamental, 'inferiores' esta vacio y (2) no hace nada.
    Devuelve la energia E(tau_k) = -ln(psi.M psi)/eps (cociente de Rayleigh de M) y psi en cada paso."""
    Q = np.array(inferiores).T if len(inferiores) else np.zeros((len(psi0), 0))
    quitar = lambda v: v - Q @ (Q.T @ v)
    psi = quitar(psi0)
    psi = psi / np.linalg.norm(psi)
    E, P = [], [psi]
    for _ in range(int(round(tau / eps))):
        Mpsi = M @ psi
        E.append(-np.log(psi @ Mpsi) / eps)
        psi = quitar(Mpsi)                  # se repite en CADA paso: el redondeo reintroduce los inferiores
        psi = psi / np.linalg.norm(psi)
        P.append(psi)
    E.append(-np.log(psi @ (M @ psi)) / eps)
    return np.array(E), np.array(P)


def escalera(M, x, eps, nniveles, tau_bajos, tau_alto, prueba):
    """Proyecta los niveles 0..nniveles-1 en orden; cada uno usa los anteriores como 'inferiores'.
    El ultimo nivel se proyecta con tau_alto. Devuelve las historias E(tau) y las funciones finales."""
    psi0 = gaussiana(x, *prueba)
    hist, estados = [], []
    for n in range(nniveles):
        tau = tau_alto if n == nniveles - 1 else tau_bajos
        E, P = proyectar(M, psi0, eps, tau, estados)
        hist.append(E)
        estados.append(P[-1])
    return hist, estados


def schrodinger_df(x, V, k):
    """Referencia independiente: diferencias finitas de tres puntos, psi = 0 en los bordes."""
    h = x[1] - x[0]
    E, U = eigh_tridiagonal(1 / h**2 + V(x), -1 / (2 * h**2) * np.ones(len(x) - 1),
                            select="i", select_range=(0, k - 1))
    return E, U * np.sign(U[np.argmax(np.abs(U), axis=0), np.arange(k)])


def momentos(x, psi):
    """<x>, sigma^2 y asimetria gamma_1 de rho = psi^2."""
    p = psi**2 / np.sum(psi**2)
    m = p @ x
    s2 = p @ (x - m)**2
    return m, s2, (p @ (x - m)**3) / s2**1.5


def resuelto(E_a, E_b, eps, tol, ventana=5.0):
    """Criterio de 'resuelto' con las historias E(tau) de dos dominios:
    (a) estabilidad: cambio < tol en las ultimas 'ventana' unidades de tau, en ambos dominios;
    (b) independencia del dominio: los dos dominios coinciden a menos de tol."""
    k = int(round(ventana / eps))
    cambio = max(abs(E_a[-1] - E_a[-1 - k]), abs(E_b[-1] - E_b[-1 - k]))
    dif = abs(E_a[-1] - E_b[-1])
    return cambio, dif, (cambio < tol and dif < tol)


def tau_necesario(E, eps, tol):
    """Primer tau a partir del cual E(tau) queda a menos de tol de su valor final."""
    fuera = np.where(np.abs(E - E[-1]) > tol)[0]
    return eps * (fuera[-1] + 1) if len(fuera) else 0.0


# =============================================================================================
# 3. CALCULOS
# =============================================================================================
def calcular():
    os.makedirs(CARPETA, exist_ok=True)
    salida = open(os.path.join(CARPETA, "tablas.txt"), "w")

    def log(s=""):
        print(s)
        salida.write(s + "\n")

    V = potencial(D)
    Ex = energias_exactas(D)
    datos = {}

    # ---- 3.1 Verificacion con un caso conocido ---------------------------------------------
    log("1. VERIFICACION CON EL OSCILADOR ARMONICO (V = x^2/2: E0 = 0.5, <x> = 0)")
    xo = rejilla(8.0, -8.0)
    Vo = lambda t: 0.5 * t**2
    Eo, Po = proyectar(matriz_transferencia(xo, Vo, EPS), gaussiana(xo, 1.0, 0.6), EPS, TAU_REF)
    log(f"   E0(eps={EPS}, tau=10) = {Eo[-1]:.6f}   <x> = {momentos(xo, Po[-1])[0]:+.1e}")
    E0o = [proyectar(matriz_transferencia(xo, Vo, e), gaussiana(xo, 1.0, 0.6), e, TAU_REF)[0][-1] for e in EPS_LISTA]
    log(f"   extrapolacion eps -> 0 (recta en eps^2, 3 menores): E0 = "
        f"{np.polyfit(np.array(EPS_LISTA[1:])**2, E0o[1:], 1)[1]:.6f}")

    # ---- 3.2 Fundamental: dos dominios x dos duraciones ------------------------------------
    log(f"\n2. FUNDAMENTAL DE MORSE (D = {D}, eps = {EPS}, h = {H})")
    log(f"   niveles exactos E_n = {np.round(Ex, 4)};  espaciamientos exactos = {np.diff(Ex)};"
        f"  armonico: {np.sqrt(2*D):.0f} constante")
    xr = rejilla(20.0)
    mx, s2x, gx = momentos(xr, psi0_exacta(xr, D))
    log(f"   momentos exactos: <x> = {mx:.5f}  sigma^2 = {s2x:.5f}  gamma1 = {gx:.4f}")
    w = np.sqrt(2 * D)
    log(f"   oscilador armonico de igual curvatura (omega = {w:.0f}): E0 = {w/2:.3f}, <x> = 0, "
        f"sigma = {1/np.sqrt(2*w):.3f}, gamma1 = 0;  E0_Morse - E0_arm = {Ex[0]-w/2:+.3f}")
    log("   dominio  tau   E0(tau)     <x>       sigma^2   gamma1")
    for L in DOMINIOS:
        x = rejilla(L)
        M = matriz_transferencia(x, V, EPS)
        E, P = proyectar(M, gaussiana(x, *PRUEBA_FUND), EPS, TAU_REF)
        for t in TAUS:
            k = int(round(t / EPS))
            m_, s2_, g_ = momentos(x, P[k])
            log(f"   {L:5.0f}   {t:3.0f}   {E[k]:.6f}   {m_:.5f}   {s2_:.5f}   {g_:.4f}")
        if L == DOMINIOS[0]:
            Edf, Udf = schrodinger_df(x, V, 1)
            k5 = int(round(5 / EPS))
            rho5 = P[k5]**2 / H
            acum = np.cumsum(rho5) * H
            log(f"   tau = 5: maximo de rho en x = {x[np.argmax(rho5)]:.2f}, mediana en x = "
                f"{x[np.searchsorted(acum, 0.5)]:.2f}, media <x> = {np.sum(x*rho5)*H:.3f}, "
                f"sigma = {np.sqrt(momentos(x, P[k5])[1]):.3f}")
            log(f"   DF: E0 = {Edf[0]:.6f} (error O(h^2): {Edf[0]-Ex[0]:+.1e});  "
                f"max |rho_proy(tau=5) - rho_DF| = {np.max(np.abs(rho5 - Udf[:, 0]**2 / H)):.1e}")
            log("   convergencia en tau:")
            for t in [1.0, 2.0, 3.0, 5.0]:
                k = int(round(t / EPS))
                log(f"      tau = {t:.0f}:  E(tau) - E(10) = {E[k]-E[-1]:.1e}   "
                    f"||psi(tau) - psi(10)|| = {np.linalg.norm(P[k]-P[-1]):.1e}")
            datos.update(x=x, tau=EPS * np.arange(len(E)), E_tau=E,
                         dist_tau=np.linalg.norm(P - P[-1], axis=1),
                         tau_snap=np.array([0, 0.25, 0.5, 1.0, 5.0]),
                         rho_tau=np.array([P[int(round(t / EPS))]**2 / H for t in [0, 0.25, 0.5, 1.0, 5.0]]),
                         rho_df=Udf[:, 0]**2 / H)

    # ---- 3.3 Control del paso temporal --------------------------------------------------------
    log("\n3. CONTROL DEL PASO TEMPORAL: E0(tau=5) contra eps^2 (dominio 12)")
    x = rejilla(DOMINIOS[0])
    E0e = []
    for e in EPS_LISTA:
        E0e.append(proyectar(matriz_transferencia(x, V, e), gaussiana(x, *PRUEBA_FUND), e, 5.0)[0][-1])
        log(f"   eps = {e:<8} E0 = {E0e[-1]:.7f}   E0 - exacto = {E0e[-1]-Ex[0]:+.2e}")
    err = np.array(E0e) - Ex[0]
    log(f"   cociente del error al dividir eps por 2: {np.round(err[1:]/err[:-1], 4)} (esperado 0.25)")
    e2 = np.array(EPS_LISTA)**2
    c4, a4 = np.polyfit(e2, E0e, 1)
    c3, a3 = np.polyfit(e2[1:], E0e[1:], 1)
    log(f"   recta con 4 puntos: E0(eps->0) = {a4:.7f};  con los 3 menores: {a3:.7f}")
    log(f"   => E0 = {a3:.6f} +/- {abs(a3-a4):.1e}  (exacto 1.875; desviacion {a3-Ex[0]:+.1e})")
    datos.update(eps=np.array(EPS_LISTA), E0_eps=np.array(E0e), fit3=np.array([c3, a3]))

    # ---- 3.4 Control de rejilla -----------------------------------------------------------------
    log("\n4. CONTROL DE REJILLA: E0(tau=5, eps=0.0125, dominio 12)")
    E0h = []
    for h in H_LISTA:
        xh = rejilla(DOMINIOS[0], h=h)
        E0h.append(proyectar(matriz_transferencia(xh, V, EPS), gaussiana(xh, *PRUEBA_FUND), EPS, 5.0)[0][-1])
        log(f"   h = {h}: E0 = {E0h[-1]:.10f}")
    log(f"   variacion maxima: {max(E0h)-min(E0h):.1e}")

    # ---- 3.5 Control de caja --------------------------------------------------------------------
    log("\n5. CONTROL DE CAJA (borde derecho): E0 y E3 proyectados")
    cajas = []
    for L in XMAX_LISTA:
        x = rejilla(L)
        hist, _ = escalera(matriz_transferencia(x, V, EPS), x, EPS, 4, TAU_REF, max(TAU_EXC), PRUEBA_EXC)
        cajas.append([L, hist[0][-1], hist[3][-1]])
        log(f"   x_max = {L:4.0f}   E0 = {hist[0][-1]:.13f}   E3 = {hist[3][-1]:.6f}")
    cajas = np.array(cajas)
    log(f"   max |E0(x_max) - E0(20)| = {np.max(np.abs(cajas[:-1,1]-cajas[-1,1])):.1e}")
    log(f"   longitud de la cola 1/kappa_n = {np.round(1/np.sqrt(2*(D-Ex)), 2)} para n = 0..3")
    datos["cajas"] = cajas

    # ---- 3.6 Excitados por Gram-Schmidt ---------------------------------------------------------
    log(f"\n6. EXCITADOS POR GRAM-SCHMIDT (dominios {DOMINIOS}, prueba centro {PRUEBA_EXC[0]} ancho {PRUEBA_EXC[1]})")
    Hh = {}
    for L in DOMINIOS:
        x = rejilla(L)
        hist, est = escalera(matriz_transferencia(x, V, EPS), x, EPS, 4, max(TAU_EXC), max(TAU_EXC), PRUEBA_EXC)
        Hh[L] = hist
        Edf, Udf = schrodinger_df(x, V, 5)
        log(f"   dominio {L:.0f}: primer estado de caja (DF) E4 = {Edf[4]:.4f};  hueco E4 - E3 = {Edf[4]-Edf[3]:.3f}")
        if L == DOMINIOS[1]:
            S = np.array(est)
            g = gaussiana(x, *PRUEBA_EXC)
            log(f"   validacion: max |<psi_n|psi_m>| = {np.max(np.abs(S @ S.T - np.eye(4))):.0e};  "
                f"|<psi_n|phi_n DF>|^2 = {np.round([(S[n] @ Udf[:, n])**2 for n in range(4)], 6)}")
            log(f"   peso de la funcion de prueba en cada nivel: {np.round((Udf.T @ (g/np.linalg.norm(g)))[:4]**2, 4)}")
    for t in TAU_EXC:
        k = int(round(t / EPS)) + 1
        log(f"   criterio con tau = {t:.0f}:")
        for n in range(4):
            ch, df, ok = resuelto(Hh[DOMINIOS[0]][n][:k], Hh[DOMINIOS[1]][n][:k], EPS, TOL)
            log(f"      n = {n}: E = {Hh[DOMINIOS[1]][n][k-1]:.6f} (exacto {Ex[n]:.3f})  cambio en 5 tau = {ch:.1e}"
                f"  dif. dominios = {df:.1e}  {'RESUELTO' if ok else 'NO RESUELTO'}")
    log("   tau a partir del cual E_n queda a menos de TOL de su valor final (peor dominio): " +
        "  ".join(f"n={n}: {max(tau_necesario(Hh[L][n], EPS, TOL) for L in DOMINIOS):.2f}" for n in range(4)))
    Ef = [Hh[DOMINIOS[1]][n][-1] for n in range(4)]
    log(f"   espaciamientos (tau = 40): {np.round(np.diff(Ef), 5)}   exactos {np.diff(Ex)}")
    datos.update(tau_exc=EPS * np.arange(len(Hh[DOMINIOS[1]][3])),
                 E3_12=Hh[DOMINIOS[0]][3], E3_20=Hh[DOMINIOS[1]][3])

    # ---- 3.7 Desaparicion del nivel mas alto ----------------------------------------------------
    log(f"\n7. DESAPARICION DEL NIVEL n = 3 (h = {H_D}, dominios {DOMINIOS_D}, tau = 40)")
    log("   D     lambda  cola 1/k  (E3-D)exacto   (E3-D)proy   (E3-D)DF(40)  cambio   dif.dom.  estado")
    desap = []
    for Dv in D_LISTA:
        Vd = potencial(Dv)
        lam = np.sqrt(2 * Dv)
        hh = {}
        for L in DOMINIOS_D:
            x = rejilla(L, h=H_D)
            hh[L] = escalera(matriz_transferencia(x, Vd, EPS), x, EPS, 4, TAU_REF, max(TAU_EXC), PRUEBA_EXC)[0][3]
        ch, df, ok = resuelto(hh[DOMINIOS_D[0]], hh[DOMINIOS_D[1]], EPS, TOL)
        existe = lam > 3.5
        ex = (lam * 3.5 - 3.5**2 / 2) - Dv if existe else np.nan
        Ep = hh[DOMINIOS_D[1]][-1] - Dv
        Edf = schrodinger_df(rejilla(DOMINIOS_D[1], h=0.01), Vd, 4)[0][3] - Dv
        cola = f"{1/(lam-3.5):7.1f}" if existe else "     --"
        estado = "resuelto" if ok else ("no resuelto: E > D" if Ep > 0 else "no resuelto: no converge en tau")
        log(f"   {Dv:4.1f}  {lam:.3f}  {cola}   {ex:+.4e}   {Ep:+.4e}  {Edf:+.4e}   {ch:.1e}  {df:.1e}  {estado}")
        desap.append([Dv, ex, Ep, ok])
    datos["desap"] = np.array(desap, dtype=float)

    np.savez(os.path.join(CARPETA, "datos.npz"), **datos)
    salida.close()


# =============================================================================================
# 4. FIGURAS
# =============================================================================================
def figuras():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(os.path.join(CARPETA, "datos.npz"))
    Ex = energias_exactas(D)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "font.family": "serif", "mathtext.fontset": "cm",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                         "lines.linewidth": 1.2, "legend.frameon": False})
    AZ, NA, GR, NE = "#2a78d6", "#eb6834", "#898781", "#0b0b0b"
    RAMPA = ["#86b6ef", "#5598e7", "#2a78d6", "#0d366b"]
    UX, UE, UT = r"$x\ [1/a]$", r"$[\hbar^2a^2/\mu]$", r"$\tau\ [\mu/\hbar a^2]$"

    def panel(ax, letra):
        ax.text(-0.16, 1.03, f"({letra})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")

    def guardar(fig, nombre):
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(CARPETA, f"{nombre}.{ext}"), bbox_inches="tight", dpi=300)
        plt.close(fig)

    x = d["x"]

    # Fig. 1: proyeccion del fundamental
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.4, 3.9), gridspec_kw=dict(hspace=0.5))
    a.plot(x, d["rho_tau"][0], "--", color=GR, lw=1, label=r"$\tau=0$ (prueba)")
    for t, r, c in zip(d["tau_snap"][1:], d["rho_tau"][1:], RAMPA):
        a.plot(x, r, color=c, label=rf"$\tau={t:g}$")
    a.axvline(0, color=GR, lw=0.5, ls=":")
    a.set_xlim(-1, 3); a.set_ylim(0, 1.15); a.set_xlabel(UX); a.set_ylabel(r"$\rho(x,\tau)$")
    a.legend(loc="upper right")
    panel(a, "a")
    tau, dE, dpsi = d["tau"], d["E_tau"] - d["E_tau"][-1], d["dist_tau"]
    gap = Ex[1] - Ex[0]
    m = (dE > 1e-13) & (tau <= 6)
    b.semilogy(tau[m], dE[m], color=AZ, label=r"$E(\tau)-E(10)$")
    m2 = (dpsi > 1e-13) & (tau <= 6)
    b.semilogy(tau[m2], dpsi[m2], color=NA, label=r"$\|\psi(\tau)-\psi(10)\|$")
    k = int(round(0.5 / EPS))
    tt = tau[tau <= 6]
    b.semilogy(tt, dE[k] * np.exp(-2 * gap * (tt - 0.5)), ":", color=NE, lw=0.8,
               label=r"$e^{-2\Delta\tau}$ y $e^{-\Delta\tau}$, $\Delta=3$")
    b.semilogy(tt, dpsi[k] * np.exp(-gap * (tt - 0.5)), ":", color=NE, lw=0.8)
    for t in TAUS:
        b.axvline(t, color=GR, lw=0.6, ls="--")
    b.set_xlim(0, 6); b.set_ylim(1e-13, 2); b.set_xlabel(UT); b.set_ylabel(r"diferencia con $\tau=10$")
    b.legend(loc="lower left")
    panel(b, "b")
    guardar(fig, "fig1_proyeccion")

    # Fig. 2: figura central
    fig, a = plt.subplots(figsize=(3.4, 2.5))
    w = np.sqrt(2 * D)
    rho = d["rho_tau"][-1]
    a.plot(x, np.sqrt(w / np.pi) * np.exp(-w * x**2), "--", color=GR, lw=1, label=r"oscilador armónico ($\omega=4$)")
    a.plot(x, d["rho_df"], color=NA, lw=3, alpha=0.5, label="Schrödinger (DF)")
    a.plot(x, rho, color=AZ, label=r"proyección, $\tau=5$")
    xm = np.sum(x * rho) * H
    imax = np.argmax(rho)
    a.axvline(0, color=GR, lw=0.5, ls=":"); a.axvline(xm, color=AZ, lw=0.6, ls=":")
    a.plot(x[imax], rho[imax], "o", color=NE, ms=3.5, zorder=5)
    a.text(1.25, 0.5, rf"máximo: $x={x[imax]:.2f}$" "\n" rf"media: $\langle x\rangle={xm:.3f}$" "\n"
           r"asimetría: $\gamma_1=0.39$" "\n(armónico: 0, 0 y 0)", fontsize=7)
    a.set_xlim(-1, 3); a.set_ylim(0, 1.25); a.set_xlabel(UX); a.set_ylabel(r"$\rho_0(x)$")
    a.legend(loc="upper right")
    guardar(fig, "fig2_densidad")

    # Fig. 3: controles y nivel mas alto
    fig, ax = plt.subplots(2, 2, figsize=(6.4, 4.3), gridspec_kw=dict(wspace=0.38, hspace=0.55))
    ax = ax.ravel()
    a = ax[0]                                                   # (a) paso temporal
    e2 = d["eps"]**2
    c3, a3 = d["fit3"]
    s = np.linspace(0, e2.max() * 1.05, 50)
    a.plot(e2 * 1e3, d["E0_eps"], "o", color=AZ, ms=4)
    a.plot(s * 1e3, a3 + c3 * s, "-", color=AZ, lw=0.8)
    a.axhline(Ex[0], color=GR, ls="--", lw=0.8)
    a.text(1.0, Ex[0] - 5e-4, "exacto", fontsize=6.5, color=GR)
    a.set_xlabel(r"$\varepsilon^2\ [10^{-3}]$"); a.set_ylabel(r"$E_0$ " + UE)
    panel(a, "a")
    a = ax[1]                                                   # (b) caja
    cj = d["cajas"]
    dE3 = np.abs(cj[:-1, 2] - cj[-1, 2])
    a.semilogy(cj[:-1, 0], dE3, "o-", color=NE, ms=3, label=r"$n=3$")
    Ls = np.linspace(6, 16, 20)
    a.semilogy(Ls, dE3[2] * np.exp(-2 * 0.5 * (Ls - cj[2, 0])), ":", color=GR, lw=0.8,
               label=r"$\propto e^{-2\kappa_3 x_{\max}}$")
    a.axhline(TOL, color=GR, ls="--", lw=0.6)
    a.text(6.2, 3e-12, r"$n=0$: $<10^{-13}$", fontsize=6.5)
    a.set_ylim(1e-12, 1); a.set_xlabel(r"$x_{\max}\ [1/a]$"); a.set_ylabel(r"$|E_n(x_{\max})-E_n(20)|$", labelpad=1)
    a.legend(loc="upper right")
    panel(a, "b")
    a = ax[2]                                                   # (c) nivel n = 3
    te = d["tau_exc"]
    a.plot(te, d["E3_12"], "--", color=AZ, lw=1, label=r"$x_{\max}=12$")
    a.plot(te, d["E3_20"], color=NA, lw=1, label=r"$x_{\max}=20$")
    a.axhline(Ex[3], color=GR, ls=":", lw=0.8)
    for t in TAU_EXC:
        a.axvline(t, color=GR, lw=0.6, ls="--")
    a.set_xlim(5, 40); a.set_ylim(7.873, 7.881)
    a.set_xlabel(UT); a.set_ylabel(r"$E_3$ " + UE); a.legend(loc="upper right")
    panel(a, "c")
    a = ax[3]                                                   # (d) desaparicion
    ds = d["desap"]
    Dc = np.linspace(6.125, 8.2, 100)
    a.plot(Dc, -(np.sqrt(2 * Dc) - 3.5)**2 / 2, color=GR, label="exacto")
    ok = ds[:, 3] == 1
    a.plot(ds[ok, 0], ds[ok, 2], "o", color=AZ, ms=4, label="resuelto")
    a.plot(ds[~ok, 0], ds[~ok, 2], "x", color=NA, ms=5, mew=1.3, label="no resuelto")
    a.axhline(0, color=GR, ls="--", lw=0.6); a.axvline(6.125, color=GR, ls=":", lw=0.6)
    a.set_ylim(-0.14, 0.03); a.set_xlabel(r"$D$ " + UE); a.set_ylabel(r"$E_3-D$ " + UE)
    a.legend(loc="lower left")
    panel(a, "d")
    guardar(fig, "fig3_controles")
    print(f"Figuras guardadas en {CARPETA}/")


if __name__ == "__main__":
    if "--figuras" not in sys.argv:
        calcular()
    figuras()
