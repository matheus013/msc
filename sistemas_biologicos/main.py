"""
Simulação: Circuito RC no Espaço de Estados
Disciplina: Modelagem de Sistemas Biológicos

Sistema:
    ẋ(t) = A·x(t) + B·u(t)
    y(t) = C·x(t) + D·u(t)

Circuito RC com dois capacitores e dois resistores:
    Estados: x = [V1, V2]  (tensões nos capacitores)
    Entrada: u = V          (tensão da fonte)
    Saída:   y = V2         (tensão no segundo capacitor)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================
# PARÂMETROS DO CIRCUITO
# =============================================================
R1 = 1e3    # 1 kΩ
R2 = 2e3    # 2 kΩ
C1 = 1e-6   # 1 µF
C2 = 2e-6   # 2 µF

# =============================================================
# MATRIZES DO ESPAÇO DE ESTADOS
# =============================================================
# Conforme derivado na aula:
#
#  V̇1 = -(1/R1C1 + 1/R2C1)*V1 + (1/R2C1)*V2 + (1/R1C1)*V
#  V̇2 =  (1/R2C2)*V1 - (1/R2C2)*V2

a11 = -(1/(R1*C1) + 1/(R2*C1))
a12 =  1/(R2*C1)
a21 =  1/(R2*C2)
a22 = -1/(R2*C2)

A = np.array([[a11, a12],
              [a21, a22]])

B = np.array([[1/(R1*C1)],
              [0.0      ]])

# Saída: y = V2
C_mat = np.array([[0, 1]])
D_mat = np.array([[0]])

print("=" * 55)
print("  CIRCUITO RC - REPRESENTAÇÃO EM ESPAÇO DE ESTADOS")
print("=" * 55)
print(f"\nParâmetros:")
print(f"  R1 = {R1/1e3:.1f} kΩ  |  R2 = {R2/1e3:.1f} kΩ")
print(f"  C1 = {C1*1e6:.1f} µF  |  C2 = {C2*1e6:.1f} µF")
print(f"\nMatriz A:\n{A}")
print(f"\nMatriz B:\n{B}")
print(f"\nMatriz C: {C_mat}")
print(f"\nAutovalores de A: {np.linalg.eigvals(A)}")

# =============================================================
# SIMULAÇÃO POR EULER (DISCRETIZAÇÃO)
# =============================================================
# Conforme visto na aula:
#   x[k+1] = x[k] + Ts * (A·x[k] + B·u[k])

Ts   = 1e-4          # passo de tempo (100 µs)
tmax = 0.05          # tempo total (50 ms)
t    = np.arange(0, tmax, Ts)
N    = len(t)

# --- Entrada degrau unitário (5 V) ---
V_fonte = 5.0
u = V_fonte * np.ones((1, N))

# Condições iniciais
x = np.zeros((2, N))   # [V1; V2]
y = np.zeros((1, N))

for k in range(N - 1):
    x_k = x[:, k:k+1]
    u_k = u[:, k:k+1]
    # Euler explícito
    x[:, k+1:k+2] = x_k + Ts * (A @ x_k + B @ u_k)

y = C_mat @ x   # saída

V1 = x[0, :]
V2 = x[1, :]

# Valores de regime permanente (teórico)
# Em regime: ẋ = 0  →  x_ss = -A⁻¹ B u
x_ss = -np.linalg.inv(A) @ B @ np.array([[V_fonte]])
print(f"\nRegime permanente teórico:")
print(f"  V1_ss = {x_ss[0,0]:.4f} V")
print(f"  V2_ss = {x_ss[1,0]:.4f} V")

# =============================================================
# VISUALIZAÇÃO
# =============================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Simulação: Circuito RC no Espaço de Estados\n"
             "Modelagem de Sistemas Biológicos", fontsize=14, fontweight='bold')

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# --- Plot 1: Tensões V1 e V2 ---
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t*1e3, V1, color='steelblue', linewidth=2, label='$V_1(t)$ — Capacitor C₁')
ax1.plot(t*1e3, V2, color='tomato',    linewidth=2, label='$V_2(t)$ — Capacitor C₂')
ax1.axhline(x_ss[0,0], color='steelblue', linestyle='--', alpha=0.5, label=f'$V_{{1,ss}}$ = {x_ss[0,0]:.2f} V')
ax1.axhline(x_ss[1,0], color='tomato',    linestyle='--', alpha=0.5, label=f'$V_{{2,ss}}$ = {x_ss[1,0]:.2f} V')
ax1.axhline(V_fonte,   color='gray',      linestyle=':',  alpha=0.7, label=f'Entrada = {V_fonte} V')
ax1.set_xlabel('Tempo (ms)')
ax1.set_ylabel('Tensão (V)')
ax1.set_title('Resposta ao Degrau — Estados do Sistema')
ax1.legend(loc='center right', fontsize=9)

# --- Plot 2: Derivadas (velocidades dos estados) ---
ax2 = fig.add_subplot(gs[1, 0])
dV1 = np.diff(V1) / Ts
dV2 = np.diff(V2) / Ts
ax2.plot(t[:-1]*1e3, dV1, color='steelblue', linewidth=1.5, label=r'$\dot{V}_1$')
ax2.plot(t[:-1]*1e3, dV2, color='tomato',    linewidth=1.5, label=r'$\dot{V}_2$')
ax2.set_xlabel('Tempo (ms)')
ax2.set_ylabel('dV/dt (V/s)')
ax2.set_title('Derivadas dos Estados')
ax2.legend(fontsize=9)

# --- Plot 3: Plano de fase V1 x V2 ---
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(V1, V2, color='purple', linewidth=2)
ax3.plot(V1[0], V2[0], 'go', markersize=8, label='Início (0,0)')
ax3.plot(x_ss[0,0], x_ss[1,0], 'r*', markersize=12, label='Equilíbrio')
ax3.set_xlabel('$V_1$ (V)')
ax3.set_ylabel('$V_2$ (V)')
ax3.set_title('Plano de Fase $V_1 \\times V_2$')
ax3.legend(fontsize=9)

# --- Plot 4: Diagrama do circuito ---
ax4 = fig.add_subplot(gs[2, 0])
ax4.axis('off')
circuit_text = (
    "DIAGRAMA DO CIRCUITO\n\n"
    "  V ──┬── R₁ ──┬── R₂ ──┬\n"
    "      │        │        │\n"
    "     GND      C₁       C₂\n"
    "              │        │\n"
    "             GND      GND\n\n"
    f"  R₁={R1/1e3:.0f}kΩ  R₂={R2/1e3:.0f}kΩ\n"
    f"  C₁={C1*1e6:.0f}µF   C₂={C2*1e6:.0f}µF"
)
ax4.text(0.05, 0.95, circuit_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# --- Plot 5: Equações do espaço de estados ---
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')
eq_text = (
    "EQUAÇÕES DE ESTADO\n\n"
    "  ẋ(t) = A·x(t) + B·u(t)\n"
    "  y(t) = C·x(t)\n\n"
    "  A[0][0] = -(1/R1C1 + 1/R2C1)"
    f" = {a11:.4e}\n"
    f"  A[0][1] =  1/R2C1           = {a12:.4e}\n"
    f"  A[1][0] =  1/R2C2           = {a21:.4e}\n"
    f"  A[1][1] = -1/R2C2           = {a22:.4e}\n\n"
    f"  B[0]    =  1/R1C1           = {1/(R1*C1):.4e}\n"
    "  B[1]    =  0\n\n"
    "  Discretização (Euler):\n"
    "  x[k+1] = x[k] + Ts·(Ax[k]+Bu[k])\n"
    f"  Ts = {Ts*1e6:.0f} µs"
)
ax5.text(0.05, 0.95, eq_text, transform=ax5.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

plt.savefig('outputs/simulacao_espaco_estados.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\nSimulação concluída! Gráfico salvo.")
