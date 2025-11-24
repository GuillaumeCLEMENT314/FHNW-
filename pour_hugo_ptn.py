import numpy as np
import matplotlib.pyplot as plt

# ---- Physical Constants ----

sigma = 1.082e-13 #m^2 from Table 7, [2]
mu_B = 9.2740100657e-24 # J/T Bohr Magneton
g_F = 0.5006
mu_eff = g_F * mu_B  # g_F depends on hyperfine level (±1/2 for Rb-87)
gamma = 7e9             # Gyromagnetic ratio (Hz/T)
Gamma_natural = 36.
10e6 # Natural linewidth (Hz)

# ---- Function ----
def R_op(I):
    return Gamma_natural * I / (I + I_sat)

# ---- Experimental Parameters that works ----
""" T = 310                    # Cell temperature (K)
P_laser = 2e-6             # Laser power (W)
B0 = 50e-6                 # Static magnetic field (T)
B1 = 10e-6                 # RF magnetic field (T)
theta = np.deg2rad(45)     # Angle (rad)
Gamma = 50e3               # Spin relaxation rate (Hz)
beam_diameter = 3e-3       # Beam diameter (m)
z_cell = 1.6e-3            # Cell length (m)
A_beam = np.pi * (beam_diameter / 2)**2
I_sat = 44.84              # Saturation intensity (W/m²)
R = 0.568                  # Photodiode responsivity (A/W)
 """

T = 273+40                 # Cell temperature (K)
P_laser = 20e-6             # Laser power (W)
B0 = 100e-9                # Static magnetic field (T)
B1 = 100e-6                 # RF magnetic field (T)
theta = np.deg2rad(20)     # Angle (rad) between B0 (0°) and B1 (theta)
Gamma = 50e3               # Spin relaxation rate (Hz)
beam_diameter = 3e-3       # Beam diameter (m)
z_cell = 1.6e-3            # Cell length (m)
A_beam = np.pi * (beam_diameter / 2)**2
I_sat = 44.84              # Saturation intensity (W/m²)
R = 0.568                  # Photodiode responsivity (A/W)


# ---- Derived Quantities ----
n = (1/T)*10**(21.866+4.312-4040/T) * 10**6                # Vapor density
M0 = n * mu_eff                  # Max magnetization
I = P_laser / A_beam             # Laser intensity
R_op_val = R_op(I)              # Optical pumping rate
Gamma_tot = Gamma + R_op_val    # Total relaxation
omega_0 = gamma * B0              # Larmor frequency (rad/s)
omega = omega_0 +0.1*omega_0
delta = (1 / (2 * np.sqrt(2))) * gamma * B1 * np.sin(theta)  # RF coupling
Delta = omega - omega_0                       # Resonant case

# ---- Time Vector ----
t_span = [0, 1000/(omega_0)]
sampling_rate = int(3*omega_0)
t = np.linspace(t_span[0], t_span[1], sampling_rate)  

# ---- Time-Dependent Magnetization ----
My = R_op_val * M0 * delta * np.cos(theta) * np.sqrt(2) * (Gamma_tot * np.sin(omega * t) - Delta * np.cos(omega * t)) / (Gamma_tot * (Gamma_tot**2 + Delta**2 + 2 * delta**2))
Mz = R_op_val * M0 * np.cos(theta) * (Gamma_tot**2 + Delta**2) / (Gamma_tot * (Gamma_tot**2 + Delta**2 + 2 * delta**2))
M_real = np.sin(theta) * My + np.cos(theta) * Mz

# ---- Polarization and Detection ----
P_t = M_real / M0
alpha_0 = n * sigma
alpha_t = alpha_0 * (1 - P_t)
I_out = I * np.exp(-alpha_t * z_cell)
P_out = I_out * A_beam
I_PD = R * P_out  # Time-varying photodiode current
# ---- Plotting ----
%matplotlib widget
plt.figure(figsize=(10, 5))
plt.plot(t * 1e3, I_PD * 1e9)  # ms vs. nA
plt.xlabel("Time [ms]")
plt.ylabel("Photodiode Current [nA]")
plt.title("Simulated OPM Signal in Self-Oscillation Mode")
plt.grid(True)
plt.tight_layout()
plt.show()