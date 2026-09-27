"""
Standalone replica of the Entity pgen print_setup() startup diagnostics.

Edit the USER PARAMETERS block below, then run:  python3 print_params.py

"""

import math

# ============================================================
# USER PARAMETERS  (edit here to match your TOML)
# ============================================================
resolution = [2000, 1000, 1000]
extent     = [[0.0, 750.0], [0.0, 375.0], [0.0, 375.0]]

skindepth0   = 1.0                         # [scales] skindepth0
larmor0      = 0.5                         # [scales] larmor0
mass_ratio   = 1.0                         # species[1].mass / species[0].mass ; 1.0 = pair

cs_density   = 10.0                        # [setup] cs_density  (n_CS/n_BG, total)
cs_width     = 1.0                         # [setup] cs_width    (half-thickness, d0)
guide_field  = 0.0                         # [setup] guide_field (Bg/B0)
bg_theta_i   = 0.1                         # [setup] bg_theta_i  (background theta_i)

runtime      = 2500.0                      # [simulation] runtime

# turbulence (only used if turb_amp > 0; mode COUNT is resolution-independent)
turb_amp       = 0.0                       # [setup] turb_amp (dB/B0 per component)
turb_seed      = 12345                     # [setup] turb_seed (unused for the count)
kmin           = 1                         # [setup] kmin
kmax           = 2                         # [setup] kmax
turb_plane     = 0                         # [setup] turb_plane (0=xy,1=yz,2=zx)
spectral_index = 1.6667                    # [setup] spectral_index
# ============================================================


# ---- geometry ----
D    = len(resolution)                                     # dimensionality from list length
nx1  = resolution[0]
nx2  = resolution[1]
nx3  = resolution[2] if D == 3 else None
xmin, xmax = extent[0]
ymin, ymax = extent[1]
zmin, zmax = (extent[2] if D == 3 else (0.0, 1.0))
Lx   = xmax - xmin
Ly   = ymax - ymin
Lz   = zmax - zmin
cs_y = 0.5 * (ymin + ymax)                                 # sheet at Y centre
dx   = Lx / nx1                                            # uniform Minkowski cell size

is_pair = (mass_ratio == 1.0)

# ---- DERIVED (mirrors make_init, fixed n_per = cs_density/2) ----
n_per     = 0.5 * cs_density                               # per-species peak sheet density (injector HALF split)
sigma_ion = (skindepth0 / larmor0) ** 2 / mass_ratio      # sigma_ion = sigma0 / mass_ratio
B_BG      = math.sqrt(sigma_ion)                           # in-plane background field, B_BG = sqrt(sigma_ion)

beta_d    = B_BG / (2.0 * n_per * cs_width)                # Harris drift 3-velocity (uses n_per, not cs_density)
gamma_d   = 1.0 / math.sqrt(1.0 - beta_d * beta_d)         # drift Lorentz factor
drift_u   = beta_d * gamma_d                               # drift four-velocity

T_cs_i    = (B_BG * B_BG) * gamma_d / (4.0 * n_per)        # current-sheet theta_i (pressure balance)
T_cs_e    = T_cs_i * mass_ratio                            # theta_e^CS = theta_i^CS * mass_ratio
T_bg_i    = bg_theta_i                                     # background theta_i
T_bg_e    = T_bg_i * mass_ratio                            # theta_e^BG = theta_i^BG * mass_ratio


def gamma_mean(theta):                                     # <gamma> of a Maxwell-Juttner at theta (iPIC3D closed form)
    return 1.0 + theta * (6.0 + 15.0 * theta) / (4.0 + 5.0 * theta)


gamma_e_mean = gamma_mean(T_bg_e)                          # electron <gamma> from theta_e
gamma_i_mean = gamma_mean(T_bg_i)                          # sp2 <gamma> from theta_i

d_e       = skindepth0 * math.sqrt(gamma_e_mean)           # electron skin depth = sqrt(<g_e>) * d0
d_i       = skindepth0 * math.sqrt(mass_ratio * gamma_i_mean)   # sp2 skin depth = sqrt(mr*<g_sp2>) * d0
rho_e     = larmor0 * gamma_e_mean                         # electron Larmor = <g_e> * rho_cold
rho_i     = larmor0 * mass_ratio * gamma_i_mean            # sp2 Larmor = mr * <g_sp2> * larmor0
lambda_De = math.sqrt(T_bg_e) * d_e                        # electron Debye (code uses HOT d_e; see note at end)
lambda_Di = math.sqrt(T_bg_i) * d_i                        # sp2 Debye = sqrt(theta_i) * d_i

beta_d_disp  = drift_u / math.sqrt(1.0 + drift_u * drift_u)  # recover beta from u (as print_setup does)
gamma_d_disp = math.sqrt(1.0 + drift_u * drift_u)           # recover gamma from u

t_wpe = runtime                                           # runtime in 1/omega_pe
t_wpi = runtime / math.sqrt(mass_ratio)                   # runtime in 1/omega_p(sp2)
t_Lx  = runtime / Lx                                      # runtime in light-crossing times of Lx

sigma_e_cold = sigma_ion * mass_ratio                     # cold electron sigma
sigma_e_hot  = sigma_e_cold / gamma_e_mean                # hot electron sigma

P_mag = B_BG * B_BG / 2.0                                 # upstream magnetic pressure
P_th  = T_bg_i                                            # upstream thermal pressure (total, both species; n_bg=1)
P_cs  = cs_density * T_cs_i / gamma_d                     # CS thermal pressure (total, both species)
plasma_beta = P_th / P_mag                                # upstream plasma beta


def count_turb_modes():                                   # ported mode COUNT from initial_turbulence (RNG-independent)
    if turb_amp <= 0.0 or kmax < kmin:
        return 0
    vary_x = vary_y = vary_z = False
    if   turb_plane == 0: vary_x, vary_y = True, True
    elif turb_plane == 1: vary_y, vary_z = True, True
    elif turb_plane == 2: vary_z, vary_x = True, True
    k0x, k0y, k0z = 2.0 * math.pi / Lx, 2.0 * math.pi / Ly, 2.0 * math.pi / Lz
    range_x = kmax if vary_x else 0
    range_y = kmax if vary_y else 0
    range_z = kmax if vary_z else 0
    count = 0
    for nx in range(0, range_x + 1):
        for ny in range(-range_y, range_y + 1):
            for nz in range(-range_z, range_z + 1):
                if nx < 0: continue
                if nx == 0 and ny < 0: continue
                if nx == 0 and ny == 0 and nz <= 0: continue
                n_mag = math.sqrt(nx * nx + ny * ny + nz * nz)
                if n_mag < kmin or n_mag > kmax: continue
                kx = nx * k0x if vary_x else 0.0
                ky = ny * k0y if vary_y else 0.0
                kz = nz * k0z if vary_z else 0.0
                if math.sqrt(kx * kx + ky * ky + kz * kz) == 0.0: continue
                count += 1
    return count


n_modes = count_turb_modes()

# ---- formatting helpers (LW1=34, LW2=30; one tab = 4 spaces, as in the C++) ----
def g(x):  return f"{x:g}"                                # ~6 sig figs, like C++ default ostream
def kv1(label):  return "    " + label.ljust(34) + "= "
def kv2(label):  return "        " + label.ljust(30) + "= "

sp2  = "positron"  if is_pair else "ion"
SP2u = "Positrons" if is_pair else "Ions"
sp2s = "positrons" if is_pair else "ions"
title = "Relativistic Harris sheet (pair plasma)" if is_pair else "Relativistic Harris sheet (ion-electron plasma)"

L = []
L.append("")
L.append("===========================================================")
L.append(title)
L.append("===========================================================")
L.append("")
L.append("GEOMETRY")
L.append(kv1("Dimensionality") + f"{D}D")
if D == 3:
    L.append(kv1("Resolution") + f"{nx1} x {nx2} x {nx3}")
    L.append(kv1("Box (Lx x Ly x Lz) [d0]") + f"{g(Lx)} x {g(Ly)} x {g(Lz)}")
else:
    L.append(kv1("Resolution") + f"{nx1} x {nx2}")
    L.append(kv1("Box (Lx x Ly) [d0]") + f"{g(Lx)} x {g(Ly)}")
L.append(kv1("Grid size [d0]") + g(dx))
L.append(kv1("Current sheet at y [d0]") + g(cs_y))
L.append(kv1("Y boundaries") + "REFLECTION")
L.append(kv1("Runtime [cold electron freq.] ") + g(t_wpe))
L.append(kv1("Runtime [cold " + sp2 + " freq.] ") + g(t_wpi))
L.append(kv1("Runtime [c/L]") + g(t_Lx))
L.append("")
L.append("MASS / MAGNETISATION")
L.append(kv1("Mass ratio [m_i/m_e]") + g(mass_ratio))
L.append(kv1("Sigma (" + sp2 + ")") + g(sigma_ion))
L.append(kv1("Sigma (electron, cold)") + g(sigma_e_cold))
L.append(kv1("Sigma (electron, hot)") + g(sigma_e_hot))
L.append(kv1("Mean electron Lorentz factor") + g(gamma_e_mean))
L.append(kv1("B_BG [B0]") + g(B_BG))
L.append(kv1("Guide field [Bg/B0]") + g(guide_field))
L.append("")
L.append("CHARACTERISTIC SCALES")
L.append("    Electrons")
L.append(kv2("Skin depth") + f"{g(d_e)}  (resolved with {g(d_e / dx)} cells)")
L.append(kv2("Larmor radius") + f"{g(rho_e)}  (resolved with {g(rho_e / dx)} cells)")
L.append(kv2("Debye length") + f"{g(lambda_De)}  (resolved with {g(lambda_De / dx)} cells)")
L.append("    " + SP2u)
L.append(kv2("Skin depth") + f"{g(d_i)}  (resolved with {g(d_i / dx)} cells)")
L.append(kv2("Larmor radius") + f"{g(rho_i)}  (resolved with {g(rho_i / dx)} cells)")
L.append(kv2("Debye length") + f"{g(lambda_Di)}  (resolved with {g(lambda_Di / dx)} cells)")
L.append("")
L.append("TEMPERATURES (theta = kT/mc^2)")
L.append("    Background")
L.append(kv2("electrons") + g(T_bg_e))
L.append(kv2(sp2s) + g(T_bg_i))
L.append("    Current Sheet")
L.append(kv2("electrons") + g(T_cs_e))
L.append(kv2(sp2s) + g(T_cs_i))
L.append("")
L.append("TURBULENCE")
L.append(kv1("Amplitude [dB/B0] (per comp)") + g(turb_amp))
L.append(kv1("Mode band [kmin, kmax]") + f"[{kmin}, {kmax}]")
L.append(kv1("Plane (0=xy,1=yz,2=zx)") + f"{turb_plane}")
L.append(kv1("Spectral index [p]") + g(spectral_index))
L.append(kv1("Number of modes") + f"{n_modes}")
L.append("")
L.append("CURRENT SHEET")
L.append(kv1("Overdensity [n_CS/n_BG]") + g(cs_density))
L.append(kv1("Half-thickness [d0]") + g(cs_width))
L.append(kv1("Thermal spread of " + sp2s) + g(T_cs_i))
L.append(kv1("Thermal spread of electrons") + g(T_cs_e))
L.append(kv1("Drift four-velocity [u]") + g(drift_u))
L.append(kv1("beta = v/c ") + g(beta_d_disp))
L.append(kv1("Lorentz factor of particles") + g(gamma_d_disp))
L.append("")
L.append("PRESSURE BALANCE")
L.append(kv1("Thermal pressure (CS)") + g(P_cs))
L.append(kv1("Thermal pressure (upstream)") + g(P_th))
L.append(kv1("Magnetic pressure (upstream)") + g(P_mag))
L.append(kv1("Plasma beta (upstream)") + g(plasma_beta))
L.append("")
L.append("===========================================================")

print("\n".join(L))