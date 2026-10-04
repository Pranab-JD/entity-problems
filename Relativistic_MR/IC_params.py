"""
Standalone replica of the Entity pgen make_init() + print_setup() startup diagnostics.

Edit the USER PARAMETERS block below, then run:  python3 IC_params.py

"""

import math

#! ============================================================
#! USER PARAMETERS  (edit to match your TOML)
#! ============================================================

resolution      = [1000, 500, 50]
extent          = [[0.0, 200.0], [0.0, 100.0], [0.0, 10.0]]

skindepth0      = 1.0        # [scales] skindepth0
larmor0         = 0.5        # [scales] larmor0
mass_ratio      = 1.0        # species[1].mass / species[0].mass ; 1.0 = pair

ppc0            = 10.0       # [particles] ppc0  (TOTAL over both species, per cell, at n_hat = 1)

cs_density      = 20.0       # [setup] cs_density  (n_CS/n_BG, TOTAL over both species)
cs_width        = 0.5        # [setup] cs_width    (half-thickness, d0)
guide_field     = 0.0        # [setup] guide_field (Bg/B0)
bg_theta_i      = 0.01       # [setup] bg_theta_i  (background theta of species 2)
inject_y        = False      # [setup] inject_y

runtime         = 200.0      # [simulation] runtime

turb_amp        = 0.0        # [setup] turb_amp (dB/B0 per component; 0 disables the rest)
kmin            = 1          # [setup] kmin
kmax            = 4          # [setup] kmax
turb_plane      = 0          # [setup] turb_plane (0=xy, 1=yz, 2=zx)
spectral_index  = 1.6667     # [setup] spectral_index

#! ============================================================


# ---- geometry ----
D   = len(resolution)
nx1, nx2 = resolution[0], resolution[1]
nx3 = resolution[2] if D == 3 else None
xmin, xmax = extent[0]
ymin, ymax = extent[1]
zmin, zmax = extent[2] if D == 3 else (0.0, 1.0)
Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
cs_y = 0.5 * (ymin + ymax)                                      # sheet at the Y centre
dx   = Lx / nx1                                                 # uniform Minkowski cell size

is_pair = (mass_ratio == 1.0)

# ---- normalisation constants (make_init) ----
sigma0    = (skindepth0 / larmor0) ** 2                         # [scales] magnetisation
sigma_ion = sigma0 / mass_ratio
AMP_COEFF = skindepth0 ** 2 / larmor0                           # J_written / curl B
B_BG      = 1.0                                                 # fields are in units of B0 = 1/larmor0
n_bg_per  = 0.5                                                 # per species, n0 units
n_cs_per  = 0.5 * cs_density                                    # per species, sheet peak

# ---- drift: 2 * n_per * beta_d must equal AMP_COEFF * B_BG / cs_width ----
beta_d = AMP_COEFF * B_BG / (2.0 * n_cs_per * cs_width)
if beta_d >= 1.0:
    raise SystemExit(f"drift velocity >= c (beta_d = {beta_d:g}): raise cs_density, raise cs_width, or lower sigma")
gamma_d = 1.0 / math.sqrt(1.0 - beta_d * beta_d)
drift_u = beta_d * gamma_d

# ---- temperatures: 2 * n_per * theta_e / gamma_d must equal sigma0 * B^2 / 2 ----
T_cs_e = sigma0 * B_BG ** 2 * gamma_d / (4.0 * n_cs_per)        # theta is per species rest mass
T_cs_i = T_cs_e / mass_ratio
T_bg_i = bg_theta_i
T_bg_e = T_bg_i * mass_ratio


def gamma_mean(theta):                                          # <gamma> of a Maxwell-Juttner at theta
    return 1.0 + theta * (6.0 + 15.0 * theta) / (4.0 + 5.0 * theta)


gamma_e_mean = gamma_mean(T_bg_e)
gamma_i_mean = gamma_mean(T_bg_i)

# ---- characteristic scales; the sqrt(2) is n_e = n0/2 after the injector's species split ----
d_e_cold  = skindepth0 * math.sqrt(2.0)
d_e       = d_e_cold * math.sqrt(gamma_e_mean)
d_i       = skindepth0 * math.sqrt(2.0 * mass_ratio * gamma_i_mean)
rho_e     = larmor0 * gamma_e_mean / B_BG                       # Larmor ~ 1/B
rho_i     = larmor0 * mass_ratio * gamma_i_mean / B_BG
lambda_De = math.sqrt(T_bg_e) * d_e
lambda_Di = math.sqrt(T_bg_i) * d_i

# ---- times; code time is in units of skindepth0/c ----
t_wpe = runtime * skindepth0 / d_e_cold
t_wpi = runtime / math.sqrt(2.0 * mass_ratio)
t_Lx  = runtime / Lx

# ---- magnetisation: sigma_s = B^2 * sigma0 / (n_s * m_s), with n_s = 0.5 per species ----
sigma_e_cold   = 2.0 * B_BG ** 2 * sigma0
sigma_sp2_cold = 2.0 * B_BG ** 2 * sigma0 / mass_ratio
sigma_tot_cold = 2.0 * B_BG ** 2 * sigma0 / (1.0 + mass_ratio)
sigma_e_hot    = sigma_e_cold / gamma_e_mean
sigma_sp2_hot  = sigma_sp2_cold / gamma_i_mean
sigma_tot_hot  = 2.0 * B_BG ** 2 * sigma0 / (gamma_e_mean + mass_ratio * gamma_i_mean)

# ---- pressure balance, in units of n0 * m_e * c^2 ----
P_mag       = sigma0 * B_BG ** 2 / 2.0
P_th        = T_bg_e                                            # n_e*theta_e + n_i*theta_i*mr, n_bg total = 1
P_cs        = cs_density * T_cs_e / gamma_d
plasma_beta = P_th / P_mag

# ---- the two invariants that must hold at t = 0 ----
J_required = AMP_COEFF * B_BG / cs_width
J_supplied = 2.0 * n_cs_per * beta_d
J_ratio    = J_supplied / J_required
P_ratio    = P_cs / P_mag


def count_turb_modes():                                         # mode COUNT from initial_turbulence, RNG-independent
    if turb_amp <= 0.0 or kmax < kmin:
        return 0
    vary_x = vary_y = vary_z = False
    if   turb_plane == 0: vary_x, vary_y = True, True
    elif turb_plane == 1: vary_y, vary_z = True, True
    elif turb_plane == 2: vary_z, vary_x = True, True
    k0x, k0y, k0z = 2.0 * math.pi / Lx, 2.0 * math.pi / Ly, 2.0 * math.pi / Lz
    range_x, range_y, range_z = (kmax if vary_x else 0), (kmax if vary_y else 0), (kmax if vary_z else 0)
    count = 0
    for nx in range(0, range_x + 1):
        for ny in range(-range_y, range_y + 1):
            for nz in range(-range_z, range_z + 1):
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

# ---- formatting: LW1 = 34, LW2 = 30, one tab = 4 spaces, as in the C++ ----
def g(x):       return f"{x:g}"
def kv1(label): return "    " + label.ljust(34) + "= "
def kv2(label): return "        " + label.ljust(30) + "= "

sp2   = "positron"  if is_pair else "ion"
SP2u  = "Positrons" if is_pair else "Ions"
sp2s  = "positrons" if is_pair else "ions"
title = "Relativistic Harris sheet (pair plasma)" if is_pair else "Relativistic Harris sheet (ion-electron plasma)"

L = ["", "===========================================================", title, "===========================================================", ""]

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
L.append(kv1("CS half-thickness [cells]") + g(cs_width / dx))
L.append(kv1("CS full thickness [cells]") + g(2.0 * cs_width / dx))
L.append(kv1("Y boundaries") + ("INJECTION" if inject_y else "REFLECTION"))
L.append(kv1("Runtime [cold electron freq.]") + g(t_wpe))
L.append(kv1("Runtime [cold " + sp2 + " freq.]") + g(t_wpi))
L.append(kv1("Runtime [c/Lx]") + g(t_Lx))
L.append("")

L.append("NORMALISATIONS")
L.append(kv1("sigma0 [(d0/larmor0)^2]") + g(sigma0))
L.append(kv1("J/curl(B) factor [d0^2/larmor0]") + g(AMP_COEFF))
L.append(kv1("n_bg per species [n0]") + g(n_bg_per))
L.append(kv1("n_CS per species [n0]") + g(n_cs_per))
L.append(kv1("ppc0 per species") + g(0.5 * ppc0))
L.append("")

L.append("MASS / MAGNETISATION")
L.append(kv1("Mass ratio [m_i/m_e]") + g(mass_ratio))
L.append(kv1("Background field [B0 = 1/larmor0]") + g(B_BG))
L.append(kv1("Guide field [Bg/B0]") + g(guide_field))
L.append("    Magnetisation (cold)")
L.append(kv2("total") + g(sigma_tot_cold))
L.append(kv2("electrons") + g(sigma_e_cold))
L.append(kv2(sp2s) + g(sigma_sp2_cold))
L.append("    Magnetisation (hot)")
L.append(kv2("total") + g(sigma_tot_hot))
L.append(kv2("electrons") + g(sigma_e_hot))
L.append(kv2(sp2s) + g(sigma_sp2_hot))
L.append(kv1("Mean Lorentz factor (electrons)") + g(gamma_e_mean))
L.append(kv1("Mean Lorentz factor (" + sp2s + ")") + g(gamma_i_mean))
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

L.append("CURRENT SHEET")
L.append(kv1("Overdensity [n_CS/n_BG]") + g(cs_density))
L.append(kv1("Half-thickness [d0]") + g(cs_width))
L.append(kv1("Drift four-velocity [u]") + g(drift_u))
L.append(kv1("beta = v/c") + g(beta_d))
L.append(kv1("Lorentz factor of drift") + g(gamma_d))
L.append(kv1("J required [curl(B)]") + g(J_required))
L.append(kv1("J supplied [2 n_CS beta]") + g(J_supplied))
L.append(kv1("2 n_CS beta/curl(B) (must be 1)") + g(J_ratio))
L.append("")

L.append("PRESSURE BALANCE")
L.append(kv1("Thermal (upstream)") + g(P_th))
L.append(kv1("Magnetic (upstream)") + g(P_mag))
L.append(kv1("Thermal (CS)") + g(P_cs))
L.append(kv1("P(CS,th)/P(US,mag) (must be 1)") + g(P_ratio))
L.append(kv1("Plasma beta (upstream)") + g(plasma_beta))
L.append("")

L.append("TURBULENCE")
L.append(kv1("Amplitude [dB/B0] (per comp)") + g(turb_amp))
L.append(kv1("Mode band [kmin, kmax]") + f"[{kmin}, {kmax}]")
L.append(kv1("Plane (0=xy,1=yz,2=zx)") + f"{turb_plane}")
L.append(kv1("Spectral index [p]") + g(spectral_index))
L.append(kv1("Number of modes") + f"{n_modes}")
L.append("")
L.append("===========================================================")

print("\n".join(L))

# ---- the same two guards the C++ raises, plus the resolution warnings ----
if abs(J_ratio - 1.0) > 1.0e-3:
    print(f"\nERROR: Ampere mismatch, J supplied / J required = {J_ratio:g}")
if abs(P_ratio - 1.0) > 1.0e-3:
    print(f"\nERROR: Harris pressure balance violated, P_cs / P_mag = {P_ratio:g}")
if 2.0 * cs_width / dx < 5.0:
    print(f"\nWARNING: the full CS thickness is only {2.0 * cs_width / dx:g} cells")
if lambda_De / dx < 0.3:
    print(f"WARNING: Debye length is {lambda_De / dx:g} cells; expect numerical heating")
print()