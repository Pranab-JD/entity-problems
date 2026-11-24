#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "traits/pgen.h"
#include "utils/numeric.h"
#include "arch/kokkos_aliases.h"

//! Requires an MPI build (-D mpi=ON defines MPI_ENABLED)
#if !defined(MPI_ENABLED)
    #error "This pgen requires an MPI build: configure Entity with -D mpi=ON"
#endif
#include "arch/mpi_tags.h"
#include <mpi.h>

#include "archetypes/utils.h"
#include "kernels/divergences.hpp"
#include "archetypes/energy_dist.h"
#include "archetypes/spatial_dist.h"
#include "framework/domain/metadomain.h"
#include "archetypes/particle_injector.h"

#include <any>
#include <array>
#include <cmath>
#include <string>
#include <vector>
#include <limits>
#include <random>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <typeinfo>
#include <algorithm>

//! =============================================================================
//!  PROBLEM DESCRIPTION
//!
//!  Decaying turbulence in a uniform, collisionless two-species plasma (electron-positron or 
//! electron-ion), relativistic or non-relativistic, in 2D or 3D. Periodic in every direction.
//!
//!  INITIAL STATE (t = 0)
//!   - B = B_guide z + B_perturbed. B_perturbed: solenoidal Fourier modes, kmin <= |n| <= kmax (k = 2 pi n / L),
//!     shell spectrum E(k) ~ k^spectral_slope, random phases, exact RMS. Amplitudes use the Yee-matched k~, so div(B) = 0 on the grid.
//!   - Flow u_perturbed: same construction, independent phases (v_seed). RMS = (B_perturbed_rms / B_rms) vA (relativistic),
//!     scaled by s <= 1 so that each species stays below u_total_cap [< c].
//!   - Particles: uniform n_s = n_tot/2, exact Maxwell-Juttner at T_s, drifting at u_perturbed + u_drift_species.
//!     The drifts carry J (InitFields::loaded_current), shared between the species by enthalpy at zero net momentum.
//!   - E = -(u_motional x B), u_motional = field-line velocity (u_perturbed, plus the Hall part when the species differ).
//!   - Charge rho = AMP div(E), loaded by displacing the two species in opposite directions along E;
//!     velocities are redrawn at the displaced positions.
//!   - Startup printout, also written to <simulation.name>/Simulation_params.txt, followed by a Gauss-law check.
//!
//!  INPUTS ([setup]; fields in code units)
//!     B_guide                         Guide field along Z
//!     B_perturbed_rms                 RMS |B_perturbed|
//!     kmin, kmax                      mode-number band, k = 2 pi n / L
//!     spectral_slope                  E(k) ~ k^spectral_slope; 0 = flat, -5/3 = Kolmogorov
//!     B_seed, v_seed                  phase seeds of B_perturbed and u_perturbed; must differ
//!     turb_kplane                     k-space: -1 = 3D, 0 = X-Y, 1 = Y-Z, 2 = Z-X
//!     turb_a_axis                     vector potential: -1 = random _|_ k, 0/1/2 = X/Y/Z (2 in 2D gives B_perturbed in the X-Y plane)
//!     turb_rand_amp                   Gaussian random mode amplitudes
//!     temperature                     species-1 temperature kT_1 / (m0 c^2)
//!     temperature_ratio               T_2 / T_1
//!     perturb_velocity                load u_perturbed
//!     u_total_cap                     cap on each species' total speed [c], in (0, 0.99)
//!
//! =============================================================================
//!
//!  LIMITATIONS
//!
//!  NON-RELATIVISTIC  (sigma_h < 0.1; for ion-electron also vA_e = sqrt(m_i/m_e) vA << c)
//!   - Particle noise: the current drift and the polarisation charge are small against thermal noise.
//!     The Gauss-law check is noise-dominated here.
//!   - E is first order in the drifts (electron inertia omitted): relative size ~ (k_max d)^2 B_perturbed/B,
//!     d = electron (or pair) skin depth. Keep L/kmax >~ 20-40 d.
//!   - Ion-electron: the unloaded dE/dt is ~ (k_max d_i)^2 (vA/c)^2 B_perturbed/B, negligible while vA_e << c.
//!   - current_filters > 0 makes the filtered current differ from the loaded J at high k.
//!
//!  RELATIVISTIC  (sigma_h >= 0.1, or B_guide = 0)
//!   - Loaded: E = -(u_motional x B), including the Hall field (InitFields::motional_E), and its charge rho = AMP div(E)
//!     (load_charge_density); in J (InitFields::loaded_current), the sigma_h split and the excess convection current of the
//!     displaced charge, both from the first-order fields.
//!   - NOT loaded: two parts of dE/dt at t = 0. Both exist only when E != 0, (a) in a pair plasma with u_perturbed or T_1 != T_2,
//!     and (b) always in an ion-electron plasma (the loaded Hall field makes E != 0 even without u_perturbed). With u = u_motional:
//!       * the parallel displacement current that keeps E.B = 0, J_par -> J_par - AMP E.(curl E)/|B|; grows as (u/c)^2;
//!       * the advective part of dE_perp/dt, (u.grad)u x B - u x curl(u x B); grows with vA/c.
//!   - The u_total cap uses the first-order current; print_setup reports the peak speed with the convection term.
//!   - Current starvation: the field-aligned current is carried in full, so a high sigma0 with a large k_max d0 can exceed the cap (abort).
//!   - Charge displacements of several cells at sigma_h ~ 1 limit the number of ranks per direction (guarded in load_charge_density).
//!
//!  ION-ELECTRON  (mass ratio mu = m_i/m_e; species 1 = electrons)
//!   - Electrons carry ~ mu/(1 + mu) of J: |u_drift,e| ~ k d_e sqrt(sigma_e) B_perturbed/B with sigma_e ~ mu sigma, so current
//!     starvation (abort) and electron streaming instabilities (u_drift,e >~ v_th,e) tighten as mu grows.
//!   - Omega_ce dt = CFL (dx/d0) sqrt(sigma0) |B| / sqrt(D) grows with sigma0 ~ sigma (1 + mu)/2: check it in the printout.
//!   - The Hall E and its charge displacement grow as AMP^2 = d0^2 sigma0: fewer ranks per direction.
//!   - The sigma_h split uses the MHD inertia w, approximate below d_i; the Hall part of curl(u_motional) ignores the sigma_h split.
//!
//! =============================================================================

namespace user
{
    using namespace ntt;

    //~ u_perturbed has phases independent of B_perturbed (v_seed != B_seed), so <u_perturbed . B_perturbed> = 0 on average over seeds:
    //~ both Elsasser families z+- are excited. For one seed pair the normalised cross-helicity scatters by ~ 1/sqrt(2 N_eff),
    //~ N_eff = effective number of energy-carrying modes; print_setup reports it.

    //* Mode m of B_perturbed or u_perturbed: amplitude (bx, by, bz) times cos(k.x + phase). The curl of the vector potential is
    //* folded into the amplitudes, so the device kernel is a bare cosine sum. kc are the Yee-matched wavenumbers used for curl(B).
    struct TurbMode
    {
        real_t kx, ky, kz;
        real_t bx, by, bz;
        real_t phase;
        real_t kcx, kcy, kcz;
    };

    //! Dimensionality of k-SPACE, not of the box: a plane-restricted spectrum in a 3D box still has N(k)dk ~ k dk per shell
    inline auto kspace_dim(int kplane, bool is_3d) -> int
    {
        bool vary_x = true, vary_y = true, vary_z = is_3d;
        if      (kplane == 0) { vary_x = true;  vary_y = true;  vary_z = false; }
        else if (kplane == 1) { vary_x = false; vary_y = true;  vary_z = true;  }
        else if (kplane == 2) { vary_x = true;  vary_y = false; vary_z = true;  }
        if (!is_3d) { vary_z = false; }
        return (vary_x ? 1 : 0) + (vary_y ? 1 : 0) + (vary_z ? 1 : 0);
    }

    //* Yee-matched wavenumber k~ = (2/h) sin(k h/2): the one-cell centred difference of cos(k x) is exactly -k~ sin(k x); h = 0 -> continuum k
    inline auto yee_wavenumber(double k, double h) -> double
    {
        return (k == 0.0 || h <= 0.0) ? k : (2.0 / h) * std::sin(0.5 * k * h);
    }

    //? Solenoidal mode table with total RMS rms_target; returns the per-component RMS.
    //?   A_m = a_k e_A sin(k.x + ph)  =>  b_m = a_k (k~ x e_A) cos(k.x + ph): k~ . b_m = 0, so div(B) = 0 exactly on the Yee grid.
    //?   a_k ~ (k/k0)^((shell_spectrum_slope - dim - 1)/2) with N(k)dk ~ k^(dim-1) modes per shell gives E(k) ~ k^shell_spectrum_slope.
    //? cell_size = 0 -> continuum k (used for u_perturbed).
    inline auto init_turbulence(real_t                 Lx,
                                real_t                 Ly,
                                real_t                 Lz,
                                bool                   is_3d,
                                real_t                 rms_target,            //* total RMS of the mode sum
                                int                    kmin,
                                int                    kmax,
                                long long              seed,
                                real_t                 shell_spectrum_slope,  //* E(k) ~ k^shell_spectrum_slope (shell spectrum)
                                int                    kplane,                //* -1 = full 3D, 0 = x1x2, 1 = x2x3, 2 = x3x1
                                int                    a_axis,                //* -1 = random e_A _|_ k, else 0/1/2 = fixed axis
                                bool                   random_amp,            //* Gaussian mode amplitudes
                                const std::array<double, 3>& cell_size,       //* Yee cell (dx, dy, dz) for k~, or zeros for continuum k
                                std::vector<TurbMode>& modes_out) -> std::array<real_t, 3>
    {
        modes_out.clear();
        const std::array<real_t, 3> zero { (real_t)0.0, (real_t)0.0, (real_t)0.0 };

        if (rms_target <= (real_t)0.0 || kmax < kmin || kmin < 1) return zero;

        std::mt19937_64                        rng(static_cast<uint64_t>(seed));
        std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * M_PI);
        std::normal_distribution<double>       gauss_dist(0.0, 1.0);

        //! In 2D only the x1-x2 plane is admissible: a mode with k3 != 0 would be sampled at x3 = 0 alone
        bool vary_x = true, vary_y = true, vary_z = is_3d;
        if      (kplane == 0) { vary_x = true;  vary_y = true;  vary_z = false; }
        else if (kplane == 1) { vary_x = false; vary_y = true;  vary_z = true;  }
        else if (kplane == 2) { vary_x = true;  vary_y = false; vary_z = true;  }
        if (!is_3d) { vary_z = false; }

        const double k0x = vary_x ? 2.0 * M_PI / (double)Lx : 0.0;
        const double k0y = vary_y ? 2.0 * M_PI / (double)Ly : 0.0;
        const double k0z = vary_z ? 2.0 * M_PI / (double)Lz : 0.0;

        //* longest active wavelength; non-dimensionalises the power law
        double k0_ref = 0.0;
        if (vary_x) k0_ref = (k0_ref == 0.0) ? k0x : std::min(k0_ref, k0x);
        if (vary_y) k0_ref = (k0_ref == 0.0) ? k0y : std::min(k0_ref, k0y);
        if (vary_z) k0_ref = (k0_ref == 0.0) ? k0z : std::min(k0_ref, k0z);
        if (k0_ref <= 0.0) return zero;

        const double dim = (double)kspace_dim(kplane, is_3d);

        //~ Energy of ONE mode goes as k^(shell_spectrum_slope - (dim-1)). Two reference values:
        //~   shell_spectrum_slope = 0       -> equal energy per |k| SHELL (individual modes still weaken with k)
        //~   shell_spectrum_slope = dim - 1 -> a_k ~ 1/k, |b_m| = const: equal energy per MODE (+1 in 2D, +2 in 3D; iPIC3D's AlfvTurb2D)
        const double weight_exp = 0.5 * ((double)shell_spectrum_slope - dim - 1.0);

        const int nx_range = vary_x ? kmax : 0;
        const int ny_range = vary_y ? kmax : 0;
        const int nz_range = vary_z ? kmax : 0;

        for (int nx = 0; nx <= nx_range; ++nx)
            for (int ny = -ny_range; ny <= ny_range; ++ny)
                for (int nz = -nz_range; nz <= nz_range; ++nz)
                {
                    //* half-space: one of each (+k,-k) pair, and no k = 0
                    if (nx == 0 && ny < 0) continue;
                    if (nx == 0 && ny == 0 && nz <= 0) continue;

                    const double n_mag = std::sqrt((double)(nx * nx + ny * ny + nz * nz));
                    if (n_mag < (double)kmin || n_mag > (double)kmax) continue;

                    const double kx    = (double)nx * k0x;
                    const double ky    = (double)ny * k0y;
                    const double kz    = (double)nz * k0z;
                    const double k_mag = std::sqrt(kx * kx + ky * ky + kz * kz);
                    if (k_mag <= 0.0) continue;

                    //* wavenumbers for the amplitude direction and the curl (k~ on the Yee grid, k otherwise)
                    const double kcx    = yee_wavenumber(kx, cell_size[0]);
                    const double kcy    = yee_wavenumber(ky, cell_size[1]);
                    const double kcz    = yee_wavenumber(kz, cell_size[2]);
                    const double kc_mag = std::sqrt(kcx * kcx + kcy * kcy + kcz * kcz);

                    double ex, ey, ez;          //* e_A, direction of this mode's vector potential
                    if (a_axis >= 0 && a_axis <= 2)
                    {
                        //* fixed e_A: b_m _|_ e_A, so a_axis = 2 with k in x1-x2 gives an in-plane mode (2D flux-function setup)
                        ex = (a_axis == 0) ? 1.0 : 0.0;
                        ey = (a_axis == 1) ? 1.0 : 0.0;
                        ez = (a_axis == 2) ? 1.0 : 0.0;
                    }
                    else
                    {
                        //* random e_A with the k~-parallel part projected out, so |b_m| = a_k |k~| exactly
                        double gx = 0.0, gy = 0.0, gz = 0.0, norm = 0.0;
                        int    tries = 0;
                        do
                        {
                            gx = gauss_dist(rng); gy = gauss_dist(rng); gz = gauss_dist(rng);
                            const double proj = (gx * kcx + gy * kcy + gz * kcz) / (kc_mag * kc_mag);
                            gx -= proj * kcx; gy -= proj * kcy; gz -= proj * kcz;
                            norm = std::sqrt(gx * gx + gy * gy + gz * gz);
                        }
                        while (norm < 1.0e-12 && ++tries < 10);

                        if (norm < 1.0e-12) continue;
                        ex = gx / norm; ey = gy / norm; ez = gz / norm;
                    }

                    const double rand_fac = random_amp ? gauss_dist(rng) : 1.0;
                    const double a_k      = rand_fac * std::pow(k_mag / k0_ref, weight_exp);

                    const double bx = a_k * (kcy * ez - kcz * ey);
                    const double by = a_k * (kcz * ex - kcx * ez);
                    const double bz = a_k * (kcx * ey - kcy * ex);

                    if (std::sqrt(bx * bx + by * by + bz * bz) < 1.0e-12 * std::fabs(a_k) * kc_mag) continue;

                    modes_out.push_back(TurbMode { (real_t)kx, (real_t)ky, (real_t)kz,
                                                   (real_t)bx, (real_t)by, (real_t)bz,
                                                   (real_t)phase_dist(rng),
                                                   (real_t)kcx, (real_t)kcy, (real_t)kcz });
                }

        //~ RMS is EXACT: no two modes share +/-k (half-space), so cross terms vanish over the box and each component's mean square is 0.5 sum_m b_i,m^2
        double s_bx = 0.0, s_by = 0.0, s_bz = 0.0;
        for (const auto& m : modes_out)
        {
            s_bx += 0.5 * (double)m.bx * (double)m.bx;
            s_by += 0.5 * (double)m.by * (double)m.by;
            s_bz += 0.5 * (double)m.bz * (double)m.bz;
        }

        const double s_tot = s_bx + s_by + s_bz;
        if (s_tot <= 0.0) { modes_out.clear(); return zero; }

        const double scale = (double)rms_target / std::sqrt(s_tot);
        for (auto& m : modes_out)
        {
            m.bx = (real_t)((double)m.bx * scale);
            m.by = (real_t)((double)m.by * scale);
            m.bz = (real_t)((double)m.bz * scale);
        }

        return { (real_t)(std::sqrt(s_bx) * scale), (real_t)(std::sqrt(s_by) * scale), (real_t)(std::sqrt(s_bz) * scale) };
    }

    //~ Host mode sums, shared by the u_total cap (make_init) and the startup diagnostics (print_setup); they mirror the device kernels.

    //* sum_m a_m cos(k_m.x + phase_m), with a_m the (bx, by, bz) slot of the mode
    inline void host_cos_sum(const std::vector<TurbMode>& modes, double X, double Y, double Z, double& sum_x, double& sum_y, double& sum_z)
    {
        sum_x = 0.0; sum_y = 0.0; sum_z = 0.0;
        for (const auto& m : modes)
        {
            const double c = std::cos((double)m.kx * X + (double)m.ky * Y + (double)m.kz * Z + (double)m.phase);
            sum_x += (double)m.bx * c;
            sum_y += (double)m.by * c;
            sum_z += (double)m.bz * c;
        }
    }

    //* curl of the cosine sum with the Yee-matched kc:  -sum_m (kc_m x b_m) sin(k_m.x + phase_m)
    inline void host_curl_sum(const std::vector<TurbMode>& modes, double X, double Y, double Z, double& curl_x, double& curl_y, double& curl_z)
    {
        curl_x = 0.0; curl_y = 0.0; curl_z = 0.0;
        for (const auto& m : modes)
        {
            const double s = std::sin((double)m.kx * X + (double)m.ky * Y + (double)m.kz * Z + (double)m.phase);
            curl_x -= ((double)m.kcy * (double)m.bz - (double)m.kcz * (double)m.by) * s;
            curl_y -= ((double)m.kcz * (double)m.bx - (double)m.kcx * (double)m.bz) * s;
            curl_z -= ((double)m.kcx * (double)m.by - (double)m.kcy * (double)m.bx) * s;
        }
    }

    //* sigma_h split (mirrors InitFields::sigma_split): V -> V - f V_perp, f = sigma_local / (1 + sigma_local)
    inline void host_sigma_split(double sigma_h_per_B_sq, const double (&B)[3], double (&V)[3])
    {
        if (sigma_h_per_B_sq <= 0.0) { return; }
        const double B_sq = B[0] * B[0] + B[1] * B[1] + B[2] * B[2];
        if (B_sq <= 0.0) { return; }
        const double sigma_local     = sigma_h_per_B_sq * B_sq;
        const double perp_reduction  = sigma_local / (1.0 + sigma_local);
        const double parallel_over_B = (V[0] * B[0] + V[1] * B[1] + V[2] * B[2]) / B_sq;
        for (int i = 0; i < 3; ++i) { V[i] -= perp_reduction * (V[i] - parallel_over_B * B[i]); }
    }

    //* first-order current J/AMP = curl(B) - f (curl B)_perp (mirrors InitFields::first_order_current)
    inline void host_first_order_current(const std::vector<TurbMode>& modes, double B_guide, double sigma_h_per_B_sq, double X, double Y, double Z, double& current_x, double& current_y, double& current_z)
    {
        double current[3], B[3];
        host_curl_sum(modes, X, Y, Z, current[0], current[1], current[2]);
        host_cos_sum(modes, X, Y, Z, B[0], B[1], B[2]);
        B[2] += B_guide;
        host_sigma_split(sigma_h_per_B_sq, B, current);
        current_x = current[0]; current_y = current[1]; current_z = current[2];
    }

    //* u_perturbed and its curl, continuum k (mirrors TurbVelocity::with_curl)
    inline void host_u_and_curl(const std::vector<TurbMode>& modes, double X, double Y, double Z, double (&u)[3], double (&curl_u)[3])
    {
        for (int i = 0; i < 3; ++i) { u[i] = 0.0; curl_u[i] = 0.0; }
        for (const auto& m : modes)
        {
            const double phase = (double)m.kx * X + (double)m.ky * Y + (double)m.kz * Z + (double)m.phase;
            const double c = std::cos(phase), s = std::sin(phase);
            u[0] += (double)m.bx * c; u[1] += (double)m.by * c; u[2] += (double)m.bz * c;
            curl_u[0] -= ((double)m.ky * (double)m.bz - (double)m.kz * (double)m.by) * s;
            curl_u[1] -= ((double)m.kz * (double)m.bx - (double)m.kx * (double)m.bz) * s;
            curl_u[2] -= ((double)m.kx * (double)m.by - (double)m.ky * (double)m.bx) * s;
        }
    }

    //* curl(curl B) = sum_m |kc_m|^2 b_m cos(k_m.x + ph_m) (mirrors InitFields::curl_curl_B)
    inline void host_curl_curl_sum(const std::vector<TurbMode>& modes, double X, double Y, double Z, double (&curl_curl)[3])
    {
        for (int i = 0; i < 3; ++i) { curl_curl[i] = 0.0; }
        for (const auto& m : modes)
        {
            const double c       = std::cos((double)m.kx * X + (double)m.ky * Y + (double)m.kz * Z + (double)m.phase);
            const double kc_sq_c = ((double)m.kcx * (double)m.kcx + (double)m.kcy * (double)m.kcy + (double)m.kcz * (double)m.kcz) * c;
            curl_curl[0] += (double)m.bx * kc_sq_c; curl_curl[1] += (double)m.by * kc_sq_c; curl_curl[2] += (double)m.bz * kc_sq_c;
        }
    }

    //* drift current J_drift/AMP = sigma_h split of [curl(B) - div(E) u_convection] (mirrors InitFields::loaded_current)
    inline void host_loaded_current(const std::vector<TurbMode>& B_modes, const std::vector<TurbMode>& u_modes, double B_guide, double sigma_h_per_B_sq, double u_motional_coeff, double X, double Y, double Z, double& current_x, double& current_y, double& current_z, double* div_E_out = nullptr)
    {
        double B[3], curl_B[3], u[3], curl_u[3], curl_hall[3] = { 0.0, 0.0, 0.0 };
        host_cos_sum(B_modes, X, Y, Z, B[0], B[1], B[2]);
        B[2] += B_guide;
        host_curl_sum(B_modes, X, Y, Z, curl_B[0], curl_B[1], curl_B[2]);
        host_u_and_curl(u_modes, X, Y, Z, u, curl_u);

        double first_order[3] = { curl_B[0], curl_B[1], curl_B[2] };
        host_sigma_split(sigma_h_per_B_sq, B, first_order);
        if (u_motional_coeff != 0.0) { host_curl_curl_sum(B_modes, X, Y, Z, curl_hall); }

        double u_motional[3], curl_u_motional[3];
        for (int i = 0; i < 3; ++i)
        {
            u_motional[i]      = u[i] + u_motional_coeff * first_order[i];
            curl_u_motional[i] = curl_u[i] + u_motional_coeff * curl_hall[i];
        }
        const double div_E = (u_motional[0] * curl_B[0] + u_motional[1] * curl_B[1] + u_motional[2] * curl_B[2])
                           - (B[0] * curl_u_motional[0] + B[1] * curl_u_motional[1] + B[2] * curl_u_motional[2]);

        if (div_E_out != nullptr) { *div_E_out = div_E; }

        double current[3];
        for (int i = 0; i < 3; ++i) { current[i] = curl_B[i] - div_E * 0.5 * (u[i] + u_motional[i]); }
        host_sigma_split(sigma_h_per_B_sq, B, current);
        current_x = current[0]; current_y = current[1]; current_z = current[2];
    }

    //* Host sampling grid for peaks and checks. Fields are band-limited at |n| <= kmax, so 16 points per shortest wavelength (8 in 3D)
    //* locate peaks to within 1 - cos(pi/16) ~ 2% (8% in 3D). Never finer than the simulation grid. Cost ~ kmax^4 in 2D.
    inline auto sampling_points_per_side(int kmax, std::size_t n_cells, bool is_3d) -> int
    {
        const int points_per_wavelength = is_3d ? 8 : 16;
        const int minimum_points        = is_3d ? 24 : 64;
        const int wanted_points         = std::max(minimum_points, points_per_wavelength * kmax);
        return std::max(1, std::min(wanted_points, static_cast<int>(n_cells)));
    }

    //* cell-centred sample coordinate i of n along an axis [x_min, x_min + length)
    inline auto sample_coordinate(double x_min, double length, int i, int n) -> double
    {
        return x_min + ((double)i + 0.5) / (double)n * length;
    }

    //~ =========================================================================
    //~  VELOCITY PERTURBATION u_perturbed (common to both species)
    //~
    //~    u_perturbed(x) = s * sum_m u_m cos(k_m.x + psi_m)
    //~
    //~  Zero mean flow (plasma rest frame). Same solenoidal construction as B_perturbed (same band, slope and polarisation axis)
    //~  but with INDEPENDENT phases (setup.v_seed). s <= 1 is the u_total cap factor (make_init); it scales every mode alike.
    //~
    //~  Reusing B_perturbed's seed would give u_perturbed proportional to B_perturbed everywhere: a pure Elsasser state (z- = 0),
    //~  for which z-.grad(z+) vanishes identically. It is an exact nonlinear solution of incompressible MHD and does not cascade.
    //~ =========================================================================
    template <Dimension D>
    struct TurbVelocity
    {
        TurbVelocity() = default;

        TurbVelocity(array_t<real_t*> kx, array_t<real_t*> ky, array_t<real_t*> kz,
                     array_t<real_t*> ux, array_t<real_t*> uy, array_t<real_t*> uz,
                     array_t<real_t*> ph, std::size_t nmodes) :
                     kx { kx }, ky { ky }, kz { kz },
                     ux { ux }, uy { uy }, uz { uz },
                     ph { ph }, nmodes { nmodes } {}

        //* u_perturbed [c] at x_Ph; zero when no modes were built (perturb_velocity = false or B_perturbed_rms = 0)
        Inline void operator()(const coord_t<D>& x_Ph, real_t& u_perturbed_x, real_t& u_perturbed_y, real_t& u_perturbed_z) const
        {
            u_perturbed_x = ZERO; u_perturbed_y = ZERO; u_perturbed_z = ZERO;
            if (nmodes == 0) { return; }

            const real_t X = x_Ph[0];
            const real_t Y = x_Ph[1];
            real_t       Z = ZERO;
            if constexpr (D == Dim::_3D) { Z = x_Ph[2]; }

            for (std::size_t m = 0; m < nmodes; ++m)
            {
                const real_t c = math::cos(kx(m) * X + ky(m) * Y + kz(m) * Z + ph(m));
                u_perturbed_x += ux(m) * c;
                u_perturbed_y += uy(m) * c;
                u_perturbed_z += uz(m) * c;
            }
        }

        //* u_perturbed and curl(u_perturbed) = -sum_m (k_m x u_m) sin(k_m.x + psi_m), continuum k
        Inline void with_curl(const coord_t<D>& x_Ph, real_t (&u_perturbed)[3], real_t (&curl_u_perturbed)[3]) const
        {
            for (int i = 0; i < 3; ++i) { u_perturbed[i] = ZERO; curl_u_perturbed[i] = ZERO; }
            if (nmodes == 0) { return; }

            const real_t X = x_Ph[0];
            const real_t Y = x_Ph[1];
            real_t       Z = ZERO;
            if constexpr (D == Dim::_3D) { Z = x_Ph[2]; }

            for (std::size_t m = 0; m < nmodes; ++m)
            {
                const real_t phase = kx(m) * X + ky(m) * Y + kz(m) * Z + ph(m);
                const real_t c = math::cos(phase), s = math::sin(phase);
                u_perturbed[0] += ux(m) * c;
                u_perturbed[1] += uy(m) * c;
                u_perturbed[2] += uz(m) * c;
                curl_u_perturbed[0] -= (ky(m) * uz(m) - kz(m) * uy(m)) * s;
                curl_u_perturbed[1] -= (kz(m) * ux(m) - kx(m) * uz(m)) * s;
                curl_u_perturbed[2] -= (kx(m) * uy(m) - ky(m) * ux(m)) * s;
            }
        }

    private:
        array_t<real_t*> kx, ky, kz, ux, uy, uz, ph;
        std::size_t      nmodes { 0 };
    };


    template <Dimension D>
    struct InitFields
    {
        InitFields() = default;

        InitFields( real_t B_guide,
                    array_t<real_t*> kx, array_t<real_t*> ky, array_t<real_t*> kz,
                    array_t<real_t*> bx, array_t<real_t*> by, array_t<real_t*> bz,
                    array_t<real_t*> ph,
                    array_t<real_t*> kcx, array_t<real_t*> kcy, array_t<real_t*> kcz,
                    std::size_t nmodes, const TurbVelocity<D>& vel, real_t u_motional_coeff, real_t sigma_h_per_B_sq) :
                    B_guide { B_guide },
                    kx { kx }, ky { ky }, kz { kz },
                    bx { bx }, by { by }, bz { bz },
                    ph { ph },
                    kcx { kcx }, kcy { kcy }, kcz { kcz },
                    nmodes { nmodes },
                    vel { vel },
                    u_motional_coeff { u_motional_coeff },
                    sigma_h_per_B_sq { sigma_h_per_B_sq } {}

        //* B_perturbed component; comp: 0 -> x1, 1 -> x2, 2 -> x3
        Inline auto turb(const coord_t<D>& x_Ph, int comp) const -> real_t
        {
            real_t       acc = ZERO;
            const real_t X   = x_Ph[0];
            const real_t Y   = x_Ph[1];
            real_t       Z   = ZERO;
            if constexpr (D == Dim::_3D) { Z = x_Ph[2]; }

            for (std::size_t m = 0; m < nmodes; ++m)
            {
                const real_t c = math::cos(kx(m) * X + ky(m) * Y + kz(m) * Z + ph(m));
                const real_t b = (comp == 0) ? bx(m) : ((comp == 1) ? by(m) : bz(m));
                acc += b * c;
            }
            return acc;
        }

        //! MAGNETIC FIELD   B = B_guide (Z) + B_perturbed
        Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t { return turb(x_Ph, 0); }
        Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t { return turb(x_Ph, 1); }
        Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t { return B_guide + turb(x_Ph, 2); }

        //! =========================================================================
        //! MOTIONAL ELECTRIC FIELD    E = -(u_motional x B)
        //!
        //! E gives both species the same acceleration per enthalpy, q_s (E + u_total_s x B) / h_s, so the plasma as a whole feels only J x B:
        //!     q_1 (u_total_1 - u_motional) / h_1 = q_2 (u_total_2 - u_motional) / h_2
        //!     =>  u_motional = (h_2 u_total_1 + h_1 u_total_2) / (h_1 + h_2) = u_perturbed + u_drift_1 + u_drift_2,
        //! with u_drift_s = sign(q_s) lambda / h_s (TurbCurrentDist).
        //!     pair plasma, equal T : u_drift_1 = -u_drift_2  ->  u_motional = u_perturbed (ideal MHD)
        //!     ion-electron plasma  : u_motional ~ electron velocity -> E = -(u_perturbed x B) + Hall field
        //!
        //!     u_motional = u_perturbed + u_motional_coeff J_code/AMP,  u_motional_coeff = u_drift_coeff_1 + u_drift_coeff_2,
        //! with J_code/AMP from first_order_current. Velocities are 3-velocities [c].
        //! =========================================================================
        Inline void motional_E(const coord_t<D>& x_Ph, real_t& Ex, real_t& Ey, real_t& Ez) const
        {
            real_t u_perturbed_x = ZERO, u_perturbed_y = ZERO, u_perturbed_z = ZERO;
            vel(x_Ph, u_perturbed_x, u_perturbed_y, u_perturbed_z);

            real_t curl_B_x = ZERO, curl_B_y = ZERO, curl_B_z = ZERO;
            if (u_motional_coeff != ZERO) { first_order_current(x_Ph, curl_B_x, curl_B_y, curl_B_z); }

            const real_t u_motional_x = u_perturbed_x + u_motional_coeff * curl_B_x;
            const real_t u_motional_y = u_perturbed_y + u_motional_coeff * curl_B_y;
            const real_t u_motional_z = u_perturbed_z + u_motional_coeff * curl_B_z;

            const real_t B_x = turb(x_Ph, 0);
            const real_t B_y = turb(x_Ph, 1);
            const real_t B_z = B_guide + turb(x_Ph, 2);

            //* E = -(u_motional x B)
            Ex = -(u_motional_y * B_z - u_motional_z * B_y);
            Ey = -(u_motional_z * B_x - u_motional_x * B_z);
            Ez = -(u_motional_x * B_y - u_motional_y * B_x);
        }

        Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t
        {
            real_t Ex, Ey, Ez; motional_E(x_Ph, Ex, Ey, Ez); return Ex;
        }
        Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t
        {
            real_t Ex, Ey, Ez; motional_E(x_Ph, Ex, Ey, Ez); return Ey;
        }
        Inline auto ex3(const coord_t<D>& x_Ph) const -> real_t
        {
            real_t Ex, Ey, Ez; motional_E(x_Ph, Ex, Ey, Ez); return Ez;
        }

        //! =========================================================================
        //! curl(B):  curl(b cos(k.x + ph)) = -(k x b) sin(k.x + ph)
        //!
        //! kc replaces k so that the DISCRETE Yee curl is matched exactly (see init_turbulence). B_guide has no curl.
        //! The currents built on it are first_order_current (sigma_h split) and loaded_current (adds the convection term).
        //! =========================================================================
        Inline void curl_B(const coord_t<D>& x_Ph, real_t& curl_B_x, real_t& curl_B_y, real_t& curl_B_z) const
        {
            curl_B_x = ZERO; curl_B_y = ZERO; curl_B_z = ZERO;

            const real_t X = x_Ph[0];
            const real_t Y = x_Ph[1];
            real_t       Z = ZERO;
            if constexpr (D == Dim::_3D) { Z = x_Ph[2]; }

            for (std::size_t m = 0; m < nmodes; ++m)
            {
                const real_t s = math::sin(kx(m) * X + ky(m) * Y + kz(m) * Z + ph(m));
                curl_B_x -= (kcy(m) * bz(m) - kcz(m) * by(m)) * s;
                curl_B_y -= (kcz(m) * bx(m) - kcx(m) * bz(m)) * s;
                curl_B_z -= (kcx(m) * by(m) - kcy(m) * bx(m)) * s;
            }
        }

        //! =========================================================================
        //! SIGMA_H SPLIT OF THE CURRENT:  J_code = AMP [ curl(B) - f (curl B)_perp ],  f = sigma_h / (1 + sigma_h)  (first order)
        //!
        //!  A current across the local B accelerates the plasma (J x B), and the growing E = -(u x B) supplies part of curl(B) as
        //!  displacement current:
        //!      c (curl B)_perp = 4 pi J_perp + dE_perp/dt,  dE_perp/dt = J_perp B^2 / (c w)
        //!   => J_perp = (c/4 pi) (curl B)_perp / (1 + sigma_h),   sigma_h = B^2 / (4 pi w), with the LOCAL |B|^2 (f -> 0 at nulls).
        //!  The field-aligned part is carried in full (E_par = 0). f -> sigma_h -> 0 for sigma_h << 0.1, so it is applied unconditionally.
        //!  2D (k in X-Y, B_guide along Z): J_z is reduced and an in-plane current along the tilted field lines appears.
        //! =========================================================================
        Inline void sigma_split(const real_t (&B)[3], real_t (&V)[3]) const
        {
            if (sigma_h_per_B_sq <= ZERO) { return; }
            const real_t B_sq = SQR(B[0]) + SQR(B[1]) + SQR(B[2]);
            if (B_sq <= ZERO) { return; }

            const real_t sigma_local     = sigma_h_per_B_sq * B_sq;
            const real_t perp_reduction  = sigma_local / (ONE + sigma_local);                       //* f
            const real_t parallel_over_B = (V[0] * B[0] + V[1] * B[1] + V[2] * B[2]) / B_sq;       //* (V . B) / B^2
            for (int i = 0; i < 3; ++i) { V[i] -= perp_reduction * (V[i] - parallel_over_B * B[i]); }
        }

        //* J/AMP at first order (used for the Hall part of E)
        Inline void first_order_current(const coord_t<D>& x_Ph, real_t& current_x, real_t& current_y, real_t& current_z) const
        {
            real_t B[3] = { turb(x_Ph, 0), turb(x_Ph, 1), B_guide + turb(x_Ph, 2) };
            real_t current[3];
            curl_B(x_Ph, current[0], current[1], current[2]);
            sigma_split(B, current);
            current_x = current[0]; current_y = current[1]; current_z = current[2];
        }

        //* curl(curl B) = sum_m |kc_m|^2 b_m cos(k_m.x + ph_m), using kc . b = 0
        Inline void curl_curl_B(const coord_t<D>& x_Ph, real_t (&curl_curl)[3]) const
        {
            for (int i = 0; i < 3; ++i) { curl_curl[i] = ZERO; }
            const real_t X = x_Ph[0];
            const real_t Y = x_Ph[1];
            real_t       Z = ZERO;
            if constexpr (D == Dim::_3D) { Z = x_Ph[2]; }

            for (std::size_t m = 0; m < nmodes; ++m)
            {
                const real_t c       = math::cos(kx(m) * X + ky(m) * Y + kz(m) * Z + ph(m));
                const real_t kc_sq_c = (SQR(kcx(m)) + SQR(kcy(m)) + SQR(kcz(m))) * c;
                curl_curl[0] += bx(m) * kc_sq_c;
                curl_curl[1] += by(m) * kc_sq_c;
                curl_curl[2] += bz(m) * kc_sq_c;
            }
        }

        //! =========================================================================
        //! CURRENT CARRIED BY THE DRIFTS:  J_drift / AMP = sigma_h split of [ curl(B) - div(E) u_convection ]
        //!
        //!  The displaced charge rho = AMP div(E) moves with the mean species velocity u_convection = (u_perturbed + u_motional)/2 and
        //!  carries rho u_convection on top of the drifts. Subtracting it BEFORE the split makes the total current
        //!      along B:   AMP (curl B)_par
        //!      across B:  [AMP (curl B)_perp + sigma_h rho u_convection_perp] / (1 + sigma_h)
        //!  Nonzero only where E != 0; relative size ~ (u/c)^2. rho and u_motional are taken at first order (no recursion).
        //! =========================================================================
        Inline void loaded_current(const coord_t<D>& x_Ph, real_t& current_x, real_t& current_y, real_t& current_z) const
        {
            real_t B[3] = { turb(x_Ph, 0), turb(x_Ph, 1), B_guide + turb(x_Ph, 2) };
            real_t curl_B_value[3];
            curl_B(x_Ph, curl_B_value[0], curl_B_value[1], curl_B_value[2]);

            real_t u_perturbed[3], curl_u_perturbed[3];
            vel.with_curl(x_Ph, u_perturbed, curl_u_perturbed);

            //* u_motional = u_perturbed + u_motional_coeff J_first_order, and its curl
            real_t first_order[3] = { curl_B_value[0], curl_B_value[1], curl_B_value[2] };
            sigma_split(B, first_order);
            real_t curl_hall[3] = { ZERO, ZERO, ZERO };
            if (u_motional_coeff != ZERO) { curl_curl_B(x_Ph, curl_hall); }

            real_t u_motional[3], curl_u_motional[3];
            for (int i = 0; i < 3; ++i)
            {
                u_motional[i]      = u_perturbed[i] + u_motional_coeff * first_order[i];
                curl_u_motional[i] = curl_u_perturbed[i] + u_motional_coeff * curl_hall[i];
            }

            //* rho / AMP = div(E) = u_motional . curl(B) - B . curl(u_motional),  E = -(u_motional x B)
            const real_t div_E = (u_motional[0] * curl_B_value[0] + u_motional[1] * curl_B_value[1] + u_motional[2] * curl_B_value[2])
                               - (B[0] * curl_u_motional[0] + B[1] * curl_u_motional[1] + B[2] * curl_u_motional[2]);

            //* remove the convection current rho u_convection, then split by sigma_h
            real_t current[3];
            for (int i = 0; i < 3; ++i) { current[i] = curl_B_value[i] - div_E * HALF * (u_perturbed[i] + u_motional[i]); }
            sigma_split(B, current);
            current_x = current[0]; current_y = current[1]; current_z = current[2];
        }

    private:
        real_t            B_guide { ZERO };
        array_t<real_t*>  kx, ky, kz, bx, by, bz, ph, kcx, kcy, kcz;
        std::size_t       nmodes { 0 };
        TurbVelocity<D>   vel {};                     //* empty -> u_perturbed = 0
        real_t            u_motional_coeff { ZERO };  //* 0 for a pair plasma at equal temperatures
        real_t            sigma_h_per_B_sq { ZERO };  //* sigma0 / w: local sigma_h = sigma_h_per_B_sq |B|^2 (0 -> J = AMP curl B)
    };

    //! =========================================================================
    //!  Energy distribution: Maxwell-Juttner at theta_s, drifting at the local LAB-frame velocity u_total_species = u_perturbed + u_drift_species
    //!
    //!  Entity integrates dE/dt = curl(B) - (rho0/d0^2) J, so the particles must carry J_code = AMP J_loaded, AMP = d0^2/rho0 = d0 sqrt(sigma0),
    //!  J_loaded = InitFields::loaded_current (= curl(B) for sigma_h << 0.1).
    //!
    //!  With |q_s| = 1:  J_code = sum_s n_s q_s u_drift_species. Each species drifts with the same momentum per particle lambda in opposite
    //!  directions, h_s u_drift_species = sign(q_s) lambda (h_s = m_s <gamma_s> + T_s), so sum_s n_s h_s u_drift_species = 0 and
    //!        u_drift_species = sign(q_s) AMP J_loaded / ( h_s sum_t n_t/h_t ) = u_drift_coeff_s J_loaded.
    //!  Cold: the light species carries ~ m_heavy/(m_1 + m_2) of J. Pair at equal T: equal and opposite drifts.
    //!
    //!  The sampler gives each species the mean 3-velocity u_total_species and the injector loads uniform n_s, so the loaded current is
    //!  sum_s q_s n_s u_total_species = AMP J_loaded: u_perturbed cancels in a neutral plasma. After the charge-loading displacement the
    //!  plasma is not exactly neutral; that convection current is what J_loaded subtracts.
    //! =========================================================================
    template <class M>
    struct TurbCurrentDist
    {
        static constexpr auto D = M::Dim;

        TurbCurrentDist(random_number_pool_t& pool, real_t theta_s, real_t u_drift_coeff,
                        const InitFields<M::Dim>& flds, const TurbVelocity<M::Dim>& vel):
                        maxwellian { pool, theta_s }, pool { pool }, theta { theta_s }, u_drift_coeff { u_drift_coeff },
                        flds { flds }, vel { vel } {}

        //? ISOTROPIC Maxwell-Juttner four-velocity at theta = kT_s/(m_s c^2), rest frame.
        //? theta >= 0.5: Entity's Maxwellian (exact Sobol method).
        //? theta <  0.5: Entity's Maxwellian is a non-relativistic Gaussian (loads <gamma> - 1 at 0.98, 0.93, 0.81, 0.62 of the exact value at
        //?   theta = 0.01, 0.03, 0.1, 0.3), so eps = gamma - 1 is sampled exactly here. Its distribution
        //?       f(eps) ~ sqrt(eps) (1 + eps) sqrt(1 + eps/2) exp(-eps/theta)
        //?   is bounded (sqrt(1 + x) <= 1 + x/2) by the envelope sqrt(eps) (1 + eps)(1 + eps/4) exp(-eps/theta): a mixture of Gamma(3/2, 5/2, 7/2)
        //?   of scale theta with weights 1 : 15 theta/8 : 15 theta^2/16. Gamma(k + 1/2) = (k unit exponentials) + Z^2/2. Accept with
        //?   probability sqrt(1 + eps/2) / (1 + eps/4) (rate > 0.97 for theta < 0.5). Isotropic direction, |u| = sqrt(eps (eps + 2)).
        Inline void thermal_sample(const coord_t<D>& x_Ph, vec_t<Dim::_3D>& v) const
        {
            if (theta >= HALF) { maxwellian(x_Ph, v); return; }

            auto rand_gen = pool.get_state();

            //* mixture weights of the envelope: Gamma shape 3/2, 5/2, 7/2
            const real_t weight_shape_3_2 = ONE;
            const real_t weight_shape_5_2 = static_cast<real_t>(15.0 / 8.0) * theta;
            const real_t weight_shape_7_2 = static_cast<real_t>(15.0 / 16.0) * SQR(theta);
            const real_t weight_total     = weight_shape_3_2 + weight_shape_5_2 + weight_shape_7_2;

            real_t energy   = ZERO;   //* eps = gamma - 1
            bool   accepted = false;
            while (!accepted)
            {
                //* component: number of unit exponentials beyond the first (0, 1 or 2)
                const real_t pick          = Random<real_t>(rand_gen) * weight_total;
                const int    extra_shapes  = (pick < weight_shape_3_2) ? 0 : ((pick < weight_shape_3_2 + weight_shape_5_2) ? 1 : 2);

                //* Gamma(3/2 + extra_shapes) = (1 + extra_shapes) unit exponentials + Z^2/2,  Z^2/2 = -ln(U) cos^2(2 pi U') (Box-Muller)
                real_t gamma_variate = ZERO;
                for (int i = 0; i <= extra_shapes; ++i)
                {
                    real_t uniform = Random<real_t>(rand_gen);
                    while (!(uniform > ZERO)) { uniform = Random<real_t>(rand_gen); }
                    gamma_variate -= math::log(uniform);
                }
                real_t uniform = Random<real_t>(rand_gen);
                while (!(uniform > ZERO)) { uniform = Random<real_t>(rand_gen); }
                const real_t box_muller_cos = math::cos(static_cast<real_t>(constant::TWO_PI) * Random<real_t>(rand_gen));
                gamma_variate -= math::log(uniform) * SQR(box_muller_cos);

                //* accept with probability sqrt(1 + eps/2) / (1 + eps/4)
                energy   = theta * gamma_variate;
                accepted = (Random<real_t>(rand_gen) * (ONE + static_cast<real_t>(0.25) * energy) < math::sqrt(ONE + HALF * energy));
            }

            //* isotropic direction, |u| = sqrt(eps (eps + 2))
            const real_t four_velocity_magnitude = math::sqrt(energy * (energy + TWO));
            const real_t cos_polar               = TWO * Random<real_t>(rand_gen) - ONE;
            const real_t sin_polar_sq            = ONE - SQR(cos_polar);
            const real_t sin_polar               = (sin_polar_sq > ZERO) ? math::sqrt(sin_polar_sq) : ZERO;
            const real_t azimuth                 = static_cast<real_t>(constant::TWO_PI) * Random<real_t>(rand_gen);
            v[0] = four_velocity_magnitude * sin_polar * math::cos(azimuth);
            v[1] = four_velocity_magnitude * sin_polar * math::sin(azimuth);
            v[2] = four_velocity_magnitude * cos_polar;

            pool.free_state(rand_gen);
        }

        //? Isotropic Juttner sample -> drifting at u_total_species (lab frame) by the flipping method of Zenitani (2015), arXiv:1504.03910,
        //? as in Entity's Maxwellian, generalised to a position-dependent drift direction. A plain Lorentz boost of each sample misses
        //? the Jacobian gamma'/gamma and undersamples forward-moving particles.
        Inline void drift_sample(vec_t<Dim::_3D>& v, real_t u_total_species_x, real_t u_total_species_y, real_t u_total_species_z) const
        {
            real_t u_total_species_sq = SQR(u_total_species_x) + SQR(u_total_species_y) + SQR(u_total_species_z);

            //! Safety net: make_init keeps |u_total_species| <= u_total_cap (< 0.99) on the sampling grid; this clamp acts at most between samples
            constexpr real_t MAX_DRIFT_SPEED    = static_cast<real_t>(0.99);
            constexpr real_t MAX_DRIFT_SPEED_SQ = MAX_DRIFT_SPEED * MAX_DRIFT_SPEED;
            if (u_total_species_sq >= MAX_DRIFT_SPEED_SQ)
            {
                const real_t rescale = MAX_DRIFT_SPEED / math::sqrt(u_total_species_sq);
                u_total_species_x *= rescale; u_total_species_y *= rescale; u_total_species_z *= rescale;
                u_total_species_sq = MAX_DRIFT_SPEED_SQ;
            }
            if (u_total_species_sq <= ZERO) { return; }

            const real_t u_total_species_magnitude = math::sqrt(u_total_species_sq);
            const real_t direction_x = u_total_species_x / u_total_species_magnitude;
            const real_t direction_y = u_total_species_y / u_total_species_magnitude;
            const real_t direction_z = u_total_species_z / u_total_species_magnitude;
            const real_t lorentz_factor_drift = ONE / math::sqrt(ONE - u_total_species_sq);

            const real_t four_velocity_x = v[0], four_velocity_y = v[1], four_velocity_z = v[2];
            const real_t lorentz_factor  = math::sqrt(ONE + SQR(four_velocity_x) + SQR(four_velocity_y) + SQR(four_velocity_z));

            //* four-velocity component along the drift, before any flip
            const real_t four_velocity_parallel_initial = four_velocity_x * direction_x + four_velocity_y * direction_y + four_velocity_z * direction_z;

            //* flip backward-going particles with probability -|u_total_species| u_parallel / gamma
            real_t four_velocity_parallel = four_velocity_parallel_initial;
            {
                auto rand_gen = pool.get_state();
                if (-u_total_species_magnitude * four_velocity_parallel > lorentz_factor * Random<real_t>(rand_gen)) { four_velocity_parallel = -four_velocity_parallel; }
                pool.free_state(rand_gen);
            }

            //* boost the parallel component: u_par' = Gamma_V (u_par + |V| gamma); the perpendicular part is untouched
            const real_t four_velocity_parallel_boosted = lorentz_factor_drift * (four_velocity_parallel + u_total_species_magnitude * lorentz_factor);

            v[0] = four_velocity_x - four_velocity_parallel_initial * direction_x + four_velocity_parallel_boosted * direction_x;
            v[1] = four_velocity_y - four_velocity_parallel_initial * direction_y + four_velocity_parallel_boosted * direction_y;
            v[2] = four_velocity_z - four_velocity_parallel_initial * direction_z + four_velocity_parallel_boosted * direction_z;
        }

        Inline void operator()(const coord_t<D>& x_Ph, vec_t<Dim::_3D>& v) const
        {
            //* 1. isotropic thermal four-velocity: exact Maxwell-Juttner at theta_s
            thermal_sample(x_Ph, v);

            //* 2. current-carrying drift of this species, u_drift_species = u_drift_coeff_s J_loaded
            real_t u_drift_species_x = ZERO, u_drift_species_y = ZERO, u_drift_species_z = ZERO;
            if (u_drift_coeff != ZERO)
            {
                real_t current_x, current_y, current_z;
                flds.loaded_current(x_Ph, current_x, current_y, current_z);
                u_drift_species_x = u_drift_coeff * current_x;
                u_drift_species_y = u_drift_coeff * current_y;
                u_drift_species_z = u_drift_coeff * current_z;
            }

            //* 3. velocity perturbation common to both species, already capped in make_init
            real_t u_perturbed_x, u_perturbed_y, u_perturbed_z;
            vel(x_Ph, u_perturbed_x, u_perturbed_y, u_perturbed_z);

            //* 4. lab-frame mean velocity u_total_species = u_perturbed + u_drift_species, sampled in one step
            drift_sample(v, u_perturbed_x + u_drift_species_x, u_perturbed_y + u_drift_species_y, u_perturbed_z + u_drift_species_z);
        }

    private:
        const arch::energy_dist::Maxwellian<M::Dim, M::CoordType> maxwellian;
        random_number_pool_t       pool;
        const real_t               theta;           //* kT_s / (m_s c^2)
        const real_t               u_drift_coeff;
        const InitFields<M::Dim>   flds;
        const TurbVelocity<M::Dim> vel;
    };

    //! Uniform spatial distribution
    template <Dimension D>
    struct UniformDist
    {
        Inline auto operator()(const coord_t<D>&) const -> real_t { return ONE; }
    };


    template <SimEngine::type S, class M>
    struct PGen
    {
        static constexpr auto D { M::Dim };

        static constexpr auto engines { ::traits::pgen::compatible_with<SimEngine::SRPIC> {} };
        static constexpr auto metrics { ::traits::pgen::compatible_with<Metric::Minkowski> {} };
        static constexpr auto dimensions { ::traits::pgen::compatible_with<Dim::_2D, Dim::_3D> {} };

        const SimulationParams& params;
        Metadomain<S, M>&       metadomain;

        //? Scales
        const real_t larmor0;                   //* rho0 [code length]
        const real_t skindepth0;                //* d0   [code length]
        const real_t sp1_mass;                  //* species 1 (electrons) mass, fiducial units
        const real_t sp2_mass;                  //* species 2 (positrons / ions) mass
        const real_t sp1_charge;                //* signed, |q| must be 1
        const real_t sp2_charge;
        const real_t mass_ratio;                //* sp2_mass / sp1_mass (= 1 for pair plasma)
        const real_t sigma0;                    //* (skindepth0/larmor0)^2 = magnetisation of the FIDUCIAL species at n0, |B| = 1

        //? Setup
        const real_t    B_guide;                //* guide field along x3 (0 -> zero-mean-field turbulence)
        const real_t    B_perturbed_rms;        //* RMS |B_perturbed|
        const int       kmin, kmax;             //* mode-number band (inclusive)
        const real_t    energy_spectrum_slope;  //* E(k) ~ k^energy_spectrum_slope, shell spectrum (required input)
        const long long B_seed;                 //* phases of B_perturbed
        const long long v_seed;                 //* phases of u_perturbed; MUST differ from B_seed
        const int       turb_kplane;            //* k-space restriction: -1 = 3D, 0 = x1x2, 1 = x2x3, 2 = x3x1
        const int       turb_a_axis;            //* vector potential axis: -1 = random _|_ k, 0/1/2 = x1/x2/x3
        const bool      turb_rand_amp;          //* true -> Gaussian random mode amplitudes
        const real_t    temperature;            //* species-1 temperature kT_1 / (m0 c^2)
        const real_t    temperature_ratio;      //* T_2 / T_1 (ratio of kT, not of theta)
        const bool      perturb_velocity;       //* false -> u_perturbed = 0
        const real_t    u_total_cap;            //* cap on |u_total_species| for both species [c]

        //? Geometry
        const real_t xmin, xmax, ymin, ymax, zmin, zmax, Lx, Ly, Lz;

        //? Derived quantities
        real_t                temperature_1 { ZERO }, temperature_2 { ZERO };                     //* kT_s / (m0 c^2)
        real_t                theta_1 { ZERO }, theta_2 { ZERO };                                 //* kT_s / (m_s c^2)
        real_t                enthalpy_per_particle_1 { ONE }, enthalpy_per_particle_2 { ONE };   //* h_s = m_s <gamma_s> + T_s [m0 c^2]
        std::array<real_t, 3> B_perturbed_rms_component { ZERO, ZERO, ZERO };
        real_t                AMP { ONE };        //* d0^2/rho0 = d0 sqrt(sigma0): J_code = AMP curl(B)
        real_t                u_drift_coeff_1 { ZERO }, u_drift_coeff_2 { ZERO };
        real_t                u_motional_coeff { ZERO };
        real_t                sigma_h_per_B_sq { ZERO };         //* sigma0 / w: local sigma_h = sigma_h_per_B_sq |B|^2

        real_t                u_perturbed_rms_alfven { ZERO };   //* Alfven-relation target for RMS |u_perturbed| [c]
        real_t                u_perturbed_cap_factor { ONE };    //* s <= 1 applied to u_perturbed by the u_total cap
        real_t                u_perturbed_rms { ZERO };          //* RMS |u_perturbed| loaded = s u_perturbed_rms_alfven (0 if perturb_velocity = false)
        std::array<real_t, 3> u_perturbed_rms_component { ZERO, ZERO, ZERO };

        //* peaks on the host sampling grid, after the cap [c]
        real_t                peak_u_drift_1 { ZERO }, peak_u_drift_2 { ZERO };
        real_t                peak_u_total_1 { ZERO }, peak_u_total_2 { ZERO };
        real_t                peak_charge_displacement { ZERO };   //* max |xi_s| [code length], set in print_setup, checked in load_charge_density

        std::vector<TurbMode> B_perturbed_modes_host;
        array_t<real_t*>      kx_d, ky_d, kz_d, bx_d, by_d, bz_d, ph_d, kcx_d, kcy_d, kcz_d;

        std::vector<TurbMode> u_perturbed_modes_host;
        array_t<real_t*>      u_perturbed_kx_d, u_perturbed_ky_d, u_perturbed_kz_d;
        array_t<real_t*>      u_perturbed_mode_x_d, u_perturbed_mode_y_d, u_perturbed_mode_z_d, u_perturbed_phase_d;

        //! Must be declared BEFORE init_flds: make_init() fills it as a side effect
        TurbVelocity<D> turb_vel;

        InitFields<D> init_flds;

        //* number of host samples along each axis; axes the spectrum does not span get one sample
        static auto sampling_counts(const PGen& g) -> std::array<int, 3>
        {
            const bool is_3d  = (D == Dim::_3D);
            const bool vary_x = (g.turb_kplane != 1);
            const bool vary_y = (g.turb_kplane != 2);
            const bool vary_z = is_3d && (g.turb_kplane != 0);

            const auto& mesh = g.metadomain.mesh();
            std::array<int, 3> counts { 1, 1, 1 };
            if (vary_x) counts[0] = sampling_points_per_side(g.kmax, static_cast<std::size_t>(mesh.n_active(in::x1)), is_3d);
            if (vary_y) counts[1] = sampling_points_per_side(g.kmax, static_cast<std::size_t>(mesh.n_active(in::x2)), is_3d);
            if constexpr (D == Dim::_3D)
            {
                if (vary_z) counts[2] = sampling_points_per_side(g.kmax, static_cast<std::size_t>(mesh.n_active(in::x3)), is_3d);
            }
            return counts;
        }

        static auto make_init(PGen& g) -> InitFields<D>
        {
            //! HARD FAILURES
            raise::ErrorIf(g.mass_ratio < (real_t)1.0,
                           "species 2 (positron/ion) must be at least as heavy as species 1 "
                           "(electron); mass_ratio = 1 is a pair plasma", HERE);
            raise::ErrorIf(g.kmin < 1 || g.kmax < g.kmin, "require 1 <= setup.kmin <= setup.kmax", HERE);
            raise::ErrorIf(g.turb_kplane > 2, "setup.turb_kplane must be -1 (isotropic 3D), 0 (X-Y), 1 (Y-Z) or 2 (Z-X)", HERE);
            raise::ErrorIf((D == Dim::_2D) && (g.turb_kplane != 0),
                           "in 2D only setup.turb_kplane = 0 (X-Y) is admissible; "
                           "there is no Z grid direction. Use setup.turb_a_axis to "
                           "choose the polarisation of B_perturbed instead", HERE);
            raise::ErrorIf(g.turb_a_axis > 2,
                           "setup.turb_a_axis must be -1 (random, perpendicular to k) "
                           "or 0 / 1 / 2 (fixed axis X / Y / Z)", HERE);
            raise::ErrorIf(g.temperature <= ZERO, "setup.temperature must be > 0", HERE);
            raise::ErrorIf(g.temperature_ratio <= ZERO, "setup.temperature_ratio (T_2/T_1) must be > 0", HERE);
            raise::ErrorIf(g.u_total_cap <= ZERO || g.u_total_cap >= static_cast<real_t>(0.99), "setup.u_total_cap must lie in (0, 0.99) [c]", HERE);

            //? u_drift_species = sign(q_s) AMP J / (h_s sum n/h) assumes |q_s| = 1 and opposite signs (counter-drift at zero net charge)
            raise::ErrorIf(math::abs(math::abs(g.sp1_charge) - ONE) > (real_t)1.0e-5 ||
                           math::abs(math::abs(g.sp2_charge) - ONE) > (real_t)1.0e-5,
                           "current loading requires |q_s| = 1 for both species: "
                           "u_drift_species = sign(q_s) * AMP * curl(B) / (h_s sum n/h) assumes it, "
                           "so the loaded current would be wrong otherwise.", HERE);
            raise::ErrorIf(g.sp1_charge * g.sp2_charge >= ZERO,
                           "current loading requires the two species to have OPPOSITE charge "
                           "signs: they must counter-drift to carry J at zero net charge.", HERE);

            //? T_s = kT_s / (m0 c^2) enters pressures, enthalpies and Debye lengths; theta_s = T_s / m_s enters the sampler and <gamma_s>
            g.temperature_1 = g.temperature;
            g.temperature_2 = g.temperature * g.temperature_ratio;
            g.theta_1       = g.temperature_1 / g.sp1_mass;
            g.theta_2       = g.temperature_2 / g.sp2_mass;

            g.enthalpy_per_particle_1 = enthalpy_per_particle(g.sp1_mass, g.temperature_1);
            g.enthalpy_per_particle_2 = enthalpy_per_particle(g.sp2_mass, g.temperature_2);

            //* sigma_h = sigma0 |B|^2 / w, w = enthalpy density; sets the perpendicular-current correction
            g.sigma_h_per_B_sq = g.sigma0 / enthalpy(g.sp1_mass, g.temperature_1, g.sp2_mass, g.temperature_2);

            const bool is_3d = (D == Dim::_3D);

            //? B_perturbed with the Yee-matched k~: exactly divergence-free on the grid, and -(k~ x b) sin(k.x + ph) equals the discrete Yee curl
            const auto&  mesh_for_k = g.metadomain.mesh();
            const double cell_dx    = (double)g.Lx / (double)mesh_for_k.n_active(in::x1);
            const double cell_dy    = (double)g.Ly / (double)mesh_for_k.n_active(in::x2);
            double       cell_dz    = 0.0;
            if constexpr (D == Dim::_3D) { cell_dz = (double)g.Lz / (double)mesh_for_k.n_active(in::x3); }

            g.B_perturbed_rms_component = init_turbulence(g.Lx, g.Ly, g.Lz, is_3d, g.B_perturbed_rms,
                                                          g.kmin, g.kmax, g.B_seed,
                                                          g.energy_spectrum_slope, g.turb_kplane, g.turb_a_axis,
                                                          g.turb_rand_amp, { cell_dx, cell_dy, cell_dz }, g.B_perturbed_modes_host);

            g.AMP = SQR(g.skindepth0) / g.larmor0;      //* AMP = d0^2/rho0 = d0 sqrt(sigma0)

            //* u_drift_coeff_s = sign(q_s) AMP / (h_s sum_t n_t/h_t), n_t = N_PER_SP (InjectNonUniform splits the total density evenly)
            const real_t inv_enthalpy_sum = N_PER_SP * (ONE / g.enthalpy_per_particle_1 + ONE / g.enthalpy_per_particle_2);
            const real_t sign_1           = (g.sp1_charge < ZERO) ? -ONE : ONE;
            const real_t sign_2           = (g.sp2_charge < ZERO) ? -ONE : ONE;
            g.u_drift_coeff_1 = sign_1 * g.AMP / (g.enthalpy_per_particle_1 * inv_enthalpy_sum);
            g.u_drift_coeff_2 = sign_2 * g.AMP / (g.enthalpy_per_particle_2 * inv_enthalpy_sum);

            //* u_motional = u_perturbed + u_drift_1 + u_drift_2 (InitFields::motional_E): zero correction for a pair plasma at equal T
            g.u_motional_coeff = g.u_drift_coeff_1 + g.u_drift_coeff_2;

            raise::ErrorIf(g.perturb_velocity && (g.v_seed == g.B_seed),
                           "setup.v_seed must differ from setup.B_seed. Equal seeds produce "
                           "the same phases and 'random' amplitudes for each mode, yielding "
                           "u_perturbed || B_perturbed everywhere -- a pure Elsasser state (z- = 0) "
                           "that is an exact nonlinear solution and does not cascade at all.", HERE);

            //~ Relativistic Alfven relation, w = relativistic enthalpy density:
            //~     u_perturbed_rms = B_perturbed_rms / sqrt(4 pi w + B^2) c  =  (B_perturbed_rms / B_rms) vA,   vA / c = sqrt(sigma_h / (1 + sigma_h))
            //~ In 2D with k _|_ B_guide the modes do not propagate, so this sets the kinetic/magnetic energy partition.
            {
                const real_t B_rms_ = (real_t)std::sqrt((double)(g.B_guide * g.B_guide + g.B_perturbed_rms * g.B_perturbed_rms));
                const real_t B_ref_ = (B_rms_ > ZERO) ? B_rms_ : ONE;
                const real_t w_     = enthalpy(g.sp1_mass, g.temperature_1, g.sp2_mass, g.temperature_2);
                const real_t sig_h  = g.sigma0 * B_ref_ * B_ref_ / w_;
                const real_t vA_h   = (real_t)std::sqrt((double)sig_h / (1.0 + (double)sig_h));

                //* computed even when perturb_velocity = false: the eddy-time estimate uses it
                g.u_perturbed_rms_alfven = vA_h * g.B_perturbed_rms / B_ref_;

                //* perturb_velocity = false -> target 0 -> empty table -> u_perturbed = 0 (drifts unaffected)
                const real_t u_perturbed_rms_target = g.perturb_velocity ? g.u_perturbed_rms_alfven : ZERO;
                g.u_perturbed_rms           = u_perturbed_rms_target;
                g.u_perturbed_rms_component = init_turbulence(g.Lx, g.Ly, g.Lz, is_3d, u_perturbed_rms_target,
                                                              g.kmin, g.kmax, g.v_seed,
                                                              g.energy_spectrum_slope, g.turb_kplane, g.turb_a_axis,
                                                              g.turb_rand_amp, { 0.0, 0.0, 0.0 }, g.u_perturbed_modes_host);
            }

            //~ u_total cap. Each species moves at u_total_species = s u_perturbed + u_drift_species; where modes add in phase the local
            //~ speed is several times the RMS and can exceed c at large vA/c. u_perturbed is scaled by
            //~     s = largest value in [0, 1] with  max_{x,species} |s u_perturbed + u_drift_species| <= u_total_cap,
            //~ which keeps it solenoidal; the kinetic/magnetic fluctuation energy drops by s^2. f(s) is convex, so {f <= cap} is an interval
            //~ containing 0 and bisection finds its upper end. If f(0) > cap, the field needs more current than this density can carry
            //~ (u_drift ~ J / (n q)): lower kmax or B_perturbed_rms, or lower sigma0. Raising ppc0 does not help.
            {
                const std::array<int, 3> counts = sampling_counts(g);
                const std::size_t n_samples = (std::size_t)counts[0] * (std::size_t)counts[1] * (std::size_t)counts[2];

                std::vector<double> u_perturbed_samples(3 * n_samples, 0.0);
                std::vector<double> first_order_current_samples(3 * n_samples, 0.0);   //* first-order J_code/AMP

                std::size_t sample_index = 0;
                for (int ix = 0; ix < counts[0]; ++ix)
                    for (int iy = 0; iy < counts[1]; ++iy)
                        for (int iz = 0; iz < counts[2]; ++iz)
                        {
                            const double X = sample_coordinate((double)g.xmin, (double)g.Lx, ix, counts[0]);
                            const double Y = sample_coordinate((double)g.ymin, (double)g.Ly, iy, counts[1]);
                            const double Z = (D == Dim::_3D) ? sample_coordinate((double)g.zmin, (double)g.Lz, iz, counts[2]) : 0.0;

                            host_cos_sum(g.u_perturbed_modes_host, X, Y, Z, u_perturbed_samples[3 * sample_index], u_perturbed_samples[3 * sample_index + 1], u_perturbed_samples[3 * sample_index + 2]);
                            //* first-order current: the convection term depends on s itself and is second order (checked in print_setup)
                            host_first_order_current(g.B_perturbed_modes_host, (double)g.B_guide, (double)g.sigma_h_per_B_sq, X, Y, Z, first_order_current_samples[3 * sample_index], first_order_current_samples[3 * sample_index + 1], first_order_current_samples[3 * sample_index + 2]);
                            ++sample_index;
                        }

                const double u_drift_coeff_1 = (double)g.u_drift_coeff_1;
                const double u_drift_coeff_2 = (double)g.u_drift_coeff_2;

                //* peak speeds over the sampling grid for a given cap factor
                auto peak_speeds = [&](double cap_factor, double& peak_u_perturbed, double& peak_u_drift_1, double& peak_u_drift_2, double& peak_u_total_1, double& peak_u_total_2)
                {
                    peak_u_perturbed = 0.0; peak_u_drift_1 = 0.0; peak_u_drift_2 = 0.0; peak_u_total_1 = 0.0; peak_u_total_2 = 0.0;
                    for (std::size_t n = 0; n < n_samples; ++n)
                    {
                        const double u_perturbed_x = cap_factor * u_perturbed_samples[3 * n], u_perturbed_y = cap_factor * u_perturbed_samples[3 * n + 1], u_perturbed_z = cap_factor * u_perturbed_samples[3 * n + 2];
                        const double current_x = first_order_current_samples[3 * n], current_y = first_order_current_samples[3 * n + 1], current_z = first_order_current_samples[3 * n + 2];

                        const double u_drift_1_x = u_drift_coeff_1 * current_x, u_drift_1_y = u_drift_coeff_1 * current_y, u_drift_1_z = u_drift_coeff_1 * current_z;
                        const double u_drift_2_x = u_drift_coeff_2 * current_x, u_drift_2_y = u_drift_coeff_2 * current_y, u_drift_2_z = u_drift_coeff_2 * current_z;

                        peak_u_perturbed = std::max(peak_u_perturbed, std::sqrt(SQR(u_perturbed_x) + SQR(u_perturbed_y) + SQR(u_perturbed_z)));
                        peak_u_drift_1   = std::max(peak_u_drift_1, std::sqrt(SQR(u_drift_1_x) + SQR(u_drift_1_y) + SQR(u_drift_1_z)));
                        peak_u_drift_2   = std::max(peak_u_drift_2, std::sqrt(SQR(u_drift_2_x) + SQR(u_drift_2_y) + SQR(u_drift_2_z)));
                        peak_u_total_1   = std::max(peak_u_total_1, std::sqrt(SQR(u_perturbed_x + u_drift_1_x) + SQR(u_perturbed_y + u_drift_1_y) + SQR(u_perturbed_z + u_drift_1_z)));
                        peak_u_total_2   = std::max(peak_u_total_2, std::sqrt(SQR(u_perturbed_x + u_drift_2_x) + SQR(u_perturbed_y + u_drift_2_y) + SQR(u_perturbed_z + u_drift_2_z)));
                    }
                };

                //* f(s) = max over species and samples of |s u_perturbed + u_drift_species|
                auto peak_u_total_both_species = [&](double cap_factor) -> double
                {
                    double peak_u_perturbed, peak_u_drift_1, peak_u_drift_2, peak_u_total_1, peak_u_total_2;
                    peak_speeds(cap_factor, peak_u_perturbed, peak_u_drift_1, peak_u_drift_2, peak_u_total_1, peak_u_total_2);
                    return std::max(peak_u_total_1, peak_u_total_2);
                };

                const double u_total_cap = (double)g.u_total_cap;

                double peak_u_total_without_perturbation = peak_u_total_both_species(0.0);
                MPI_Bcast(&peak_u_total_without_perturbation, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                raise::ErrorIf(peak_u_total_without_perturbation > u_total_cap,
                               "u_drift_species alone exceeds setup.u_total_cap: the field asks "
                               "for more current than this density can carry. Lower setup.kmax, enlarge "
                               "the box, lower setup.B_perturbed_rms, or lower sigma0 (raise scales.larmor0); "
                               "or raise setup.u_total_cap (< 0.99).", HERE);

                double cap_factor = 1.0;
                if (peak_u_total_both_species(1.0) > u_total_cap)
                {
                    double cap_factor_inside = 0.0, cap_factor_outside = 1.0;   //* f(inside) <= cap < f(outside)
                    for (int iteration = 0; iteration < 60; ++iteration)
                    {
                        const double cap_factor_mid = 0.5 * (cap_factor_inside + cap_factor_outside);
                        if (peak_u_total_both_species(cap_factor_mid) <= u_total_cap) { cap_factor_inside = cap_factor_mid; }
                        else                                                          { cap_factor_outside = cap_factor_mid; }
                    }
                    cap_factor = cap_factor_inside;
                }
                MPI_Bcast(&cap_factor, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                //* apply s to every u_perturbed mode; the RMS scales linearly
                for (auto& m : g.u_perturbed_modes_host)
                {
                    m.bx = (real_t)((double)m.bx * cap_factor);
                    m.by = (real_t)((double)m.by * cap_factor);
                    m.bz = (real_t)((double)m.bz * cap_factor);
                }
                g.u_perturbed_cap_factor = (real_t)cap_factor;
                g.u_perturbed_rms        = (real_t)((double)g.u_perturbed_rms * cap_factor);
                for (auto& component : g.u_perturbed_rms_component) { component = (real_t)((double)component * cap_factor); }

                double peak_u_perturbed, peak_u_drift_1, peak_u_drift_2, peak_u_total_1, peak_u_total_2;
                peak_speeds(cap_factor, peak_u_perturbed, peak_u_drift_1, peak_u_drift_2, peak_u_total_1, peak_u_total_2);
                g.peak_u_drift_1   = (real_t)peak_u_drift_1;
                g.peak_u_drift_2   = (real_t)peak_u_drift_2;
                g.peak_u_total_1   = (real_t)peak_u_total_1;
                g.peak_u_total_2   = (real_t)peak_u_total_2;
            }

            const std::size_t nv = g.u_perturbed_modes_host.size();
            g.u_perturbed_kx_d     = array_t<real_t*> { "u_perturbed_kx",     nv };
            g.u_perturbed_ky_d     = array_t<real_t*> { "u_perturbed_ky",     nv };
            g.u_perturbed_kz_d     = array_t<real_t*> { "u_perturbed_kz",     nv };
            g.u_perturbed_mode_x_d = array_t<real_t*> { "u_perturbed_mode_x", nv };
            g.u_perturbed_mode_y_d = array_t<real_t*> { "u_perturbed_mode_y", nv };
            g.u_perturbed_mode_z_d = array_t<real_t*> { "u_perturbed_mode_z", nv };
            g.u_perturbed_phase_d  = array_t<real_t*> { "u_perturbed_phase",  nv };

            {
                auto h_kx     = Kokkos::create_mirror_view(g.u_perturbed_kx_d);
                auto h_ky     = Kokkos::create_mirror_view(g.u_perturbed_ky_d);
                auto h_kz     = Kokkos::create_mirror_view(g.u_perturbed_kz_d);
                auto h_mode_x = Kokkos::create_mirror_view(g.u_perturbed_mode_x_d);
                auto h_mode_y = Kokkos::create_mirror_view(g.u_perturbed_mode_y_d);
                auto h_mode_z = Kokkos::create_mirror_view(g.u_perturbed_mode_z_d);
                auto h_phase  = Kokkos::create_mirror_view(g.u_perturbed_phase_d);

                for (std::size_t m = 0; m < nv; ++m)
                {
                    h_kx(m)     = g.u_perturbed_modes_host[m].kx;
                    h_ky(m)     = g.u_perturbed_modes_host[m].ky;
                    h_kz(m)     = g.u_perturbed_modes_host[m].kz;
                    h_mode_x(m) = g.u_perturbed_modes_host[m].bx;
                    h_mode_y(m) = g.u_perturbed_modes_host[m].by;
                    h_mode_z(m) = g.u_perturbed_modes_host[m].bz;
                    h_phase(m)  = g.u_perturbed_modes_host[m].phase;
                }

                Kokkos::deep_copy(g.u_perturbed_kx_d, h_kx);         Kokkos::deep_copy(g.u_perturbed_ky_d, h_ky);
                Kokkos::deep_copy(g.u_perturbed_kz_d, h_kz);         Kokkos::deep_copy(g.u_perturbed_mode_x_d, h_mode_x);
                Kokkos::deep_copy(g.u_perturbed_mode_y_d, h_mode_y); Kokkos::deep_copy(g.u_perturbed_mode_z_d, h_mode_z);
                Kokkos::deep_copy(g.u_perturbed_phase_d, h_phase);
            }

            g.turb_vel = TurbVelocity<D> { g.u_perturbed_kx_d, g.u_perturbed_ky_d, g.u_perturbed_kz_d,
                                           g.u_perturbed_mode_x_d, g.u_perturbed_mode_y_d, g.u_perturbed_mode_z_d,
                                           g.u_perturbed_phase_d, nv };

            const std::size_t nm = g.B_perturbed_modes_host.size();
            g.kx_d  = array_t<real_t*> { "turb_kx",  nm };
            g.ky_d  = array_t<real_t*> { "turb_ky",  nm };
            g.kz_d  = array_t<real_t*> { "turb_kz",  nm };
            g.bx_d  = array_t<real_t*> { "turb_bx",  nm };
            g.by_d  = array_t<real_t*> { "turb_by",  nm };
            g.bz_d  = array_t<real_t*> { "turb_bz",  nm };
            g.ph_d  = array_t<real_t*> { "turb_ph",  nm };
            g.kcx_d = array_t<real_t*> { "turb_kcx", nm };
            g.kcy_d = array_t<real_t*> { "turb_kcy", nm };
            g.kcz_d = array_t<real_t*> { "turb_kcz", nm };

            auto h_kx  = Kokkos::create_mirror_view(g.kx_d);
            auto h_ky  = Kokkos::create_mirror_view(g.ky_d);
            auto h_kz  = Kokkos::create_mirror_view(g.kz_d);
            auto h_bx  = Kokkos::create_mirror_view(g.bx_d);
            auto h_by  = Kokkos::create_mirror_view(g.by_d);
            auto h_bz  = Kokkos::create_mirror_view(g.bz_d);
            auto h_ph  = Kokkos::create_mirror_view(g.ph_d);
            auto h_kcx = Kokkos::create_mirror_view(g.kcx_d);
            auto h_kcy = Kokkos::create_mirror_view(g.kcy_d);
            auto h_kcz = Kokkos::create_mirror_view(g.kcz_d);

            for (std::size_t m = 0; m < nm; ++m)
            {
                h_kx(m)  = g.B_perturbed_modes_host[m].kx;
                h_ky(m)  = g.B_perturbed_modes_host[m].ky;
                h_kz(m)  = g.B_perturbed_modes_host[m].kz;
                h_bx(m)  = g.B_perturbed_modes_host[m].bx;
                h_by(m)  = g.B_perturbed_modes_host[m].by;
                h_bz(m)  = g.B_perturbed_modes_host[m].bz;
                h_ph(m)  = g.B_perturbed_modes_host[m].phase;
                h_kcx(m) = g.B_perturbed_modes_host[m].kcx;
                h_kcy(m) = g.B_perturbed_modes_host[m].kcy;
                h_kcz(m) = g.B_perturbed_modes_host[m].kcz;
            }

            Kokkos::deep_copy(g.kx_d, h_kx);   Kokkos::deep_copy(g.ky_d, h_ky);
            Kokkos::deep_copy(g.kz_d, h_kz);   Kokkos::deep_copy(g.bx_d, h_bx);
            Kokkos::deep_copy(g.by_d, h_by);   Kokkos::deep_copy(g.bz_d, h_bz);
            Kokkos::deep_copy(g.ph_d, h_ph);
            Kokkos::deep_copy(g.kcx_d, h_kcx); Kokkos::deep_copy(g.kcy_d, h_kcy);
            Kokkos::deep_copy(g.kcz_d, h_kcz);

            //* g.turb_vel is filled above, so InitFields can hold it and serve E = -(u_motional x B) from ex1/ex2/ex3
            return InitFields<D> { g.B_guide,
                                   g.kx_d, g.ky_d, g.kz_d,
                                   g.bx_d, g.by_d, g.bz_d, g.ph_d,
                                   g.kcx_d, g.kcy_d, g.kcz_d, nm,
                                   g.turb_vel, g.u_motional_coeff, g.sigma_h_per_B_sq };
        }

        //? Densities in units of n0 (= ppc0 / V0, the TOTAL fiducial density). InitPrtls injects n_tot = 1 split evenly: n_s = 1/2.
        //? Pressures and enthalpy densities use n_s; species sigma, skin depth and Debye length (print_setup) use n_tot.
        static constexpr real_t N_TOTAL  = static_cast<real_t>(1.0);
        static constexpr real_t N_PER_SP = static_cast<real_t>(0.5);

        //? <gamma> of a 3D Maxwell-Juttner distribution, K1(1/theta)/K2(1/theta) + 3 theta, by quadrature (no special functions, no underflow).
        //? With eps = gamma - 1 = theta t^2 the weight in t is t^2 (1 + eps) sqrt(1 + eps/2) exp(-t^2), negligible beyond t = 9.
        //? Simpson's rule with 4000 intervals matches the Bessel form to ~1e-15 for 0.001 <= theta <= 100. Returned in double: for ions
        //? <gamma> - 1 can be ~1e-6, below single precision.
        static auto gamma_mean(real_t theta) -> double
        {
            constexpr int    n_intervals = 4000;
            constexpr double t_max       = 9.0;
            const double     h           = t_max / (double)n_intervals;
            double weight_sum = 0.0, gamma_weight_sum = 0.0;
            for (int i = 0; i <= n_intervals; ++i)
            {
                const double t       = (double)i * h;
                const double eps     = (double)theta * t * t;
                const double weight  = t * t * (1.0 + eps) * std::sqrt(1.0 + 0.5 * eps) * std::exp(-t * t);
                const double simpson = (i == 0 || i == n_intervals) ? 1.0 : ((i % 2 == 1) ? 4.0 : 2.0);
                weight_sum       += simpson * weight;
                gamma_weight_sum += simpson * (1.0 + eps) * weight;
            }
            return gamma_weight_sum / weight_sum;
        }

        //? Enthalpy per particle [m0 c^2]:  h_s = m_s <gamma_s> + T_s. Inertia of a drifting thermal species: momentum density = n_s h_s u
        static auto enthalpy_per_particle(real_t mass, real_t temperature_s) -> real_t
        {
            return (real_t)((double)mass * gamma_mean(temperature_s / mass) + (double)temperature_s);
        }

        //? Enthalpy density [n0 m0 c^2]:  w = sum_s n_s h_s; sets the Alfven speed
        static auto enthalpy(real_t mass_1, real_t temperature_1_, real_t mass_2, real_t temperature_2_) -> real_t
        {
            return N_PER_SP * enthalpy_per_particle(mass_1, temperature_1_) + N_PER_SP * enthalpy_per_particle(mass_2, temperature_2_);
        }

        static auto read_real_setting(const SimulationParams& p, const std::string& key, real_t default_value) -> real_t
        {
            if (!p.contains(key)) { return p.template get<real_t>(key, default_value); }
            const std::any& value = p.allVars().at(key);
            if (value.type() == typeid(int)) { return static_cast<real_t>(std::any_cast<int>(value)); }
            return p.template get<real_t>(key);
        }

        static auto read_required_real_setting(const SimulationParams& p, const std::string& key, const std::string& description) -> real_t
        {
            raise::ErrorIf(!p.contains(key), key + " is required in the input: " + description, HERE);
            return read_real_setting(p, key, ZERO);
        }

        static auto read_integer_setting(const SimulationParams& p, const std::string& key, int default_value) -> int
        {
            if (!p.contains(key)) { return p.template get<int>(key, default_value); }
            const std::any& value = p.allVars().at(key);
            if (value.type() == typeid(int)) { return std::any_cast<int>(value); }
            const real_t real_value = p.template get<real_t>(key);
            raise::ErrorIf(std::floor((double)real_value) != (double)real_value,
                           key + " must be a whole number, but the input gives " + std::to_string((double)real_value), HERE);
            return static_cast<int>(real_value);
        }

        //? Parameter set by Entity itself (not the input) as float or double (real_t or simtime_t); get<T> needs the exact type, so accept both. Aborts if missing.
        static auto read_stored_real(const SimulationParams& p, const std::string& key) -> real_t
        {
            raise::ErrorIf(!p.contains(key), key + " not found in the simulation parameters", HERE);
            const std::any& value = p.allVars().at(key);
            if (value.type() == typeid(float))  { return static_cast<real_t>(std::any_cast<float>(value)); }
            if (value.type() == typeid(double)) { return static_cast<real_t>(std::any_cast<double>(value)); }
            raise::Error(key + " is stored with a type other than float or double", HERE);
            return ZERO;
        }

        //? Integer parameter set by Entity itself, stored under an integer type that differs between keys (e.g. unsigned short); -1 if missing or not an integer
        static auto read_stored_integer(const SimulationParams& p, const std::string& key) -> long long
        {
            if (!p.contains(key)) { return -1; }
            const std::any& value = p.allVars().at(key);
            if (value.type() == typeid(unsigned short))     { return static_cast<long long>(std::any_cast<unsigned short>(value)); }
            if (value.type() == typeid(unsigned int))       { return static_cast<long long>(std::any_cast<unsigned int>(value)); }
            if (value.type() == typeid(unsigned long))      { return static_cast<long long>(std::any_cast<unsigned long>(value)); }
            if (value.type() == typeid(unsigned long long)) { return static_cast<long long>(std::any_cast<unsigned long long>(value)); }
            if (value.type() == typeid(short))              { return static_cast<long long>(std::any_cast<short>(value)); }
            if (value.type() == typeid(int))                { return static_cast<long long>(std::any_cast<int>(value)); }
            if (value.type() == typeid(long))               { return static_cast<long long>(std::any_cast<long>(value)); }
            if (value.type() == typeid(long long))          { return std::any_cast<long long>(value); }
            return -1;
        }

        static auto species_mass(Metadomain<S, M>& m, std::size_t idx) -> real_t
        {
            const auto& species = m.species_params();
            raise::ErrorIf(species.size() < 2, "ERROR in species_mass: pgen needs >= 2 species (1 = electrons, 2 = positrons/ions)", HERE);
            const real_t ms = (real_t)species[idx].mass();
            raise::ErrorIf(ms <= (real_t)0.0, "ERROR in species_mass: species mass must be > 0", HERE);
            return ms;
        }

        static auto species_charge(Metadomain<S, M>& m, std::size_t idx) -> real_t
        {
            const auto& species = m.species_params();
            raise::ErrorIf(species.size() < 2, "ERROR in species_charge: pgen needs >= 2 species (1 = electrons, 2 = positrons/ions)", HERE);
            return (real_t)species[idx].charge();
        }

        PGen(const SimulationParams& p, Metadomain<S, M>& m) :
            params                { p }, metadomain { m },
            larmor0               { p.template get<real_t>("scales.larmor0") },
            skindepth0            { p.template get<real_t>("scales.skindepth0") },
            sp1_mass              { species_mass(m, 0) },
            sp2_mass              { species_mass(m, 1) },
            sp1_charge            { species_charge(m, 0) },
            sp2_charge            { species_charge(m, 1) },
            mass_ratio            { sp2_mass / sp1_mass },
            sigma0                { (skindepth0 / larmor0) * (skindepth0 / larmor0) },
            B_guide               { read_real_setting(p, "setup.B_guide", (real_t)1.0) },
            B_perturbed_rms       { read_real_setting(p, "setup.B_perturbed_rms", (real_t)1.0) },
            kmin                  { read_integer_setting(p, "setup.kmin", 1) },
            kmax                  { read_integer_setting(p, "setup.kmax", 4) },
            energy_spectrum_slope { read_required_real_setting(p, "setup.spectral_slope", "slope of the shell spectrum E(k) ~ k^spectral_slope (0 = flat, -5/3 = Kolmogorov)") },
            B_seed                { static_cast<long long>(read_integer_setting(p, "setup.B_seed", 12345)) },
            v_seed                { static_cast<long long>(read_integer_setting(p, "setup.v_seed", 67890)) },
            turb_kplane           { read_integer_setting(p, "setup.turb_kplane", (D == Dim::_3D) ? -1 : 0) },
            turb_a_axis           { read_integer_setting(p, "setup.turb_a_axis", -1) },
            turb_rand_amp         { p.template get<bool>("setup.turb_rand_amp", true) },
            temperature           { read_real_setting(p, "setup.temperature", (real_t)0.01) },
            temperature_ratio     { read_real_setting(p, "setup.temperature_ratio", (real_t)1.0) },
            perturb_velocity      { p.template get<bool>("setup.perturb_velocity", true) },
            u_total_cap           { read_real_setting(p, "setup.u_total_cap", (real_t)0.7) },
            xmin                  { m.mesh().extent(in::x1).first },
            xmax                  { m.mesh().extent(in::x1).second },
            ymin                  { m.mesh().extent(in::x2).first },
            ymax                  { m.mesh().extent(in::x2).second },
            zmin                  { (D == Dim::_3D) ? m.mesh().extent(in::x3).first : (real_t)0.0 },
            zmax                  { (D == Dim::_3D) ? m.mesh().extent(in::x3).second : (real_t)1.0 },
            Lx                    { xmax - xmin },
            Ly                    { ymax - ymin },
            Lz                    { zmax - zmin },
            init_flds             { make_init(*this) } { print_setup(); }

        //? Startup diagnostics. Lengths, times and fields in code units; c = 1.
        //? Species sigma, skin depth and Debye length use the TOTAL density n_tot = n_1 + n_2.
        void print_setup()
        {
            int mpi_rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

            const bool        is_pair = (mass_ratio == ONE);
            const std::string sp2s    = is_pair ? "positrons" : "ions";
            const std::string SP2u    = is_pair ? "Positrons" : "Ions";
            const std::string sp2_tag = is_pair ? "e+" : "i";   //* subscript of omega_p for species 2
            const std::string title   = is_pair ? "Decaying turbulence (pair plasma)" : "Decaying turbulence (ion-electron plasma)";

            const auto   nx1 = metadomain.mesh().n_active(in::x1);
            const auto   nx2 = metadomain.mesh().n_active(in::x2);
            const real_t dx  = Lx / (real_t)nx1;
            const real_t dy  = Ly / (real_t)nx2;
            real_t       dz  = ONE;
            if constexpr (D == Dim::_3D) { dz = Lz / (real_t)metadomain.mesh().n_active(in::x3); }
            //* numerics set by Entity (Algorithms::setParams): dt = CFL dx0, dx0 = dx/sqrt(D); filter passes and shape order affect numerical heating
            const real_t    dt                = read_stored_real(params, "algorithms.timestep.dt");
            const long long current_filters   = read_stored_integer(params, "algorithms.current_filters");
            const long long deposit_order     = read_stored_integer(params, "algorithms.deposit.order");   //* = SHAPE_ORDER (build option)
            auto integer_text = [](long long value) -> std::string { return (value >= 0) ? std::to_string(value) : std::string("n/a (key or type not found)"); };

            const double g1 = gamma_mean(theta_1);   //* <gamma> of electrons, rest frame
            const double g2 = gamma_mean(theta_2);   //* <gamma> of positrons / ions, rest frame

            //* field scale for magnetisations, betas and gyroradii: B_rms (1 if B = 0)
            const real_t B_rms = (real_t)std::sqrt((double)(B_guide * B_guide + B_perturbed_rms * B_perturbed_rms));
            const real_t B_ref = (B_rms > ZERO) ? B_rms : ONE;

            //! ---- magnetisation:  sigma = sigma0 B^2 / inertia ------------------------------------
            //!   species:  inertia = n_tot m_s (cold),  n_tot h_s (hot),  h_s = m_s <g_s> + T_s
            //!   plasma:   inertia = sum_s n_s m_s (cold),  w = sum_s n_s h_s (hot)   =>   1/sigma_plasma = sum_s (n_s/n_tot) / sigma_s
            //!   The pressure T_s adds 2.9% to the inertia at theta = 0.03, more at higher theta.
            const real_t w_enth = enthalpy(sp1_mass, temperature_1, sp2_mass, temperature_2);
            auto magnetisation  = [&](real_t B, real_t inertia) -> real_t { return sigma0 * B * B / inertia; };

            const real_t sigma_1_cold       = magnetisation(B_ref, N_TOTAL * sp1_mass);
            const real_t sigma_2_cold       = magnetisation(B_ref, N_TOTAL * sp2_mass);
            const real_t sigma_cold         = magnetisation(B_ref, N_PER_SP * (sp1_mass + sp2_mass));
            const real_t sigma_1_hot        = magnetisation(B_ref, N_TOTAL * enthalpy_per_particle_1);
            const real_t sigma_2_hot        = magnetisation(B_ref, N_TOTAL * enthalpy_per_particle_2);
            const real_t sigma_hot          = magnetisation(B_ref, w_enth);
            const real_t sigma_1_cold_guide = magnetisation(B_guide, N_TOTAL * sp1_mass);
            const real_t sigma_2_cold_guide = magnetisation(B_guide, N_TOTAL * sp2_mass);
            const real_t sigma_cold_guide   = magnetisation(B_guide, N_PER_SP * (sp1_mass + sp2_mass));
            const real_t sigma_1_hot_guide  = magnetisation(B_guide, N_TOTAL * enthalpy_per_particle_1);
            const real_t sigma_2_hot_guide  = magnetisation(B_guide, N_TOTAL * enthalpy_per_particle_2);
            const real_t sigma_hot_guide    = magnetisation(B_guide, w_enth);
            const real_t vA_hot             = (real_t)std::sqrt((double)sigma_hot / (1.0 + (double)sigma_hot));   //* vA/c = sqrt(sigma_h / (1 + sigma_h))

            //! ---- kinetic scales --------------------------------------------------------------------
            //!   d_s        = d0 sqrt(m_s / n_tot)        (cold)
            //!   d_s        = d0 sqrt(m_s <g_s> / n_tot)  (hot, relativistic inertia)
            //!   lambda_D,s = d0 sqrt(T_s / n_tot)        (no inertia factor; no cold value)
            //!   plasma:    1/x^2 = sum_s (n_s/n_tot) / x_s^2   (NOT sum_s 1/x_s^2, which double-counts)
            const double density_fraction = (double)N_PER_SP / (double)N_TOTAL;   //* n_s / n_tot, equal for both species
            auto plasma_scale = [&](real_t scale_1, real_t scale_2) -> real_t { return (real_t)(1.0 / std::sqrt(density_fraction / SQR((double)scale_1) + density_fraction / SQR((double)scale_2))); };

            const real_t skin_depth_1_cold      = skindepth0 * (real_t)std::sqrt((double)sp1_mass / (double)N_TOTAL);
            const real_t skin_depth_2_cold      = skindepth0 * (real_t)std::sqrt((double)sp2_mass / (double)N_TOTAL);
            const real_t skin_depth_plasma_cold = plasma_scale(skin_depth_1_cold, skin_depth_2_cold);
            const real_t skin_depth_1_hot       = skindepth0 * (real_t)std::sqrt(g1 * (double)sp1_mass / (double)N_TOTAL);
            const real_t skin_depth_2_hot       = skindepth0 * (real_t)std::sqrt(g2 * (double)sp2_mass / (double)N_TOTAL);
            const real_t skin_depth_plasma_hot  = plasma_scale(skin_depth_1_hot, skin_depth_2_hot);

            const real_t debye_length_1         = skindepth0 * (real_t)std::sqrt((double)temperature_1 / (double)N_TOTAL);
            const real_t debye_length_2         = skindepth0 * (real_t)std::sqrt((double)temperature_2 / (double)N_TOTAL);
            const real_t debye_length_plasma    = plasma_scale(debye_length_1, debye_length_2);   //* the grid criterion uses this (the finite-grid instability aliases the TOTAL charge)

            //! thermal gyroradius with the thermal FOUR-velocity sqrt(<g>^2 - 1): rho_s = rho0 m_s sqrt(<g>^2 - 1) / |B|
            const double thermal_four_velocity_1 = std::sqrt(std::max(0.0, g1 * g1 - 1.0));
            const double thermal_four_velocity_2 = std::sqrt(std::max(0.0, g2 * g2 - 1.0));
            const real_t rho_1                   = (real_t)((double)larmor0 * thermal_four_velocity_1 * (double)sp1_mass / (double)B_ref);
            const real_t rho_2                   = (real_t)((double)larmor0 * thermal_four_velocity_2 * (double)sp2_mass / (double)B_ref);
            //* thermal three-speed v_th,s = sqrt(<g>^2 - 1) / <g>; drifts above it can drive streaming instabilities (two-stream, Buneman)
            const double thermal_speed_1         = thermal_four_velocity_1 / g1;
            const double thermal_speed_2         = thermal_four_velocity_2 / g2;
            const double drift_over_thermal_1    = (thermal_speed_1 > 0.0) ? (double)peak_u_drift_1 / thermal_speed_1 : 0.0;
            const double drift_over_thermal_2    = (thermal_speed_2 > 0.0) ? (double)peak_u_drift_2 / thermal_speed_2 : 0.0;

            //* confinement: gyroradius of a species-s particle holding the whole magnetic energy per particle, gamma_s = 1 + sigma_s,hot
            //* (pair: 1 + sigma_hot, the usual gamma ~ sigma ceiling; ion-electron: electrons up to gamma ~ sigma_e ~ mu sigma). Must stay well inside the box.
            auto confinement_radius = [&](real_t sigma_species_hot, real_t mass) -> real_t
            {
                const double gamma_ceiling = 1.0 + (double)sigma_species_hot;
                return (real_t)((double)larmor0 * std::sqrt(gamma_ceiling * gamma_ceiling - 1.0) * (double)mass / (double)B_ref);
            };
            const real_t confinement_radius_1 = confinement_radius(sigma_1_hot, sp1_mass);
            const real_t confinement_radius_2 = confinement_radius(sigma_2_hot, sp2_mass);

            //* shortest box side among the directions the spectrum spans
            real_t L_min = std::numeric_limits<real_t>::max();
            if (turb_kplane != 1) L_min = std::min(L_min, Lx);
            if (turb_kplane != 2) L_min = std::min(L_min, Ly);
            if ((D == Dim::_3D) && (turb_kplane != 0)) L_min = std::min(L_min, Lz);
            const real_t l_outer = L_min / (real_t)kmin;   //* largest driven wavelength
            const real_t l_inner = L_min / (real_t)kmax;   //* smallest driven wavelength

            //* largest scale against the heavy-species skin depth (pair: plasma skin depth), smallest against the electron one
            const real_t      skin_depth_large = is_pair ? skin_depth_plasma_hot : skin_depth_2_hot;
            const std::string large_scale_name = is_pair ? "plasma" : "ion";

            //* outer-scale eddy time t = 1 / (k_min u_perturbed_rms), k_min = 2 pi / l_outer (the Alfven target if no flow is loaded)
            const real_t u_eddy  = (u_perturbed_rms > ZERO) ? u_perturbed_rms : u_perturbed_rms_alfven;
            const real_t k_outer = (real_t)(2.0 * std::acos(-1.0)) / l_outer;
            const real_t t_eddy  = (u_eddy > ZERO) ? ONE / (k_outer * u_eddy) : ZERO;

            //* omega_p,s t = t c / d_s with the species skin depth at n_tot (hot or cold inertia)
            const real_t runtime    = params.template get<simtime_t>("simulation.runtime");
            const real_t t_wp1_hot  = runtime / skin_depth_1_hot;
            const real_t t_wp2_hot  = runtime / skin_depth_2_hot;
            const real_t t_wp1_cold = runtime / skin_depth_1_cold;
            const real_t t_wp2_cold = runtime / skin_depth_2_cold;
            const real_t t_cross    = runtime / L_min;                                    //* light crossings
            const real_t n_eddy     = (t_eddy > ZERO) ? runtime / t_eddy : ZERO;

            //! beta_s = P_s / U_B = 2 n_s T_s / (sigma0 B^2),  P_s = n_s T_s,  U_B = sigma0 B^2 / 2   [n0 m0 c^2]
            const real_t P_th_1 = N_PER_SP * temperature_1;
            const real_t P_th_2 = N_PER_SP * temperature_2;
            auto magnetic_pressure = [&](real_t B) -> real_t { return sigma0 * B * B / (real_t)2.0; };
            const real_t U_B_rms      = magnetic_pressure(B_ref);
            const real_t beta_1_rms   = (U_B_rms > ZERO) ? P_th_1 / U_B_rms : (real_t)-1.0;
            const real_t beta_2_rms   = (U_B_rms > ZERO) ? P_th_2 / U_B_rms : (real_t)-1.0;
            const real_t U_B_guide    = magnetic_pressure(B_guide);
            const real_t beta_1_guide = (U_B_guide > ZERO) ? P_th_1 / U_B_guide : (real_t)-1.0;
            const real_t beta_2_guide = (U_B_guide > ZERO) ? P_th_2 / U_B_guide : (real_t)-1.0;

            //! =================================================================
            //! SAMPLED DIAGNOSTICS on the host grid of the u_total cap (sampling_counts)
            //!   curl_B     = -sum (kc x b) sin(k.x + ph)   mode-table curl (the drifts carry its sigma_h-corrected form)
            //!   curl_B_yee = one-cell centred difference   what the Yee solver demands; must agree to round-off
            //! =================================================================
            auto dot   = [](const double (&a)[3], const double (&b)[3]) -> double { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
            auto cross = [](const double (&a)[3], const double (&b)[3], double (&result)[3]) { result[0] = a[1] * b[2] - a[2] * b[1]; result[1] = a[2] * b[0] - a[0] * b[2]; result[2] = a[0] * b[1] - a[1] * b[0]; };

            auto B_host = [&](double X, double Y, double Z, double& B_x, double& B_y, double& B_z)
            {
                host_cos_sum(B_perturbed_modes_host, X, Y, Z, B_x, B_y, B_z);
                B_z += (double)B_guide;
            };

            auto u_perturbed_host = [&](double X, double Y, double Z, double& u_perturbed_x, double& u_perturbed_y, double& u_perturbed_z)
            {
                host_cos_sum(u_perturbed_modes_host, X, Y, Z, u_perturbed_x, u_perturbed_y, u_perturbed_z);
            };

            //* centred Yee difference over one cell, as the FDTD curl does it
            auto curl_B_yee_host = [&](double X, double Y, double Z, double& curl_B_yee_x, double& curl_B_yee_y, double& curl_B_yee_z)
            {
                const double hx = (double)dx, hy = (double)dy, hz = (double)dz;
                double bxp, byp, bzp, bxm, bym, bzm;

                B_host(X + 0.5 * hx, Y, Z, bxp, byp, bzp);
                B_host(X - 0.5 * hx, Y, Z, bxm, bym, bzm);
                const double dBy_dx = (byp - bym) / hx;
                const double dBz_dx = (bzp - bzm) / hx;

                B_host(X, Y + 0.5 * hy, Z, bxp, byp, bzp);
                B_host(X, Y - 0.5 * hy, Z, bxm, bym, bzm);
                const double dBx_dy = (bxp - bxm) / hy;
                const double dBz_dy = (bzp - bzm) / hy;

                double dBx_dz = 0.0, dBy_dz = 0.0;
                if constexpr (D == Dim::_3D)
                {
                    B_host(X, Y, Z + 0.5 * hz, bxp, byp, bzp);
                    B_host(X, Y, Z - 0.5 * hz, bxm, bym, bzm);
                    dBx_dz = (bxp - bxm) / hz;
                    dBy_dz = (byp - bym) / hz;
                }

                curl_B_yee_x = dBz_dy - dBy_dz;
                curl_B_yee_y = dBx_dz - dBz_dx;
                curl_B_yee_z = dBy_dx - dBx_dy;
            };

            //* discrete div(B) with the Yee one-cell differences; zero to round-off with the k~-built amplitudes
            auto divB_host = [&](double X, double Y, double Z) -> double
            {
                const double hx = (double)dx, hy = (double)dy, hz = (double)dz;
                double bp[3], bm[3];
                B_host(X + 0.5 * hx, Y, Z, bp[0], bp[1], bp[2]);
                B_host(X - 0.5 * hx, Y, Z, bm[0], bm[1], bm[2]);
                double div = (bp[0] - bm[0]) / hx;
                B_host(X, Y + 0.5 * hy, Z, bp[0], bp[1], bp[2]);
                B_host(X, Y - 0.5 * hy, Z, bm[0], bm[1], bm[2]);
                div += (bp[1] - bm[1]) / hy;
                if constexpr (D == Dim::_3D)
                {
                    B_host(X, Y, Z + 0.5 * hz, bp[0], bp[1], bp[2]);
                    B_host(X, Y, Z - 0.5 * hz, bm[0], bm[1], bm[2]);
                    div += (bp[2] - bm[2]) / hz;
                }
                return div;
            };

            //* E = -(u_motional x B), u_motional = u_perturbed + u_motional_coeff J_first_order (mirrors InitFields::motional_E)
            auto E_host = [&](double X, double Y, double Z, double& Ex, double& Ey, double& Ez)
            {
                double B_x, B_y, B_z, u_perturbed_x, u_perturbed_y, u_perturbed_z, curl_B_x, curl_B_y, curl_B_z;
                B_host(X, Y, Z, B_x, B_y, B_z);
                u_perturbed_host(X, Y, Z, u_perturbed_x, u_perturbed_y, u_perturbed_z);
                host_first_order_current(B_perturbed_modes_host, (double)B_guide, (double)sigma_h_per_B_sq, X, Y, Z, curl_B_x, curl_B_y, curl_B_z);
                const double u_motional_x = u_perturbed_x + (double)u_motional_coeff * curl_B_x;
                const double u_motional_y = u_perturbed_y + (double)u_motional_coeff * curl_B_y;
                const double u_motional_z = u_perturbed_z + (double)u_motional_coeff * curl_B_z;
                Ex = -(u_motional_y * B_z - u_motional_z * B_y);
                Ey = -(u_motional_z * B_x - u_motional_x * B_z);
                Ez = -(u_motional_x * B_y - u_motional_y * B_x);
            };

            const std::array<int, 3> counts = sampling_counts(*this);

            long   n_samples = 0;
            double curl_B_yee_max = 0.0, curl_B_deviation_max = 0.0;              //* max |curl_B_yee|, max |curl_B - curl_B_yee|
            double divB_max = 0.0;                                                //* max |discrete div(B)|
            double B_peak = 0.0;                                                  //* max |B|
            double peak_u_total_loaded = 0.0;                                     //* max |u_perturbed + u_drift_species| with the drift as loaded
            double perp_reduction_sum = 0.0, perp_reduction_max = 0.0;            //* f = sigma_local / (1 + sigma_local)
            double flow_lorentz_sum[2] = { 0.0, 0.0 }, inverse_flow_lorentz_sum[2] = { 0.0, 0.0 };   //* Gamma_s and 1/Gamma_s of u_total_species

            //* current budget: A = AMP curl(B) = particle drift + convection + displacement at every point [n0 e c]
            double required_sq = 0.0, drift_dot_required = 0.0, convection_dot_required = 0.0, displacement_dot_required = 0.0;   //* projections on A: these add up
            double particle_across_sq = 0.0;                                                                                       //* across A (= displacement current across A)
            double displacement_sq = 0.0, displacement_along_B_sq = 0.0, displacement_theory_error_sq = 0.0;                       //* checks

            double E_ideal_sq = 0.0, E_hall_sq = 0.0;                             //* |u_perturbed x B|^2, |(u_motional - u_perturbed) x B|^2
            double xi_max = 0.0;                                                  //* max |xi_s| = (AMP/n_tot) |E|

            if (mpi_rank == 0)
            for (int ix = 0; ix < counts[0]; ++ix)
                for (int iy = 0; iy < counts[1]; ++iy)
                    for (int iz = 0; iz < counts[2]; ++iz)
                    {
                        const double X = sample_coordinate((double)xmin, (double)Lx, ix, counts[0]);
                        const double Y = sample_coordinate((double)ymin, (double)Ly, iy, counts[1]);
                        const double Z = (D == Dim::_3D) ? sample_coordinate((double)zmin, (double)Lz, iz, counts[2]) : 0.0;
                        ++n_samples;

                        //* fields at the sample
                        double B_sample[3], u_perturbed[3], curl_B[3], curl_B_yee[3];
                        B_host(X, Y, Z, B_sample[0], B_sample[1], B_sample[2]);
                        u_perturbed_host(X, Y, Z, u_perturbed[0], u_perturbed[1], u_perturbed[2]);
                        host_curl_sum(B_perturbed_modes_host, X, Y, Z, curl_B[0], curl_B[1], curl_B[2]);
                        curl_B_yee_host(X, Y, Z, curl_B_yee[0], curl_B_yee[1], curl_B_yee[2]);
                        const double B_sq = dot(B_sample, B_sample);
                        B_peak = std::max(B_peak, std::sqrt(B_sq));

                        //* mode-table curl vs Yee curl, and the discrete div(B)
                        curl_B_yee_max       = std::max(curl_B_yee_max, std::sqrt(dot(curl_B_yee, curl_B_yee)));
                        curl_B_deviation_max = std::max(curl_B_deviation_max, std::sqrt(SQR(curl_B[0] - curl_B_yee[0]) + SQR(curl_B[1] - curl_B_yee[1]) + SQR(curl_B[2] - curl_B_yee[2])));
                        divB_max             = std::max(divB_max, std::fabs(divB_host(X, Y, Z)));

                        //* first-order current (sigma_h split) and drift current (adds the convection term), both J/AMP
                        double first_order_current[3], loaded_current[3], div_E_analytic = 0.0;
                        host_first_order_current(B_perturbed_modes_host, (double)B_guide, (double)sigma_h_per_B_sq, X, Y, Z, first_order_current[0], first_order_current[1], first_order_current[2]);
                        host_loaded_current(B_perturbed_modes_host, u_perturbed_modes_host, (double)B_guide, (double)sigma_h_per_B_sq, (double)u_motional_coeff, X, Y, Z, loaded_current[0], loaded_current[1], loaded_current[2], &div_E_analytic);

                        //* f = sigma_local / (1 + sigma_local): share of J_perp carried by dE/dt
                        const double sigma_local = (double)sigma_h_per_B_sq * B_sq;
                        const double f_local     = sigma_local / (1.0 + sigma_local);
                        perp_reduction_sum += f_local;
                        perp_reduction_max  = std::max(perp_reduction_max, f_local);

                        //* u_total_species = u_perturbed + u_drift_coeff_s J_loaded: peak speed, and Gamma_s = 1/sqrt(1 - u^2) below the 0.99c clamp
                        const double u_drift_coeff[2] = { (double)u_drift_coeff_1, (double)u_drift_coeff_2 };
                        for (int s = 0; s < 2; ++s)
                        {
                            double u_total[3];
                            for (int i = 0; i < 3; ++i) { u_total[i] = u_perturbed[i] + u_drift_coeff[s] * loaded_current[i]; }
                            const double u_total_sq   = dot(u_total, u_total);
                            const double flow_lorentz = 1.0 / std::sqrt(1.0 - std::min(u_total_sq, 0.9801));
                            peak_u_total_loaded          = std::max(peak_u_total_loaded, std::sqrt(u_total_sq));
                            flow_lorentz_sum[s]         += flow_lorentz;
                            inverse_flow_lorentz_sum[s] += 1.0 / flow_lorentz;
                        }

                        //* current budget: A = AMP curl(B) = drift (AMP J_loaded) + convection (rho u_convection) + displacement (AMP dE/dt)
                        {
                            const double amp        = (double)AMP;
                            const double rho_sample = amp * div_E_analytic;   //* polarisation charge [n0 e]
                            double required[3], drift[3], convection[3], particle[3], displacement[3];
                            for (int i = 0; i < 3; ++i)
                            {
                                const double u_convection = u_perturbed[i] + 0.5 * (double)u_motional_coeff * first_order_current[i];   //* (u_perturbed + u_motional)/2
                                required[i]     = amp * curl_B[i];
                                drift[i]        = amp * loaded_current[i];
                                convection[i]   = rho_sample * u_convection;
                                particle[i]     = drift[i] + convection[i];
                                displacement[i] = required[i] - particle[i];
                            }

                            //* projections on A add up because A = drift + convection + displacement at every point
                            const double required_magnitude_sq = dot(required, required);
                            required_sq               += required_magnitude_sq;
                            drift_dot_required        += dot(drift, required);
                            convection_dot_required   += dot(convection, required);
                            displacement_dot_required += dot(displacement, required);

                            //* across A, particle current and displacement current are equal and opposite
                            if (required_magnitude_sq > 0.0)
                            {
                                const double particle_along = dot(particle, required) / required_magnitude_sq;
                                for (int i = 0; i < 3; ++i) { particle_across_sq += SQR(particle[i] - particle_along * required[i]); }
                            }

                            //* checks: no displacement current along B; displacement current = f (A - rho u_convection)_perp
                            displacement_sq += dot(displacement, displacement);
                            if (B_sq > 0.0)
                            {
                                displacement_along_B_sq += SQR(dot(displacement, B_sample)) / B_sq;
                                double source[3];
                                for (int i = 0; i < 3; ++i) { source[i] = required[i] - convection[i]; }
                                const double source_along_B_over_B = dot(source, B_sample) / B_sq;
                                for (int i = 0; i < 3; ++i) { displacement_theory_error_sq += SQR(displacement[i] - f_local * (source[i] - source_along_B_over_B * B_sample[i])); }
                            }
                        }

                        //* motional E = -(u x B) = B x u, split into ideal (u_perturbed) and Hall (u_motional_coeff J_first_order) parts; |xi_s| = (AMP/n_tot) |E|
                        double hall_velocity[3], E_ideal[3], E_hall[3];
                        for (int i = 0; i < 3; ++i) { hall_velocity[i] = (double)u_motional_coeff * first_order_current[i]; }
                        cross(B_sample, u_perturbed, E_ideal);
                        cross(B_sample, hall_velocity, E_hall);
                        double E_total[3];
                        for (int i = 0; i < 3; ++i) { E_total[i] = E_ideal[i] + E_hall[i]; }
                        E_ideal_sq += dot(E_ideal, E_ideal);
                        E_hall_sq  += dot(E_hall, E_hall);
                        xi_max      = std::max(xi_max, (double)AMP / (double)N_TOTAL * std::sqrt(dot(E_total, E_total)));
                    }

            const double inv_n = (n_samples > 0) ? 1.0 / (double)n_samples : 0.0;
            //* lab-frame mean Lorentz factor of a Juttner species boosted by Gamma: <gamma_lab> = T^00 / N = Gamma <g> + theta (Gamma - 1/Gamma)
            const double gamma_lab_1 = (flow_lorentz_sum[0] * g1 + (double)theta_1 * (flow_lorentz_sum[0] - inverse_flow_lorentz_sum[0])) * inv_n;
            const double gamma_lab_2 = (flow_lorentz_sum[1] * g2 + (double)theta_2 * (flow_lorentz_sum[1] - inverse_flow_lorentz_sum[1])) * inv_n;

            const double ampere_deviation    = (curl_B_yee_max > 0.0) ? curl_B_deviation_max / curl_B_yee_max : 0.0;
            const double divB_over_curlB     = (curl_B_yee_max > 0.0) ? divB_max / curl_B_yee_max : 0.0;
            const bool   E_ideal_present     = (E_ideal_sq > 0.0);
            const double E_hall_over_ideal   = E_ideal_present ? std::sqrt(E_hall_sq / E_ideal_sq) : 0.0;
            const double perp_reduction_mean = perp_reduction_sum * inv_n;
            //* bulk-kinetic / magnetic perturbation energy, (w u^2 / 2) / (sigma0 dB^2 / 2), non-relativistic flow
            const double bulk_to_magnetic    = (B_perturbed_rms > ZERO) ? (double)w_enth * SQR((double)u_perturbed_rms) / ((double)sigma0 * SQR((double)B_perturbed_rms)) : 0.0;
            //* values along A [n0 e c]: <X . A> / A_rms, so drift + convection + displacement = A_rms exactly
            const double required_rms            = std::sqrt(required_sq * inv_n);
            const double drift_along             = (required_rms > 0.0) ? drift_dot_required * inv_n / required_rms : 0.0;
            const double convection_along        = (required_rms > 0.0) ? convection_dot_required * inv_n / required_rms : 0.0;
            const double displacement_along      = (required_rms > 0.0) ? displacement_dot_required * inv_n / required_rms : 0.0;
            const double particle_across_rms     = std::sqrt(particle_across_sq * inv_n);
            const double displacement_rms        = std::sqrt(displacement_sq * inv_n);
            const double displacement_along_B    = (required_rms > 0.0) ? std::sqrt(displacement_along_B_sq * inv_n) / required_rms : 0.0;
            const double displacement_theory_rel = (displacement_rms > 0.0) ? std::sqrt(displacement_theory_error_sq * inv_n) / displacement_rms : 0.0;
            //* Boris accuracy: cold electron gyration per step at the peak |B|, Omega_ce dt = dt |B|_max / (rho0 m_e); phase error ~ (Omega dt)^2 / 12
            const double gyration_per_step       = (double)dt * B_peak / ((double)larmor0 * (double)sp1_mass);
            const double peak_displacement_cells = xi_max / (double)dx;

            //! =================================================================
            //! UNLOADED DISPLACEMENT CURRENT AT t = 0 (estimate)
            //!
            //! The drifts carry curl(B) minus the displacement current of the J x B acceleration (sigma_h split). The rest of dE/dt at t = 0,
            //! with E = -(u_m x B) and u_m the field-line velocity, is not loaded:
            //!     D = [(u_m . grad) u_m] x B - u_m x dB/dt,        dB/dt = -curl(E).
            //! Across B: the advective dE_perp/dt. Along B: E.curl(E)/|B|, the parallel displacement current that keeps E.B = 0.
            //! Reported as RMS fractions of AMP curl(B) (AMP cancels). Centred differences of the host mode sums on at most 64 points per axis
            //! (32 in 3D); dB/dt is cross-checked with curl(u_m x B) = u_m div(B) - B div(u_m) + (B.grad)u_m - (u_m.grad)B.
            //! =================================================================
            auto u_motional_host = [&](double X, double Y, double Z, double (&u_motional)[3])
            {
                double u_perturbed_value[3], current[3];
                u_perturbed_host(X, Y, Z, u_perturbed_value[0], u_perturbed_value[1], u_perturbed_value[2]);
                host_first_order_current(B_perturbed_modes_host, (double)B_guide, (double)sigma_h_per_B_sq, X, Y, Z, current[0], current[1], current[2]);
                for (int i = 0; i < 3; ++i) { u_motional[i] = u_perturbed_value[i] + (double)u_motional_coeff * current[i]; }
            };

            const bool   E_present = E_ideal_present || (u_motional_coeff != ZERO);
            const double fd_step   = 1.0e-3 * (double)l_inner;          //* well below the shortest injected wavelength
            const int    n_axes    = (D == Dim::_3D) ? 3 : 2;            //* simulated directions; d/dz = 0 in 2D
            double unloaded_perp_sq = 0.0, unloaded_par_sq = 0.0, unloaded_curl_B_sq = 0.0;
            double dB_dt_sq = 0.0, dB_dt_mismatch_sq = 0.0;

            if (mpi_rank == 0 && E_present)
            {
                std::array<int, 3> estimate_counts;
                const int max_points_per_axis = (D == Dim::_3D) ? 32 : 64;
                for (int a = 0; a < 3; ++a) { estimate_counts[a] = std::min(counts[a], max_points_per_axis); }

                for (int ix = 0; ix < estimate_counts[0]; ++ix)
                    for (int iy = 0; iy < estimate_counts[1]; ++iy)
                        for (int iz = 0; iz < estimate_counts[2]; ++iz)
                        {
                            const double X = sample_coordinate((double)xmin, (double)Lx, ix, estimate_counts[0]);
                            const double Y = sample_coordinate((double)ymin, (double)Ly, iy, estimate_counts[1]);
                            const double Z = (D == Dim::_3D) ? sample_coordinate((double)zmin, (double)Lz, iz, estimate_counts[2]) : 0.0;

                            double u_motional[3], B[3];
                            u_motional_host(X, Y, Z, u_motional);
                            B_host(X, Y, Z, B[0], B[1], B[2]);

                            //* derivatives d/dx_a of E, u_m and B: [axis][component]
                            double dE[3][3] = {}, du[3][3] = {}, dB[3][3] = {};
                            for (int a = 0; a < n_axes; ++a)
                            {
                                double shift[3] = { 0.0, 0.0, 0.0 };
                                shift[a] = fd_step;
                                double E_plus[3], E_minus[3], u_plus[3], u_minus[3], B_plus[3], B_minus[3];
                                E_host(X + shift[0], Y + shift[1], Z + shift[2], E_plus[0], E_plus[1], E_plus[2]);
                                E_host(X - shift[0], Y - shift[1], Z - shift[2], E_minus[0], E_minus[1], E_minus[2]);
                                u_motional_host(X + shift[0], Y + shift[1], Z + shift[2], u_plus);
                                u_motional_host(X - shift[0], Y - shift[1], Z - shift[2], u_minus);
                                B_host(X + shift[0], Y + shift[1], Z + shift[2], B_plus[0], B_plus[1], B_plus[2]);
                                B_host(X - shift[0], Y - shift[1], Z - shift[2], B_minus[0], B_minus[1], B_minus[2]);
                                for (int c = 0; c < 3; ++c)
                                {
                                    dE[a][c] = (E_plus[c] - E_minus[c]) / (2.0 * fd_step);
                                    du[a][c] = (u_plus[c] - u_minus[c]) / (2.0 * fd_step);
                                    dB[a][c] = (B_plus[c] - B_minus[c]) / (2.0 * fd_step);
                                }
                            }

                            //* dB/dt = -curl(E)
                            const double dB_dt[3] = { -(dE[1][2] - dE[2][1]), -(dE[2][0] - dE[0][2]), -(dE[0][1] - dE[1][0]) };

                            //* the same from the product rule, as a check of the derivatives and signs
                            double div_B = 0.0, div_u = 0.0;
                            for (int a = 0; a < 3; ++a) { div_B += dB[a][a]; div_u += du[a][a]; }
                            for (int c = 0; c < 3; ++c)
                            {
                                double B_grad_u = 0.0, u_grad_B = 0.0;
                                for (int a = 0; a < 3; ++a) { B_grad_u += B[a] * du[a][c]; u_grad_B += u_motional[a] * dB[a][c]; }
                                const double dB_dt_product = u_motional[c] * div_B - B[c] * div_u + B_grad_u - u_grad_B;
                                dB_dt_mismatch_sq += SQR(dB_dt[c] - dB_dt_product);
                                dB_dt_sq          += SQR(dB_dt[c]);
                            }

                            //* (u_m . grad) u_m
                            double advection[3] = { 0.0, 0.0, 0.0 };
                            for (int c = 0; c < 3; ++c)
                                for (int a = 0; a < 3; ++a) { advection[c] += u_motional[a] * du[a][c]; }

                            //* D = [(u_m . grad) u_m] x B - u_m x dB/dt
                            const double unloaded[3] = { (advection[1] * B[2] - advection[2] * B[1]) - (u_motional[1] * dB_dt[2] - u_motional[2] * dB_dt[1]),
                                                         (advection[2] * B[0] - advection[0] * B[2]) - (u_motional[2] * dB_dt[0] - u_motional[0] * dB_dt[2]),
                                                         (advection[0] * B[1] - advection[1] * B[0]) - (u_motional[0] * dB_dt[1] - u_motional[1] * dB_dt[0]) };

                            //* split along and across the local B
                            const double B_sq = SQR(B[0]) + SQR(B[1]) + SQR(B[2]);
                            if (B_sq > 0.0)
                            {
                                const double along_over_B = (unloaded[0] * B[0] + unloaded[1] * B[1] + unloaded[2] * B[2]) / B_sq;
                                unloaded_par_sq  += SQR(along_over_B) * B_sq;
                                unloaded_perp_sq += SQR(unloaded[0] - along_over_B * B[0]) + SQR(unloaded[1] - along_over_B * B[1]) + SQR(unloaded[2] - along_over_B * B[2]);
                            }

                            double curl_B_x, curl_B_y, curl_B_z;
                            host_curl_sum(B_perturbed_modes_host, X, Y, Z, curl_B_x, curl_B_y, curl_B_z);
                            unloaded_curl_B_sq += SQR(curl_B_x) + SQR(curl_B_y) + SQR(curl_B_z);
                        }
            }

            //* RMS fractions of the required current AMP curl(B)
            const double unloaded_perp_fraction = (unloaded_curl_B_sq > 0.0) ? std::sqrt(unloaded_perp_sq / unloaded_curl_B_sq) : 0.0;
            const double unloaded_par_fraction  = (unloaded_curl_B_sq > 0.0) ? std::sqrt(unloaded_par_sq / unloaded_curl_B_sq) : 0.0;
            const double dB_dt_check            = (dB_dt_sq > 0.0) ? std::sqrt(dB_dt_mismatch_sq / dB_dt_sq) : 0.0;

            //* fraction as a percentage, two decimals (own stream, so the printout's number format is untouched)
            auto percent_text = [](double fraction) -> std::string
            {
                std::ostringstream text_stream;
                text_stream << std::fixed << std::setprecision(2) << 100.0 * fraction << "%";
                return text_stream.str();
            };

            //* x as a percentage of the RMS required current, one decimal
            auto percent_of_required = [&](double x) -> std::string
            {
                if (required_rms <= 0.0) { return "n/a"; }
                std::ostringstream text_stream;
                text_stream << std::fixed << std::setprecision(1) << 100.0 * x / required_rms << "%";
                return text_stream.str();
            };

            double peak_displacement = xi_max;
            MPI_Bcast(&peak_displacement, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            peak_charge_displacement = (real_t)peak_displacement;

            const double peak_u_total_both = std::max((double)peak_u_total_1, (double)peak_u_total_2);

            //* share of J carried by each species: (n_s/h_s) / sum_t (n_t/h_t)
            const double inv_h_1         = 1.0 / (double)enthalpy_per_particle_1;
            const double inv_h_2         = 1.0 / (double)enthalpy_per_particle_2;
            const double current_share_1 = inv_h_1 / (inv_h_1 + inv_h_2);
            const double current_share_2 = inv_h_2 / (inv_h_1 + inv_h_2);

            //! Normalised cross-helicity sigma_c = <u_perturbed . B_perturbed> / (u_perturbed_rms B_perturbed_rms). Distinct modes are orthogonal
            //! over the box, so <u . B> = 0.5 sum_m (u_m . b_m) cos(psi_m - phi_m) when both tables list the same k in the same order (checked).
            double cross_correlation = 0.0;
            bool   tables_paired     = !u_perturbed_modes_host.empty() && (u_perturbed_modes_host.size() == B_perturbed_modes_host.size());
            for (std::size_t m = 0; tables_paired && m < B_perturbed_modes_host.size(); ++m)
            {
                const auto& mode_B_perturbed = B_perturbed_modes_host[m];
                const auto& mode_u_perturbed = u_perturbed_modes_host[m];
                if (mode_B_perturbed.kx != mode_u_perturbed.kx || mode_B_perturbed.ky != mode_u_perturbed.ky || mode_B_perturbed.kz != mode_u_perturbed.kz) { tables_paired = false; break; }
                const double amplitude_dot = (double)mode_u_perturbed.bx * (double)mode_B_perturbed.bx + (double)mode_u_perturbed.by * (double)mode_B_perturbed.by + (double)mode_u_perturbed.bz * (double)mode_B_perturbed.bz;
                cross_correlation += 0.5 * amplitude_dot * std::cos((double)mode_u_perturbed.phase - (double)mode_B_perturbed.phase);
            }
            const bool   sigma_c_valid = tables_paired && (u_perturbed_rms > ZERO) && (B_perturbed_rms > ZERO);
            const double sigma_c       = sigma_c_valid ? cross_correlation / ((double)u_perturbed_rms * (double)B_perturbed_rms) : 0.0;

            //* empirical floor of the finite-grid instability, applied to the COLLECTIVE Debye length
            constexpr real_t DEBYE_FLOOR = static_cast<real_t>(1.0) / static_cast<real_t>(3.0);
            const real_t     lD_ratio    = debye_length_plasma / dx;

            constexpr int LW1 = 44;
            constexpr int LW2 = 40;
            auto kv1 = [&](const std::string& label) -> std::string
            { std::ostringstream t; t << "\t" << std::left << std::setw(LW1) << label << "= "; return t.str(); };
            auto kv2 = [&](const std::string& label) -> std::string
            { std::ostringstream t; t << "\t\t" << std::left << std::setw(LW2) << label << "= "; return t.str(); };

            const char* axis_name[3] = { "X", "Y", "Z" };

            const std::string kplane_name = (turb_kplane == 0) ? "k in XY plane"
                                          : (turb_kplane == 1) ? "k in YZ plane"
                                          : (turb_kplane == 2) ? "k in ZX plane" : "isotropic 3D";

            const std::string apol_name = (turb_a_axis >= 0 && turb_a_axis <= 2)
                                        ? std::string("A = A_") + axis_name[turb_a_axis] + " e_" + axis_name[turb_a_axis] + "  (B_perturbed _|_ " + axis_name[turb_a_axis] + ")"
                                        : std::string("random, A _|_ k");

            //* curl(B) lies along Z in 2D (k in the XY plane, A along Z); in 3D it can point anywhere
            const std::string budget_along  = (D == Dim::_2D) ? "along Z" : "parallel to curl(B)";
            const std::string budget_across = (D == Dim::_2D) ? "in the XY plane" : "perpendicular to curl(B)";
            const std::string across_note   = (D == Dim::_2D) ? "; along the tilted B-field lines" : "";

            std::ostringstream oss;
            oss << "\n"
                << "===========================================================\n"
                << title << "\n"
                << "===========================================================\n\n";

            //! ---- GEOMETRY ----
            oss << "GEOMETRY (periodic; lengths, times and fields in code units, c = 1)\n"
                << kv1("Dimensionality") << (int)D << "D\n";
            if constexpr (D == Dim::_3D)
            {
                const auto nx3 = metadomain.mesh().n_active(in::x3);
                oss << kv1("Resolution") << nx1 << " x " << nx2 << " x " << nx3 << "\n"
                    << kv1("Box (Lx x Ly x Lz)") << Lx << " x " << Ly << " x " << Lz << "\n";
            }
            else
            {
                oss << kv1("Resolution") << nx1 << " x " << nx2 << "\n"
                    << kv1("Box (Lx x Ly)") << Lx << " x " << Ly << "\n";
            }
            oss << kv1("Grid size dx") << dx << "\n"
                << kv1("Time step dt") << dt << "\n"
                << kv1("Current filter passes") << integer_text(current_filters) << "\n"
                << kv1("Deposit shape order") << integer_text(deposit_order) << "\n"
                << kv1("d0 = skindepth0, rho0 = larmor0") << skindepth0 << ", " << larmor0 << "\n"
                << kv1("Runtime [omega_p,e t]") << t_wp1_hot << "  (hot; cold = " << t_wp1_cold << ")\n"
                << kv1("Runtime [omega_p," + sp2_tag + " t]") << t_wp2_hot << "  (hot; cold = " << t_wp2_cold << ")\n"
                << kv1("Runtime [L/c]") << t_cross << "\n"
                << kv1("Runtime [outer eddy times]") << n_eddy << "\n\n";

            //! ---- MAGNETIC FIELD AND MAGNETISATION ----
            oss << "MAGNETIC FIELD AND MAGNETISATION (at B = B_rms unless stated)\n"
                << kv1("B_rms") << B_rms << "\n"
                << kv1("Guide field (along Z)") << B_guide << "\n"
                << kv1("B_perturbed RMS") << B_perturbed_rms << "\n"
                << kv1("B_perturbed RMS (X, Y, Z)") << "(" << B_perturbed_rms_component[0] << ", " << B_perturbed_rms_component[1] << ", " << B_perturbed_rms_component[2] << ")\n"
                << kv1("B_perturbed^2 / B_guide^2") << ((B_guide != ZERO) ? (B_perturbed_rms * B_perturbed_rms) / (B_guide * B_guide) : (real_t)-1.0) << "  (-1: B_guide = 0)\n"
                << kv1("Mass ratio m_2/m_1") << mass_ratio << "\n"
                << kv1("sigma0 = (d0/rho0)^2") << sigma0 << "\n"
                << "\tMagnetisation, cold (at B_rms)\n"
                << kv2("Electrons") << sigma_1_cold << "\n"
                << kv2(SP2u) << sigma_2_cold << "\n"
                << kv2("Plasma") << sigma_cold << "\n"
                << "\tMagnetisation, hot (at B_rms)\n"
                << kv2("Electrons") << sigma_1_hot << "\n"
                << kv2(SP2u) << sigma_2_hot << "\n"
                << kv2("Plasma") << sigma_hot << "\n"
                << "\tMagnetisation, cold (at B_guide)\n"
                << kv2("Electrons") << sigma_1_cold_guide << "\n"
                << kv2(SP2u) << sigma_2_cold_guide << "\n"
                << kv2("Plasma") << sigma_cold_guide << "\n"
                << "\tMagnetisation, hot (at B_guide)\n"
                << kv2("Electrons") << sigma_1_hot_guide << "\n"
                << kv2(SP2u) << sigma_2_hot_guide << "\n"
                << kv2("Plasma") << sigma_hot_guide << "\n"
                << kv1("Enthalpy w [n0 m0 c^2]") << w_enth << "  (= sum_s n_s (m_s <g_s> + T_s))\n"
                << kv1("Alfven speed (hot) [c]") << vA_hot << "\n"
                << "\tMean Lorentz factor, species rest frame (thermal only)\n"
                << kv2("Electrons") << g1 << "\n"
                << kv2(SP2u) << g2 << "\n"
                << "\tMean Lorentz factor, simulation frame (box mean, with bulk flow and drift)\n"
                << kv2("Electrons") << gamma_lab_1 << "\n"
                << kv2(SP2u) << gamma_lab_2 << "\n"
                << "\tPlasma beta (at B_rms)\n"
                << kv2("Electrons") << beta_1_rms << "\n"
                << kv2(SP2u) << beta_2_rms << "\n"
                << kv2("Plasma") << beta_1_rms + beta_2_rms << "\n";
            if (B_guide != ZERO)
            {
                oss << "\tPlasma beta (at B_guide)\n"
                    << kv2("Electrons") << beta_1_guide << "\n"
                    << kv2(SP2u) << beta_2_guide << "\n"
                    << kv2("Plasma") << beta_1_guide + beta_2_guide << "\n\n";
            }
            else
            {
                oss << kv1("Plasma beta (at B_guide)") << "n/a (B_guide = 0)\n\n";
            }

            //! ---- TURBULENCE SPECTRUM ----
            oss << "TURBULENCE SPECTRUM\n"
                << kv1("Mode band [kmin, kmax]") << "[" << kmin << ", " << kmax << "]\n"
                << kv1("Number of modes") << B_perturbed_modes_host.size() << "\n"
                << kv1("Phase seed of B_perturbed") << B_seed << "\n"
                << kv1("Spectral slope (E(k) ~ k^slope)") << energy_spectrum_slope << "  (0 = flat; " << kspace_dim(turb_kplane, D == Dim::_3D) - 1 << " = equal energy per mode)\n"
                << kv1("Random mode amplitudes") << (turb_rand_amp ? "YES" : "NO") << "\n"
                << kv1("k-space") << kplane_name << "\n"
                << kv1("Vector potential") << apol_name << "\n"
                << kv1("Largest driven wavelength L/kmin") << l_outer << "  (" << l_outer / skin_depth_large << " " << large_scale_name << " skin depths)\n"
                << kv1("Smallest driven wavelength L/kmax") << l_inner << "  (" << l_inner / skin_depth_1_hot << " electron skin depths; " << l_inner / dx << " cells)\n"
                << kv1("Eddy time 1/(k_min u_perturbed_rms)") << t_eddy << ((u_perturbed_rms > ZERO) ? "" : "  (Alfven target; no flow loaded)") << "\n\n";

            //! ---- CURRENT ----
            oss << "CURRENT\n"
                << kv1("AMP = d0^2/rho0") << AMP << "\n";
            if (E_ideal_present) { oss << kv1("Hall / ideal part of E (RMS)") << E_hall_over_ideal << "\n"; }
            else                 { oss << kv1("Hall / ideal part of E (RMS)") << "n/a (u_perturbed = 0)\n"; }
            oss << kv1("Displacement share f of J_perp") << "mean " << perp_reduction_mean << ", max " << perp_reduction_max << "  (f = sigma_h/(1 + sigma_h), local)\n"
                << "\tCurrent distribution " << budget_along << "  [n0 e c]\n"
                << kv2("Total = AMP curl(B)") << required_rms << "  (100%)\n"
                << kv2("Particle drift") << drift_along << "  (" << percent_of_required(drift_along) << ")\n"
                << kv2("Convection (polarisation charge)") << convection_along << "  (" << percent_of_required(convection_along) << ")\n"
                << kv2("Displacement current (dE/dt)") << displacement_along << "  (" << percent_of_required(displacement_along) << ")\n"
                << "\tCurrent distribution " << budget_across << "  [n0 e c, RMS]\n"
                << kv2("Particles (= -displacement current)") << particle_across_rms << "  (" << percent_of_required(particle_across_rms) << across_note << ")\n"
                << kv1("Share of J (electrons, " + sp2s + ")") << "(" << current_share_1 << ", " << current_share_2 << ")  (zero net momentum)\n"
                << kv1("max |u_drift| [c] (electrons, " + sp2s + ")") << "(" << peak_u_drift_1 << ", " << peak_u_drift_2 << ")\n"
                << kv1("max |u_drift| / v_th (electrons, " + sp2s + ")") << "(" << drift_over_thermal_1 << ", " << drift_over_thermal_2 << ")  (>~ 1: streaming instabilities)\n\n";

            //! ---- VELOCITY PERTURBATION ----
            oss << "VELOCITY PERTURBATION\n"
                << kv1("Enabled") << (perturb_velocity ? "YES" : "NO  (u_perturbed = 0 everywhere)") << "\n";
            if (perturb_velocity)
            {
                oss << kv1("Phase seed of u_perturbed") << v_seed << "\n"
                    << kv1("u_total cap [c]") << u_total_cap << "\n"
                    << kv1("u_perturbed_rms [c]") << u_perturbed_rms << "  (= s vA_hot B_perturbed_rms / B_rms)\n"
                    << kv1("u_perturbed RMS (X, Y, Z) [c]") << "(" << u_perturbed_rms_component[0] << ", " << u_perturbed_rms_component[1] << ", " << u_perturbed_rms_component[2] << ")\n"
                    << kv1("Cap factor s") << u_perturbed_cap_factor << ((u_perturbed_cap_factor < ONE) ? "  (REDUCED below the Alfven target)" : "  (Alfven target loaded)") << "\n"
                    << kv1("Turbulent kinetic / magnetic energy") << bulk_to_magnetic << "  (w u^2 / (sigma0 dB^2), non-relativistic)\n";
                if (sigma_c_valid) { oss << kv1("Cross-helicity sigma_c") << sigma_c << "  (0 on average over seeds)\n"; }
                else               { oss << kv1("Cross-helicity sigma_c") << "n/a\n"; }
            }
            oss << "\n";

            //! ---- UNLOADED DISPLACEMENT CURRENT ----
            oss << "UNLOADED DISPLACEMENT CURRENT AT t = 0 (estimates, RMS fraction of AMP curl(B))\n"
                << "\tThe run starts with a transient of about this relative size; both vanish when E = 0 at t = 0\n";
            if (E_present)
            {
                oss << kv1("Advective dE_perp/dt") << percent_text(unloaded_perp_fraction) << "  (flow and field advection; grows with vA/c)\n"
                    << kv1("Parallel dE/dt (keeps E.B = 0)") << percent_text(unloaded_par_fraction) << "  (= E.curl(E)/|B|; grows as (u/c)^2)\n\n";
            }
            else
            {
                oss << kv1("Advective dE_perp/dt") << "0  (E = 0 at t = 0)\n"
                    << kv1("Parallel dE/dt (keeps E.B = 0)") << "0  (E = 0 at t = 0)\n\n";
            }

            //! ---- CHARACTERISTIC SCALES ----
            oss << "CHARACTERISTIC SCALES\n"
                << "\tSkin depth, cold (inertia m_s)\n"
                << kv2("Electrons") << skin_depth_1_cold << "  (resolved with " << skin_depth_1_cold / dx << " cells)\n"
                << kv2(SP2u) << skin_depth_2_cold << "  (resolved with " << skin_depth_2_cold / dx << " cells)\n"
                << kv2("Plasma") << skin_depth_plasma_cold << "  (resolved with " << skin_depth_plasma_cold / dx << " cells)\n"
                << "\tSkin depth, hot (inertia m_s <g_s>)\n"
                << kv2("Electrons") << skin_depth_1_hot << "  (resolved with " << skin_depth_1_hot / dx << " cells)\n"
                << kv2(SP2u) << skin_depth_2_hot << "  (resolved with " << skin_depth_2_hot / dx << " cells)\n"
                << kv2("Plasma") << skin_depth_plasma_hot << "  (resolved with " << skin_depth_plasma_hot / dx << " cells)\n"
                << "\tDebye length (no inertia factor)\n"
                << kv2("Electrons") << debye_length_1 << "  (resolved with " << debye_length_1 / dx << " cells)\n"
                << kv2(SP2u) << debye_length_2 << "  (resolved with " << debye_length_2 / dx << " cells)\n"
                << kv2("Plasma") << debye_length_plasma << "  (resolved with " << debye_length_plasma / dx << " cells)\n"
                << "\tLarmor radius (thermal, u = sqrt(<g>^2 - 1), at B_rms)\n"
                << kv2("Electrons") << rho_1 << "  (resolved with " << rho_1 / dx << " cells)\n"
                << kv2(SP2u) << rho_2 << "  (resolved with " << rho_2 / dx << " cells)\n"
                << "\tConfinement: gyroradius at gamma_s = 1 + sigma_s,hot (at B_rms)\n"
                << kv2("Electrons") << confinement_radius_1 << "  (" << confinement_radius_1 / L_min << " x box)\n"
                << kv2(SP2u) << confinement_radius_2 << "  (" << confinement_radius_2 / L_min << " x box)\n\n";

            //! ---- TEMPERATURES ----
            oss << "TEMPERATURES\n"
                << kv1("T_1 = kT_1/(m0 c^2) of electrons") << temperature_1 << "  (setup.temperature)\n"
                << kv1("T_2 = kT_2/(m0 c^2) of " + sp2s) << temperature_2 << "  (temperature_ratio = " << temperature_ratio << ")\n"
                << kv1("theta_1 = kT_1/(m_1 c^2) of electrons") << theta_1 << "\n"
                << kv1("theta_2 = kT_2/(m_2 c^2) of " + sp2s) << theta_2 << "\n\n";

            //! ---- CONSISTENCY CHECKS ----
            oss << "CONSISTENCY CHECKS\n"
                << kv1("max |u_total_species| (<= cap)") << peak_u_total_both << "\n"
                << kv1("max|curl B - curl B Yee| / max (->0)") << ampere_deviation << "\n"
                << kv1("max|div B| / max|curl B|, Yee (->0)") << divB_over_curlB << "\n"
                << kv1("Displacement current along B / required") << displacement_along_B << "  (->0: E.B stays 0)\n"
                << kv1("Displacement current vs f (A - rho u)_perp") << displacement_theory_rel << "  (->0, relative)\n"
                << kv1("dB/dt: curl(E) vs product rule") << (E_present ? std::to_string(dB_dt_check) : std::string("n/a (E = 0)")) << "  (->0, relative)\n"
                << kv1("lambda_D / dx (>~ 1/3)") << lD_ratio << "\n"
                << kv1("Omega_ce dt at max|B| (<~ 0.1-0.2)") << gyration_per_step << "\n"
                << kv1("max charge displacement [cells]") << peak_displacement_cells << "  (must be << the local subdomain)\n\n";

            if (lD_ratio < DEBYE_FLOOR)
                oss << "!! WARNING: lambda_D / dx = " << lD_ratio << " is BELOW the finite-grid instability floor of " << DEBYE_FLOOR << ".\n"
                    << "!! Expect numerical heating; refine the grid or raise the temperature.\n\n";
            else if (lD_ratio < ONE)
                oss << "!  NOTE: lambda_D is under one cell (" << lD_ratio << "): Debye-scale structure is not resolved.\n"
                    << "!  The floor of " << DEBYE_FLOOR << " is met, so heating should be bounded -- confirm it on a short run.\n\n";

            if (gyration_per_step > 0.2)
                oss << "!  NOTE: Omega_ce dt = " << gyration_per_step << " at max|B|: Boris phase error ~ (Omega dt)^2/12 per step.\n"
                    << "!  Omega_ce dt = CFL (dx/d0) sqrt(sigma0) |B| / sqrt(D): lower CFL or refine the grid.\n\n";

            if (peak_u_total_loaded > (double)u_total_cap)
                oss << "!  NOTE: with the convection term the peak |u_total| is " << peak_u_total_loaded << " c, above setup.u_total_cap = " << u_total_cap << "\n"
                    << "!  (the cap uses the first-order current). The 0.99c clamp still holds; lower u_total_cap slightly if this matters.\n\n";

            if (u_perturbed_cap_factor < ONE)
                oss << "!  NOTE: u_perturbed was scaled by s = " << u_perturbed_cap_factor << " to keep |u_total_species| <= " << u_total_cap << " c.\n"
                    << "!  The kinetic/magnetic fluctuation energy starts at s^2 = " << u_perturbed_cap_factor * u_perturbed_cap_factor << " of the Alfven value.\n\n";

            oss << "===========================================================\n";

            if (mpi_rank == 0)
            {
                std::cout << oss.str() << std::endl;

                //! the <simulation.name> directory must already exist
                const std::string sim_name = params.template get<std::string>("simulation.name");
                std::string       path     = sim_name;
                if (!path.empty() && path.back() != '/') path += '/';
                path += "Simulation_params.txt";
                std::ofstream fout(path);
                if (fout.is_open()) { fout << oss.str(); fout.close(); }
                else { std::cout << "WARNING: could not write " << path << std::endl; }
            }

            //! HARD FAILURE: the loaded J must be the curl of the field written to the grid (catches a sign or index error in kc x b).
            //! Rank 0 only: raise::Error calls MPI_Abort(MPI_COMM_WORLD), which stops every rank.
            if (mpi_rank == 0)
            {
                raise::ErrorIf(ampere_deviation > 1.0e-2,
                               "Ampere mismatch: curl(B) of the mode table differs from the Yee curl of the initialised field.\n"
                               "Check InitFields::curl_B() against InitFields::bx1/bx2/bx3.", HERE);
            }
        }

        //! =====================================================================
        //!  INITIALISE PARTICLES
        //!  Maxwell-Juttner at theta_s, drifting at u_total_species = u_perturbed + u_drift_species (TurbCurrentDist); after the
        //!  charge-loading displacement each velocity is drawn again at the particle's final position (load_charge_density).
        //! =====================================================================
        void InitPrtls(Domain<S, M>& local_domain)
        {
            metadomain.CommunicateFields(local_domain, Comm::E | Comm::B);

            const auto edist_1 = TurbCurrentDist<M>(local_domain.random_pool(), theta_1, u_drift_coeff_1, init_flds, turb_vel);
            const auto edist_2 = TurbCurrentDist<M>(local_domain.random_pool(), theta_2, u_drift_coeff_2, init_flds, turb_vel);
            const auto sdist   = UniformDist<M::Dim>();

            //? total injected density N_TOTAL = 1, split evenly (n_s = N_PER_SP)
            arch::InjectNonUniform<S, M, decltype(edist_1), decltype(edist_2), decltype(sdist)>(params, local_domain, { 1, 2 }, { edist_1, edist_2 }, sdist, N_TOTAL);

            //? charge density of the motional E field, then velocities redrawn at the displaced positions
            load_charge_density(local_domain, edist_1, edist_2);

            //? Gauss-law check of the loaded state: deposited charge vs AMP div(E)
            check_gauss_law(local_domain);
        }

        //! =====================================================================
        //!  GAUSS-LAW CHECK (t = 0): deposited charge vs AMP div(E)
        //*  Charge: Entity's charge moment without smoothing, in units of n0 e, at cell centres.
        //*  div(E): Entity's Yee divergence at the nodes, averaged over each cell's corners.
        //*  They differ at O((k dx)^2) and the charge carries particle noise, so global statistics are compared:
        //*  slope -> 1 when the loading is right; a low correlation only means the charge is noise-dominated.
        //! =====================================================================
        void check_gauss_law(Domain<S, M>& local_domain)
        {
            int mpi_rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

            auto buffer = local_domain.fields.buff;   //* scratch: [0] deposited charge, [1] div(E) at nodes

            //* deposited charge density [n0 e], smoothing order 0
            arch::ComputeMomentWithSpecies<S, M, FldsID::Charge, 3>(params, local_domain, { 1, 2 }, buffer, {}, 0u, 0u);

            //* div(E) at the nodes, one layer past the active range so every active cell has all its corners
            const ncells_t ni1 = local_domain.mesh.n_active(in::x1);
            const ncells_t ni2 = local_domain.mesh.n_active(in::x2);
            ncells_t       ni3 = 1;
            if constexpr (D == Dim::_3D) { ni3 = local_domain.mesh.n_active(in::x3); }
            tuple_t<ncells_t, M::Dim> range_min { 0 }, range_max { 0 };
            range_min[0] = N_GHOSTS; range_max[0] = N_GHOSTS + ni1 + 1;
            range_min[1] = N_GHOSTS; range_max[1] = N_GHOSTS + ni2 + 1;
            if constexpr (D == Dim::_3D) { range_min[2] = N_GHOSTS; range_max[2] = N_GHOSTS + ni3 + 1; }
            Kokkos::parallel_for("GaussCheckDivE", CreateRangePolicy<M::Dim>(range_min, range_max), kernel::ComputeDivergence_kernel<M, 3>(local_domain.mesh.metric, local_domain.fields.em, buffer, 1));

            //* sums over the active cells (one-off host copy)
            auto         buffer_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), buffer);
            const double amp         = (double)AMP;
            double       sums[5]     = { 0.0, 0.0, 0.0, 0.0, 0.0 };   //* deposited^2, expected^2, deposited*expected, (deposited - expected)^2, cells
            auto accumulate = [&](double rho_deposited, double rho_expected)
            {
                sums[0] += rho_deposited * rho_deposited;
                sums[1] += rho_expected * rho_expected;
                sums[2] += rho_deposited * rho_expected;
                sums[3] += SQR(rho_deposited - rho_expected);
                sums[4] += 1.0;
            };
            if constexpr (D == Dim::_2D)
            {
                for (ncells_t i1 = N_GHOSTS; i1 < N_GHOSTS + ni1; ++i1)
                for (ncells_t i2 = N_GHOSTS; i2 < N_GHOSTS + ni2; ++i2)
                {
                    const double divE_centre = 0.25 * ((double)buffer_host(i1, i2, 1) + (double)buffer_host(i1 + 1, i2, 1) + (double)buffer_host(i1, i2 + 1, 1) + (double)buffer_host(i1 + 1, i2 + 1, 1));
                    accumulate((double)buffer_host(i1, i2, 0), amp * divE_centre);
                }
            }
            else
            {
                for (ncells_t i1 = N_GHOSTS; i1 < N_GHOSTS + ni1; ++i1)
                for (ncells_t i2 = N_GHOSTS; i2 < N_GHOSTS + ni2; ++i2)
                for (ncells_t i3 = N_GHOSTS; i3 < N_GHOSTS + ni3; ++i3)
                {
                    double divE_centre = 0.0;
                    for (ncells_t c1 = 0; c1 < 2; ++c1)
                    for (ncells_t c2 = 0; c2 < 2; ++c2)
                    for (ncells_t c3 = 0; c3 < 2; ++c3) { divE_centre += 0.125 * (double)buffer_host(i1 + c1, i2 + c2, i3 + c3, 1); }
                    accumulate((double)buffer_host(i1, i2, i3, 0), amp * divE_centre);
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, sums, 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            if (mpi_rank != 0) { return; }

            const double n_cells       = std::max(1.0, sums[4]);
            const double rms_deposited = std::sqrt(sums[0] / n_cells);
            const double rms_expected  = std::sqrt(sums[1] / n_cells);

            std::ostringstream oss;
            oss << "GAUSS-LAW CHECK  (t = 0, after charge loading; rho in units of n0 e)\n";
            if (sums[1] > 0.0)
            {
                //* least-squares slope of deposited vs expected, with its 1-sigma error from the residual scatter
                const double slope       = sums[2] / sums[1];
                const double correlation = (sums[0] > 0.0) ? sums[2] / std::sqrt(sums[0] * sums[1]) : 0.0;
                const double residual    = std::sqrt(sums[3] / sums[1]);
                const double slope_error = std::sqrt(std::max(0.0, sums[3] / n_cells) / sums[1]);
                oss << "\trho RMS, expected (AMP div E)          = " << rms_expected << "\n"
                    << "\trho RMS, deposited (charge moment)     = " << rms_deposited << "\n"
                    << "\tbest-fit slope deposited/expected (->1) = " << slope << " +/- " << slope_error << "\n"
                    << "\tcorrelation                             = " << correlation << "  (low = noise-dominated, not wrong)\n"
                    << "\tresidual RMS / expected RMS             = " << residual << "\n";
                if (std::fabs(slope - 1.0) > std::max(0.05, 3.0 * slope_error))
                    oss << "!  NOTE: slope differs from 1 by more than 3 sigma and 5%: the loaded charge does not match div(E).\n";
            }
            else
            {
                oss << "\trho expected = 0 (E carries no charge); deposited RMS = " << rms_deposited << "\n";
            }
            oss << "===========================================================\n";

            std::cout << oss.str() << std::endl;
            const std::string sim_name = params.template get<std::string>("simulation.name");
            std::string       path     = sim_name;
            if (!path.empty() && path.back() != '/') path += '/';
            path += "Simulation_params.txt";
            std::ofstream fout(path, std::ios::app);
            if (fout.is_open()) { fout << oss.str(); fout.close(); }
        }

        //! =====================================================================
        //!  GAUSS LAW CONSTRAINT: load the charge density the motional E field needs
        //!
        //!  WHY A CHARGE DENSITY?  With E = -(u_motional x B),
        //*     div(E) = -B.(curl u_motional) + u_motional.(curl B) != 0.
        //*  E.g. a solenoidal in-plane u_perturbed = grad(psi) x z in a guide field gives E_perp = +B_z grad(psi). This is the polarisation
        //*  charge of a relativistic Alfvenic fluctuation; its ideal part scales as rho/n ~ (k d0) sqrt(sigma0) u_perturbed B, vanishing
        //*  in the MHD (k d0 -> 0) and non-relativistic (sigma0 -> 0) limits.
        //*
        //!  WHY AT t = 0?  Entity has no Poisson solver or divergence cleaner, and the deposition (zigzag or Esirkepov) is charge
        //*  conserving, so div(E) - rho/AMP is preserved exactly for all time: an initial violation is frozen in.
        //*
        //!  HOW?  The injector places the two species in pairs at the same positions (exactly neutral). Displacing species s by xi_s
        //*  changes its density by dn_s = -n_s div(xi_s). With xi_s = -sign(q_s) a E, n_s = n/2 and |q_s| = 1:
        //*      rho = sum_s q_s dn_s = n a div(E),
        //*  so a = AMP / n reproduces rho = AMP div(E) to first order in xi, with no solver. Negative charges move along +E.
        //*
        //!  ACCURACY: the linear response dn_s = -n_s div(xi_s) leaves O((div xi)^2). Velocities are redrawn at the displaced positions.
        //!  2D: d/dz = 0, so E_z carries no charge and no z-displacement is applied.
        //!
        //! =====================================================================
        //!  BOUNDARY BOOKKEEPING (normally done by the pusher, kernels/pushers/sr.hpp)
        //*   1. PERIODIC WRAP on faces where this subdomain's particle BC is PERIODIC (it spans the box in that direction).
        //*   2. SEND TAG otherwise (mpi::SendTag). CommunicateParticles only moves particles with tag != alive; an untagged
        //*      out-of-range particle corrupts memory at the first sort or deposit.
        //*  CommunicateParticles moves a particle to the ADJACENT subdomain only, so the sampled peak |xi| is checked against the local
        //*  subdomain size first (2x margin). A final guard keeps any particle that would stay out of range and untagged where it was.
        //! =====================================================================
        void load_charge_density(Domain<S, M>& local_domain, const TurbCurrentDist<M>& edist_1, const TurbCurrentDist<M>& edist_2)
        {
            //* local copies: a device lambda must not capture `this`
            const auto flds   = init_flds;
            const auto metric = local_domain.mesh.metric;

            //* local active cell counts, for the wrap and the range test
            const int ni1 = static_cast<int>(local_domain.mesh.n_active(in::x1));
            const int ni2 = static_cast<int>(local_domain.mesh.n_active(in::x2));
            const int ni3 = (D == Dim::_3D)? static_cast<int>(local_domain.mesh.n_active(in::x3)) : 1;

            //* displacements must stay well inside one neighbouring subdomain
            {
                const auto&  global_mesh         = metadomain.mesh();
                const double cell_size           = (double)Lx / (double)global_mesh.n_active(in::x1);
                int          smallest_local_side = std::min(ni1, ni2);
                if constexpr (D == Dim::_3D) { smallest_local_side = std::min(smallest_local_side, ni3); }
                const double peak_displacement_cells = (double)peak_charge_displacement / cell_size;
                raise::ErrorIf(2.0 * peak_displacement_cells >= (double)smallest_local_side,
                               "Charge-loading displacement (peak " + std::to_string(peak_displacement_cells) +
                               " cells) is not small compared with this MPI subdomain (" + std::to_string(smallest_local_side) +
                               " cells): particles could skip a whole neighbour. Use fewer ranks along the "
                               "short direction, a finer grid, or a smaller u_perturbed / sigma0.", HERE);
            }

            //* PERIODIC on a face => this subdomain spans the box in that direction: wrap locally. Otherwise a neighbour owns it.
            const auto bcs     = local_domain.mesh.prtl_bc();
            const bool per1min = (bcs[0].first  == PrtlBC::PERIODIC);
            const bool per1max = (bcs[0].second == PrtlBC::PERIODIC);
            const bool per2min = (bcs[1].first  == PrtlBC::PERIODIC);
            const bool per2max = (bcs[1].second == PrtlBC::PERIODIC);
            bool       per3min = false, per3max = false;
            if constexpr (D == Dim::_3D)
            {
                per3min = (bcs[2].first  == PrtlBC::PERIODIC);
                per3max = (bcs[2].second == PrtlBC::PERIODIC);
            }

            for (auto s { 0u }; s < 2u; ++s)
            {
                const real_t q_s  = (s == 0u) ? sp1_charge : sp2_charge;
                const real_t sgn  = (q_s < ZERO) ? -ONE : ONE;
                //* xi_s = -sign(q_s) (AMP / n_tot) E
                const real_t coef = -sgn * AMP / N_TOTAL;

                //* velocity distribution of this species, redrawn at the displaced position
                const auto edist = (s == 0u) ? edist_1 : edist_2;

                auto& sp  = local_domain.species[s];
                auto  i1  = sp.i1;   auto dx1 = sp.dx1;
                auto  i2  = sp.i2;   auto dx2 = sp.dx2;
                auto  i3  = sp.i3;   auto dx3 = sp.dx3;
                auto  ux1 = sp.ux1;  auto ux2 = sp.ux2;  auto ux3 = sp.ux3;
                auto  tag = sp.tag;

                Kokkos::parallel_for("LoadChargeDensity", sp.rangeActiveParticles(), Lambda(prtlidx_t p)
                {
                    if (tag(p) == ParticleTag::dead) { return; }

                    //* code coords -> physical coords
                    coord_t<D> x_Cd { ZERO }, x_Ph { ZERO };
                    x_Cd[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
                    x_Cd[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
                    if constexpr (D == Dim::_3D)
                    {
                        x_Cd[2] = static_cast<real_t>(i3(p)) + static_cast<real_t>(dx3(p));
                    }
                    metric.template convert<Crd::Cd, Crd::XYZ>(x_Cd, x_Ph);

                    //* the same E the field setter wrote, at the particle
                    real_t Ex = ZERO, Ey = ZERO, Ez = ZERO;
                    flds.motional_E(x_Ph, Ex, Ey, Ez);

                    x_Ph[0] += coef * Ex;
                    x_Ph[1] += coef * Ey;
                    if constexpr (D == Dim::_3D) { x_Ph[2] += coef * Ez; }

                    //* back to code coords; floor (not truncation) keeps dx in [0, 1) for negative x_Cd
                    metric.template convert<Crd::XYZ, Crd::Cd>(x_Ph, x_Cd);

                    //* split into (cell, offset); prtldx_t is float, so an offset that rounds to 1 is carried into the cell index
                    auto split = [](real_t x, int& cell, prtldx_t& off)
                    {
                        cell = static_cast<int>(math::floor(x));
                        off  = static_cast<prtldx_t>(x - static_cast<real_t>(cell));
                        if (off >= static_cast<prtldx_t>(1)) { off = static_cast<prtldx_t>(0); ++cell; }
                    };

                    int c1 = 0, c2 = 0, c3 = 0;
                    prtldx_t o1 = ZERO, o2 = ZERO, o3 = ZERO;
                    split(x_Cd[0], c1, o1);
                    split(x_Cd[1], c2, o2);
                    if constexpr (D == Dim::_3D) { split(x_Cd[2], c3, o3); }

                    //* (1) periodic wrap on faces this subdomain owns outright
                    if      (c1 < 0   && per1min) { c1 += ni1; }
                    else if (c1 >= ni1 && per1max) { c1 -= ni1; }
                    if      (c2 < 0   && per2min) { c2 += ni2; }
                    else if (c2 >= ni2 && per2max) { c2 -= ni2; }
                    if constexpr (D == Dim::_3D)
                    {
                        if      (c3 < 0   && per3min) { c3 += ni3; }
                        else if (c3 >= ni3 && per3max) { c3 -= ni3; }
                    }

                    const bool im1 = (c1 < 0),  ip1 = (c1 >= ni1);
                    const bool jm1 = (c2 < 0),  jp1 = (c2 >= ni2);
                    const bool km1 = (D == Dim::_3D) && (c3 < 0);
                    const bool kp1 = (D == Dim::_3D) && (c3 >= ni3);
                    const bool outside = im1 || ip1 || jm1 || jp1 || km1 || kp1;

                    bool tagged = false;
                    //* (2) hand anything still out of range to the exchange
                    if (outside)
                    {
                        if constexpr (D == Dim::_2D)
                            tag(p) = mpi::SendTag(tag(p), im1, ip1, jm1, jp1);
                        else
                            tag(p) = mpi::SendTag(tag(p), im1, ip1, jm1, jp1, km1, kp1);

                        tagged = (tag(p) != ParticleTag::alive);
                    }
                    //* (3) guard: never leave an untagged particle out of range
                    if (outside && !tagged) { return; }

                    i1(p) = c1; dx1(p) = o1;
                    i2(p) = c2; dx2(p) = o2;
                    if constexpr (D == Dim::_3D) { i3(p) = c3; dx3(p) = o3; }

                    //* (4) redraw the velocity at the displaced position (the mode sums are box-periodic, so an unwrapped x_Ph is fine)
                    vec_t<Dim::_3D> four_velocity { ZERO };
                    edist(x_Ph, four_velocity);
                    ux1(p) = four_velocity[0];
                    ux2(p) = four_velocity[1];
                    ux3(p) = four_velocity[2];
                });
            }

            //! MANDATORY: Entity's own exchange runs only after the first push and deposit, so without this the first deposit would index out of range
            metadomain.CommunicateParticles(local_domain);
        }
    };

} // namespace user

#endif