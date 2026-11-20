#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "traits/pgen.h"
#include "utils/numeric.h"
#include "arch/kokkos_aliases.h"

//! Assumes an MPI build (Entity sets -D MPI_ENABLED with -D mpi=ON)
#if !defined(MPI_ENABLED)
    #error "This pgen requires an MPI build: configure Entity with -D mpi=ON"
#endif
#include "arch/mpi_tags.h"
#include <mpi.h>

#include "archetypes/utils.h"
#include "archetypes/energy_dist.h"
#include "archetypes/spatial_dist.h"
#include "framework/domain/metadomain.h"
#include "archetypes/particle_injector.h"
#include "kernels/divergences.hpp"

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
#include <algorithm>

//! =============================================================================
//!  PROBLEM DESCRIPTION
//!
//!  Decaying turbulence in a uniform, collisionless two-species plasma 
//!  (electron-positron or electron-ion), relativistic or non-relativistic, in 2D or 3D
//!
//!  BOUNDARIES: periodic in every direction
//!
//!  INITIAL STATE (t = 0)
//!   - Magnetic field: B = B_guide z + B_perturbed. B_perturbed is a sum of solenoidal
//!     Fourier modes with mode numbers kmin <= |n| <= kmax (k = 2 pi n / L), shell spectrum
//!     E(k) ~ k^spectral_slope, random phases and an exact RMS. The amplitudes use the Yee-matched
//!     wavenumber k~, so div(B) = 0 on the grid.
//!   - Flow: u_perturbed, built the same way with independent phases (v_seed). Its RMS
//!     follows the relativistic Alfven relation (B_perturbed_rms / B_rms) vA, scaled by
//!     s <= 1 so that each species stays below u_total_cap.
//!   - Particles: uniform density n_s = n0/2 per species, Maxwell-Juttner at T_s, drifting
//!     at u_perturbed + u_drift_species. The thermal spread is an exact Maxwell-Juttner sample at
//!     every temperature (TurbCurrentDist::thermal_sample), and <gamma_s> is computed exactly. The drifts carry the current J (see
//!     InitFields::loaded_current), shared between the species by enthalpy at zero net
//!     momentum.
//!   - Electric field: E = -(u_motional x B), where u_motional is the field-line velocity
//!     (u_perturbed, plus the Hall part when the two species differ).
//!   - Charge: the polarisation charge rho = AMP div(E) is loaded by displacing the two
//!     species in opposite directions along E; velocities are then redrawn at the
//!     displaced positions.
//!   - Startup printout, also written to <simulation.name>/Simulation_params.txt: scales,
//!     spectrum, speeds, current, consistency checks and a Gauss-law check of the charge.
//!
//!  INPUTS
//!     B_guide                         Guide field along Z [B0]
//!     B_perturbed_rms                 RMS |B_perturbed| [B0]
//!     kmin, kmax                      mode-number band, k = 2 pi n / L
//!     spectral_slope                  slope of the shell spectrum E(k) ~ k^spectral_slope
//!                                     0 = flat, -5/3 = Kolmogorov
//!     B_seed, v_seed                  Phase seeds of B_perturbed and u_perturbed; must differ
//!     turb_kplane                     k-space: -1 = 3D, 0 = X-Y, 1 = Y-Z, 2 = Z-X
//!     turb_a_axis                     vector potential: -1 = random _|_ k, 0/1/2 = X/Y/Z
//!                                     (2 in 2D gives B_perturbed in the X-Y plane)
//!     turb_rand_amp                   Gaussian random mode amplitudes
//!     temperature                     Species-1 temperature kT_1 / (m0 c^2)
//!     temperature_ratio               T_2 / T_1
//!     perturb_velocity                Load u_perturbed
//!     u_total_cap                     Cap on each species' total speed [c], in (0, 0.99)
//!
//! =============================================================================
//!
//!  LIMITATIONS
//!
//!  NON-RELATIVISTIC  (sigma_h < 0.1; for ion-electron also vA_e = sqrt(m_i/m_e) vA << c)
//!   - Particle noise: the current drift and the polarisation charge are small against
//!     thermal noise. No quiet start; both species share ppc0 (Entity allows no particle
//!     weights in Cartesian coordinates). The Gauss-law check is noise-dominated here.
//!   - E is first order in the drifts (electron inertia omitted): relative size
//!     ~ (k_max d)^2 B_perturbed/B, d = electron (or pair) skin depth. Keep L/kmax >~ 20-40 d.
//!   - Ion-electron: the unloaded dE/dt is ~ (k_max d_i)^2 (vA/c)^2 B_perturbed/B,
//!     negligible while vA_e << c.
//!   - current_filters > 0 would make the filtered current differ from the loaded J at high k.
//!
//!  RELATIVISTIC  (sigma_h >= 0.1, or B_guide = 0)
//!   - Loaded (InitFields::loaded_current): the sigma_h split of J and the excess convection
//!     current of the displaced charge, both evaluated from the first-order fields.
//!   - NOT loaded, both needing E != 0 -- in a pair plasma only when u_perturbed is loaded
//!     (or T_1 != T_2), in an ion-electron plasma always (Hall field):
//!       * the parallel displacement current that keeps E.B = 0 as field lines rotate,
//!         J_par -> J_par - AMP E.(curl E)/|B|; it grows as (u/c)^2;
//!       * the advective part of dE_perp/dt, (u.grad)u x B - u x curl(u x B); it grows with vA/c.
//!     Measured with print_setup for B_perturbed/B ~ 0.4 (RMS fraction of AMP curl(B), advective / parallel):
//!     ion-electron at sigma_h = 0.02: 1.1% / 0.35%; pair at sigma_h = 1.1: 12-17% / 6-8% (kmax = 2-4);
//!     pair at sigma_h = 11: 21% / 10%.
//!     Both scale with s^2 when the u_total cap acts. Expect an initial transient of these
//!     sizes and measure it before trusting early times. print_setup estimates both from the
//!     loaded fields (UNLOADED DISPLACEMENT CURRENT AT t = 0).
//!   - The u_total cap uses the first-order current; print_setup reports the peak speed with
//!     the convection term and warns if it exceeds the cap.
//!   - Current starvation: the field-aligned current is carried in full, so a high sigma0
//!     with a large k_max d0 can exceed the cap (abort). Lower kmax or skindepth0.
//!   - Charge displacements of several cells at sigma_h ~ 1 limit the number of ranks per
//!     direction (guarded in load_charge_density).
//!   - Ion-electron: the sigma_h split uses the MHD inertia w, approximate at sub-ion scales,
//!     and the Hall part of curl(u_motional) ignores the sigma_h split.
//!   - 3D: the sigma_h split covers k_par != 0 at first order, but 3D is untested.
//! =============================================================================

namespace user
{
    using namespace ntt;

    //~ u_perturbed has phases independent of B_perturbed (v_seed != B_seed), so <u_perturbed . B_perturbed> = 0 on AVERAGE over seeds:
    //~ a BALANCED seed, both Elsasser families z+- excited. For one seed pair the normalised cross-helicity
    //~ scatters by ~ 1/sqrt(2 N_eff), N_eff = effective number of energy-carrying modes; print_setup reports it.
    //~ E = -(u_motional x B) at t = 0 (see InitFields::motional_E).

    //* Mode m of B_perturbed or u_perturbed: amplitude (bx, by, bz) times cos(k.x + phase). The curl of the
    //* vector potential is already folded into the amplitudes, so the device kernel is a bare cosine sum.
    //* kc are the Yee-matched wavenumbers used for curl(B).
    struct TurbMode
    {
        real_t kx, ky, kz;
        real_t bx, by, bz;
        real_t phase;
        real_t kcx, kcy, kcz;
    };

    //! Dimensionality of k-SPACE is not the dimensionality of the box: a
    //! plane-restricted spectrum inside a 3D box still has N(k)dk ~ k dk per shell
    inline auto kspace_dim(int kplane, bool is_3d) -> int
    {
        bool vary_x = true, vary_y = true, vary_z = is_3d;
        if      (kplane == 0) { vary_x = true;  vary_y = true;  vary_z = false; }
        else if (kplane == 1) { vary_x = false; vary_y = true;  vary_z = true;  }
        else if (kplane == 2) { vary_x = true;  vary_y = false; vary_z = true;  }
        if (!is_3d) { vary_z = false; }
        return (vary_x ? 1 : 0) + (vary_y ? 1 : 0) + (vary_z ? 1 : 0);
    }

    //* A_m = a_k e_A sin(k.x + ph)  =>  b_m cos(k.x + ph) with b_m = a_k (k x e_A):
    //* divergence-free by construction, with |b_m| = a_k |k| since e_A _|_ k.
    //*
    //* a_k ~ (k/k0)^((spectral_slope - dim - 1)/2): with N(k)dk ~ k^(dim-1)dk modes per shell this
    //* yields E(k) ~ k^(dim-1)|b_k|^2 ~ k^spectral_slope. dim = 2 -> (spectral_slope - 3)/2, dim = 3 -> (spectral_slope - 4)/2

    //* Yee-matched wavenumber k~ = (2/h) sin(k h/2): the one-cell centred difference of cos(k x) is exactly -k~ sin(k x); h = 0 -> continuum k
    inline auto yee_wavenumber(double k, double h) -> double
    {
        return (k == 0.0 || h <= 0.0) ? k : (2.0 / h) * std::sin(0.5 * k * h);
    }

    //? Solenoidal mode table with total RMS rms_target; returns the per-component RMS.
    //? cell_size > 0: amplitudes b_m = a_k (k~ x e_A), so k~ . b_m = 0 and the field is exactly divergence-free on the Yee grid; cell_size = 0: continuum k
    inline auto init_turbulence(real_t                 Lx,
                                real_t                 Ly,
                                real_t                 Lz,
                                bool                   is_3d,
                                real_t                 rms_target,      //* total RMS of the mode sum
                                int                    kmin,
                                int                    kmax,
                                long long              seed,
                                real_t                 spectral_slope,  //* E(k) ~ k^spectral_slope (shell spectrum)
                                int                    kplane,          //* -1 = full 3D, 0 = x1x2, 1 = x2x3, 2 = x3x1
                                int                    a_axis,          //* -1 = random e_A _|_ k, else 0/1/2 = fixed axis
                                bool                   random_amp,      //* Gaussian mode amplitudes
                                const std::array<double, 3>& cell_size, //* Yee cell (dx, dy, dz) for k~, or zeros for continuum k
                                std::vector<TurbMode>& modes_out) -> std::array<real_t, 3>
    {
        modes_out.clear();
        const std::array<real_t, 3> zero { (real_t)0.0, (real_t)0.0, (real_t)0.0 };

        if (rms_target <= (real_t)0.0 || kmax < kmin || kmin < 1) return zero;

        std::mt19937_64                        rng(static_cast<uint64_t>(seed));
        std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * M_PI);
        std::normal_distribution<double>       gauss_dist(0.0, 1.0);

        //! In 2D, only the x1-x2 plane is admissible: a mode with k3 != 0 would be
        //! sampled at x3 = 0 alone, breaking periodicity and the zero divergence
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

        //! dim is the dimensionality of K-SPACE, not of the simulation
        const double dim = (double)kspace_dim(kplane, is_3d);

        //~ E(k) ~ k^spectral_slope: the energy summed over a |k| shell, i.e. the slope read off a
        //~ power-spectrum plot. A shell holds ~ k^(dim-1) modes and |b_m| = a_k |k|, so the energy of
        //~ ONE mode goes as k^(spectral_slope - (dim-1)). Two values worth knowing:
        //~
        //~   spectral_slope = 0       -> E(k) = const: equal energy in each |k| SHELL. The number
        //~                               of modes per shell grows as k^(dim-1), so individual
        //~                               modes still get weaker with k.
        //~   spectral_slope = dim - 1 -> weight_exp = -1 exactly, a_k ~ 1/k, |b_m| = const:
        //~                               equal energy in every MODE. That is +1 in 2D and +2 in
        //~                               3D, and it is what iPIC3D's AlfvTurb2D does.
        //~                               NB dim is the K-SPACE dimension, so turb_kplane moves it.
        const double weight_exp = 0.5 * ((double)spectral_slope - dim - 1.0);

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

                    double ex, ey, ez;          //* components of e_A, the direction of this mode's vector potential
                    if (a_axis >= 0 && a_axis <= 2)
                    {
                        //* fixed e_A: b_m = a_k (k x e_A) is perpendicular to e_A, so a_axis = 2
                        //* with k in x1-x2 gives an in-plane mode (b_z = 0), the 2D flux-function setup
                        ex = (a_axis == 0) ? 1.0 : 0.0;
                        ey = (a_axis == 1) ? 1.0 : 0.0;
                        ez = (a_axis == 2) ? 1.0 : 0.0;
                    }
                    else
                    {
                        //* project out the k~-parallel part: it drops out of k~ x e
                        //* anyway, and removing it keeps |b_m| = a_k|k~| exact
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

        //~ RMS is EXACT, not sampled: half-space selection means no two modes share
        //~ +/-k, so cross terms vanish over the box and each component's mean square is 0.5 sum_m b_i,m^2.
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

    //~ Host mode sums, shared by the u_total cap (make_init) and the startup
    //~ diagnostics (print_setup). Both evaluate exactly what the device kernels load.

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

    //* sigma_h split on the host (mirrors InitFields::sigma_split): V -> V - f V_perp, f = sigma_local / (1 + sigma_local)
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

    //* Host sampling grid for peak and consistency checks. The fields are band-limited at |n| <= kmax, so
    //* 16 points per shortest wavelength (8 in 3D) locate peaks to within 1 - cos(pi/16) ~ 2% (8% in 3D).
    //* Never finer than the simulation grid. Cost ~ (points per side)^dim x (number of modes) ~ kmax^4 in 2D.
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
    //~  The mean flow is zero (plasma rest frame, no k = 0 mode), so u_perturbed is
    //~  the whole velocity the two species share. Same solenoidal construction as
    //~  B_perturbed (u_m = c_k (k x e_A), same band, same spectral slope, same
    //~  polarisation axis) but from an INDEPENDENT phase set seeded by
    //~  setup.v_seed. s <= 1 is the u_total cap factor (make_init); it
    //~  scales every mode alike, so u_perturbed stays solenoidal.
    //~
    //~  Reusing B_perturbed's seed would reproduce the same phases and the same
    //~  random amplitudes mode for mode, giving u_perturbed proportional to
    //~  B_perturbed everywhere: a pure Elsasser state with z- = 0, for which the
    //~  nonlinear term z-.grad(z+) vanishes identically. That is an exact nonlinear
    //~  solution of incompressible MHD -- it propagates and does not cascade.
    //~  Independent phases give <u_perturbed . B_perturbed> = 0 on average over seeds.
    //~
    //~  This mirrors iPIC3D's AlfvTurb2D, which keeps rphb (B and J) and
    //~  rphv (velocity) as separate phase arrays.
    //~
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

        //* u_perturbed [c] at x_Ph; identically zero when no modes were built
        //* (setup.perturb_velocity = false or B_perturbed_rms = 0), leaving u_total_species = u_drift_species.
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

        //! =========================================================================
        //! MAGNETIC FIELD   B = B_guide (Z) + B_perturbed
        //! =========================================================================

        Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t { return turb(x_Ph, 0); }
        Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t { return turb(x_Ph, 1); }
        Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t { return B_guide + turb(x_Ph, 2); }

        //! =========================================================================
        //! MOTIONAL ELECTRIC FIELD    E = -(u_motional x B)
        //!
        //! Each species moves at u_total_species = u_perturbed + u_drift_species.
        //! E is chosen so that the Lorentz force q_s (E + u_total_species x B) gives
        //! both species the SAME acceleration (= Lorentz force per particle / h_s):
        //! the plasma as a whole feels only J x B (h_s = enthalpy per particle) 
        //! 
        //! With E = -(u_motional x B), this requires 
        //!     q_1 (u_total_1 - u_motional) / h_1 = q_2 (u_total_2 - u_motional) / h_2
        //!     =>  u_motional = (h_2 u_total_1 + h_1 u_total_2) / (h_1 + h_2) = u_perturbed + u_drift_1 + u_drift_2,
        //! where u_drift_species = sign(q_s) lambda / h_s (TurbCurrentDist)
        //!
        //!     pair plasma + equal T : u_drift_1 = -u_drift_2  ->  u_motional = u_perturbed (ideal MHD)
        //!     ion-electron plasma   : u_motional ~ electron velocity -> E = -(u_perturbed x B) + Hall field
        //!
        //! Velocities are 3-velocities in units of c, and c = 1 in code units.
        //!
        //!     u_motional = u_perturbed + u_motional_coeff * J_code/AMP,   u_motional_coeff = u_drift_coeff_1 + u_drift_coeff_2
        //!  with J_code/AMP from first_order_current (curl(B) with its sigma_h split)
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
        //! curl(B) (~ J)
        //!
        //!     B_m(x) = b_m cos(k_m.x + ph_m)
        //!     curl(b cos(k.x + ph)) = -(k x b) sin(k.x + ph)
        //!
        //! The uniform B_guide has zero curl, so only the mode sum contributes.
        //! kc is used in place of k so that the DISCRETE Yee curl is matched
        //! exactly (see init_turbulence); with kc = k this is the continuum curl.
        //!
        //! Returns curl(B) only: J_code = AMP * curl(B) and u_drift_species = u_drift_coeff_s * curl(B)
        //!  are computed by the callers.
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
        //! SIGMA_H SPLIT OF THE CURRENT:  J_code = AMP * [ curl(B) - f (curl B)_perp ]  (first order)
        //!
        //!  NEEDED ONLY FOR sigma_h >= 0.1, OR FOR B_guide = 0 (where the error without
        //!  it is sigma_h itself). For sigma_h << 0.1 with a guide field, f -> sigma_h -> 0
        //!  and this reduces to J_code = AMP curl(B), so it is applied unconditionally.
        //!
        //!  A current perpendicular to the local B accelerates the plasma (J x B), and the
        //!  growing E = -(u x B) supplies part of curl(B) as displacement current:
        //!      c (curl B)_perp = 4 pi J_perp + dE_perp/dt,  dE_perp/dt = J_perp B^2 / (c w)
        //!   => J_perp = (c/4 pi) (curl B)_perp / (1 + sigma_h),   sigma_h = B^2 / (4 pi w)
        //!  The field-aligned part is carried in full (E_par = 0). sigma_h is evaluated with
        //!  the LOCAL |B|^2, so f -> 0 at magnetic nulls, where the plasma carries all of J.
        //!
        //!  2D (k in X-Y, B_guide along Z): J_z is reduced and an in-plane current along the
        //!  tilted field lines appears. 3D: the in-plane current from d/dZ is also reduced
        //!  (J = curl(B) (1 - vA^2/c^2) for waves along B). First order in the fluctuations.
        //!  Used for the Hall part of E; the drifts use loaded_current (adds the convection term).
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

        Inline void first_order_current(const coord_t<D>& x_Ph, real_t& current_x, real_t& current_y, real_t& current_z) const
        {
            real_t B[3] = { turb(x_Ph, 0), turb(x_Ph, 1), B_guide + turb(x_Ph, 2) };
            real_t current[3];
            curl_B(x_Ph, current[0], current[1], current[2]);
            sigma_split(B, current);
            current_x = current[0]; current_y = current[1]; current_z = current[2];
        }

        //* curl(curl B) = sum_m |kc_m|^2 b_m cos(k_m.x + ph_m), using kc . b = 0 (k~-built amplitudes)
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
        //! CURRENT CARRIED BY THE DRIFTS (TurbCurrentDist):
        //!     J_drift / AMP = sigma_h split of [ curl(B) - div(E) u_convection ]
        //!
        //!  The displaced charge rho = AMP div(E) (load_charge_density) moves with the mean
        //!  velocity of the two species, u_convection = (u_perturbed + u_motional)/2, so it
        //!  carries the convection current rho u_convection on top of the drifts. Subtracting
        //!  it BEFORE the sigma_h split makes the TOTAL current
        //!      along B:   AMP (curl B)_par
        //!      across B:  [AMP (curl B)_perp + sigma_h rho u_convection_perp] / (1 + sigma_h)
        //!  i.e. the drift current of the polarisation charge is kept where it is physical, and
        //!  only its excess is removed: rho u_par in full, rho u_perp / (1 + sigma_h).
        //!
        //!  Nonzero only where E != 0: in a pair plasma only when u_perturbed is loaded (or
        //!  T_1 != T_2); in an ion-electron plasma always (Hall field). Relative size ~ (u/c)^2,
        //!  so negligible for sigma_h << 0.1. rho and u_motional are taken at first order
        //!  (first_order_current), so there is no recursion; the Hall part of curl(u_motional)
        //!  uses curl(curl B), ignoring the sigma_h split there.
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
    //!  Energy distribution: Maxwell-Juttner at theta_s, drifting at the local
    //!  LAB-frame velocity  u_total_species = u_perturbed + u_drift_species
    //!
    //!  Entity integrates   dE/dt = curl(B) - (r0/d0^2) J,  AMP = d0^2 / r0 = d0 * sqrt(sigma0).
    //!  The drifts carry J_code = AMP * J_loaded, J_loaded = sigma_h split of [curl(B) - div(E) u_convection]
    //!  (InitFields::loaded_current; = curl(B) for sigma_h << 0.1). Below, curl(B) stands for J_loaded.
    //!
    //!  The two species carry J at fixed density n_s and |q_s| = 1:
    //!
    //!        J_code = sum_s n_s q_s u_drift_species
    //!
    //!  A slowly drifting thermal species carries momentum density n_s h_s u_drift_species,
    //!  where h_s = m_s <gamma_s> + T_s is its enthalpy per particle. Each species drifts  
    //!  with the same momentum per particle, lambda, in opposite directions,  
    //!  h_s u_drift_species = sign(q_s) lambda. With equal densities the momenta cancel,
    //!  sum_s n_s h_s u_drift_species = 0, and the current is J_code = lambda sum_s n_s / h_s
    //!
    //!        u_drift_species = sign(q_s) * AMP * curl(B) / ( h_s * sum_t n_t/h_t )
    //!                        = u_drift_coeff_s * curl(B).
    //!
    //!  Cold plasma: h_s -> m_s (current carried mostly by the light species).
    //!  Pair plasma at equal temperatures: equal and opposite drifts.
    //!
    //!  Velocity sampling in TurbCurrentDist makes each species' mean 3-velocity 
    //!  <v> = u_total_species (on average, up to particle noise), and the injector 
    //!  loads uniform lab densities n_s. The loaded current is therefore  
    //!  sum_s q_s n_s u_total_species = sum_s q_s n_s u_perturbed + sum_s q_s n_s u_drift_species = 0 + AMP * curl(B):
    //!  the u_perturbed terms cancel because the plasma is neutral (equal n_s, opposite q_s), 
    //!  for any regime and any orientation of u_perturbed relative to J. After the charge-loading
    //!  displacement it is not exactly neutral; that convection current is what J_loaded subtracts.
    //!  
    //! =========================================================================
    template <class M>
    struct TurbCurrentDist
    {
        static constexpr auto D = M::Dim;

        TurbCurrentDist(random_number_pool_t& pool, real_t theta_s, real_t u_drift_coeff,
                        const InitFields<M::Dim>& flds, const TurbVelocity<M::Dim>& vel):
                        maxwellian { pool, theta_s }, pool { pool }, theta { theta_s }, u_drift_coeff { u_drift_coeff },
                        flds { flds }, vel { vel } {}

        //? ISOTROPIC Maxwell-Juttner four-velocity at temperature theta = kT_s/(m_s c^2), in the rest frame.
        //? theta >= 0.5: Entity's Maxwellian, which uses the exact Sobol method there.
        //? theta <  0.5: Entity's Maxwellian draws a non-relativistic Gaussian four-velocity, which loads
        //?   <gamma> - 1 at 0.98, 0.93, 0.81, 0.62 of the Maxwell-Juttner value at theta = 0.01, 0.03, 0.1, 0.3,
        //?   so the energy eps = gamma - 1 is sampled exactly here instead. Its distribution is
        //?       f(eps) ~ sqrt(eps) (1 + eps) sqrt(1 + eps/2) exp(-eps/theta),
        //?   and since sqrt(1 + x) <= 1 + x/2 it is bounded by the envelope sqrt(eps) (1 + eps)(1 + eps/4) exp(-eps/theta):
        //?   a mixture of Gamma distributions of shape 3/2, 5/2, 7/2 and scale theta, with weights 1 : 15 theta/8 : 15 theta^2/16.
        //?   Gamma(k + 1/2) = (sum of k unit exponentials) + Z^2/2 with Z normal. A candidate is accepted with
        //?   probability f/envelope = sqrt(1 + eps/2) / (1 + eps/4); the acceptance rate is above 0.97 for theta < 0.5.
        //?   The direction is isotropic and |u| = sqrt(eps (eps + 2)).
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
                //* choose the component: number of unit exponentials beyond the first (0, 1 or 2)
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

        //? Turn an ISOTROPIC Juttner sample into one drifting at u_total_species (lab
        //? frame) by the flipping method of Zenitani (2015), arXiv:1504.03910 -- the
        //? same scheme Entity's arch::energy_dist::Maxwellian uses, generalised
        //? to a drift along an arbitrary, position-dependent direction.
        //?
        //? A naive Lorentz boost of each sampled four-velocity is NOT equivalent: it
        //? misses the Jacobian g'/g between the rest-frame and lab-frame momentum
        //? measures and undersamples forward-moving particles.
        Inline void drift_sample(vec_t<Dim::_3D>& v, real_t u_total_species_x, real_t u_total_species_y, real_t u_total_species_z) const
        {
            real_t u_total_species_sq = SQR(u_total_species_x) + SQR(u_total_species_y) + SQR(u_total_species_z);

            //! Safety net: make_init scales u_perturbed so that |u_total_species| <=  setup.u_total_cap (< 0.99)
            //! on the sampling grid, so this clamp can act at most between sampling points.
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

            //* boost the parallel component; the perpendicular part is untouched
            const real_t four_velocity_parallel_boosted = lorentz_factor_drift * (four_velocity_parallel + u_total_species_magnitude * lorentz_factor);

            v[0] = four_velocity_x - four_velocity_parallel_initial * direction_x + four_velocity_parallel_boosted * direction_x;
            v[1] = four_velocity_y - four_velocity_parallel_initial * direction_y + four_velocity_parallel_boosted * direction_y;
            v[2] = four_velocity_z - four_velocity_parallel_initial * direction_z + four_velocity_parallel_boosted * direction_z;
        }

        Inline void operator()(const coord_t<D>& x_Ph, vec_t<Dim::_3D>& v) const
        {
            //* 1. isotropic thermal four-velocity: exact Maxwell-Juttner at theta_s
            thermal_sample(x_Ph, v);

            //* 2. current-carrying drift of this species, u_drift_species = u_drift_coeff_s * J_code/AMP
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
        const real_t larmor0;                   //* rho0 [d0]
        const real_t skindepth0;                //* d0   [d0]
        const real_t sp1_mass;                  //* fiducial units
        const real_t sp2_mass;                  //* species 2 mass (positrons / ions)
        const real_t sp1_charge;                //* signed, |q| must be 1
        const real_t sp2_charge;
        const real_t mass_ratio;                //* DERIVED = sp2_mass / sp1_mass  (= 1 for pair plasma)
        const real_t sigma0;                    //* (skindepth0/larmor0)^2 = magnetisation of the FIDUCIAL species at B = 1

        //? Setup
        const real_t    B_guide;                //* guide field along x3, setup.B_guide (0 -> zero-mean-field turbulence) in code B units
        const real_t    B_perturbed_rms;        //* RMS |B_perturbed|, setup.B_perturbed_rms in code B units
        const int       kmin, kmax;             //* mode-number band (inclusive)
        const real_t    spectral_slope;         //* E(k) ~ k^spectral_slope, shell spectrum (0 default = flat)
        const long long B_seed;                 //* phases of B_perturbed
        const long long v_seed;                 //* phases of u_perturbed; MUST differ from B_seed
        const int       turb_kplane;            //* k-space restriction: -1 = 3D, 0 = x1x2, 1 = x2x3, 2 = x3x1
        const int       turb_a_axis;            //* vector potential axis: -1 = random _|_ k, 0/1/2 = x1/x2/x3
        const bool      turb_rand_amp;          //* true -> Gaussian random mode amplitudes
        const real_t    temperature;            //* species-1 temperature kT_1 / (m0 c^2)
        const real_t    temperature_ratio;      //* T_2 / T_1 (ratio of kT, not of theta); 1 -> equal temperatures
        const bool      perturb_velocity;       //* false -> u_perturbed = 0
        const real_t    u_total_cap;            //* cap on |u_total_species| for both species [c]

        //? Geometry
        const real_t xmin, xmax, ymin, ymax, zmin, zmax, Lx, Ly, Lz;

        //? Derived quantities
        real_t                temperature_1 { ZERO }, temperature_2 { ZERO };                     //* kT_s / (m0 c^2)
        real_t                theta_1 { ZERO }, theta_2 { ZERO };                                 //* kT_s / (m_s c^2)
        real_t                enthalpy_per_particle_1 { ONE }, enthalpy_per_particle_2 { ONE };   //* h_s = m_s <gamma_s> + T_s [m0 c^2]
        std::array<real_t, 3> B_perturbed_rms_component { ZERO, ZERO, ZERO };
        real_t                AMP { ONE };        //* d0^2/r0 = d0 sqrt(sigma0): J_code = AMP curl(B)
        real_t                u_drift_coeff_1 { ZERO }, u_drift_coeff_2 { ZERO };
        real_t                u_motional_coeff { ZERO };
        real_t                sigma_h_per_B_sq { ZERO };     //* sigma0 / w: local sigma_h = sigma_h_per_B_sq |B|^2 (InitFields::loaded_current)

        real_t                u_perturbed_rms_alfven { ZERO };   //* Alfven-relation target for RMS |u_perturbed| [c]
        real_t                u_perturbed_cap_factor { ONE };    //* s <= 1 applied to u_perturbed by the u_total cap
        real_t                u_perturbed_rms { ZERO };          //* RMS |u_perturbed| actually loaded = s * u_perturbed_rms_alfven (0 if perturb_velocity = false)
        std::array<real_t, 3> u_perturbed_rms_component { ZERO, ZERO, ZERO };

        //* peaks on the host sampling grid, after the cap [c]
        real_t                peak_u_perturbed { ZERO };
        real_t                peak_u_drift_1 { ZERO }, peak_u_drift_2 { ZERO };
        real_t                peak_u_total_1 { ZERO }, peak_u_total_2 { ZERO };
        real_t                peak_charge_displacement { ZERO };   //* max |xi_s| [d0], set in print_setup, checked in load_charge_density

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

            //? u_drift_species = sign(q_s) * AMP * curl(B) / (h_s sum_t n_t/h_t) assumes |q_s| = 1 for 
            //? both species, and opposite signs so that they can counter-drift at zero net charge.
            raise::ErrorIf(math::abs(math::abs(g.sp1_charge) - ONE) > (real_t)1.0e-5 ||
                           math::abs(math::abs(g.sp2_charge) - ONE) > (real_t)1.0e-5,
                           "current loading requires |q_s| = 1 for both species: "
                           "u_drift_species = sign(q_s) * AMP * curl(B) / (h_s sum n/h) assumes it, "
                           "so the loaded current would be wrong otherwise.", HERE);
            raise::ErrorIf(g.sp1_charge * g.sp2_charge >= ZERO,
                           "current loading requires the two species to have OPPOSITE charge "
                           "signs: they must counter-drift to carry J at zero net charge.", HERE);

            //? Temperature
            //? T_s = kT_s / (m0 c^2) is what enters pressures, enthalpies and Debye
            //? lengths; theta_s = T_s / m_s = kT_s / (m_s c^2) is what the
            //? Maxwell-Juettner sampler and <gamma_s> take.
            g.temperature_1 = g.temperature;
            g.temperature_2 = g.temperature * g.temperature_ratio;
            g.theta_1       = g.temperature_1 / g.sp1_mass;
            g.theta_2       = g.temperature_2 / g.sp2_mass;

            //* h_s = m_s <gamma_s> + T_s: inertia per particle of a drifting thermal species
            g.enthalpy_per_particle_1 = enthalpy_per_particle(g.sp1_mass, g.temperature_1);
            g.enthalpy_per_particle_2 = enthalpy_per_particle(g.sp2_mass, g.temperature_2);

            //* sigma_h = sigma0 |B|^2 / w, w = enthalpy density; sets the perpendicular-current correction
            g.sigma_h_per_B_sq = g.sigma0 / enthalpy(g.sp1_mass, g.temperature_1, g.sp2_mass, g.temperature_2);

            const bool is_3d = (D == Dim::_3D);

            //? B_perturbed is built with the Yee-matched k~ = (2/dx) sin(k dx/2): it is then exactly divergence-free on the grid (k~ . b = 0),
            //? and the curl it supplies, -(k~ x b) sin(k.x + ph), equals the discrete Yee curl of the initialised B to round-off.
            const auto&  mesh_for_k = g.metadomain.mesh();
            const double cell_dx    = (double)g.Lx / (double)mesh_for_k.n_active(in::x1);
            const double cell_dy    = (double)g.Ly / (double)mesh_for_k.n_active(in::x2);
            double       cell_dz    = 0.0;
            if constexpr (D == Dim::_3D) { cell_dz = (double)g.Lz / (double)mesh_for_k.n_active(in::x3); }

            g.B_perturbed_rms_component = init_turbulence(g.Lx, g.Ly, g.Lz, is_3d, g.B_perturbed_rms,
                                                          g.kmin, g.kmax, g.B_seed,
                                                          g.spectral_slope, g.turb_kplane, g.turb_a_axis,
                                                          g.turb_rand_amp, { cell_dx, cell_dy, cell_dz }, g.B_perturbed_modes_host);

            g.AMP = SQR(g.skindepth0) / g.larmor0;      //* AMP = d0^2/r0 = d0 sqrt(sigma0)

            //! n_s = N_PER_SP is set by the injector: CONFIRMED against
            //! src/archetypes/particle_injector.h, the last argument of
            //! InjectNonUniform is the "Total number density (in units of n0)", and
            //! the kernel takes ppc = number_density * ppc0 * HALF, i.e. an even split.
            //*   u_drift_coeff_s = sign(q_s) AMP / (h_s sum_t n_t/h_t)
            const real_t inv_enthalpy_sum = N_PER_SP * (ONE / g.enthalpy_per_particle_1 + ONE / g.enthalpy_per_particle_2);
            const real_t sign_1           = (g.sp1_charge < ZERO) ? -ONE : ONE;
            const real_t sign_2           = (g.sp2_charge < ZERO) ? -ONE : ONE;
            g.u_drift_coeff_1 = sign_1 * g.AMP / (g.enthalpy_per_particle_1 * inv_enthalpy_sum);
            g.u_drift_coeff_2 = sign_2 * g.AMP / (g.enthalpy_per_particle_2 * inv_enthalpy_sum);

            //* u_motional = u_perturbed + u_drift_1 + u_drift_2 (see InitFields::motional_E):
            //* exactly zero correction for a pair plasma at equal temperatures
            g.u_motional_coeff = g.u_drift_coeff_1 + g.u_drift_coeff_2;

            //~ Velocity perturbation: Alfven-relation target
            raise::ErrorIf(g.perturb_velocity && (g.v_seed == g.B_seed),
                           "setup.v_seed must differ from setup.B_seed. Equal seeds produce "
                           "the same phases and 'random' amplitudes for each mode, yielding "
                           "u_perturbed || B_perturbed everywhere -- a pure Elsasser state (z- = 0) "
                           "that is an exact nonlinear solution and does not cascade at all.", HERE);

            //~ u_perturbed_rms follows from setup.B_perturbed_rms through the relativistic Alfven
            //~ eigenrelation (setting it equal to B_perturbed_rms in code units would 
            //~ ignore vA). Setting w as the relativistic ENTHALPY density (see enthalpy()),
            //~
            //~     u_perturbed_rms = B_perturbed_rms / sqrt(4 pi w + B^2) * c  =  (B_perturbed_rms / B_rms) * vA,
            //~     vA / c = sqrt( sigma_h / (1 + sigma_h) ), sigma_h = B_rms^2 sigma0 / w.
            //~
            //~ For sigma_h << 1, B_rms cancels and u_perturbed_rms = B_perturbed_rms / sqrt(4 pi rho).
            //~ In 2D with k _|_ B_guide the modes do not propagate, so this sets the
            //~ kinetic/magnetic energy partition rather than describing Alfven waves
            {
                const real_t B_rms_ = (real_t)std::sqrt((double)(g.B_guide * g.B_guide + g.B_perturbed_rms * g.B_perturbed_rms));
                const real_t B_ref_ = (B_rms_ > ZERO) ? B_rms_ : ONE;
                const real_t w_     = enthalpy(g.sp1_mass, g.temperature_1, g.sp2_mass, g.temperature_2);
                const real_t sig_h  = g.sigma0 * B_ref_ * B_ref_ / w_;
                const real_t vA_h   = (real_t)std::sqrt((double)sig_h / (1.0 + (double)sig_h));

                //* computed even when perturb_velocity = false: the eddy-time estimate uses it
                g.u_perturbed_rms_alfven = vA_h * g.B_perturbed_rms / B_ref_;

                //* perturb_velocity = false -> target 0 -> init_turbulence returns an empty
                //* table -> u_perturbed = 0. u_drift_species is unaffected.
                const real_t u_perturbed_rms_target = g.perturb_velocity ? g.u_perturbed_rms_alfven : ZERO;
                g.u_perturbed_rms           = u_perturbed_rms_target;
                g.u_perturbed_rms_component = init_turbulence(g.Lx, g.Ly, g.Lz, is_3d, u_perturbed_rms_target,
                                                              g.kmin, g.kmax, g.v_seed,
                                                              g.spectral_slope, g.turb_kplane, g.turb_a_axis,
                                                              g.turb_rand_amp, { 0.0, 0.0, 0.0 }, g.u_perturbed_modes_host);
            }

            //~ Capping particles' velocity perturbation
            //~
            //~ Each species moves at u_total_species(x) = s u_perturbed(x) + u_drift_species(x)
            //~ in the lab frame. u_perturbed is a sum of modes, so where many modes add in phase
            //~ its local speed is several times u_perturbed_rms; at large vA/c (high
            //~ sigma, or weak guide field) those peaks may exceed c. To prevent this,
            //~ u_perturbed pattern is scaled by a factor
            //~
            //~     s = largest value in [0, 1] with  max_{x,species} |s u_perturbed + u_drift_species| <= u_total_cap,
            //~
            //~ which keeps u_perturbed exactly solenoidal and band-limited; only the
            //~ kinetic-to-magnetic fluctuation energy ratio drops, to s^2 times the
            //~ Alfven value. f(s) = max|s u_perturbed + u_drift_species| is convex in s,
            //~ so {f <= cap} is an interval containing 0 and bisection finds its upper end.
            //~ 
            //~ If u_drift_species alone exceeds u_total_cap (f(0) > u_total_cap), no s 
            //~ helps: the field needs more current than this plasma density can carry,
            //~ since u_drift ~ J / (n q). Remedies: lower kmax or B_perturbed_rms, or raise the 
            //~ density in code units (smaller skindepth0 or larger larmor0, i.e. lower sigma0); 
            //~ raising ppc0 does not help.
            {
                const std::array<int, 3> counts = sampling_counts(g);
                const std::size_t n_samples = (std::size_t)counts[0] * (std::size_t)counts[1] * (std::size_t)counts[2];

                std::vector<double> u_perturbed_samples(3 * n_samples, 0.0);
                std::vector<double> first_order_current_samples(3 * n_samples, 0.0);   //* first-order J_code/AMP (sigma_h split of curl B)

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
                g.peak_u_perturbed = (real_t)peak_u_perturbed;
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

            //* g.turb_vel was filled above, so InitFields can hold it and serve
            //* the motional field E = -(u_motional x B) from ex1/ex2/ex3.
            return InitFields<D> { g.B_guide,
                                   g.kx_d, g.ky_d, g.kz_d,
                                   g.bx_d, g.by_d, g.bz_d, g.ph_d,
                                   g.kcx_d, g.kcy_d, g.kcz_d, nm,
                                   g.turb_vel, g.u_motional_coeff, g.sigma_h_per_B_sq };
        }

        //? Densities in units of n0. Entity's n0 = ppc0 / V0 is the TOTAL fiducial
        //? density, and InitPrtls injects ndens = 1 split evenly over the two
        //? species, so n_s = 0.5 n0. Every sigma, enthalpy and pressure below must
        //? carry this factor: sigma_s = B^2 sigma0 / (n_s m_s), NOT / m_s.
        static constexpr real_t N_TOTAL  = static_cast<real_t>(1.0);
        static constexpr real_t N_PER_SP = static_cast<real_t>(0.5);

        //? <gamma> of a 3D Maxwell-Juettner distribution, exactly: K1(1/theta)/K2(1/theta) + 3 theta, evaluated
        //? by quadrature so that no special functions are needed and nothing underflows at small theta. With
        //? eps = gamma - 1 = theta t^2, the distribution in t is  t^2 (1 + eps) sqrt(1 + eps/2) exp(-t^2)  (up to a
        //? constant): smooth at t = 0 and negligible beyond t = 9. Simpson's rule with 4000 intervals agrees with the
        //? Bessel-function form to ~1e-15 for 0.001 <= theta <= 100.
        static auto gamma_mean(real_t theta) -> real_t
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
            return (real_t)(gamma_weight_sum / weight_sum);
        }

        //? Enthalpy per particle in units of m0 c^2:  h_s = m_s <gamma_s> + T_s
        //? (energy + pressure per particle, theta_s = T_s / m_s). This is the inertia
        //? of a drifting thermal species: momentum density = n_s h_s u for a slow drift u.
        static auto enthalpy_per_particle(real_t mass, real_t temperature_s) -> real_t
        {
            return mass * gamma_mean(temperature_s / mass) + temperature_s;
        }

        //? Relativistic enthalpy density in units of n0 m0 c^2:  w = sum_s n_s h_s.
        //? This is the inertia that sets the Alfven speed
        static auto enthalpy(real_t mass_1, real_t temperature_1_, real_t mass_2, real_t temperature_2_) -> real_t
        {
            return N_PER_SP * enthalpy_per_particle(mass_1, temperature_1_) + N_PER_SP * enthalpy_per_particle(mass_2, temperature_2_);
        }

        //? setup.spectral_slope: slope of the shell spectrum E(k) ~ k^spectral_slope, default 0 (flat).
        //? The old key setup.spectral_index (E ~ k^-p, i.e. spectral_index = -spectral_slope) is refused,
        //? so an old input file cannot silently produce the opposite spectrum.
        static auto read_spectral_slope(const SimulationParams& p) -> real_t
        {
            const real_t old_spectral_index = p.template get<real_t>("setup.spectral_index", std::numeric_limits<real_t>::quiet_NaN());
            raise::ErrorIf(!std::isnan((double)old_spectral_index),
                           "setup.spectral_index has been replaced by setup.spectral_slope = -spectral_index, the slope of "
                           "E(k) ~ k^spectral_slope (e.g. spectral_index = 1.6667 -> spectral_slope = -1.6667; "
                           "spectral_index = -1 -> spectral_slope = 1).", HERE);
            return p.template get<real_t>("setup.spectral_slope", ZERO);
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
            params { p }, metadomain { m },
            larmor0           { p.template get<real_t>("scales.larmor0") },
            skindepth0        { p.template get<real_t>("scales.skindepth0") },
            sp1_mass          { species_mass(m, 0) },
            sp2_mass          { species_mass(m, 1) },
            sp1_charge        { species_charge(m, 0) },
            sp2_charge        { species_charge(m, 1) },
            mass_ratio        { sp2_mass / sp1_mass },
            sigma0            { (skindepth0 / larmor0) * (skindepth0 / larmor0) },
            B_guide           { p.template get<real_t>("setup.B_guide", (real_t)1.0) },
            B_perturbed_rms   { p.template get<real_t>("setup.B_perturbed_rms", (real_t)1.0) },
            kmin              { p.template get<int>("setup.kmin", 1) },
            kmax              { p.template get<int>("setup.kmax", 4) },
            spectral_slope    { read_spectral_slope(p) },
            B_seed            { static_cast<long long>(p.template get<int>("setup.B_seed", 12345)) },
            v_seed            { static_cast<long long>(p.template get<int>("setup.v_seed", 67890)) },
            turb_kplane       { p.template get<int>("setup.turb_kplane", (D == Dim::_3D) ? -1 : 0) },
            turb_a_axis       { p.template get<int>("setup.turb_a_axis", -1) },
            turb_rand_amp     { p.template get<bool>("setup.turb_rand_amp", true) },
            temperature       { p.template get<real_t>("setup.temperature", (real_t)0.01) },
            temperature_ratio { p.template get<real_t>("setup.temperature_ratio", (real_t)1.0) },
            perturb_velocity  { p.template get<bool>("setup.perturb_velocity", true) },
            u_total_cap       { p.template get<real_t>("setup.u_total_cap", (real_t)0.7) },
            xmin              { m.mesh().extent(in::x1).first },
            xmax              { m.mesh().extent(in::x1).second },
            ymin              { m.mesh().extent(in::x2).first },
            ymax              { m.mesh().extent(in::x2).second },
            zmin              { (D == Dim::_3D) ? m.mesh().extent(in::x3).first : (real_t)0.0 },
            zmax              { (D == Dim::_3D) ? m.mesh().extent(in::x3).second : (real_t)1.0 },
            Lx                { xmax - xmin },
            Ly                { ymax - ymin },
            Lz                { zmax - zmin },
            init_flds         { make_init(*this) } { print_setup(); }

        //? Startup diagnostics; all lengths in code units (d0)
        void print_setup()
        {
            int mpi_rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

            const bool        is_pair = (mass_ratio == (real_t)1.0);
            const std::string sp2     = is_pair ? "positron" : "ion";
            const std::string sp2s    = is_pair ? "positrons" : "ions";
            const std::string SP2u    = is_pair ? "Positrons" : "Ions";
            const std::string title   = is_pair ? "Decaying turbulence (pair plasma)"
                                                : "Decaying turbulence (ion-electron plasma)";

            const auto   nx1 = metadomain.mesh().n_active(in::x1);
            const auto   nx2 = metadomain.mesh().n_active(in::x2);
            const real_t dx  = Lx / (real_t)nx1;
            const real_t dy  = Ly / (real_t)nx2;
            real_t       dz  = ONE;
            if constexpr (D == Dim::_3D) { dz = Lz / (real_t)metadomain.mesh().n_active(in::x3); }

            const real_t g1 = gamma_mean(theta_1);     //* electrons
            const real_t g2 = gamma_mean(theta_2);     //* positrons / ions

            //! Everything below is evaluated at the ACTUAL initial RMS field, not at
            //! a separate reference amplitude: the field scale comes from [scales].
            const real_t B_rms = (real_t)std::sqrt((double)(B_guide * B_guide + B_perturbed_rms * B_perturbed_rms));
            const real_t B_ref = (B_rms > (real_t)0.0) ? B_rms : (real_t)1.0;   //* guard: B = 0 run

            //* sigma_s = B^2 / (4 pi n_s m_s c^2), n_s = 1/2
            const real_t sigma_1_cold = sigma0 * B_ref * B_ref / (N_PER_SP * sp1_mass); //* electrons
            const real_t sigma_2_cold = sigma0 * B_ref * B_ref / (N_PER_SP * sp2_mass); //* sp2
            //* both species contribute inertia: 1/sigma = 1/s1 + 1/s2
            const real_t sigma_cold   = sigma0 * B_ref * B_ref / (N_PER_SP * (sp1_mass + sp2_mass));
            //! hot magnetisation uses the relativistic ENTHALPY w = sum_s n_s (m_s <g_s> + T_s),
            //! not the energy density sum_s n_s m_s <g_s>: the pressure adds T_s per particle
            //! to the inertia (2.9% at theta = 0.03), and it grows with theta.
            const real_t w_enth       = enthalpy(sp1_mass, temperature_1, sp2_mass, temperature_2);
            const real_t sigma_hot    = sigma0 * B_ref * B_ref / w_enth;
            const real_t vA_cold      = (real_t)std::sqrt((double)sigma_cold / (1.0 + (double)sigma_cold));
            const real_t vA_hot       = (real_t)std::sqrt((double)sigma_hot / (1.0 + (double)sigma_hot));

            //! ---- kinetic scales -------------------------------------------------
            //! Entity's n0 = ppc0 / V0 is the TOTAL fiducial density and InitPrtls
            //! injects ndens = 1 split evenly, so n_s = n0/2:
            //!
            //!     d_s        = d0 sqrt(m_s <g_s> / n_s) = d0 sqrt(2 m_s <g_s>)   (relativistic inertia)
            //!     lambda_D,s = d0 sqrt(T_s / n_s)       = d0 sqrt(2 T_s)        (Boltzmann screening: no <g_s>)
            //!
            //! Collective values follow from 1/x^2 = sum_s 1/x_s^2. The grid criterion
            //! below is applied to the collective Debye length, because the finite-grid
            //! instability aliases the TOTAL charge density.
            const real_t d_1       = skindepth0 * (real_t)std::sqrt((double)(g1 * sp1_mass / N_PER_SP));
            const real_t d_2       = skindepth0 * (real_t)std::sqrt((double)(g2 * sp2_mass / N_PER_SP));
            const real_t d_plasma  = (real_t)(1.0 / std::sqrt(1.0 / SQR((double)d_1) + 1.0 / SQR((double)d_2)));
            //! Larmor radius uses the thermal FOUR-velocity sqrt(<g>^2 - 1), not <g>.
            const real_t thermal_four_velocity_1 = (real_t)std::sqrt(std::max(0.0, (double)(g1 * g1 - ONE)));
            const real_t thermal_four_velocity_2 = (real_t)std::sqrt(std::max(0.0, (double)(g2 * g2 - ONE)));
            const real_t rho_1     = larmor0 * thermal_four_velocity_1 * sp1_mass / B_ref;
            const real_t rho_2     = larmor0 * thermal_four_velocity_2 * sp2_mass / B_ref;
            const real_t lambda_D1 = skindepth0 * (real_t)std::sqrt((double)(temperature_1 / N_PER_SP));
            const real_t lambda_D2 = skindepth0 * (real_t)std::sqrt((double)(temperature_2 / N_PER_SP));
            const real_t lambda_Dpl = (real_t)(1.0 / std::sqrt(1.0 / SQR((double)lambda_D1) +
                                                               1.0 / SQR((double)lambda_D2)));

            //* gyroradius at gamma = sigma, the usual acceleration ceiling: must
            //* stay well inside the box or the spectrum is cut by confinement
            const real_t rho_sigma = larmor0 * sigma_cold / B_ref;

            //* shortest box side among the directions the spectrum spans
            real_t L_min = std::numeric_limits<real_t>::max();
            if (turb_kplane != 1) L_min = std::min(L_min, Lx);   //* x1 active unless k in x2-x3
            if (turb_kplane != 2) L_min = std::min(L_min, Ly);   //* x2 active unless k in x3-x1
            if ((D == Dim::_3D) && (turb_kplane != 0)) L_min = std::min(L_min, Lz);
            const real_t l_outer = L_min / (real_t)kmin;                   //* largest eddy [d0]
            const real_t l_inner = L_min / (real_t)kmax;                   //* smallest injected eddy
            const real_t B_perturbed_over_guide = (B_guide != ZERO) ? B_perturbed_rms / math::abs(B_guide) : (real_t)-1.0;

            //* outer-scale eddy turnover t = ell / u_perturbed_rms, ell = L/kmin (wavelength convention,
            //* = 2 pi / k_min), with the u_perturbed actually loaded (the Alfven target if perturb_velocity = false)
            const real_t u_eddy  = (u_perturbed_rms > ZERO) ? u_perturbed_rms : u_perturbed_rms_alfven;
            const real_t t_eddy  = (u_eddy > ZERO) ? l_outer / u_eddy : ZERO;

            const real_t runtime = params.template get<simtime_t>("simulation.runtime");
            const real_t t_w1    = runtime * (real_t)std::sqrt((double)(N_PER_SP / sp1_mass)); //* [1/omega_p,sp1], omega_p,s = omega_p0 sqrt(n_s / m_s)
            const real_t t_w2    = runtime * (real_t)std::sqrt((double)(N_PER_SP / sp2_mass)); //* [1/omega_p,sp2]
            const real_t t_cross = runtime / L_min;                                           //* light crossings
            const real_t n_eddy  = (t_eddy > (real_t)0.0) ? runtime / t_eddy : (real_t)0.0;

            //! Pressures in units of n0 m0 c^2, so ratios are normalisation independent:
            //!     P_s = n_s T_s,   U_B = sigma0 B^2 / 2,   plasma beta_s = P_s / U_B = 2 n_s T_s / (sigma0 B^2)
            const real_t P_th_1 = N_PER_SP * temperature_1;
            const real_t P_th_2 = N_PER_SP * temperature_2;
            auto magnetic_pressure = [&](real_t B) -> real_t { return sigma0 * B * B / (real_t)2.0; };
            const real_t U_B_rms      = magnetic_pressure(B_ref);
            const real_t beta_1_rms   = (U_B_rms > ZERO) ? P_th_1 / U_B_rms : (real_t)-1.0;
            const real_t beta_2_rms   = (U_B_rms > ZERO) ? P_th_2 / U_B_rms : (real_t)-1.0;
            const real_t U_B_guide    = magnetic_pressure(B_guide);
            const real_t beta_1_guide = (U_B_guide > ZERO) ? P_th_1 / U_B_guide : (real_t)-1.0;
            const real_t beta_2_guide = (U_B_guide > ZERO) ? P_th_2 / U_B_guide : (real_t)-1.0;
            const real_t turb_frac    = (B_rms > (real_t)0.0) ? (B_perturbed_rms * B_perturbed_rms) / (B_rms * B_rms) : (real_t)0.0;

            //! =================================================================
            //! SAMPLED DIAGNOSTICS, on the same host grid as the u_total cap
            //! (sampling_counts). Two forms of curl(B) are compared at every sample:
            //!
            //!   curl_B     = -sum (kc x b) sin(k.x + ph)   mode-table curl (the drifts carry its sigma_h-corrected form)
            //!   curl_B_yee = centred difference of B       what the Yee solver demands
            //!
            //! These should agree to round-off. A nonzero residual means the mode
            //! table and the k~ correction have fallen out of step.
            //! =================================================================
            auto B_host = [&](double X, double Y, double Z, double& B_x, double& B_y, double& B_z)
            {
                host_cos_sum(B_perturbed_modes_host, X, Y, Z, B_x, B_y, B_z);
                B_z += (double)B_guide;
            };

            auto u_perturbed_host = [&](double X, double Y, double Z, double& u_perturbed_x, double& u_perturbed_y, double& u_perturbed_z)
            {
                host_cos_sum(u_perturbed_modes_host, X, Y, Z, u_perturbed_x, u_perturbed_y, u_perturbed_z);
            };

            //* centred Yee difference over one cell, exactly as the FDTD curl does it
            auto curl_B_yee_host = [&](double X, double Y, double Z, double& curl_B_yee_x, double& curl_B_yee_y, double& curl_B_yee_z)
            {
                const double hx = (double)dx, hy = (double)dy, hz = (double)dz;
                double bxp, byp, bzp, bxm, bym, bzm;

                //* d/dx
                B_host(X + 0.5 * hx, Y, Z, bxp, byp, bzp);
                B_host(X - 0.5 * hx, Y, Z, bxm, bym, bzm);
                const double dBy_dx = (byp - bym) / hx;
                const double dBz_dx = (bzp - bzm) / hx;

                //* d/dy
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

            //* E = -(u_motional x B), u_motional = u_perturbed + u_motional_coeff curl(B), mirroring InitFields::motional_E exactly
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

            //* div(E) by the same one-cell centred differences the Yee divergence
            //* uses, summed only over the SIMULATED directions (in 2D d/dz = 0, so
            //* E_z carries no charge).
            auto divE_host = [&](double X, double Y, double Z) -> double
            {
                const double hx = (double)dx, hy = (double)dy, hz = (double)dz;
                double ep[3], em[3];

                E_host(X + 0.5 * hx, Y, Z, ep[0], ep[1], ep[2]);
                E_host(X - 0.5 * hx, Y, Z, em[0], em[1], em[2]);
                double div = (ep[0] - em[0]) / hx;

                E_host(X, Y + 0.5 * hy, Z, ep[0], ep[1], ep[2]);
                E_host(X, Y - 0.5 * hy, Z, em[0], em[1], em[2]);
                div += (ep[1] - em[1]) / hy;

                if constexpr (D == Dim::_3D)
                {
                    E_host(X, Y, Z + 0.5 * hz, ep[0], ep[1], ep[2]);
                    E_host(X, Y, Z - 0.5 * hz, em[0], em[1], em[2]);
                    div += (ep[2] - em[2]) / hz;
                }
                return div;
            };

            const std::array<int, 3> counts = sampling_counts(*this);

            long   n_samples = 0;
            double curl_B_max = 0.0, curl_B_yee_max = 0.0;
            double curl_B_deviation_max = 0.0;                          //* max |curl_B - curl_B_yee|
            double divB_max = 0.0;                                      //* max |discrete div(B)|
            double u_drift_sum_1 = 0.0, u_drift_sum_2 = 0.0;            //* |u_drift_species|
            double u_perturbed_sum = 0.0;                               //* |u_perturbed|
            double loaded_current_sq = 0.0, curl_B_sq_sum = 0.0;        //* |J_loaded|^2, |curl B|^2 (J_loaded = J_code/AMP)
            double current_correction_sq = 0.0;                         //* |J_first_order - curl B|^2 (sigma_h split)
            double convection_correction_sq = 0.0;                      //* |J_loaded - J_first_order|^2 (convection term)
            double peak_u_total_loaded = 0.0;                           //* max |u_perturbed + u_drift_species| with the drift as loaded
            double rho_analytic_sq = 0.0;                               //* (AMP div(E) / n)^2 from the analytic div(E) the convection term uses
            double perp_reduction_sum = 0.0, perp_reduction_max = 0.0;  //* f = sigma_local / (1 + sigma_local)

            //* current budget at t = 0:  required current A = AMP curl(B) = particle drift + convection + displacement, at every point [n0 e c]
            double required_sq = 0.0, drift_dot_required = 0.0, convection_dot_required = 0.0, displacement_dot_required = 0.0;   //* projections on A: these add up
            double particle_across_sq = 0.0, displacement_across_sq = 0.0;                                                         //* across A (the XY plane in 2D with turb_a_axis = 2)
            double displacement_sq = 0.0, displacement_along_B_sq = 0.0, displacement_theory_error_sq = 0.0;                       //* checks

            double E_max = 0.0;                                         //* max |E| = |u_motional x B|
            double E_ideal_sq = 0.0, E_hall_sq = 0.0;                   //* |u_perturbed x B|^2 and |(u_motional - u_perturbed) x B|^2
            double rho_sq = 0.0, rho_max = 0.0;                         //* rho/n_tot = AMP div(E) / n_tot
            double xi_sq = 0.0, xi_max = 0.0;                           //* |xi_s| = (AMP/n) |E|, one species

            if (mpi_rank == 0)
            for (int ix = 0; ix < counts[0]; ++ix)
                for (int iy = 0; iy < counts[1]; ++iy)
                    for (int iz = 0; iz < counts[2]; ++iz)
                    {
                        const double X = sample_coordinate((double)xmin, (double)Lx, ix, counts[0]);
                        const double Y = sample_coordinate((double)ymin, (double)Ly, iy, counts[1]);
                        const double Z = (D == Dim::_3D) ? sample_coordinate((double)zmin, (double)Lz, iz, counts[2]) : 0.0;

                        double curl_B_x, curl_B_y, curl_B_z;
                        host_curl_sum(B_perturbed_modes_host, X, Y, Z, curl_B_x, curl_B_y, curl_B_z);
                        const double curl_B_magnitude = std::sqrt(curl_B_x * curl_B_x + curl_B_y * curl_B_y + curl_B_z * curl_B_z);

                        double curl_B_yee_x, curl_B_yee_y, curl_B_yee_z;
                        curl_B_yee_host(X, Y, Z, curl_B_yee_x, curl_B_yee_y, curl_B_yee_z);
                        const double curl_B_yee_magnitude = std::sqrt(curl_B_yee_x * curl_B_yee_x + curl_B_yee_y * curl_B_yee_y + curl_B_yee_z * curl_B_yee_z);

                        const double curl_B_deviation = std::sqrt(SQR(curl_B_x - curl_B_yee_x) + SQR(curl_B_y - curl_B_yee_y) + SQR(curl_B_z - curl_B_yee_z));

                        ++n_samples;
                        curl_B_max     = std::max(curl_B_max, curl_B_magnitude);
                        curl_B_yee_max = std::max(curl_B_yee_max, curl_B_yee_magnitude);
                        curl_B_deviation_max = std::max(curl_B_deviation_max, curl_B_deviation);
                        divB_max = std::max(divB_max, std::fabs(divB_host(X, Y, Z)));

                        //* first-order current (sigma_h split; Hall part of E) and the drift current (adds the convection term)
                        double first_order_current_x, first_order_current_y, first_order_current_z;
                        host_first_order_current(B_perturbed_modes_host, (double)B_guide, (double)sigma_h_per_B_sq, X, Y, Z, first_order_current_x, first_order_current_y, first_order_current_z);
                        double loaded_current_x, loaded_current_y, loaded_current_z;
                        double div_E_analytic = 0.0;
                        host_loaded_current(B_perturbed_modes_host, u_perturbed_modes_host, (double)B_guide, (double)sigma_h_per_B_sq, (double)u_motional_coeff, X, Y, Z, loaded_current_x, loaded_current_y, loaded_current_z, &div_E_analytic);
                        rho_analytic_sq += SQR((double)AMP * div_E_analytic / (double)N_TOTAL);
                        const double loaded_current_magnitude = std::sqrt(SQR(loaded_current_x) + SQR(loaded_current_y) + SQR(loaded_current_z));
                        loaded_current_sq     += SQR(loaded_current_magnitude);
                        curl_B_sq_sum         += SQR(curl_B_magnitude);
                        current_correction_sq += SQR(first_order_current_x - curl_B_x) + SQR(first_order_current_y - curl_B_y) + SQR(first_order_current_z - curl_B_z);
                        convection_correction_sq += SQR(loaded_current_x - first_order_current_x) + SQR(loaded_current_y - first_order_current_y) + SQR(loaded_current_z - first_order_current_z);
                        {
                            double B_sample_x, B_sample_y, B_sample_z;
                            B_host(X, Y, Z, B_sample_x, B_sample_y, B_sample_z);
                            const double sigma_local = (double)sigma_h_per_B_sq * (SQR(B_sample_x) + SQR(B_sample_y) + SQR(B_sample_z));
                            const double f_local     = sigma_local / (1.0 + sigma_local);
                            perp_reduction_sum += f_local;
                            perp_reduction_max  = std::max(perp_reduction_max, f_local);
                        }

                        //* u_drift_species = u_drift_coeff_s J_loaded
                        u_drift_sum_1 += std::fabs((double)u_drift_coeff_1) * loaded_current_magnitude;
                        u_drift_sum_2 += std::fabs((double)u_drift_coeff_2) * loaded_current_magnitude;

                        double u_perturbed_x, u_perturbed_y, u_perturbed_z;
                        u_perturbed_host(X, Y, Z, u_perturbed_x, u_perturbed_y, u_perturbed_z);
                        u_perturbed_sum += std::sqrt(u_perturbed_x * u_perturbed_x + u_perturbed_y * u_perturbed_y + u_perturbed_z * u_perturbed_z);
                        for (const double u_drift_coeff_s : { (double)u_drift_coeff_1, (double)u_drift_coeff_2 })
                        {
                            peak_u_total_loaded = std::max(peak_u_total_loaded, std::sqrt(SQR(u_perturbed_x + u_drift_coeff_s * loaded_current_x) + SQR(u_perturbed_y + u_drift_coeff_s * loaded_current_y) + SQR(u_perturbed_z + u_drift_coeff_s * loaded_current_z)));
                        }

                        //* current budget:  A = AMP curl(B) = particle drift (AMP J_loaded) + convection (rho u_convection) + displacement (AMP dE/dt)
                        {
                            const double amp             = (double)AMP;
                            const double rho_sample      = amp * div_E_analytic;                                                 //* polarisation charge [n0 e]
                            const double u_convection[3] = { u_perturbed_x + 0.5 * (double)u_motional_coeff * first_order_current_x,   //* (u_perturbed + u_motional)/2
                                                             u_perturbed_y + 0.5 * (double)u_motional_coeff * first_order_current_y,
                                                             u_perturbed_z + 0.5 * (double)u_motional_coeff * first_order_current_z };
                            const double required[3]     = { amp * curl_B_x, amp * curl_B_y, amp * curl_B_z };
                            const double drift[3]        = { amp * loaded_current_x, amp * loaded_current_y, amp * loaded_current_z };
                            double convection[3], particle[3], displacement[3];
                            for (int i = 0; i < 3; ++i)
                            {
                                convection[i]   = rho_sample * u_convection[i];
                                particle[i]     = drift[i] + convection[i];
                                displacement[i] = required[i] - particle[i];
                            }
                            auto dot = [](const double (&a)[3], const double (&b)[3]) -> double { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };

                            //* projections on A add up, because A = drift + convection + displacement at every point
                            const double required_magnitude_sq = dot(required, required);
                            required_sq               += required_magnitude_sq;
                            drift_dot_required        += dot(drift, required);
                            convection_dot_required   += dot(convection, required);
                            displacement_dot_required += dot(displacement, required);

                            //* parts across A: A has none there, so particles and displacement current are equal and opposite
                            if (required_magnitude_sq > 0.0)
                            {
                                const double particle_along     = dot(particle, required) / required_magnitude_sq;
                                const double displacement_along = dot(displacement, required) / required_magnitude_sq;
                                for (int i = 0; i < 3; ++i)
                                {
                                    particle_across_sq     += SQR(particle[i] - particle_along * required[i]);
                                    displacement_across_sq += SQR(displacement[i] - displacement_along * required[i]);
                                }
                            }

                            //* checks: no displacement current along B; displacement current = f (A - rho u_convection) across B
                            double B_sample[3];
                            B_host(X, Y, Z, B_sample[0], B_sample[1], B_sample[2]);
                            const double B_sq = dot(B_sample, B_sample);
                            displacement_sq += dot(displacement, displacement);
                            if (B_sq > 0.0)
                            {
                                const double sigma_local = (double)sigma_h_per_B_sq * B_sq;
                                const double f_local     = sigma_local / (1.0 + sigma_local);
                                displacement_along_B_sq += SQR(dot(displacement, B_sample)) / B_sq;
                                double source[3];
                                for (int i = 0; i < 3; ++i) { source[i] = required[i] - convection[i]; }
                                const double source_along_B_over_B = dot(source, B_sample) / B_sq;
                                for (int i = 0; i < 3; ++i)
                                {
                                    const double displacement_theory = f_local * (source[i] - source_along_B_over_B * B_sample[i]);
                                    displacement_theory_error_sq += SQR(displacement[i] - displacement_theory);
                                }
                            }
                        }

                        //! ---- motional field and the charge density it demands ----
                        //! E = -(u_motional x B) is what InitFields::ex1/ex2/ex3 writes, split
                        //! here into its ideal part -(u_perturbed x B) and its Hall part
                        //! -(u_motional_coeff curl(B) x B). rho is sampled as AMP * div(E) with
                        //! the SAME one-cell centred differences the Yee divergence uses, so
                        //! this is the discrete constraint the code will actually see.
                        double B_x, B_y, B_z;
                        B_host(X, Y, Z, B_x, B_y, B_z);
                        const double E_ideal_x = -(u_perturbed_y * B_z - u_perturbed_z * B_y);
                        const double E_ideal_y = -(u_perturbed_z * B_x - u_perturbed_x * B_z);
                        const double E_ideal_z = -(u_perturbed_x * B_y - u_perturbed_y * B_x);
                        const double hall_x    = (double)u_motional_coeff * first_order_current_x, hall_y = (double)u_motional_coeff * first_order_current_y, hall_z = (double)u_motional_coeff * first_order_current_z;
                        const double E_hall_x  = -(hall_y * B_z - hall_z * B_y);
                        const double E_hall_y  = -(hall_z * B_x - hall_x * B_z);
                        const double E_hall_z  = -(hall_x * B_y - hall_y * B_x);
                        E_ideal_sq += SQR(E_ideal_x) + SQR(E_ideal_y) + SQR(E_ideal_z);
                        E_hall_sq  += SQR(E_hall_x) + SQR(E_hall_y) + SQR(E_hall_z);

                        const double E_magnitude = std::sqrt(SQR(E_ideal_x + E_hall_x) + SQR(E_ideal_y + E_hall_y) + SQR(E_ideal_z + E_hall_z));
                        E_max = std::max(E_max, E_magnitude);

                        const double divE = divE_host(X, Y, Z);
                        const double rho  = (double)AMP * divE / (double)N_TOTAL;
                        rho_sq += rho * rho;
                        rho_max = std::max(rho_max, std::fabs(rho));

                        //* per-species displacement applied in load_charge_density
                        const double xi = (double)AMP / (double)N_TOTAL * E_magnitude;
                        xi_sq += xi * xi;
                        xi_max = std::max(xi_max, xi);
                    }

            const double inv_n              = (n_samples > 0) ? 1.0 / (double)n_samples : 0.0;
            const double u_drift_mean_1     = u_drift_sum_1   * inv_n;
            const double u_drift_mean_2     = u_drift_sum_2   * inv_n;
            const double u_perturbed_mean   = u_perturbed_sum * inv_n;
            const double rho_rms            = std::sqrt(rho_sq * inv_n);
            const double rho_analytic_rms   = std::sqrt(rho_analytic_sq * inv_n);
            const double xi_rms             = std::sqrt(xi_sq  * inv_n);
            const double dx_d               = (double)dx;
            const double xi_cells           = (dx_d > 0.0) ? xi_rms / dx_d : 0.0;
            const double ampere_deviation   = (curl_B_yee_max > 0.0) ? curl_B_deviation_max / curl_B_yee_max : 0.0;
            const double curl_B_ratio_max   = (curl_B_yee_max > 0.0) ? curl_B_max / curl_B_yee_max : 0.0;
            const double divB_over_curlB    = (curl_B_yee_max > 0.0) ? divB_max / curl_B_yee_max : 0.0;
            const bool   E_ideal_present    = (E_ideal_sq > 0.0);
            const double E_hall_over_ideal  = E_ideal_present ? std::sqrt(E_hall_sq / E_ideal_sq) : 0.0;
            const double loaded_over_curl   = (curl_B_sq_sum > 0.0) ? std::sqrt(loaded_current_sq / curl_B_sq_sum) : 0.0;        //* RMS |J_loaded| / RMS |curl B|
            const double correction_size    = (curl_B_sq_sum > 0.0) ? std::sqrt(current_correction_sq / curl_B_sq_sum) : 0.0;    //* RMS |J_first_order - curl B| / RMS |curl B|
            const double convection_size    = (curl_B_sq_sum > 0.0) ? std::sqrt(convection_correction_sq / curl_B_sq_sum) : 0.0; //* RMS |J_loaded - J_first_order| / RMS |curl B|
            const double perp_reduction_mean = perp_reduction_sum * inv_n;
            //* values along A [n0 e c]: <X . A> / A_rms, so particle drift + convection + displacement = A_rms exactly
            const double required_rms             = std::sqrt(required_sq * inv_n);
            const double drift_along              = (required_rms > 0.0) ? drift_dot_required * inv_n / required_rms : 0.0;
            const double convection_along         = (required_rms > 0.0) ? convection_dot_required * inv_n / required_rms : 0.0;
            const double displacement_along       = (required_rms > 0.0) ? displacement_dot_required * inv_n / required_rms : 0.0;
            const double particle_across_rms      = std::sqrt(particle_across_sq * inv_n);
            const double displacement_across_rms  = std::sqrt(displacement_across_sq * inv_n);
            const double displacement_rms         = std::sqrt(displacement_sq * inv_n);
            const double displacement_along_B     = (required_rms > 0.0) ? std::sqrt(displacement_along_B_sq * inv_n) / required_rms : 0.0;
            const double displacement_theory_rel  = (displacement_rms > 0.0) ? std::sqrt(displacement_theory_error_sq * inv_n) / displacement_rms : 0.0;

            //! =================================================================
            //! UNLOADED DISPLACEMENT CURRENT AT t = 0 (estimate)
            //!
            //! The drifts carry curl(B) minus the displacement current of the J x B acceleration (sigma_h
            //! split). The rest of dE/dt at t = 0 is not loaded. With E = -(u_m x B), u_m the field-line
            //! velocity, it is the change of E due to advection of the flow and of the field,
            //!     D = [(u_m . grad) u_m] x B - u_m x dB/dt,        dB/dt = -curl(E).
            //! Its part across B is the advective dE_perp/dt; its part along B equals E.curl(E)/|B|, the
            //! parallel displacement current that keeps E.B = 0. Relative to the required current
            //! AMP curl(B), each is |D|/|curl B| (AMP cancels), as box RMS values. Derivatives are centred
            //! differences of the host mode sums, on at most 64 points per axis (32 in 3D). As a check, dB/dt is also
            //! computed from the product rule, curl(u_m x B) = u_m div(B) - B div(u_m) + (B.grad)u_m - (u_m.grad)B.
            //! =================================================================
            auto u_motional_host = [&](double X, double Y, double Z, double (&u_motional)[3])
            {
                double u_perturbed_value[3], current[3];
                u_perturbed_host(X, Y, Z, u_perturbed_value[0], u_perturbed_value[1], u_perturbed_value[2]);
                host_first_order_current(B_perturbed_modes_host, (double)B_guide, (double)sigma_h_per_B_sq, X, Y, Z, current[0], current[1], current[2]);
                for (int i = 0; i < 3; ++i) { u_motional[i] = u_perturbed_value[i] + (double)u_motional_coeff * current[i]; }
            };

            const bool   E_present    = E_ideal_present || (u_motional_coeff != ZERO);
            const double fd_step      = 1.0e-3 * (double)l_inner;          //* well below the shortest injected wavelength
            const int    n_axes       = (D == Dim::_3D) ? 3 : 2;            //* simulated directions; d/dz = 0 in 2D
            double unloaded_perp_sq = 0.0, unloaded_par_sq = 0.0, unloaded_curl_B_sq = 0.0;
            double dB_dt_sq = 0.0, dB_dt_mismatch_sq = 0.0;

            if (mpi_rank == 0 && E_present)
            {
                std::array<int, 3> estimate_counts;
                const int max_points_per_axis = (D == Dim::_3D) ? 32 : 64;   //* the RMS needs few samples; keeps the 3D setup cheap
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

            //* x as a percentage of the RMS required current, one decimal (own stream, so the printout's number format is untouched)
            auto percent_of_required = [&](double x) -> std::string
            {
                if (required_rms <= 0.0) { return "n/a"; }
                std::ostringstream percent_text;
                percent_text << std::fixed << std::setprecision(1) << 100.0 * x / required_rms << "%";
                return percent_text.str();
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

            //! Normalised cross-helicity sigma_c = <u_perturbed . B_perturbed> / (u_perturbed_rms B_perturbed_rms).
            //! Distinct modes are orthogonal over the box, so <u_perturbed . B_perturbed> = 0.5 sum_m (u_m . b_m)
            //! cos(psi_m - phi_m) for tables that list the same k in the same order
            //! (same band, kplane and a_axis), which is checked mode by mode.
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

            //* Empirical estimates for MOMENTUM-conserving PIC cluster around lambda_D / dx = 1/pi ~ 0.32
            constexpr real_t DEBYE_FLOOR = static_cast<real_t>(1.0) / static_cast<real_t>(3.0);
            //* applied to the COLLECTIVE Debye length, not a per-species one
            const real_t     lD_ratio    = lambda_Dpl / dx;

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
                                        ? std::string("A = A_") + axis_name[turb_a_axis] + " e_" +
                                          axis_name[turb_a_axis] + "  (B_perturbed _|_ " + axis_name[turb_a_axis] + ")"
                                        : std::string("random, A _|_ k");

            //* curl(B) lies along Z in 2D (k in the XY plane, vector potential along Z); in 3D it can point in any direction
            const std::string budget_along  = (D == Dim::_2D) ? "along Z" : "parallel to curl(B)";
            const std::string budget_across = (D == Dim::_2D) ? "in the XY plane" : "perpendicular to curl(B)";
            const std::string across_note   = (D == Dim::_2D) ? "; along the tilted B-field lines" : "";

            std::ostringstream sampling_grid_text;
            sampling_grid_text << counts[0] << " x " << counts[1];
            if constexpr (D == Dim::_3D) { sampling_grid_text << " x " << counts[2]; }

            std::ostringstream oss;
            oss << "\n"
                << "===========================================================\n"
                << title << "\n"
                << "===========================================================\n\n"
                << "GEOMETRY (Periodic Boundaries) \n"
                << kv1("Dimensionality") << (int)D << "D\n";
            if constexpr (M::Dim == Dim::_3D)
            {
                const auto nx3 = metadomain.mesh().n_active(in::x3);
                oss << kv1("Resolution") << nx1 << " x " << nx2 << " x " << nx3 << "\n"
                    << kv1("Box (Lx x Ly x Lz) [d0]") << Lx << " x " << Ly << " x " << Lz << "\n";
            }
            else
            {
                oss << kv1("Resolution") << nx1 << " x " << nx2 << "\n"
                    << kv1("Box (Lx x Ly) [d0]") << Lx << " x " << Ly << "\n";
            }
            oss << kv1("Grid size [d0]") << dx << "\n"
                << kv1("Runtime [cold electron freq.]") << t_w1 << "\n"
                << kv1("Runtime [cold " + sp2 + " freq.]") << t_w2 << "\n"
                << kv1("Runtime [L/c]") << t_cross << "\n"
                << kv1("Runtime [outer eddy times]") << n_eddy << "\n\n"


                << "MASS, MAGNETIC FIELD, AND MAGNETISATION (at B = B_rms unless stated) \n"
                << kv1("B_rms [B0]") << B_rms << "\n"
                << kv1("Guide field [B0]") << B_guide << "  (along Z)\n"
                << kv1("B_perturbed RMS (target) [B0]") << B_perturbed_rms << "\n"
                << kv1("B_perturbed RMS (X, Y, Z) [B0]") << "(" << B_perturbed_rms_component[0] << ", " 
                                                                << B_perturbed_rms_component[1] << ", " 
                                                                << B_perturbed_rms_component[2] << ")\n"
                << kv1("B_perturbed_rms / |B_guide|") << B_perturbed_over_guide << "  (-1 => zero guide field)\n"
                << kv1("Mass ratio [m_2/m_1]") << mass_ratio << "\n"
                << kv1("sigma0 = (d0/rho0)^2") << sigma0 << "  (fiducial species at B = 1)\n"
                << kv1("Sigma (electron, cold)") << sigma_1_cold << "\n"
                << kv1("Sigma (" + sp2 + ", cold)") << sigma_2_cold << "\n"
                << kv1("Sigma (plasma, cold)") << sigma_cold << "\n"
                << kv1("Sigma (plasma, hot)") << sigma_hot << "\n"
                << kv1("Enthalpy w [n0 m0 c^2]") << w_enth << "  (= sum_s n_s (m_s <g_s> + T_s))\n"
                << kv1("Alfven speed [c] (cold)") << vA_cold << "\n"
                << kv1("Alfven speed [c] (hot)") << vA_hot << "\n"
                << kv1("Mean Lorentz factor electrons") << g1 << "\n"
                << kv1("Mean Lorentz factor " + sp2s) << g2 << "\n"
                << "\tPlasma beta (at B_rms)\n"
                << kv2("Electrons") << beta_1_rms << "\n"
                << kv2(SP2u) << beta_2_rms << "\n"
                << kv2("Plasma") << beta_1_rms + beta_2_rms << "\n";
            if (B_guide != ZERO)
            {
                oss << "\tPlasma beta (at B_guide)\n"
                    << kv2("Electrons") << beta_1_guide << "\n"
                    << kv2(SP2u) << beta_2_guide << "\n"
                    << kv2("Plasma") << beta_1_guide + beta_2_guide << "\n";
            }
            else
            {
                oss << kv1("Plasma beta (at B_guide)") << "n/a (B_guide = 0)\n";
            }
            oss << kv1("B_perturbed^2 / B_rms^2 (turb frac)") << turb_frac << "\n"
                << kv1("B_perturbed^2 / B_guide^2") << ((B_guide != ZERO) ? (B_perturbed_rms * B_perturbed_rms) / (B_guide * B_guide) : (real_t)-1.0) << "\n\n"


                << "TURBULENCE SPECTRUM \n"
                << kv1("Mode band [kmin, kmax]") << "[" << kmin << ", " << kmax << "]\n"
                << kv1("Number of modes") << B_perturbed_modes_host.size() << "\n"
                << kv1("Spectral slope (E(k) ~ k^slope)") << spectral_slope << "  (0 = flat; equal energy per mode at " << kspace_dim(turb_kplane, D == Dim::_3D) - 1 << ")\n"
                << kv1("Random mode amplitudes") << (turb_rand_amp ? "YES" : "NO") << "\n"
                << kv1("k-space") << kplane_name << "\n"
                << kv1("Vector potential") << apol_name << "\n"
                << kv1("Outer scale L/kmin [d0]") << l_outer << "  (" << l_outer / d_plasma << " plasma skin depths)\n"
                << kv1("Inner scale L/kmax [d0]") << l_inner << "  (resolved with " << l_inner / dx << " cells)\n"
                << kv1("Eddy time L/(kmin u_perturbed_rms)") << t_eddy << "  [d0/c]" << ((u_perturbed_rms > ZERO) ? "" : "  (Alfven target; no flow loaded)") << "\n\n"


                << "CURRENT \n"  
                << "\tJ_code = AMP [curl(B) - f (curl B)_perp] = sum_s n_s q_s u_drift_species\n"
                << "\tf = sigma_h/(1 + sigma_h), with the local sigma_h\n"
                << kv1("AMP [d0^2/r0]") << AMP << "\n"
                << kv1("Displacement share of J_perp") << "mean " << perp_reduction_mean << ", max " << perp_reduction_max << "  (f: part of the current across B carried by dE/dt instead of particles)\n"
                << "\tCurrent Distribution " << budget_along << "  [n0 e c]\n"
                << kv2("Total = AMP curl(B)") << required_rms << "  (100%)\n"
                << kv2("Particle drift") << drift_along << "  (" << percent_of_required(drift_along) << ")\n"
                << kv2("Convection (charge polarisation)") << convection_along << "  (" << percent_of_required(convection_along) << ")\n"
                << kv2("Displacement current (dE/dt)") << displacement_along << "  (" << percent_of_required(displacement_along) << ")\n"
                << "\tCurrent Distribution " << budget_across << "  [n0 e c, RMS]\n"
                << kv2("Particle drift + convection") << particle_across_rms << "  (" << percent_of_required(particle_across_rms) << across_note << ")\n"
                << kv2("Displacement current (dE/dt)") << displacement_across_rms << "  (" << percent_of_required(displacement_across_rms) << ")\n"
                << kv1("Enthalpy/particle (electrons, " + sp2s + ")") << "(" << enthalpy_per_particle_1 << ", " << enthalpy_per_particle_2 << ")  [m0 c^2]\n"
                << kv1("Share of J (electrons, " + sp2s + ")") << "(" << current_share_1 << ", " << current_share_2 << ")  (zero net momentum)\n"
                << kv1("u_drift_coeff (electrons, " + sp2s + ")") << "(" << u_drift_coeff_1 << ", " << u_drift_coeff_2 << ")\n"
                << kv1("max |u_drift| [c] (electrons, " + sp2s + ")") << "(" << peak_u_drift_1 << ", " << peak_u_drift_2 << ")\n"
                << kv1("mean |u_drift| [c] (electrons, " + sp2s + ")") << "(" << u_drift_mean_1 << ", " << u_drift_mean_2 << ")\n\n"


                << "VELOCITY PERTURBATION\n"
                << kv1("Enabled") << (perturb_velocity ? "YES" : "NO  (u_perturbed = 0 everywhere)") << "\n";
            if (perturb_velocity)
            {
                oss << kv1("Phase seed (must differ from B's)") << v_seed << "  (B_perturbed uses " << B_seed << ")\n"
                    << kv1("Number of modes") << u_perturbed_modes_host.size() << "\n"
                    << kv1("u_total cap [c]") << u_total_cap << "\n"
                    << kv1("u_perturbed_rms [c] target") << u_perturbed_rms_alfven << "  (= vA_hot * B_perturbed_rms / B_rms)\n"
                    << kv1("u_perturbed_rms [c] loaded") << u_perturbed_rms << "\n"
                    << kv2("max |u_perturbed| [c]") << peak_u_perturbed << "\n"
                    << kv2("mean |u_perturbed| [c]") << u_perturbed_mean << "\n"
                    << kv1("u_perturbed RMS (X, Y, Z) [c]") << "(" << u_perturbed_rms_component[0] << ", " << u_perturbed_rms_component[1] << ", " << u_perturbed_rms_component[2] << ")\n"
                    << kv1("Cap factor s") << u_perturbed_cap_factor << ((u_perturbed_cap_factor < ONE) ? "  (u_perturbed REDUCED below the Alfven target)" : "  (Alfven target loaded)") << "\n"
                    << kv1("Turbulent kinetic/magnetic energy") << (double)(u_perturbed_cap_factor * u_perturbed_cap_factor) << "  (= s^2, Alfven units; 1 = equipartition)\n";
                if (sigma_c_valid) { oss << kv1("Cross-helicity, sigma_c") << sigma_c << "  (0 on average over seeds)\n"; }
                else               { oss << kv1("Cross-helicity, sigma_c") << "n/a\n"; }
            }


            oss << "\n"
                << "MOTIONAL ELECTRIC FIELD AND GAUSS CONSTRAINT \n"
                << kv1("Initial E") << "-u_motional x B,  u_motional = u_perturbed + u_drift_1 + u_drift_2\n"
                << kv1("u_motional_coeff") << u_motional_coeff << "  (0 for a pair plasma at equal T)\n";
            if (E_ideal_present) { oss << kv1("Hall / ideal part of E (RMS)") << E_hall_over_ideal << "\n"; }
            else                 { oss << kv1("Hall / ideal part of E (RMS)") << "n/a (u_perturbed = 0)\n"; }
            oss << kv1("rho/n_tot RMS (= AMP divE / n)") << rho_rms << "\n"
                << kv1("rho/n_tot RMS, analytic div(E)") << rho_analytic_rms << "  (used by the convection term; should match the line above)\n"
                << kv1("Species displacement RMS [d0]") << xi_rms << "  (" << xi_cells << " cells)\n"
                << kv1("Species displacement max [d0]") << xi_max << "  (" << ((dx_d > 0.0) ? xi_max / dx_d : 0.0) << " cells)\n"
                << kv1("Residual Gauss error (estimate)") << rho_rms * rho_rms << " of n_tot" 
                << ((rho_rms > 0.0) ? "  (~ " + std::to_string(rho_rms * 100.0) + "% of rho; measured in the GAUSS-LAW CHECK)" : "  (n/a: rho = 0)") << "\n\n";


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


            oss << "CHARACTERISTIC SCALES\n"
                << "\tPlasma (as a whole)\n"
                << kv2("Skin depth") << d_plasma << "  (resolved with " << d_plasma / dx << " cells)\n"
                << kv2("Debye length") << lambda_Dpl << "  (resolved with " << lambda_Dpl / dx << " cells; "
                    << ((lambda_Dpl / dx >= DEBYE_FLOOR) ? "greater" : "LESS") << " than the required " << DEBYE_FLOOR << " dx = " << DEBYE_FLOOR * dx << ")\n"

                << "\tElectrons\n"
                << kv2("Skin depth") << d_1 << "  (resolved with " << d_1 / dx << " cells)\n"
                << kv2("Larmor radius") << rho_1 << "  (resolved with " << rho_1 / dx << " cells)\n"
                << kv2("Debye length") << lambda_D1 << "  (resolved with " << lambda_D1 / dx << " cells)\n"

                << "\t" << SP2u << "\n"
                << kv2("Skin depth") << d_2 << "  (resolved with " << d_2 / dx << " cells)\n"
                << kv2("Larmor radius") << rho_2 << "  (resolved with " << rho_2 / dx << " cells)\n"
                << kv2("Debye length") << lambda_D2 << "  (resolved with " << lambda_D2 / dx << " cells)\n"
                << "\tConfinement\n"
                << kv2("Larmor radius at Lorentz factor = sigma") << rho_sigma << "  (" << rho_sigma / L_min << " x box)\n\n"


                << "TEMPERATURES\n"
                << kv1("T_1 = kT_1/(m0 c^2) of electrons") << temperature_1 << "  (setup.temperature)\n"
                << kv1("T_2 = kT_2/(m0 c^2) of " + sp2s) << temperature_2 << "  (temperature_ratio = " << temperature_ratio << ")\n"
                << kv1("theta_1 = kT_1/(m_1 c^2) of electrons") << theta_1 << "\n"
                << kv1("theta_2 = kT_2/(m_2 c^2) of " + sp2s) << theta_2 << "\n\n"


                << "CONSISTENCY CHECKS\n"
                << kv1("max |u_total_species| (must be <= cap)") << peak_u_total_both << "\n"
                << kv1("max|curl B| mode sum / Yee (must be 1)") << curl_B_ratio_max << "\n"
                << kv1("max|curl B - curl B Yee| / max (->0)") << ampere_deviation << "\n"
                << kv1("max|div B| / max|curl B|, Yee (->0)") << divB_over_curlB << "\n"
                << kv1("Displacement current along B / required") << displacement_along_B << "  (->0: E.B stays 0)\n"
                << kv1("Displacement current vs f (A - rho u)_perp") << displacement_theory_rel << "  (->0, relative)\n"
                << kv1("dB/dt: curl(E) vs product rule") << (E_present ? std::to_string(dB_dt_check) : std::string("n/a (E = 0)")) << "  (->0, relative; checks the unloaded estimate)\n"
                << kv1("lambda_D / dx (must be >~ 1/3)") << lD_ratio << "\n\n";

            if (lD_ratio < DEBYE_FLOOR)
                oss << "!! WARNING: lambda_D / dx = " << lD_ratio << " is BELOW the finite-grid\n"
                    << "!! instability floor of " << DEBYE_FLOOR << ". Expect numerical heating;\n"
                    << "!! refine the grid, raise the temperature, or raise ppc0.\n\n";
            else if (lD_ratio < ONE)
                oss << "!  NOTE: lambda_D is under one cell (" << lD_ratio << "), so Debye-scale\n"
                    << "!  structure (shielding, Langmuir waves at k lambda_D ~ 1) is NOT resolved.\n"
                    << "!  The grid-instability floor of " << DEBYE_FLOOR << " is still met, so heating\n"
                    << "!  should be bounded rather than explosive -- confirm it on a short run.\n\n";

            if (peak_u_total_loaded > (double)u_total_cap)
                oss << "!  NOTE: with the convection term the peak |u_total| is " << peak_u_total_loaded << " c, above\n"
                    << "!  setup.u_total_cap = " << u_total_cap << " (the cap uses the first-order current). The 0.99c clamp\n"
                    << "!  still holds; lower u_total_cap slightly if this matters.\n\n";

            if (u_perturbed_cap_factor < ONE)
                oss << "!  NOTE: u_perturbed was scaled by s = " << u_perturbed_cap_factor << " to keep\n"
                    << "!  |u_total_species| <= " << u_total_cap << " c for both species. The run starts with\n"
                    << "!  kinetic/magnetic fluctuation energy s^2 = " << u_perturbed_cap_factor * u_perturbed_cap_factor << " of the Alfven value.\n\n";

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

            //! ---- HARD FAILURE: Ampere invariant ---------------------------------
            //! The loaded J must BE the curl of the field that is actually written to
            //! the grid. Both come from the same mode table, so this mainly catches a
            //! sign or index error in the (kc x b) cross product, plus the
            //! discretisation error. Checked on rank 0 only: raise::Error calls
            //! MPI_Abort(MPI_COMM_WORLD), which stops every rank.
            if (mpi_rank == 0)
            {
                raise::ErrorIf(ampere_deviation > 1.0e-2,
                               "Ampere mismatch: curl(B) of the mode table differs from the Yee curl of the initialised field.\n"
                               "Check InitFields::curl_B() against InitFields::bx1/bx2/bx3.", HERE);
            }
        }

        //! =====================================================================
        //!  INITIALISE PARTICLES
        //!
        //!  Maxwell-Juttner at theta_s, drifting at the lab-frame velocity
        //!  u_total_species = u_perturbed + u_drift_species (TurbCurrentDist).
        //!  After the charge-loading displacement, each velocity is drawn again
        //!  at the particle's final position (load_charge_density).
        //!
        //! =====================================================================
        void InitPrtls(Domain<S, M>& local_domain)
        {
            metadomain.CommunicateFields(local_domain, Comm::E | Comm::B);

            const auto edist_1 = TurbCurrentDist<M>(local_domain.random_pool(), theta_1, u_drift_coeff_1, init_flds, turb_vel);
            const auto edist_2 = TurbCurrentDist<M>(local_domain.random_pool(), theta_2, u_drift_coeff_2, init_flds, turb_vel);
            const auto sdist   = UniformDist<M::Dim>();

            //? Total injected density = N_TOTAL = 1, split evenly (n_s = N_PER_SP)
            arch::InjectNonUniform<S, M, decltype(edist_1), decltype(edist_2), decltype(sdist)>(params, local_domain, { 1, 2 }, { edist_1, edist_2 }, sdist, N_TOTAL);
            
            //? Account for charge density created by the motional E field, then redraw velocities at the displaced positions
            load_charge_density(local_domain, edist_1, edist_2);

            //? Gauss-law check of the loaded state: deposited charge vs AMP div(E)
            check_gauss_law(local_domain);
        }

        //! =====================================================================
        //!  GAUSS-LAW CHECK (t = 0): deposited charge vs AMP div(E) of the loaded E field
        //*  Charge: Entity's charge moment without smoothing (each particle in its own cell), in units of n0, at cell centres.
        //*  div(E): Entity's Yee divergence at the nodes, averaged over each cell's corners to its centre.
        //*  The two differ at O((k dx)^2) and the deposited charge carries particle noise, so global statistics are compared:
        //*  slope -> 1 when the loading is right; a low correlation only means the charge is noise-dominated (small rho, low ppc0).
        //! =====================================================================
        void check_gauss_law(Domain<S, M>& local_domain)
        {
            int mpi_rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

            auto buffer = local_domain.fields.buff;   //* scratch: [0] deposited charge, [1] div(E) at nodes

            //* deposited charge density [n0], smoothing order 0
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
        //!  WHY IS A CHARGE DENSITY NEEDED AT ALL? 
        //*     With E = -(u_motional x B),
        //*     div(E) = -B.(curl u_motional) + u_motional.(curl B),
        //*  which is nonzero: e.g. a divergence-free (solenoidal) in-plane 
        //*  u_perturbed = grad(psi) x z in a guide field gives the pure-gradient 
        //*  E_perp = +B_z grad(psi), which demands a matching charge density. 
        //*  This is the polarisation charge that a relativistic Alfvenic fluctuation
        //*  carries. Its ideal part scales as rho/n ~ (k d0) sqrt(sigma0) u_perturbed B,
        //*  so it vanishes in the MHD limit (k d0 -> 0) and the non-relativistic 
        //*  limit (sigma0 -> 0).
        //*
        //!  WHY SHOULD THE INITIAL PARTICLES ALREADY CARRY THIS CHARGE?
        //*  entity evolves only Faraday's and Ampere's equations in time.
        //*  div(E) is a DIAGNOSTIC and there is no Poisson cleaner. The deposition
        //*  is exactly charge conserving (zigzag at O = 0, fused Esirkepov at
        //*  O >= 1, both built from i1_prev/dx1_prev, i.e. charge displacement),
        //*  so div(E) - 4 pi rho is preserved EXACTLY for all time.
        //*  An initial violation is therefore frozen in, never relaxed.
        //*
        //!  HOW IS THE CHARGE DENSITY CREATED HERE?
        //*  The injector uses a single density profile for both species and places 
        //*  them in pairs at the same positions, so n_1 = n_2 everywhere and the injected 
        //*  plasma is exactly neutral. The charge density has been created by displacing 
        //*  the particles: after injection, each species is shifted by a small distance xi_s, 
        //*  in opposite directions for the two species. For an arbitrary rho this would
        //*  require inverting div(xi) = -rho / (n q), but here rho is already the divergence 
        //*  of a known field,
        //*      rho = AMP div(E) = -AMP div(u_motional x B),  AMP = d0 sqrt(sigma0),
        //*  so a displacement along E is sufficient. Displacing species s by xi_s changes 
        //*  its density by dn_s = -n_s div(xi_s); the displacement has been chosen as 
        //*  xi_s = -sign(q_s) a E, with n_s = n/2 and |q_s| = 1:
        //*      rho = sum_s q_s dn_s = sum_s q_s (n/2) sign(q_s) a div(E) = n a div(E) 
        //*  With a = AMP / n, rho is therefore reproduced to first order in xi (see ACCURACY), 
        //*  without the need for a solver, FFT or grid inversion.
        //*  Negative charges are shifted along +E and positive charges along -E.
        //*
        //!  ACCURACY: 
        //*  dn_s = -n_s div(xi_s) is the linear density response; the true
        //*  Jacobian is 1/(1 + div xi), so the leftover is O((div xi)^2).
        //*
        //!  REMAINING SECOND-ORDER TERMS: 
        //*  Each velocity is redrawn at the particle's displaced position, so the drift
        //*  matches where the particle sits. The displacement current of the J x B
        //*  acceleration (sigma_h split) and the excess convection current of the displaced
        //*  charge are removed from the drifts (InitFields::loaded_current). What remains is
        //*  the nonlinear part of dE/dt: its parallel part (keeping E.B = 0) and its advective
        //*  perpendicular part (see LIMITATIONS). Negligible at vA << c.
        //*
        //!  NOTE ON 2D: d/dz = 0, so E_z contributes nothing to div(E) and a 
        //!  z-displacement would change no charge.
        //!
        //! =====================================================================
        //!  BOUNDARY BOOKKEEPING
        //*
        //*  Moving a particle is NOT just rewriting i1/dx1. The displacement is a
        //*  few cells, so particles near a subdomain edge leave it, and Entity
        //*  requires two further steps that normally live in the PUSHER
        //*  (kernels/pushers/sr.hpp: boundaryConditions + mpi::SendTag):
        //*
        //*   1. PERIODIC WRAP. If a face of THIS subdomain is PrtlBC::PERIODIC --
        //*      which means the subdomain spans the box in that direction, so no
        //*      neighbour owns the cell -- then i -= ni / i += ni.
        //*   2. SEND TAG. Otherwise an MPI neighbour owns it, and the particle must
        //*      be tagged via mpi::SendTag. CommunicateParticles is TAG DRIVEN
        //*      (kernels/comm.hpp PrepareOutgoingPrtls_kernel acts only on
        //*      tag != alive), so an untagged out-of-range particle is never
        //*      migrated: its i1 stays outside [0, ni1) and the first sort or
        //*      deposit indexes out of bounds. On GPU that surfaces as
        //*      "Memory access fault ... Reason: Unknown".
        //*
        //*  CommunicateParticles moves a tagged particle to the ADJACENT subdomain
        //*  only, shifting its cell index by that subdomain's size. A displacement
        //*  longer than a whole neighbouring subdomain would land out of range there,
        //*  so the sampled peak |xi| is checked against the local subdomain size
        //*  before anything moves (with a 2x margin for sampling and for neighbours
        //*  one cell smaller).
        //*
        //*  A final in-range guard undoes the displacement for any particle that is
        //*  somehow still out of range and untagged. It should never fire; it is
        //*  there so that a mistake degrades the charge loading locally instead of
        //*  corrupting memory.
        //*
        //! =====================================================================
        void load_charge_density(Domain<S, M>& local_domain, const TurbCurrentDist<M>& edist_1, const TurbCurrentDist<M>& edist_2)
        {
            //* local copies: a device lambda must not capture `this`
            const auto flds   = init_flds;                 //* carries u_perturbed, B and curl(B)
            const auto metric = local_domain.mesh.metric;

            //* local active cell counts, for the wrap and the range test
            const int ni1 = static_cast<int>(local_domain.mesh.n_active(in::x1));
            const int ni2 = static_cast<int>(local_domain.mesh.n_active(in::x2));
            const int ni3 = (D == Dim::_3D)? static_cast<int>(local_domain.mesh.n_active(in::x3)) : 1;

            //* displacements must stay well inside one neighbouring subdomain (see BOUNDARY BOOKKEEPING)
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

            //* PERIODIC on a face => this subdomain spans the box in that
            //* direction, so wrap locally. Anything else => a neighbour owns it.
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
                //* xi_s = -sign(q_s) * (AMP / n_tot) * E
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

                    //* back to code coords, then re-split into cell + offset.
                    //* floor (not truncation) so that negative x_Cd, which a
                    //* particle pushed off the low edge of the local domain will
                    //* have, still yields dx in [0, 1). Out-of-range cells are
                    //* resolved by the particle exchange below.
                    metric.template convert<Crd::XYZ, Crd::Cd>(x_Ph, x_Cd);

                    //* split x_Cd into (cell, offset). prtldx_t is float while
                    //* real_t may be double, so the narrowing cast can round an
                    //* offset just under 1 up to exactly 1; carry it into the
                    //* cell index, because the shape functions require dx in [0, 1).
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
                    if (outside && !tagged) { return; }   //* keep it where it was

                    i1(p) = c1; dx1(p) = o1;
                    i2(p) = c2; dx2(p) = o2;
                    if constexpr (D == Dim::_3D) { i3(p) = c3; dx3(p) = o3; }

                    //* (4) redraw the velocity at the displaced (periodic, global) position: thermal spread + u_perturbed + u_drift_species there
                    vec_t<Dim::_3D> four_velocity { ZERO };
                    edist(x_Ph, four_velocity);
                    ux1(p) = four_velocity[0];
                    ux2(p) = four_velocity[1];
                    ux3(p) = four_velocity[2];
                });
            }

            //! MANDATORY. The displacement is ~ a few cells, so particles near a
            //! subdomain edge now sit outside it. entity's MPI exchange does not
            //! run until AFTER the first push and deposit (srpic.hpp: ParticlePush
            //! -> CurrentsDeposit -> CommunicateParticles), so without this call the
            //! first deposit would index out of range.
            metadomain.CommunicateParticles(local_domain);
        }
    };

} // namespace user

#endif