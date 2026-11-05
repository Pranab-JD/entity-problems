#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "traits/pgen.h"
#include "utils/numeric.h"
#include "arch/kokkos_aliases.h"

#include "archetypes/utils.h"
#include "archetypes/energy_dist.h"
#include "archetypes/spatial_dist.h"
#include "framework/domain/metadomain.h"
#include "archetypes/particle_injector.h"

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

namespace user
{
    using namespace ntt;

    //! DECAYING KINETIC TURBULENCE
    //! E = 0 initially, so the seed is not Alfvenic: both wave families are excited.
    //! REQUIRES PERIODIC BOUNDARIES in every direction.
    //!
    //! CURRENT LOADING
    //! ---------------------
    //! The particles are injected with a position-dependent counter-drift that
    //! supplies curl(B) at t = 0, so Ampere's law holds from the first step

    //* dB_m(x) = (bx, by, bz) * cos(k.x + phase); the curl is already folded into
    //* the amplitudes, so the device kernel is a bare cosine sum.
    //*
    //* kcx/kcy/kcz are the wavenumbers used for the CURL only (see make_init):
    //* either k itself, or the Yee-discretised k~ = (2/dx) sin(k dx / 2).
    struct TurbMode
    {
        real_t kx, ky, kz;
        real_t bx, by, bz;
        real_t phase;
        real_t kcx, kcy, kcz;
    };

    //! A_m = a_k e_A sin(k.x + ph)  =>  dB_m = a_k (k x e_A) cos(k.x + ph),
    //! divergence-free by construction, with |dB_m| = a_k |k| since e_A _|_ k.
    //!
    //! a_k ~ (k/k0)^(-(p+dim+1)/2): with N(k)dk ~ k^(dim-1)dk modes per shell this
    //! yields E(k) ~ k^(dim-1)|dB_k|^2 ~ k^-p. dim=2 -> -(p+3)/2 (the exponent the
    //! original reconnection pgen used), dim=3 -> -(p+4)/2.
    //!
    //! Returns the achieved per-component RMS.
    inline auto init_turbulence(real_t                 Lx,
                                real_t                 Ly,
                                real_t                 Lz,
                                bool                   is_3d,
                                real_t                 dB_rms_target, //* total RMS |dB|
                                int                    kmin,
                                int                    kmax,
                                long long              seed,
                                real_t                 spectral_index,
                                int                    kplane,        //* -1 = full 3D, 0 = x1x2, 1 = x2x3, 2 = x3x1
                                int                    a_axis,        //* -1 = random e_A _|_ k, else 0/1/2 = fixed axis
                                bool                   random_amp,    //* Gaussian mode amplitudes
                                std::vector<TurbMode>& modes_out) -> std::array<real_t, 3>
    {
        modes_out.clear();
        const std::array<real_t, 3> zero { (real_t)0.0, (real_t)0.0, (real_t)0.0 };

        if (dB_rms_target <= (real_t)0.0 || kmax < kmin || kmin < 1) return zero;

        std::mt19937_64                        rng(static_cast<uint64_t>(seed));
        std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * M_PI);
        std::normal_distribution<double>       gauss_dist(0.0, 1.0);

        //! In 2D only the x1-x2 plane is admissible: a mode with k3 != 0 would be
        //! sampled at x3 = 0 alone, breaking periodicity and div(dB) = 0
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

        //! dim is the dimensionality of K-SPACE, not of the simulation: a plane-
        //! restricted spectrum in a 3D box still has N(k)dk ~ k dk per shell.
        const double dim        = (double)((vary_x ? 1 : 0) + (vary_y ? 1 : 0) + (vary_z ? 1 : 0));
        const double weight_exp = -0.5 * ((double)spectral_index + dim + 1.0);

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

                    //! shell selected in MODE-NUMBER space, not |k|: these differ
                    //! for a non-cubic box
                    const double n_mag = std::sqrt((double)(nx * nx + ny * ny + nz * nz));
                    if (n_mag < (double)kmin || n_mag > (double)kmax) continue;

                    const double kx    = (double)nx * k0x;
                    const double ky    = (double)ny * k0y;
                    const double kz    = (double)nz * k0z;
                    const double k_mag = std::sqrt(kx * kx + ky * ky + kz * kz);
                    if (k_mag <= 0.0) continue;

                    double ex, ey, ez;
                    if (a_axis >= 0 && a_axis <= 2)
                    {
                        //* dB _|_ e_a: a_axis = 2 with k in x1-x2 is the classic 2D
                        //* setup (dB in-plane, dB_3 = 0)
                        ex = (a_axis == 0) ? 1.0 : 0.0;
                        ey = (a_axis == 1) ? 1.0 : 0.0;
                        ez = (a_axis == 2) ? 1.0 : 0.0;
                    }
                    else
                    {
                        //* project out k-parallel part: it drops out of k x e
                        //* anyway, and removing it keeps |dB_m| = a_k|k| exact
                        double gx = 0.0, gy = 0.0, gz = 0.0, norm = 0.0;
                        int    tries = 0;
                        do
                        {
                            gx = gauss_dist(rng); gy = gauss_dist(rng); gz = gauss_dist(rng);
                            const double proj = (gx * kx + gy * ky + gz * kz) / (k_mag * k_mag);
                            gx -= proj * kx; gy -= proj * ky; gz -= proj * kz;
                            norm = std::sqrt(gx * gx + gy * gy + gz * gz);
                        }
                        while (norm < 1.0e-12 && ++tries < 10);

                        if (norm < 1.0e-12) continue;
                        ex = gx / norm; ey = gy / norm; ez = gz / norm;
                    }

                    const double rand_fac = random_amp ? gauss_dist(rng) : 1.0;
                    const double a_k      = rand_fac * std::pow(k_mag / k0_ref, weight_exp);

                    const double bx = a_k * (ky * ez - kz * ey);
                    const double by = a_k * (kz * ex - kx * ez);
                    const double bz = a_k * (kx * ey - ky * ex);

                    //* with a fixed A axis, k || e_A gives zero curl: drop those so
                    //* the reported mode count is honest
                    if (std::sqrt(bx * bx + by * by + bz * bz) <
                        1.0e-12 * std::fabs(a_k) * k_mag) continue;

                    //* kc filled in make_init, once dx is known
                    modes_out.push_back(TurbMode { (real_t)kx, (real_t)ky, (real_t)kz,
                                                   (real_t)bx, (real_t)by, (real_t)bz,
                                                   (real_t)phase_dist(rng),
                                                   (real_t)kx, (real_t)ky, (real_t)kz });
                }

        //! RMS is EXACT, not sampled: half-space selection means no two modes share
        //! +/-k, so cross terms vanish over the box and <dB_i^2> = 0.5 sum_m b_i,m^2.
        double s_bx = 0.0, s_by = 0.0, s_bz = 0.0;
        for (const auto& m : modes_out)
        {
            s_bx += 0.5 * (double)m.bx * (double)m.bx;
            s_by += 0.5 * (double)m.by * (double)m.by;
            s_bz += 0.5 * (double)m.bz * (double)m.bz;
        }
        const double s_tot = s_bx + s_by + s_bz;
        if (s_tot <= 0.0) { modes_out.clear(); return zero; }

        const double scale = (double)dB_rms_target / std::sqrt(s_tot);
        for (auto& m : modes_out)
        {
            m.bx = (real_t)((double)m.bx * scale);
            m.by = (real_t)((double)m.by * scale);
            m.bz = (real_t)((double)m.bz * scale);
        }

        return { (real_t)(std::sqrt(s_bx) * scale),
                 (real_t)(std::sqrt(s_by) * scale),
                 (real_t)(std::sqrt(s_bz) * scale) };
    }

    template <Dimension D>
    struct InitFields
    {
        InitFields() = default;

        InitFields(real_t b0z,
                   array_t<real_t*> kx, array_t<real_t*> ky, array_t<real_t*> kz,
                   array_t<real_t*> bx, array_t<real_t*> by, array_t<real_t*> bz,
                   array_t<real_t*> ph,
                   array_t<real_t*> kcx, array_t<real_t*> kcy, array_t<real_t*> kcz,
                   std::size_t nmodes) :
            b0z { b0z },
            kx { kx }, ky { ky }, kz { kz },
            bx { bx }, by { by }, bz { bz },
            ph { ph },
            kcx { kcx }, kcy { kcy }, kcz { kcz },
            nmodes { nmodes } {}

        //* comp: 0 -> x1, 1 -> x2, 2 -> x3
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

        Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t { return turb(x_Ph, 0); }
        Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t { return turb(x_Ph, 1); }
        Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t { return b0z + turb(x_Ph, 2); }

        //! ---------------------------------------------------------------------
        //! ANALYTIC curl(B), Cartesian components.
        //!
        //!   B_m(x) = b_m cos(k_m.x + ph_m)
        //!   curl(b cos(k.x + ph)) = -(k x b) sin(k.x + ph)
        //!
        //! The uniform b0z has zero curl, so only the mode sum contributes. kc is
        //! used in place of k so that the DISCRETE Yee curl can be matched exactly
        //! (see make_init); with kc = k this is the continuum curl.
        //!
        //! Units: this is curl(B) itself. The drift normalisation is applied by the
        //! caller (TurbCurrentDist), NOT here.
        //! ---------------------------------------------------------------------
        Inline void current(const coord_t<D>& x_Ph, real_t& Jx, real_t& Jy, real_t& Jz) const
        {
            Jx = ZERO; Jy = ZERO; Jz = ZERO;

            const real_t X = x_Ph[0];
            const real_t Y = x_Ph[1];
            real_t       Z = ZERO;
            if constexpr (D == Dim::_3D) { Z = x_Ph[2]; }

            for (std::size_t m = 0; m < nmodes; ++m)
            {
                const real_t s = math::sin(kx(m) * X + ky(m) * Y + kz(m) * Z + ph(m));
                Jx -= (kcy(m) * bz(m) - kcz(m) * by(m)) * s;
                Jy -= (kcz(m) * bx(m) - kcx(m) * bz(m)) * s;
                Jz -= (kcx(m) * by(m) - kcy(m) * bx(m)) * s;
            }
        }

    private:
        real_t            b0z { ZERO };
        array_t<real_t*>  kx, ky, kz, bx, by, bz, ph, kcx, kcy, kcz;
        std::size_t       nmodes { 0 };
    };


    //! =========================================================================
    //!  Energy distribution: Maxwellian boosted into the local current-carrying
    //!  drift frame.
    //!
    //!  Entity integrates   dE/dt = curl(B) - (r0/d0^2) J,  so at equilibrium
    //!
    //!        J_code = AMP * curl(B),       AMP = d0^2 / r0 = d0 * sqrt(sigma0).
    //!
    //!  The two species counter-drift at fixed total density n_tot and |q_s| = 1:
    //!
    //!        J_code = sum_s n_s q_s beta_s
    //!
    //!  Splitting the drift as beta_s = sign(q_s) * lambda / m_s puts most of the
    //!  current on the LIGHT species and leaves zero net momentum density. Then
    //!  J_code = lambda * sum_s (n_s / m_s), so
    //!
    //!        beta_s = sign(q_s) * AMP * curl(B) / ( m_s * sum_t n_t/m_t )
    //!               = drift_coeff_s * curl(B).
    //!
    //!  For a pair plasma (m = 1, n_s = 1/2, n_tot = 1) this reduces to the
    //!  flux-tube pgen's beta_s = sign(q_s) * AMP * curl(B).
    //! =========================================================================
    template <class M>
    struct TurbCurrentDist
    {
        static constexpr auto D = M::Dim;

        TurbCurrentDist(random_number_pool_t& pool, real_t theta_s, real_t drift_coeff,
                        const InitFields<M::Dim>& flds):
                        maxwellian { pool, theta_s }, drift_coeff { drift_coeff }, flds { flds } {}

        Inline void operator()(const coord_t<D>& x_Ph, vec_t<Dim::_3D>& v) const
        {
            //* 1. thermal four-velocity in the local rest frame
            maxwellian(x_Ph, v);

            if (drift_coeff == ZERO) { return; }

            //* 2. current-driven drift: beta = drift_coeff * curl(B)
            real_t Jx, Jy, Jz;
            flds.current(x_Ph, Jx, Jy, Jz);

            real_t drift_x = drift_coeff * Jx;
            real_t drift_y = drift_coeff * Jy;
            real_t drift_z = drift_coeff * Jz;

            //* 3. safety clamp (should never fire; print_setup asserts |beta| < 0.99)
            real_t drift_speed_squared = SQR(drift_x) + SQR(drift_y) + SQR(drift_z);
            constexpr real_t MAX_DRIFT_SPEED    = static_cast<real_t>(0.99);
            constexpr real_t MAX_DRIFT_SPEED_SQ = MAX_DRIFT_SPEED * MAX_DRIFT_SPEED;
            if (drift_speed_squared >= MAX_DRIFT_SPEED_SQ)
            {
                const real_t rescale = MAX_DRIFT_SPEED / math::sqrt(drift_speed_squared);
                drift_x *= rescale; drift_y *= rescale; drift_z *= rescale;
                drift_speed_squared = MAX_DRIFT_SPEED_SQ;
            }

            //* 4. Lorentz boost of the thermal momentum into the drift frame
            if (drift_speed_squared > ZERO)
            {
                const real_t ux = v[0], uy = v[1], uz = v[2];

                const real_t gamma_thermal = math::sqrt(ONE + SQR(ux) + SQR(uy) + SQR(uz));
                const real_t gamma_drift   = ONE / math::sqrt(ONE - drift_speed_squared);

                //* u_d = gamma_d * beta_d
                const real_t u_dx = gamma_drift * drift_x;
                const real_t u_dy = gamma_drift * drift_y;
                const real_t u_dz = gamma_drift * drift_z;

                //* f = (u . u_d)/(gamma_d + 1) + gamma_th
                const real_t boost = (ux * u_dx + uy * u_dy + uz * u_dz) / (gamma_drift + ONE) + gamma_thermal;

                v[0] = ux + boost * u_dx;
                v[1] = uy + boost * u_dy;
                v[2] = uz + boost * u_dz;
            }
        }

    private:
        const arch::energy_dist::Maxwellian<M::Dim, M::CoordType> maxwellian;
        const real_t             drift_coeff;
        const InitFields<M::Dim> flds;
    };

    //! Uniform spatial distribution (density is set by the injector's ndens argument).
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

        //? ---- [scales] ----
        const real_t larmor0;                   //* rho0 [d0]
        const real_t skindepth0;                //* d0   [d0]
        const real_t sp1_mass;                  //* fiducial units
        const real_t sp2_mass;                  //* species 2 mass (positrons / ions)
        const real_t sp1_charge;                //* signed, |q| must be 1
        const real_t sp2_charge;
        const real_t mass_ratio;                //* DERIVED = sp2_mass / sp1_mass  (= 1 for pair plasma)
        const real_t sigma0;                    //* (skindepth0/larmor0)^2 = magnetisation of the FIDUCIAL species at B = 1

        //? ---- setup block ----
        //! Both field amplitudes are ABSOLUTE, in code B units. The magnetization
        //! scale is set by [scales] alone: sigma0 = (skindepth0/larmor0)^2 is the
        //! magnetization of the fiducial species at B = 1.
        const real_t    bguide;                 //* |B_mean| along x3 (0 -> zero-mean-field turbulence)
        const real_t    turb_amp;               //* RMS |dB|
        const int       kmin, kmax;             //* mode-number band (inclusive)
        const real_t    spectral_index;         //* p in E(k) ~ k^-p  (5/3 default)
        const long long turb_seed;
        const int       turb_kplane;            //* k-space restriction: -1 = 3D, 0 = x1x2, 1 = x2x3, 2 = x3x1
        const int       turb_a_axis;            //* vector potential axis: -1 = random _|_ k, 0/1/2 = x1/x2/x3
        const bool      turb_rand_amp;          //* true -> Gaussian random mode amplitudes
        const real_t    temperature;            //* fiducial temperature kT / (m_fid c^2)

        //? ---- geometry ----
        const real_t xmin, xmax, ymin, ymax, zmin, zmax, Lx, Ly, Lz;

        //? ---- derived ----
        real_t                theta_1, theta_2;   //* kT/(m_s c^2)
        real_t                B_mean, dB_rms;
        std::array<real_t, 3> dB_rms_cmp;         //* achieved, per component
        real_t                amp { ONE };        //* d0^2/r0 = d0 sqrt(sigma0): J_code = AMP curl(B)
        real_t                drift_coeff_1 { ZERO }, drift_coeff_2 { ZERO };

        std::vector<TurbMode> modes_host;
        array_t<real_t*>      kx_d, ky_d, kz_d, bx_d, by_d, bz_d, ph_d, kcx_d, kcy_d, kcz_d;

        InitFields<D> init_flds;

        static auto make_init(PGen& g) -> InitFields<D>
        {
            raise::ErrorIf(g.mass_ratio < (real_t)1.0,
                           "species 2 (positron/ion) must be at least as heavy as species 1 "
                           "(electron); mass_ratio = 1 is a pair plasma",
                           HERE);
            raise::ErrorIf(g.kmin < 1 || g.kmax < g.kmin,
                           "require 1 <= setup.kmin <= setup.kmax", HERE);
            raise::ErrorIf(g.turb_kplane > 2,
                           "setup.turb_kplane must be -1 (isotropic 3D), 0 (x1-x2), "
                           "1 (x2-x3) or 2 (x3-x1)", HERE);
            raise::ErrorIf((D == Dim::_2D) && (g.turb_kplane != 0),
                           "in 2D only setup.turb_kplane = 0 (x1-x2) is admissible; "
                           "there is no x3 grid direction. Use setup.turb_a_axis to "
                           "choose the polarisation of dB instead", HERE);
            raise::ErrorIf(g.turb_a_axis > 2,
                           "setup.turb_a_axis must be -1 (random, perpendicular to k) "
                           "or 0 / 1 / 2 (fixed axis x1 / x2 / x3)", HERE);

            //! beta_s = sign(q_s) * AMP * curl(B) / (m_s sum_t n_t/m_t) assumes |q_s| = 1
            //! for both species, and opposite signs so that they can counter-drift at
            //! zero net charge.
            raise::ErrorIf(math::abs(math::abs(g.sp1_charge) - ONE) > (real_t)1.0e-5 ||
                           math::abs(math::abs(g.sp2_charge) - ONE) > (real_t)1.0e-5,
                           "current loading requires |q_s| = 1 for both species: "
                           "beta_s = sign(q_s) * AMP * curl(B) / (m_s sum n/m) assumes it, "
                           "so the loaded current would be wrong otherwise.", HERE);
            raise::ErrorIf(g.sp1_charge * g.sp2_charge >= ZERO,
                           "current loading requires the two species to have OPPOSITE "
                           "charge signs: they must counter-drift to carry J at zero net "
                           "charge.", HERE);

            g.theta_1 = g.temperature / g.sp1_mass;
            g.theta_2 = g.temperature / g.sp2_mass;

            g.B_mean = g.bguide;
            g.dB_rms = g.turb_amp;

            const bool is_3d = (D == Dim::_3D);

            g.dB_rms_cmp = init_turbulence(g.Lx, g.Ly, g.Lz, is_3d, g.dB_rms,
                                           g.kmin, g.kmax, g.turb_seed,
                                           g.spectral_index, g.turb_kplane, g.turb_a_axis,
                                           g.turb_rand_amp, g.modes_host);

            //! ---- AMP and the per-species drift coefficients --------------------
            //! AMP = d0^2/r0 = d0 sqrt(sigma0): see the TurbCurrentDist header.
            g.amp = SQR(g.skindepth0) / g.larmor0;

            //! n_tot is the TOTAL density handed to the injector in InitPrtls, split
            //! evenly over the two species. beta = AMP * curl(B) / n_tot for a pair
            //! plasma, so this factor matters.
            //!
            //! CONFIRMED against src/archetypes/particle_injector.h: the last
            //! argument of InjectNonUniform is documented "Total number density
            //! (in units of n0)", and the kernel takes ppc = number_density * ppc0
            //! * HALF, i.e. an even split over the two species.
            const real_t n_total  = ONE;
            const real_t n_per_sp = HALF * n_total;

            //* sum_t n_t / m_t
            const real_t inv_mass_sum = n_per_sp * (ONE / g.sp1_mass + ONE / g.sp2_mass);

            if (inv_mass_sum > ZERO)
            {
                const real_t sign_1 = (g.sp1_charge < ZERO) ? -ONE : ONE;
                const real_t sign_2 = (g.sp2_charge < ZERO) ? -ONE : ONE;
                g.drift_coeff_1 = sign_1 * g.amp / (g.sp1_mass * inv_mass_sum);
                g.drift_coeff_2 = sign_2 * g.amp / (g.sp2_mass * inv_mass_sum);
            }

            //! ---- curl wavenumbers ----------------------------------------------
            //! The Yee stagger evaluates dB/dx as [B(x + dx/2) - B(x - dx/2)]/dx. For
            //! B ~ cos(kx x + .) that is exactly -k~ sin(kx x + .) with
            //!
            //!     k~ = (2/dx) sin(k dx / 2)   =   k * sinc(k dx / 2).
            //!
            //! Using k~ instead of k makes the supplied J equal the DISCRETE curl of
            //! the initialised B to round-off, rather than to O((k dx)^2 / 24).
            //!
            //! CONFIRMED against Entity's source, so this is not optional:
            //!   - src/archetypes/field_setter.h (SRPIC branch): B sits at the standard
            //!     Yee positions, bx1 at (i, j+1/2), bx2 at (i+1/2, j), bx3 at
            //!     (i+1/2, j+1/2), each written through transform<i, Idx::T, Idx::U>.
            //!   - src/kernels/ampere_mink.hpp: the Ampere kernel differences adjacent
            //!     B over exactly ONE cell, centred on the E/J point, in 2D and 3D.
            //! So the discrete curl IS -(k~ x b) sin(k.x + ph) evaluated at the E point,
            //! with no phase shift.
            //!
            //! Two things that look like they should break this, and do not:
            //!
            //!  1. The 2D kernels' coeff1 = dT/dx^2 vs coeff2 = dT asymmetry.
            //!     minkowski.h transform() is the identity for i > D, so in 2D the
            //!     stored out-of-plane component is unscaled while the in-plane ones
            //!     carry 1/dx; faraday_mink.hpp has the matching asymmetry. With the
            //!     standard Yee stencil the product of the two coefficients gives the
            //!     correct c = 1 wave equation for both polarisations. finit.bx1/bx2/
            //!     bx3 are physical throughout, so J_code = AMP * curl(B) needs no
            //!     per-component factor in 2D.
            //!
            //!  2. The extended field-solver stencil (Blinne+ 2018): the
            //!     algorithms.fieldsolver.delta_* and beta_* coefficients, all 0.0 by
            //!     default, widen the derivative stencil and would change k~. They
            //!     appear ONLY in faraday_mink.hpp. ampere_mink.hpp carries no delta
            //!     or beta terms at all, so the curl(B) that the loaded J must balance
            //!     is the plain one-cell difference whatever those inputs are set to,
            //!     and k~ = (2/dx) sin(k dx/2) stays exact. Note that switching them on
            //!     does modify the wave dispersion, which is their purpose.
            {
                const auto&  mesh = g.metadomain.mesh();
                const real_t dx   = g.Lx / static_cast<real_t>(mesh.n_active(in::x1));
                const real_t dy   = g.Ly / static_cast<real_t>(mesh.n_active(in::x2));
                real_t       dz   = ONE;
                if constexpr (D == Dim::_3D)
                {
                    dz = g.Lz / static_cast<real_t>(mesh.n_active(in::x3));
                }

                auto k_tilde = [](double k, double h) -> double
                {
                    if (k == 0.0 || h <= 0.0) return k;
                    return (2.0 / h) * std::sin(0.5 * k * h);
                };

                for (auto& m : g.modes_host)
                {
                    m.kcx = (real_t)k_tilde((double)m.kx, (double)dx);
                    m.kcy = (real_t)k_tilde((double)m.ky, (double)dy);
                    m.kcz = (D == Dim::_3D) ? (real_t)k_tilde((double)m.kz, (double)dz) : m.kz;
                }
            }

            const std::size_t nm = g.modes_host.size();
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
                h_kx(m)  = g.modes_host[m].kx;
                h_ky(m)  = g.modes_host[m].ky;
                h_kz(m)  = g.modes_host[m].kz;
                h_bx(m)  = g.modes_host[m].bx;
                h_by(m)  = g.modes_host[m].by;
                h_bz(m)  = g.modes_host[m].bz;
                h_ph(m)  = g.modes_host[m].phase;
                h_kcx(m) = g.modes_host[m].kcx;
                h_kcy(m) = g.modes_host[m].kcy;
                h_kcz(m) = g.modes_host[m].kcz;
            }

            Kokkos::deep_copy(g.kx_d, h_kx);   Kokkos::deep_copy(g.ky_d, h_ky);
            Kokkos::deep_copy(g.kz_d, h_kz);   Kokkos::deep_copy(g.bx_d, h_bx);
            Kokkos::deep_copy(g.by_d, h_by);   Kokkos::deep_copy(g.bz_d, h_bz);
            Kokkos::deep_copy(g.ph_d, h_ph);
            Kokkos::deep_copy(g.kcx_d, h_kcx); Kokkos::deep_copy(g.kcy_d, h_kcy);
            Kokkos::deep_copy(g.kcz_d, h_kcz);

            return InitFields<D> { g.B_mean,
                                   g.kx_d, g.ky_d, g.kz_d,
                                   g.bx_d, g.by_d, g.bz_d, g.ph_d,
                                   g.kcx_d, g.kcy_d, g.kcz_d, nm };
        }

        static auto species_mass(Metadomain<S, M>& m, std::size_t idx) -> real_t
        {
            const auto& species = m.species_params();
            raise::ErrorIf(species.size() < 2,
                           "ERROR in species_mass: pgen needs >= 2 species "
                           "(1 = electrons, 2 = positrons/ions)", HERE);
            const real_t ms = (real_t)species[idx].mass();
            raise::ErrorIf(ms <= (real_t)0.0,
                           "ERROR in species_mass: species mass must be > 0", HERE);
            return ms;
        }

        static auto species_charge(Metadomain<S, M>& m, std::size_t idx) -> real_t
        {
            const auto& species = m.species_params();
            raise::ErrorIf(species.size() < 2,
                           "ERROR in species_charge: pgen needs >= 2 species "
                           "(1 = electrons, 2 = positrons/ions)", HERE);
            return (real_t)species[idx].charge();
        }

        PGen(const SimulationParams& p, Metadomain<S, M>& m) :
            params { p }, metadomain { m },
            larmor0        { p.template get<real_t>("scales.larmor0") },
            skindepth0     { p.template get<real_t>("scales.skindepth0") },
            sp1_mass       { species_mass(m, 0) },
            sp2_mass       { species_mass(m, 1) },
            sp1_charge     { species_charge(m, 0) },
            sp2_charge     { species_charge(m, 1) },
            mass_ratio     { sp2_mass / sp1_mass },
            sigma0         { (skindepth0 / larmor0) * (skindepth0 / larmor0) },
            bguide         { p.template get<real_t>("setup.bguide", (real_t)1.0) },
            turb_amp       { p.template get<real_t>("setup.turb_amp", (real_t)1.0) },
            kmin           { p.template get<int>("setup.kmin", 1) },
            kmax           { p.template get<int>("setup.kmax", 4) },
            spectral_index { (real_t)std::max(0.0, (double)p.template get<real_t>("setup.spectral_index", (real_t)1.6666667)) },
            turb_seed      { static_cast<long long>(p.template get<int>("setup.turb_seed", 12345)) },
            turb_kplane    { p.template get<int>("setup.turb_kplane", (D == Dim::_3D) ? -1 : 0) },
            turb_a_axis    { p.template get<int>("setup.turb_a_axis", -1) },
            turb_rand_amp  { p.template get<bool>("setup.turb_rand_amp", true) },
            temperature    { p.template get<real_t>("setup.temperature", (real_t)0.01) },
            xmin           { m.mesh().extent(in::x1).first },
            xmax           { m.mesh().extent(in::x1).second },
            ymin           { m.mesh().extent(in::x2).first },
            ymax           { m.mesh().extent(in::x2).second },
            zmin           { (D == Dim::_3D) ? m.mesh().extent(in::x3).first : (real_t)0.0 },
            zmax           { (D == Dim::_3D) ? m.mesh().extent(in::x3).second : (real_t)1.0 },
            Lx             { xmax - xmin },
            Ly             { ymax - ymin },
            Lz             { zmax - zmin },
            init_flds      { make_init(*this) } { print_setup(); }

        //! Startup diagnostics; all lengths in code units (d0).
        void print_setup()
        {
            int mpi_rank = 0;
            #if defined(MPI_ENABLED)
                MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            #endif

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

            //* <gamma>(theta) for a 3D Maxwell-Juettner distribution.
            //* theta -> 0 : 1 + 3/2 theta   |   theta -> inf : 3 theta
            auto gamma_mean = [](real_t theta) -> real_t
            {
                return ONE + theta * ((real_t)6.0 + (real_t)15.0 * theta) /
                             ((real_t)4.0 + (real_t)5.0 * theta);
            };
            const real_t g1 = gamma_mean(theta_1);     //* electrons
            const real_t g2 = gamma_mean(theta_2);     //* positrons / ions

            //! Everything below is evaluated at the ACTUAL initial RMS field, not at
            //! a separate reference amplitude: the field scale comes from [scales].
            const real_t B_rms = (real_t)std::sqrt((double)(B_mean * B_mean + dB_rms * dB_rms));
            const real_t B_ref = (B_rms > (real_t)0.0) ? B_rms : (real_t)1.0;   //* guard: B = 0 run

            //* sigma_s = B^2 / (4 pi n_s m_s c^2), n_s = 1
            const real_t sigma_1_cold = sigma0 * B_ref * B_ref / sp1_mass;              //* electrons
            const real_t sigma_2_cold = sigma0 * B_ref * B_ref / sp2_mass;              //* sp2
            //* both species contribute inertia: 1/sigma = 1/s1 + 1/s2
            const real_t sigma_cold   = sigma0 * B_ref * B_ref / (sp1_mass + sp2_mass);
            const real_t sigma_hot    = sigma0 * B_ref * B_ref / (sp1_mass * g1 + sp2_mass * g2);
            const real_t vA_cold      = (real_t)std::sqrt((double)sigma_cold / (1.0 + (double)sigma_cold));
            const real_t vA_hot       = (real_t)std::sqrt((double)sigma_hot / (1.0 + (double)sigma_hot));

            //* kinetic scales, with the thermal <gamma> corrections
            const real_t d_1       = skindepth0 * (real_t)std::sqrt((double)(g1 * sp1_mass));
            const real_t d_2       = skindepth0 * (real_t)std::sqrt((double)(g2 * sp2_mass));
            const real_t rho_1     = larmor0 * g1 * sp1_mass / B_ref;
            const real_t rho_2     = larmor0 * g2 * sp2_mass / B_ref;
            const real_t lambda_D1 = (real_t)std::sqrt((double)theta_1) * d_1;
            const real_t lambda_D2 = (real_t)std::sqrt((double)theta_2) * d_2;

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
            const real_t dB_tot  = dB_rms;
            const real_t dB_over_B = (B_mean > (real_t)0.0) ? dB_tot / B_mean : (real_t)-1.0;

            //* eddy turnover at the outer scale, taking dv ~ vA min(1, dB/B)
            const real_t v_turb  = (B_mean > (real_t)0.0) ? vA_cold * std::min((real_t)1.0, dB_tot / B_mean) : vA_cold;
            const real_t t_eddy  = (v_turb > (real_t)0.0) ? l_outer / v_turb : (real_t)0.0;

            const real_t runtime = params.template get<simtime_t>("simulation.runtime");
            const real_t t_w1    = runtime;                                                   //* [1/omega_p,sp1]
            const real_t t_w2    = runtime / (real_t)std::sqrt((double)mass_ratio);           //* [1/omega_p,sp2]
            const real_t t_cross = runtime / L_min;                                           //* light crossings
            const real_t n_eddy  = (t_eddy > (real_t)0.0) ? runtime / t_eddy : (real_t)0.0;

            //! beta is routed THROUGH sigma on purpose. A code-unit B^2/2 and a
            //! code-unit n*theta*m are in different normalisations unless sigma0 = 1,
            //! so dividing them directly (as the original pgen's pressure-balance
            //! block did) is only meaningful in that pgen's own convention.
            //! P_mag = n(m_1+m_2)sigma_cold/2 and P_th = sum_s n_s T_s = 2T give a
            //! ratio independent of the field normalisation.
            const real_t plasma_beta = (sigma_cold > (real_t)0.0) ? (real_t)4.0 * temperature / ((sp1_mass + sp2_mass) * sigma_cold) : (real_t)-1.0;
            const real_t turb_frac = (B_rms > (real_t)0.0) ? (dB_tot * dB_tot) / (B_rms * B_rms) : (real_t)0.0;

            //! =================================================================
            //! CURRENT / DRIFT DIAGNOSTICS
            //!
            //! Sampled on the host from modes_host, over a coarse regular grid
            //! spanning the whole box. Two currents are compared at every sample:
            //!
            //!   J_load = -sum (kc x b) sin(k.x + ph)   what the drift will supply
            //!   J_fd   = centred difference curl of B  what the Yee solver demands
            //!
            //! These should agree to round-off. A nonzero residual means the mode
            //! table and the k~ correction have fallen out of step.
            //! =================================================================
            auto B_host = [&](double X, double Y, double Z, double& Bx, double& By, double& Bz)
            {
                Bx = 0.0; By = 0.0; Bz = (double)B_mean;
                for (const auto& m : modes_host)
                {
                    const double c = std::cos((double)m.kx * X + (double)m.ky * Y +
                                              (double)m.kz * Z + (double)m.phase);
                    Bx += (double)m.bx * c;
                    By += (double)m.by * c;
                    Bz += (double)m.bz * c;
                }
            };

            auto J_load_host = [&](double X, double Y, double Z, double& Jx, double& Jy, double& Jz)
            {
                Jx = 0.0; Jy = 0.0; Jz = 0.0;
                for (const auto& m : modes_host)
                {
                    const double s = std::sin((double)m.kx * X + (double)m.ky * Y +
                                              (double)m.kz * Z + (double)m.phase);
                    Jx -= ((double)m.kcy * (double)m.bz - (double)m.kcz * (double)m.by) * s;
                    Jy -= ((double)m.kcz * (double)m.bx - (double)m.kcx * (double)m.bz) * s;
                    Jz -= ((double)m.kcx * (double)m.by - (double)m.kcy * (double)m.bx) * s;
                }
            };

            //* centred Yee difference over one cell, exactly as the FDTD curl does it
            auto J_fd_host = [&](double X, double Y, double Z, double& Jx, double& Jy, double& Jz)
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

                Jx = dBz_dy - dBy_dz;
                Jy = dBx_dz - dBz_dx;
                Jz = dBy_dx - dBx_dy;
            };

            constexpr int NS = (D == Dim::_3D) ? 24 : 64;
            const int     NSz = (D == Dim::_3D) ? NS : 1;

            long   n_samples = 0;
            double J_sum = 0.0, fd_sum = 0.0;
            double J_max = 0.0, fd_max = 0.0;
            double dev_absmax = 0.0;        //* max |J_load - J_fd|
            double beta_max = 0.0;          //* max over species of |drift_coeff * J_load|
            double beta_sum = 0.0;

            const double coeff_max = std::max(std::fabs((double)drift_coeff_1),
                                              std::fabs((double)drift_coeff_2));

            for (int ix = 0; ix < NS; ++ix)
            for (int iy = 0; iy < NS; ++iy)
            for (int iz = 0; iz < NSz; ++iz)
            {
                const double X = (double)xmin + ((double)ix + 0.5) / (double)NS  * (double)Lx;
                const double Y = (double)ymin + ((double)iy + 0.5) / (double)NS  * (double)Ly;
                const double Z = (D == Dim::_3D)
                               ? (double)zmin + ((double)iz + 0.5) / (double)NSz * (double)Lz
                               : 0.0;

                double Jx, Jy, Jz;
                J_load_host(X, Y, Z, Jx, Jy, Jz);
                const double Jmag = std::sqrt(Jx * Jx + Jy * Jy + Jz * Jz);

                double Fx, Fy, Fz;
                J_fd_host(X, Y, Z, Fx, Fy, Fz);
                const double Fmag = std::sqrt(Fx * Fx + Fy * Fy + Fz * Fz);

                const double dev = std::sqrt(SQR(Jx - Fx) + SQR(Jy - Fy) + SQR(Jz - Fz));

                ++n_samples;
                J_sum  += Jmag;  if (Jmag > J_max)  J_max  = Jmag;
                fd_sum += Fmag;  if (Fmag > fd_max) fd_max = Fmag;
                if (dev > dev_absmax) dev_absmax = dev;

                const double beta_here = coeff_max * Jmag;
                beta_sum += beta_here;
                if (beta_here > beta_max) beta_max = beta_here;
            }

            const double inv_n     = (n_samples > 0) ? 1.0 / (double)n_samples : 0.0;
            const double J_mean    = J_sum    * inv_n;
            const double fd_mean   = fd_sum   * inv_n;
            const double beta_mean = beta_sum * inv_n;
            const double amp_dev   = (fd_max > 0.0) ? dev_absmax / fd_max : 0.0;
            const double ratio_max = (fd_max > 0.0) ? J_max / fd_max : 0.0;

            const double gamma_d_max = 1.0 / std::sqrt(std::max(1.0e-12, 1.0 - beta_max * beta_max));

            //* order-of-magnitude scaling: beta ~ (k d0) sqrt(sigma0) dB
            const real_t k_inner      = (real_t)(2.0 * M_PI * (double)kmax / (double)L_min);
            const real_t beta_scaling = k_inner * skindepth0 * (real_t)std::sqrt((double)sigma0) * dB_tot;

            constexpr int LW1 = 34;
            constexpr int LW2 = 30;
            auto kv1 = [&](const std::string& label) -> std::string
            { std::ostringstream t; t << "\t" << std::left << std::setw(LW1) << label << "= "; return t.str(); };
            auto kv2 = [&](const std::string& label) -> std::string
            { std::ostringstream t; t << "\t\t" << std::left << std::setw(LW2) << label << "= "; return t.str(); };

            const char* axis_name[3] = { "x1", "x2", "x3" };

            const std::string kplane_name = (turb_kplane == 0) ? "k in x1-x2 plane"
                                          : (turb_kplane == 1) ? "k in x2-x3 plane"
                                          : (turb_kplane == 2) ? "k in x3-x1 plane" : "isotropic 3D";

            const std::string apol_name = (turb_a_axis >= 0 && turb_a_axis <= 2) ? std::string("A = A_") + std::to_string(turb_a_axis + 1) + " e_" +
                                            std::to_string(turb_a_axis + 1) + "  (dB _|_ " + axis_name[turb_a_axis] + ")" : std::string("random, A _|_ k");

            std::ostringstream oss;
            oss << "\n"
                << "===========================================================\n"
                << title << "\n"
                << "===========================================================\n\n"
                << "GEOMETRY  (periodic boundaries assumed in all directions)\n"
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
                << kv1("Runtime [cold sp1 freq.]") << t_w1 << "\n"
                << kv1("Runtime [cold " + sp2 + " freq.]") << t_w2 << "\n"
                << kv1("Runtime [L/c]") << t_cross << "\n"
                << kv1("Runtime [outer eddy times]") << n_eddy << "\n\n"

                << "MASS / MAGNETISATION  (evaluated at B = B_rms, not at B = 1)\n"
                << kv1("Mass ratio [m_2/m_1]") << mass_ratio << "\n"
                << kv1("sigma0 = (d0/rho0)^2") << sigma0 << "  (fiducial species at B = 1)\n"
                << kv1("B_rms [B0]") << B_rms << "\n"
                << kv1("Sigma (electron, cold)") << sigma_1_cold << "\n"
                << kv1("Sigma (" + sp2 + ", cold)") << sigma_2_cold << "\n"
                << kv1("Sigma (plasma, cold)") << sigma_cold << "\n"
                << kv1("Sigma (plasma, hot)") << sigma_hot << "\n"
                << kv1("Alfven speed [c] (cold)") << vA_cold << "\n"
                << kv1("Alfven speed [c] (hot)") << vA_hot << "\n"
                << kv1("Mean <gamma> electrons") << g1 << "\n"
                << kv1("Mean <gamma> " + sp2s) << g2 << "\n\n"

                << "MAGNETIC FIELD\n"
                << kv1("Mean field |B_mean| [B0]") << B_mean << "\n"
                << kv1("Mean field direction") << "x3 (fixed)" << "\n"
                << kv1("dB RMS (target) [B0]") << dB_tot << "\n"
                << kv1("dB RMS x1 / x2 / x3 [B0]") << dB_rms_cmp[0] << " / "
                                                   << dB_rms_cmp[1] << " / "
                                                   << dB_rms_cmp[2] << "\n"
                << kv1("dB_rms / B_mean") << dB_over_B << "  (-1 => zero mean field)\n"
                << kv1("B_rms total [B0]") << B_rms << "\n\n"

                << "TURBULENCE SPECTRUM\n"
                << kv1("Mode band [kmin, kmax]") << "[" << kmin << ", " << kmax << "]\n"
                << kv1("Number of modes") << modes_host.size() << "\n"
                << kv1("Spectral index p (E ~ k^-p)") << spectral_index << "\n"
                << kv1("Random mode amplitudes") << (turb_rand_amp ? "YES" : "NO") << "\n"
                << kv1("k-space") << kplane_name << "\n"
                << kv1("Vector potential") << apol_name << "\n"
                << kv1("Outer scale L/kmin [d0]") << l_outer
                    << "  (" << l_outer / d_1 << " electron skin depths)\n"
                << kv1("Inner scale L/kmax [d0]") << l_inner
                    << "  (resolved with " << l_inner / dx << " cells)\n"
                << kv1("Eddy turnover time [1/w_p1]") << t_eddy << "\n\n"

                << "CURRENT LOADING\n"
                << kv1("Curl operator matched") << "discrete (Yee), k~ = (2/dx) sin(k dx/2)" << "\n"
                << kv1("AMP [d0^2/r0] = J_code/curl(B)") << amp << "\n"
                << kv1("drift coeff sp1 / sp2") << drift_coeff_1 << " / " << drift_coeff_2 << "\n"
                << kv2("max |curl B| (loaded)") << J_max << "\n"
                << kv2("mean |curl B| (loaded)") << J_mean << "\n"
                << kv2("max |curl B| (Yee FD)") << fd_max << "\n"
                << kv2("mean |curl B| (Yee FD)") << fd_mean << "\n"
                << kv1("max |beta| (v/c)") << beta_max << "\n"
                << kv1("mean |beta| (v/c)") << beta_mean << "\n"
                << kv1("Lorentz factor at max |beta|") << gamma_d_max << "\n"
                << kv1("beta estimate (k d0) sqrt(s0) dB") << beta_scaling << "  (order of magnitude)\n\n"

                << "CHARACTERISTIC SCALES\n"
                << "\tElectrons\n"
                << kv2("Skin depth") << d_1 << "  (resolved with " << d_1 / dx << " cells)\n"
                << kv2("Larmor radius") << rho_1 << "  (resolved with " << rho_1 / dx << " cells)\n"
                << kv2("Debye length") << lambda_D1 << "  (resolved with " << lambda_D1 / dx << " cells)\n"
                << "\t" << SP2u << "\n"
                << kv2("Skin depth") << d_2 << "  (resolved with " << d_2 / dx << " cells)\n"
                << kv2("Larmor radius") << rho_2 << "  (resolved with " << rho_2 / dx << " cells)\n"
                << kv2("Debye length") << lambda_D2 << "  (resolved with " << lambda_D2 / dx << " cells)\n"
                << "\tConfinement\n"
                << kv2("rho_L(gamma = sigma) [d0]") << rho_sigma
                    << "  (" << rho_sigma / L_min << " x box)\n\n"

                << "TEMPERATURES (theta = kT / m_s c^2)\n"
                << kv1("Fiducial temperature") << temperature << "\n"
                << kv2("electrons") << theta_1 << "\n"
                << kv2(sp2s) << theta_2 << "\n\n"

                << "ENERGETICS (dimensionless; n = 1 per species)\n"
                << kv1("Plasma beta = P_th / P_mag") << plasma_beta << "\n"
                << kv1("dB^2 / B_rms^2 (turb frac)") << turb_frac << "\n"
                << kv1("dB^2 / B_mean^2") << ((B_mean > (real_t)0.0)
                        ? (dB_tot * dB_tot) / (B_mean * B_mean) : (real_t)-1.0) << "\n\n"

                << "CONSISTENCY CHECKS\n"
                << kv1("max |beta| (must be < 1)") << beta_max << "\n"
                << kv1("max|J|/max|curl B| (must be 1)") << ratio_max << "\n"
                << kv1("max|J-curl(B)|/max|curl B| (->0)") << amp_dev << "\n"
                << kv1("lambda_D / dx (should be >~ 1)") << lambda_D1 / dx << "\n\n";

            if (lambda_D1 / dx < ONE)
                oss << "!! WARNING: the electron Debye length is under-resolved ("
                    << lambda_D1 / dx << " cells).\n"
                    << "!! Expect numerical heating; refine the grid or raise the temperature.\n\n";

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

            //! ---- HARD FAILURES ---------------------------------------------
            if (mpi_rank == 0)
            {
                //! (a) feasibility: the field demands a drift faster than light.
                //!     beta ~ (k d0) sqrt(sigma0) dB, so this fires when the injected
                //!     scale approaches the skin depth at high magnetisation.
                raise::ErrorIf(beta_max >= 0.99,
                               "Current-carrying drift >= 0.99c: the field asks for more current "
                               "than this density can carry.\n"
                               "Lower setup.kmax, enlarge the box, lower setup.turb_amp, or lower "
                               "sigma0 (raise scales.larmor0).", HERE);

                //! (b) Ampere invariant: the loaded J must BE the curl of the field
                //!     that is actually written to the grid. Both come from the same
                //!     mode table, so this mainly catches a sign or index error in the
                //!     (kc x b) cross product, plus the discretisation error.
                raise::ErrorIf(amp_dev > 1.0e-2,
                               "Ampere mismatch: the loaded J differs from curl(B) of the "
                               "initialised field.\n"
                               "Check InitFields::current() against InitFields::bx1/bx2/bx3.", HERE);
            }
        }

        //! =====================================================================
        //!  INITIALISE PARTICLES
        //!
        //!  Maxwellian at theta_s, boosted into the local current-carrying drift
        //!  frame. The per-species drift is carried entirely by TurbCurrentDist,
        //!  so there is no post-injection particle loop.
        //!
        //! CONFIRMED against src/archetypes/energy_dist.h and
        //! src/archetypes/utils.h: arch::energy_dist::Maxwellian stores the
        //! temperature raw, and it is InjectUniformMaxwellians that divides by the
        //! species mass (temperature_s = temperatures.s / mass_s). Constructing the
        //! Maxwellian directly therefore requires theta_s = temperature / m_s HERE;
        //! passing `temperature` unchanged would be wrong whenever mass_ratio != 1.
        //! =====================================================================
        void InitPrtls(Domain<S, M>& local_domain)
        {
            metadomain.CommunicateFields(local_domain, Comm::E | Comm::B);

            const auto edist_1 = TurbCurrentDist<M>(local_domain.random_pool(), theta_1, drift_coeff_1, init_flds);
            const auto edist_2 = TurbCurrentDist<M>(local_domain.random_pool(), theta_2, drift_coeff_2, init_flds);
            const auto sdist   = UniformDist<M::Dim>();

            //! total injected density = ONE. The relation beta = AMP * curl(B) / n_tot
            //! depends on this; see the n_total constant in make_init.
            arch::InjectNonUniform<S, M, decltype(edist_1), decltype(edist_2), decltype(sdist)>(
                params, local_domain, { 1, 2 }, { edist_1, edist_2 }, sdist, ONE);
        }
    };

} // namespace user

#endif