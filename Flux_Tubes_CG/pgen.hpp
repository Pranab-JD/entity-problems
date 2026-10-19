#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

/**
 * @file Flux_Tubes_CG/pgen.hpp
 * @brief Force-free 2D Lundquist flux-tube collision
 *
 *~ References:
 *      1. Lyutikov et al. (2017), JPP, 83, 635830602
 *      2. Ripperda et al. (2019), MNRAS, 485, 299
 *      3. Granier  et al. (2025), ApJ, 992, 193
 *
 *~ API NOTE (this revision)
 * -------------------------------------------------------------
 * Ported to the newer Entity problem-generator interface, matching the Harris
 * pgen conventions exactly:
 *   - includes:   "traits/pgen.h" instead of "arch/traits.h"
 *   - traits:     ::traits::pgen::compatible_with<...> {}   (an OBJECT, not ::value)
 *   - PGen does NOT inherit from arch::ProblemGenerator; it holds
 *                 `const SimulationParams& params` and `Metadomain<S,M>& metadomain`
 *                 as explicit members, and exposes `static constexpr auto D { M::Dim }`
 *   - members:    all inputs are const and initialised in the ctor initialiser list;
 *                 derived quantities are filled by the static helper `make_init(PGen&)`
 *                 which is what initialises `init_flds` (declared LAST, so every input
 *                 it reads is already constructed)
 *   - functions:  `Inline auto f(...) const -> real_t` trailing-return style
 *                 throughout; plain (non-`inline`) member definitions
 *   - distributions: energy/spatial distributions are plain structs in the
 *                 arch::energy_dist / arch::spatial_dist style — they no longer
 *                 derive from arch::EnergyDistribution / arch::SpatialDistribution
 *                 and no longer carry the metric
 *
 * The PHYSICS, the diagnostics and all consistency checks are unchanged.
 *
 *~ Physics setup
 * -------------
 * Two Lundquist tubes (Bessel-function force-free equilibria) are initialised
 * side-by-side along X (tangent at the domain centre). With setup.kick_velocity
 * != 0 they are given opposite motional E = -v x B kicks so that they approach
 * each other along X (head-on) and reconnect.
 *
 *   Tube 1 = LEFT  (kicked +x, toward centre)
 *   Tube 2 = RIGHT (kicked -x, toward centre)
 *
 *~ Field initialisation   (k_t = alpha_t / tube_radius)
 * --------------------------------------------------------------
 *   Az(r) = (1/k_t) * J0(k_t r)                    r <  tube_radius
 *   Az(r) = (1/k_t) * J0(alpha_t)                  r >= tube_radius   [constant]
 *
 *   Bx   =  dAz/dy
 *   By   = -dAz/dx           ( => B_phi(r) = J1(k_t r) )
 *   Bz   =  sqrt( J0(k_t r)^2 + guide_field_floor )
 *
 *~ Current    (ANALYTIC - no grid curl is taken anywhere in this pgen)
 * --------------------------------------------------------------
 * In cylindrical coordinates, for the z-uniform axisymmetric field above:
 *
 *   J_z   = (1/r) d(r B_phi)/dr = k_t J0(k_t r)
 *   J_phi = -dBz/dr            = k_t J0(k_t r) J1(k_t r) / Bz(r)
 *   J_r   = 0
 *   J     = 0  outside the tubes (B is uniform there)
 *
 * This is exactly force-free:  (J x B)_r = J_phi Bz - J_z B_phi
 *                                        = k_t J0 J1 - k_t J0 J1 = 0.
 *
 * Cartesian, with (dx,dy) measured from the tube centre and r = |(dx,dy)|:
 *   J_x = -J_phi dy/r,   J_y = +J_phi dx/r,   J_z as above.
 *
 *~ Normalisation   (see entity_normalisations.md)
 * --------------------------------------------------------------
 * Entity integrates   dE/dt = curl(B) - (r0/d0^2) J,   so in equilibrium
 *
 *        J_code = AMP * curl(B),        AMP = d0^2 / r0 = d0 * sqrt(sigma0).
 *
 * The two species counter-drift, and with the injector's uniform total density
 * n_hat = 1 split 0.5/0.5 over |q| = 1 species,
 *
 *        J_code = sum_s n_s q_s beta_s = beta   (for beta_s = sign(q_s) * beta),
 *
 * hence the drift handed to each species is
 *
 *        beta_s = AMP * J_analytic * sign(q_s),   AMP = skindepth0 * sqrt(sigma0).
 *
 * !! This identity depends on the injected TOTAL density being n_hat = 1.
 * !! Change the density argument of the injector and the current rescales.
 *
 *~ Particle initialisation
 * -----------------------------------------------------------
 *  1. Maxwellian at setup.background_temperature, uniform density.
 *  2. Position-dependent drift beta_s (above), plus the E x B drift of the
 *     motional field when kick_velocity != 0.
 *  3. Lorentz boost of the thermal four-momentum into the drift frame.
 *     (Both done inside FluxTubeDist, so the injector does all the work.)
 *
 *~ Consistency checks performed at startup (print_setup)
 * -----------------------------------------------------------
 *  (a) |beta|_max < 1                            - feasibility
 *  (b) |J x B| / (|J||B|) = 0                    - the analytic J is force-free
 *  (c) J_analytic / curl(B_initialised) = 1      - the analytic J is the curl of
 *                                                  the field actually written to
 *                                                  the grid (the Ampere invariant,
 *                                                  analogous to the Harris J_ratio)
 *
 *! NOT IMPLEMENTED: the charge-density correction that accompanies a nonzero
 *! kick. With kick_velocity != 0, E is discontinuous at r = tube_radius and
 *! div(E) != 0 there; the weight perturbation that would neutralise it is not
 *! applied. A warning is printed. kick_velocity = 0 is unaffected.
 **/

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

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

//! ---------------------------------------------------------------------------
//! GPU-compatible Bessel functions.
//! CUDA/HIP provide device intrinsics j0/j1; on CPU we use std::cyl_bessel_j.
//! Free `Inline auto ... -> real_t` helpers, in the same style as the Harris
//! pgen's free `dki_dyc` helper.
//! ---------------------------------------------------------------------------
Inline auto bessel_J0(real_t argument) -> real_t
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    return j0(argument);
#else
    return std::cyl_bessel_j(0, argument);
#endif
}

Inline auto bessel_J1(real_t argument) -> real_t
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    return j1(argument);
#else
    return std::cyl_bessel_j(1, argument);
#endif
}

namespace user
{
    using namespace ntt;

    //! =========================================================================
    //!  E and B are initialised analytically; entity calls each component at its
    //!  own Yee-stagger position. The same object also supplies the ANALYTIC
    //!  force-free current, so field and current can never drift out of sync.
    //! =========================================================================
    template <Dimension D>
    struct InitFields
    {
        //* alpha_t: FIRST ZERO OF J1 (= 3.8317...). The tube boundary condition
        //* B_phi(tube_radius) = J1(alpha_t) = 0 (no surface current) fixes this.
        static constexpr real_t alpha_t { static_cast<real_t>(3.8317059702075125) };

        InitFields() = default;

        InitFields( real_t tube_radius_,
                    real_t tube1_x_centre_, real_t tube1_y_centre_,
                    real_t tube2_x_centre_, real_t tube2_y_centre_,
                    real_t cell_width_x_,   real_t cell_width_y_,
                    real_t guide_field_floor_,
                    real_t kick_velocity_,
                    bool   single_tube_):
                    tube_radius       { tube_radius_       },
                    tube1_x_centre    { tube1_x_centre_    },
                    tube1_y_centre    { tube1_y_centre_    },
                    tube2_x_centre    { tube2_x_centre_    },
                    tube2_y_centre    { tube2_y_centre_    },
                    cell_width_x      { cell_width_x_      },
                    cell_width_y      { cell_width_y_      },
                    guide_field_floor { guide_field_floor_ },
                    kick_velocity     { kick_velocity_     },
                    single_tube       { single_tube_       }
        {
            k_t = alpha_t / tube_radius;                                //* radial wavenumber
            Az_exterior = bessel_J0(alpha_t) / k_t;                     //* constant -> no in-plane field outside
            Bz_exterior = math::sqrt(SQR(bessel_J0(alpha_t)) + guide_field_floor);
        }

        //! Which tube (if any) contains the point, and the local radius.
        //!   returns 0 = outside both, 1 = inside tube 1, 2 = inside tube 2
        //!   on return, radius / offset_x / offset_y refer to that tube's centre
        Inline auto locate(real_t x, real_t y, real_t& radius, real_t& offset_x, real_t& offset_y) const -> int
        {
            offset_x = x - tube1_x_centre;
            offset_y = y - tube1_y_centre;
            radius   = math::sqrt(SQR(offset_x) + SQR(offset_y));
            if (radius < tube_radius) { return 1; }

            if (!single_tube)
            {
                offset_x = x - tube2_x_centre;
                offset_y = y - tube2_y_centre;
                radius   = math::sqrt(SQR(offset_x) + SQR(offset_y));
                if (radius < tube_radius) { return 2; }
            }
            return 0;
        }

        //! Az vector potential (constant outside the tubes)
        Inline auto Az(const coord_t<D>& x_Ph) const -> real_t
        {
            real_t radius, offset_x, offset_y;
            if (locate(x_Ph[0], x_Ph[1], radius, offset_x, offset_y) == 0) { return Az_exterior; }
            return bessel_J0(k_t * radius) / k_t;
        }

        //!  Bx = dAz/dy
        Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t
        {
            real_t radius, offset_x, offset_y;
            if (locate(x_Ph[0], x_Ph[1], radius, offset_x, offset_y) == 0) { return ZERO; }
            if (radius <= ZERO) { return ZERO; }                 //* J1(0) = 0
            return -bessel_J1(k_t * radius) * offset_y / radius;
        }

        //!  By = -dAz/dx
        Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t
        {
            real_t radius, offset_x, offset_y;
            if (locate(x_Ph[0], x_Ph[1], radius, offset_x, offset_y) == 0) { return ZERO; }
            if (radius <= ZERO) { return ZERO; }
            return bessel_J1(k_t * radius) * offset_x / radius;
        }

        //! Bz (guide field):  sqrt( J0(k_t r)^2 + guide_field_floor )
        Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t
        {
            real_t radius, offset_x, offset_y;
            if (locate(x_Ph[0], x_Ph[1], radius, offset_x, offset_y) == 0) { return Bz_exterior; }
            return math::sqrt(SQR(bessel_J0(k_t * radius)) + guide_field_floor);
        }

        //! Ex = -v x B |_x = 0 for an X-directed kick.
        Inline auto ex1(const coord_t<D>&) const -> real_t
        {
            return ZERO;
        }

        //! Ey = -v x B |_y = +kick * Bz (tube 1), -kick * Bz (tube 2), 0 outside.
        Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t
        {
            if (kick_velocity == ZERO) { return ZERO; }

            real_t radius, offset_x, offset_y;
            const int tube = locate(x_Ph[0], x_Ph[1], radius, offset_x, offset_y);
            if (tube == 0) { return ZERO; }

            const real_t Bz = math::sqrt(SQR(bessel_J0(k_t * radius)) + guide_field_floor);
            return (tube == 1) ? (+kick_velocity * Bz) : (-kick_velocity * Bz);
        }

        //! Ez = -v x B |_z = -kick * By (tube 1), +kick * By (tube 2), 0 outside.
        Inline auto ex3(const coord_t<D>& x_Ph) const -> real_t
        {
            if (kick_velocity == ZERO) { return ZERO; }

            real_t radius, offset_x, offset_y;
            const int tube = locate(x_Ph[0], x_Ph[1], radius, offset_x, offset_y);
            if (tube == 0) { return ZERO; }

            const real_t By = bx2(x_Ph);
            return (tube == 1) ? (-kick_velocity * By) : (+kick_velocity * By);
        }

        //! ---------------------------------------------------------------------
        //! ANALYTIC force-free current J = curl(B), Cartesian components.
        //! Units: this is curl(B) itself. The drift normalisation factor AMP is
        //! applied by the caller (FluxTubeDist), NOT here.
        //! ---------------------------------------------------------------------
        Inline void current(real_t x, real_t y, real_t& Jx, real_t& Jy, real_t& Jz) const
        {
            Jx = ZERO; Jy = ZERO; Jz = ZERO;

            real_t radius, offset_x, offset_y;
            if (locate(x, y, radius, offset_x, offset_y) == 0) { return; }  //* B uniform outside -> J = 0

            const real_t J0_r = bessel_J0(k_t * radius);
            const real_t J1_r = bessel_J1(k_t * radius);
            const real_t Bz   = math::sqrt(SQR(J0_r) + guide_field_floor);

            //* J_z = (1/r) d(r B_phi)/dr = k_t J0
            Jz = k_t * J0_r;

            //* J_phi = -dBz/dr = k_t J0 J1 / Bz   (Bz > 0 is guaranteed by the
            //* guide_field_floor > 0 check in make_init)
            if (radius > ZERO)
            {
                const real_t J_phi = k_t * J0_r * J1_r / Bz;
                Jx = -J_phi * offset_y / radius;        //* e_phi = (-sin(phi), cos(phi))
                Jy =  J_phi * offset_x / radius;
            }
        }

        //*  Data members
        real_t tube_radius       { ZERO  };
        real_t tube1_x_centre    { ZERO  };
        real_t tube1_y_centre    { ZERO  };
        real_t tube2_x_centre    { ZERO  };
        real_t tube2_y_centre    { ZERO  };
        real_t cell_width_x      { ONE   };
        real_t cell_width_y      { ONE   };
        real_t guide_field_floor { ZERO  };
        real_t kick_velocity     { ZERO  };
        real_t k_t               { ONE   };
        real_t Az_exterior       { ZERO  };
        real_t Bz_exterior       { ZERO  };
        bool   single_tube       { false };
    };


    //! =========================================================================
    //!  Energy distribution: thermal Maxwellian boosted into the local
    //!  current-carrying drift frame.
    //!
    //!      beta_s(x) = AMP * J_analytic(x) * sign(q_s)   [+ E x B when kicked]
    //!
    //!  One instance per species (they differ only by sign(q_s)).
    //!
    //!  New-API style (cf. the Harris pgen's HarrisProfile / arch::energy_dist::
    //!  Maxwellian): a plain struct, NOT derived from arch::EnergyDistribution,
    //!  and it does not hold the metric. It is templated on the metric class M
    //!  only so that `arch::energy_dist::Maxwellian<M::Dim, M::CoordType>` can be
    //!  spelled exactly as the Harris pgen spells it.
    //! =========================================================================
    template <class M>
    struct FluxTubeDist
    {
        static constexpr auto D = M::Dim;

        FluxTubeDist(random_number_pool_t&     pool,
                     real_t                    temperature,
                     real_t                    amp,
                     real_t                    charge_sign,
                     const InitFields<M::Dim>& flds)
          : maxwellian  { pool, temperature }
          , amp         { amp }
          , charge_sign { charge_sign }
          , flds        { flds } {}

        //! Signature must match arch::energy_dist::Maxwellian exactly: the injector
        //! calls `energy_dist(x_Ph, v_T)` with TWO arguments and coord_t<M::Dim>
        //! (energy_dist.h:222). No species index is passed in this API.
        Inline void operator()(const coord_t<D>& x_Ph, vec_t<Dim::_3D>& v) const
        {
            //* 1. draw the thermal four-velocity in the local rest frame
            maxwellian(x_Ph, v);

            const real_t x = x_Ph[0];
            const real_t y = x_Ph[1];

            //* 2. current-driven drift:  beta = AMP * J * sign(q)
            real_t Jx, Jy, Jz;
            flds.current(x, y, Jx, Jy, Jz);

            real_t drift_x = amp * charge_sign * Jx;
            real_t drift_y = amp * charge_sign * Jy;
            real_t drift_z = amp * charge_sign * Jz;

            //* 3. E x B drift of the motional field (only when the tubes are kicked)
            //*    v_E = (E x B)/|B|^2, which recovers the perpendicular part of the
            //*    kick velocity exactly when E = -v x B.
            if (flds.kick_velocity != ZERO)
            {
                coord_t<D> xy { ZERO };
                xy[0] = x; xy[1] = y;

                const real_t Bx = flds.bx1(xy);
                const real_t By = flds.bx2(xy);
                const real_t Bz = flds.bx3(xy);
                const real_t Ey = flds.ex2(xy);
                const real_t Ez = flds.ex3(xy);
                const real_t B2 = SQR(Bx) + SQR(By) + SQR(Bz);

                if (B2 > ZERO)
                {
                    //! NOTE: the two drifts are added as 3-velocities, not composed
                    //! relativistically. Fine while |beta_J| and |kick| are modest.
                    drift_x += (Ey * Bz - Ez * By) / B2;
                    drift_y += (Ez * Bx       ) / B2;   //* (E x B)_y = Ez Bx - Ex Bz, Ex = 0
                    drift_z += (     - Ey * Bx) / B2;   //* (E x B)_z = Ex By - Ey Bx, Ex = 0
                }
            }

            //* 4. safety clamp (should never fire; print_setup asserts |beta| < 0.99)
            real_t drift_speed_squared = SQR(drift_x) + SQR(drift_y) + SQR(drift_z);
            constexpr real_t MAX_DRIFT_SPEED    = static_cast<real_t>(0.99);
            constexpr real_t MAX_DRIFT_SPEED_SQ = MAX_DRIFT_SPEED * MAX_DRIFT_SPEED;
            if (drift_speed_squared >= MAX_DRIFT_SPEED_SQ)
            {
                const real_t rescale = MAX_DRIFT_SPEED / math::sqrt(drift_speed_squared);
                drift_x *= rescale; drift_y *= rescale; drift_z *= rescale;
                drift_speed_squared = MAX_DRIFT_SPEED_SQ;
            }

            //* 5. Lorentz boost of the thermal momentum into the drift frame
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
        const real_t                                             amp;          //* d0^2 / r0
        const real_t                                             charge_sign;  //* +-1
        const InitFields<M::Dim>                                 flds;
    };


    //! Uniform spatial distribution (density is set by the injector's ndens argument).
    //! New-API style: plain struct templated on the dimension, no metric member
    //! (cf. HarrisProfile in the Harris pgen).
    template <Dimension D>
    struct UniformDist
    {
        Inline auto operator()(const coord_t<D>&) const -> real_t
        {
            return ONE;
        }
    };


    //! =========================================================================
    //!  PGen
    //! =========================================================================
    template <SimEngine::type S, class M>
    struct PGen
    {
        static constexpr auto D { M::Dim };

        static constexpr auto engines { ::traits::pgen::compatible_with<SimEngine::SRPIC> {} };
        static constexpr auto metrics { ::traits::pgen::compatible_with<Metric::Minkowski> {} };
        static constexpr auto dimensions { ::traits::pgen::compatible_with<Dim::_2D> {} };

        const SimulationParams& params;
        Metadomain<S, M>&       metadomain;

        //* ---- user inputs ---------------------------------------------------
        const real_t background_temperature;
        const real_t kick_velocity;
        const real_t guide_field_floor;
        const bool   single_tube;

        //* ---- normalisation inputs ([scales] block) --------------------------
        const real_t skindepth0;        //* d0
        const real_t larmor0;           //* r0

        //* ---- geometry -------------------------------------------------------
        const real_t xmin, xmax, ymin, ymax, Lx, Ly;
        const real_t tube_radius;

        //* ---- DERIVED (filled by make_init) ----------------------------------
        real_t sigma0 { ONE };          //* (d0/r0)^2
        real_t amp    { ONE };          //* d0^2/r0 = d0 sqrt(sigma0): J_code = AMP curl(B)

        //* ---- runtime monitor state ------------------------------------------
        const int   monitor_interval;                 //* steps between samples
        bool        monitor_header_written { false }; //* header written once per run
        std::string monitor_path;                     //* filled in the ctor body

        InitFields<D> init_flds;        //! declared LAST: make_init reads every member above

        //! =====================================================================
        //!  Derived-quantity setup, in the Harris pgen's `make_init(PGen&)` style:
        //!  validates the inputs, fills the derived members of `g`, and returns the
        //!  fully built field initialiser.
        //! =====================================================================
        static auto make_init(PGen& g) -> InitFields<D>
        {
            //! The analytic J_phi = k_t J0 J1 / Bz divides by Bz = sqrt(J0^2 + floor),
            //! which vanishes at the zero of J0 (r/R = 0.6276) when floor = 0.
            //! A zero floor is a valid force-free solution but it is not usable with an
            //! analytic current: it is 0/0 there, and B_phi reverses discontinuously.
            raise::ErrorIf(g.guide_field_floor <= ZERO,
                           "setup.guide_field_floor must be > 0: the analytic J_phi divides "
                           "by Bz = sqrt(J0^2 + guide_field_floor), which is zero at the first "
                           "zero of J0 when the floor is zero. Use e.g. 0.01.", HERE);

            g.sigma0 = SQR(g.skindepth0 / g.larmor0);
            g.amp    = SQR(g.skindepth0) / g.larmor0;       //* == d0 * sqrt(sigma0)

            const auto&  mesh            = g.metadomain.mesh();
            const real_t domain_x_centre = g.xmin + HALF * g.Lx;
            const real_t domain_y_centre = g.ymin + HALF * g.Ly;
            const real_t dx = g.Lx / static_cast<real_t>(mesh.n_active(in::x1));
            const real_t dy = g.Ly / static_cast<real_t>(mesh.n_active(in::x2));

            return InitFields<D> { g.tube_radius,
                                   domain_x_centre - g.tube_radius, domain_y_centre,  //* Tube 1 (left)
                                   domain_x_centre + g.tube_radius, domain_y_centre,  //* Tube 2 (right)
                                   dx, dy,
                                   g.guide_field_floor, g.kick_velocity, g.single_tube };
        }

        //!  Constructor
        PGen(const SimulationParams& p, Metadomain<S, M>& m): params { p }, metadomain { m },
        background_temperature { p.template get<real_t>("setup.background_temperature", static_cast<real_t>(0.01)) },
        kick_velocity          { p.template get<real_t>("setup.kick_velocity",          ZERO) },
        guide_field_floor      { p.template get<real_t>("setup.guide_field_floor",      static_cast<real_t>(0.01)) },
        single_tube            { p.template get<bool>  ("setup.single_tube",            false) },
        skindepth0             { p.template get<real_t>("scales.skindepth0") },
        larmor0                { p.template get<real_t>("scales.larmor0") },
        xmin                   { m.mesh().extent(in::x1).first },
        xmax                   { m.mesh().extent(in::x1).second },
        ymin                   { m.mesh().extent(in::x2).first },
        ymax                   { m.mesh().extent(in::x2).second },
        Lx                     { xmax - xmin },
        Ly                     { ymax - ymin },
        //* Tube radius default = 0.99 * Lx/4 (slightly less than 1/4 to move tube edges away from boundaries)
        tube_radius            { p.template get<real_t>("setup.tube_radius", Lx * static_cast<real_t>(0.99 * 0.25)) },
        monitor_interval       { p.template get<int>("setup.monitor_interval", 10) },
        init_flds              { make_init(*this) } 
        {
            //* build the monitor path once; CustomPostStep only appends to it
            std::string sim_name = params.template get<std::string>("simulation.name");
            if (!sim_name.empty() && sim_name.back() != '/') { sim_name += '/'; }
            monitor_path = sim_name + "Force_free_monitor.txt";

            print_setup();
        }


        auto MatchFields(simtime_t) const -> InitFields<D>
        {
            return init_flds;
        }

        //! =====================================================================
        //!  Startup diagnostics and consistency checks.
        //!  All lengths in code units (grid.extent units); d0 = skindepth0.
        //! =====================================================================
        void print_setup()
        {
            int mpi_rank = 0;
            #if defined(MPI_ENABLED)
                MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            #endif

            const auto   nx1 = metadomain.mesh().n_active(in::x1);
            const auto   nx2 = metadomain.mesh().n_active(in::x2);
            const real_t dx  = init_flds.cell_width_x;
            const real_t dy  = init_flds.cell_width_y;
            const real_t k_t = init_flds.k_t;

            const real_t d0 = skindepth0;
            const real_t r0 = larmor0;

            const real_t ppc0  = params.template get<real_t>("particles.ppc0");
            const real_t theta = background_temperature;

            //* ---- densities (entity_normalisations.md, section 1 and 3) --------
            //* the injector splits the TOTAL n_hat = 1 over the two species
            const real_t n_total   = ONE;
            const real_t n_per_sp  = HALF * n_total;

            //* ---- scan the tube interior (2D, so the means are area-weighted) ----
            //* Uniform Cartesian scan over tube 1, restricted to r < 0.9 R: J_z jumps
            //* discontinuously at r = R (dJ1/dr != 0 there), so a centred finite
            //* difference of B is meaningless in the outermost 10%.
            constexpr int NS = 256;
            const real_t  r_scan_max = static_cast<real_t>(0.90) * tube_radius;

            long   n_samples  = 0;
            double J_sum      = 0.0, curl_sum  = 0.0, alpha_sum = 0.0;
            real_t J_max      = ZERO, curl_max  = ZERO, alpha_max = ZERO;
            real_t Jz_max     = ZERO, Jphi_max  = ZERO;
            real_t beta_max   = ZERO;       //* max |AMP * J|
            real_t Bphi_max   = ZERO;       //* max in-plane field
            real_t ff_residual = ZERO;      //* max |J x B| / (|J||B|)
            real_t amp_dev_absmax = ZERO;   //* max |J - curl(B)|, un-normalised
            real_t amp_dev_max    = ZERO;   //* = amp_dev_absmax / curl_max, filled after the loop


            //* finite-difference curl of the field as it is actually initialised
            auto curl_B = [&](real_t x, real_t y, real_t& cx, real_t& cy, real_t& cz)
            {
                coord_t<D> xp { x + HALF * dx, y }, xm { x - HALF * dx, y };
                coord_t<D> yp { x, y + HALF * dy }, ym { x, y - HALF * dy };
                cx =  (init_flds.bx3(yp) - init_flds.bx3(ym)) / dy;             //*  dBz/dy
                cy = -(init_flds.bx3(xp) - init_flds.bx3(xm)) / dx;             //* -dBz/dx
                cz =  (init_flds.bx2(xp) - init_flds.bx2(xm)) / dx              //*  dBy/dx - dBx/dy
                    - (init_flds.bx1(yp) - init_flds.bx1(ym)) / dy;
            };

            for (int ix = 0; ix < NS; ++ix)
            for (int iy = 0; iy < NS; ++iy)
            {
                //* normalised offsets in [-1, 1], disk-masked
                const real_t sx = (real_t)2.0 * ((real_t)ix + HALF) / (real_t)NS - ONE;
                const real_t sy = (real_t)2.0 * ((real_t)iy + HALF) / (real_t)NS - ONE;
                if (SQR(sx) + SQR(sy) >= ONE) continue;

                const real_t x = init_flds.tube1_x_centre + sx * r_scan_max;
                const real_t y = init_flds.tube1_y_centre + sy * r_scan_max;
                coord_t<D> xy { x, y };
                ++n_samples;

                //* --- analytic current ---
                real_t Jx, Jy, Jz;
                init_flds.current(x, y, Jx, Jy, Jz);
                const real_t J_inplane = (real_t)std::sqrt((double)(SQR(Jx) + SQR(Jy)));
                const real_t Jmag      = (real_t)std::sqrt((double)(SQR(Jx) + SQR(Jy) + SQR(Jz)));

                J_sum += (double)Jmag;
                if (Jmag      > J_max)    J_max    = Jmag;
                if (J_inplane > Jphi_max) Jphi_max = J_inplane;
                if (std::fabs((double)Jz) > (double)Jz_max) Jz_max = (real_t)std::fabs((double)Jz);

                //* --- field at the same point ---
                const real_t Bx = init_flds.bx1(xy);
                const real_t By = init_flds.bx2(xy);
                const real_t Bz = init_flds.bx3(xy);
                const real_t Bmag      = (real_t)std::sqrt((double)(SQR(Bx) + SQR(By) + SQR(Bz)));
                const real_t B_inplane = (real_t)std::sqrt((double)(SQR(Bx) + SQR(By)));
                if (B_inplane > Bphi_max) Bphi_max = B_inplane;

                //* --- drift magnitude ---
                const real_t beta_here = amp * Jmag;
                if (beta_here > beta_max) beta_max = beta_here;

                //* --- force-free residual: |J x B| / (|J| |B|) ---
                //*     NOTE: force-free means J x B = 0, i.e. curl(B) = alpha(x) B.
                //*     It does NOT mean J = curl(B); that is Ampere, checked separately.
                if (Jmag > ZERO && Bmag > ZERO)
                {
                    const real_t rx = Jy * Bz - Jz * By;
                    const real_t ry = Jz * Bx - Jx * Bz;
                    const real_t rz = Jx * By - Jy * Bx;
                    const real_t res = (real_t)std::sqrt((double)(SQR(rx) + SQR(ry) + SQR(rz))) / (Jmag * Bmag);
                    if (res > ff_residual) ff_residual = res;
                }

                //* --- Ampere: analytic J vs curl of the initialised field ---
                real_t cx, cy, cz;
                curl_B(x, y, cx, cy, cz);
                const real_t curl_mag = (real_t)std::sqrt((double)(SQR(cx) + SQR(cy) + SQR(cz)));

                curl_sum += (double)curl_mag;
                if (curl_mag > curl_max) curl_max = curl_mag;

                const real_t dev_abs = (real_t)std::sqrt((double)(SQR(Jx - cx) + SQR(Jy - cy) + SQR(Jz - cz)));
                if (dev_abs > amp_dev_absmax) amp_dev_absmax = dev_abs;

                //* --- force-free parameter alpha = |curl B| / |B| ---
                if (Bmag > ZERO)
                {
                    const real_t a = curl_mag / Bmag;
                    alpha_sum += (double)a;
                    if (a > alpha_max) alpha_max = a;
                }
            }

            const real_t inv_n     = (n_samples > 0) ? ONE / (real_t)n_samples : ZERO;
            const real_t J_mean    = (real_t)J_sum     * inv_n;
            const real_t curl_mean = (real_t)curl_sum  * inv_n;
            const real_t alpha_mean = (real_t)alpha_sum * inv_n;
            const real_t ratio_max  = (curl_max  > ZERO) ? J_max  / curl_max  : ZERO;
            const real_t ratio_mean = (curl_mean > ZERO) ? J_mean / curl_mean : ZERO;
            amp_dev_max = (curl_max > ZERO) ? amp_dev_absmax / curl_max : ZERO;

            //* on-axis ratio, evaluated exactly at the tube centre
            real_t amp_ratio_axis = ZERO;
            {
                real_t Jx, Jy, Jz, cx, cy, cz;
                init_flds.current(init_flds.tube1_x_centre, init_flds.tube1_y_centre, Jx, Jy, Jz);
                curl_B(init_flds.tube1_x_centre, init_flds.tube1_y_centre, cx, cy, cz);
                const real_t cm = (real_t)std::sqrt((double)(SQR(cx) + SQR(cy) + SQR(cz)));
                if (cm > ZERO) amp_ratio_axis = (real_t)std::sqrt((double)(SQR(Jx) + SQR(Jy) + SQR(Jz))) / cm;
            }

            //* add the E x B contribution to the peak drift when the tubes are kicked
            if (kick_velocity != ZERO) beta_max += std::fabs((double)kick_velocity);

            const real_t gamma_d_max = ONE / (real_t)std::sqrt(std::max(1.0e-12, 1.0 - (double)SQR(beta_max)));

            //* ---- peak analytic current on the axis -----------------------------
            const real_t Jz_axis   = k_t;                                   //* k_t J0(0) = k_t
            const real_t beta_axis = amp * Jz_axis;
            const real_t Bz_axis   = (real_t)std::sqrt(1.0 + (double)guide_field_floor);

            //* ---- thermal factors ------------------------------------------------
            //* <gamma> for a Maxwell-Juttner distribution (entity_normalisations.md, section 3)
            const real_t gamma_mean = ONE + theta * ((real_t)6.0 + (real_t)15.0 * theta) / ((real_t)4.0 + (real_t)5.0 * theta);
            const real_t u_thermal  = (real_t)std::sqrt(std::max(0.0, (double)(SQR(gamma_mean) - ONE)));

            //* ---- characteristic scales (per species, n_s = 0.5, |q_s| = 1, m_s = 1)
            //*      d_s = d0 / |q_s| * sqrt(m_s / n_s);  the sqrt(2) is the species split
            const real_t d_s_cold  = d0 * (real_t)std::sqrt(2.0);
            const real_t d_s       = d_s_cold * (real_t)std::sqrt((double)gamma_mean);
            const real_t lambda_D  = (real_t)std::sqrt((double)theta) * d_s;
            //! Larmor radius uses u_perp, NOT gamma (entity_normalisations.md, section 3)
            const real_t rho_th    = r0 * u_thermal / Bz_axis;
            const real_t rho_drift = r0 * (beta_axis * ONE / (real_t)std::sqrt(std::max(1.0e-12, 1.0 - (double)SQR(beta_axis)))) / Bz_axis;

            //* ---- magnetisation ---------------------------------------------------
            //*   sigma_s = B^2 sigma0 / (n_s m_s);  sigma_tot = B^2 sigma0 / sum_s n_s m_s
            const real_t B2_axis        = SQR(Bz_axis);                     //* B_phi = 0 on the axis
            const real_t sigma_tot_cold = B2_axis * sigma0 / n_total;
            const real_t sigma_sp_cold  = B2_axis * sigma0 / n_per_sp;
            const real_t sigma_tot_hot  = sigma_tot_cold / (gamma_mean * gamma_d_max);

            //* ---- energetics -------------------------------------------------------
            const real_t U_B         = sigma0 * B2_axis / (real_t)2.0;      //* units of n0 m0 c^2
            const real_t P_th        = n_total * theta;                     //* sum_s n_s theta_s m_s
            const real_t plasma_beta = P_th / U_B;

            //* ---- timing -------------------------------------------------------------
            const real_t runtime = (real_t)params.template get<simtime_t>("simulation.runtime");
            const real_t dt      = params.template get<real_t>("algorithms.timestep.dt");
            const real_t n_steps = (dt > ZERO) ? runtime / dt : ZERO;
            const real_t t_wp_sp = runtime * d0 / d_s_cold;                 //* in cold per-species plasma times

            constexpr int LW1 = 36;
            constexpr int LW2 = 30;
            auto kv1 = [&](const std::string& label) -> std::string { std::ostringstream t; t << "\t"   << std::left << std::setw(LW1) << label << "= "; return t.str(); };
            auto kv2 = [&](const std::string& label) -> std::string { std::ostringstream t; t << "\t\t" << std::left << std::setw(LW2) << label << "= "; return t.str(); };

            std::ostringstream oss;
            oss << "\n"
                << "===========================================================\n"
                << "Force-free Lundquist flux tubes" << (single_tube ? " (SINGLE tube)" : " (colliding pair)") << "\n"
                << "===========================================================\n\n"

                << "GEOMETRY\n"
                << kv1("Dimensionality")                   << (int)D << "D\n"
                << kv1("Resolution")                       << nx1 << " x " << nx2 << "\n"
                << kv1("Box (Lx x Ly) [d0]")               << Lx << " x " << Ly << "\n"
                << kv1("Grid size dx [d0]")                << dx << "\n"
                << kv1("Tube radius R [d0]")               << tube_radius << "\n"
                << kv1("Tube radius R [cells]")            << tube_radius / dx << "\n"
                << kv1("Tube 1 centre")                    << "(" << init_flds.tube1_x_centre << ", " << init_flds.tube1_y_centre << ")\n";
            if (!single_tube)
                oss << kv1("Tube 2 centre")                << "(" << init_flds.tube2_x_centre << ", " << init_flds.tube2_y_centre << ")\n"
                    << kv1("Centre separation [R]")        << (init_flds.tube2_x_centre - init_flds.tube1_x_centre) / tube_radius << "\n";
            oss << kv1("Timestep dt [d0/c]")               << dt << "\n"
                << kv1("Runtime [d0/c]")                   << runtime << "   (" << n_steps << " steps)\n"
                << kv1("Runtime [per-species plasma t]")   << t_wp_sp << "\n"
                << kv1("Runtime [R/c]")                    << runtime / tube_radius << "\n\n"

                << "NORMALISATIONS\n"
                << kv1("skindepth0 d0")                    << d0 << "\n"
                << kv1("larmor0 r0")                       << r0 << "\n"
                << kv1("sigma0 [(d0/r0)^2]")               << sigma0 << "\n"
                << kv1("AMP [d0^2/r0] = J_code/curl(B)")   << amp << "\n"
                << kv1("n total [n0]")                     << n_total << "\n"
                << kv1("n per species [n0]")               << n_per_sp << "\n"
                << kv1("ppc0 per species")                 << HALF * ppc0 << "\n\n"

                << "FIELD / CURRENT (analytic)\n"
                << kv1("alpha_t [1st zero of J1]")         << InitFields<D>::alpha_t << "\n"
                << kv1("k_t = alpha_t/R [1/d0]")           << k_t << "\n"
                << kv1("Guide field floor c")              << guide_field_floor << "\n"
                << kv1("Bz on axis")                       << Bz_axis << "\n"
                << kv1("Bz outside tubes")                 << init_flds.Bz_exterior << "\n"
                << kv1("max |B_in-plane| (= max J1)")      << Bphi_max << "\n"
                << kv1("J_z on axis [curl(B) units]")      << Jz_axis << "\n"
                << kv1("Kick velocity [v/c]")              << kick_velocity << "\n\n"

                << "J vs CURL(B)   [over r < 0.9R of tube 1, " << n_samples << " samples]\n"
                << "\tAnalytic J  [curl(B) units]\n"
                << kv2("max |J|")                          << J_max << "\n"
                << kv2("mean |J|")                         << J_mean << "\n"
                << kv2("max |J_z|")                        << Jz_max << "\n"
                << kv2("max |J_in-plane|")                 << Jphi_max << "\n"
                << "\tcurl(B) of the written field\n"
                << kv2("max |curl B|")                     << curl_max << "\n"
                << kv2("mean |curl B|")                    << curl_mean << "\n"
                << "\tCode current  J_code = AMP * J   (what the drift supplies)\n"
                << kv2("max |J_code|")                     << amp * J_max << "\n"
                << kv2("mean |J_code|")                    << amp * J_mean << "\n"
                << "\tCode current  AMP * curl(B)      (what the field demands)\n"
                << kv2("max |AMP curl B|")                 << amp * curl_max << "\n"
                << kv2("mean |AMP curl B|")                << amp * curl_mean << "\n"
                << "\tForce-free parameter alpha = |curl B|/|B|\n"
                << kv2("max alpha")                        << alpha_max << "\n"
                << kv2("mean alpha")                       << alpha_mean << "\n"
                << kv2("k_t (alpha of the c=0 limit)")     << k_t << "\n\n"

                << "DRIFT\n"
                << kv1("beta on axis [AMP * J_z]")         << beta_axis << "\n"
                << kv1("max |beta| over the tube")         << beta_max << "\n"
                << kv1("gamma_d at max |beta|")            << gamma_d_max << "\n\n"

                << "MAGNETISATION (on axis)\n"
                << kv2("total (cold)")                     << sigma_tot_cold << "\n"
                << kv2("per species (cold)")               << sigma_sp_cold << "\n"
                << kv2("total (hot + drift)")              << sigma_tot_hot << "\n"
                << kv1("Mean Lorentz factor <gamma>")      << gamma_mean << "\n\n"

                << "CHARACTERISTIC SCALES (per species)\n"
                << kv2("Skin depth")                       << d_s       << "  (" << d_s / dx       << " cells)\n"
                << kv2("Debye length")                     << lambda_D  << "  (" << lambda_D / dx  << " cells)\n"
                << kv2("Larmor radius (thermal u_perp)")   << rho_th    << "  (" << rho_th / dx    << " cells)\n"
                << kv2("Larmor radius (drifting beam)")    << rho_drift << "  (" << rho_drift / dx << " cells)\n\n"

                << "ENERGETICS (units of n0 m0 c^2)\n"
                << kv1("Temperature theta = kT/mc^2")      << theta << "\n"
                << kv1("Magnetic energy U_B (axis)")       << U_B << "\n"
                << kv1("Thermal pressure P_th")            << P_th << "\n"
                << kv1("Plasma beta")                      << plasma_beta << "\n\n"

                << "CONSISTENCY CHECKS\n"
                << kv1("max |beta| (must be < 1)")         << beta_max << "\n"
                << kv1("max |JxB|/(|J||B|) (must be 0)")   << ff_residual << "\n"
                << kv1("max|J|/max|curl B| (must be 1)")   << ratio_max << "\n"
                << kv1("mean|J|/mean|curl B| (must be 1)") << ratio_mean << "\n"
                << kv1("J/curl(B) on axis (must be 1)")    << amp_ratio_axis << "\n"
                << kv1("max|J-curl(B)|/max|curl B| (-> 0)") << amp_dev_max << "\n\n";

            if (kick_velocity != ZERO)
                oss << "!! WARNING: kick_velocity != 0 but the charge-density (weight) correction\n"
                    << "!! for div(E) != 0 at the tube edge is NOT implemented in this pgen.\n\n";

            oss << "===========================================================\n";

            //* Write simulation params to a .txt file
            if (mpi_rank == 0)
            {
                std::cout << oss.str() << std::endl;

                const std::string sim_name = params.template get<std::string>("simulation.name");
                std::string path = sim_name;
                if (!path.empty() && path.back() != '/') path += '/';
                path += "Simulation_params.txt";
                std::ofstream fout(path);
                if (fout.is_open()) { fout << oss.str(); fout.close(); }
                else { std::cout << "WARNING: could not write " << path << std::endl; }
            }

            //! ---- HARD FAILURES -------------------------------------------------
            if (mpi_rank == 0)
            {
                //! (a) feasibility: the field demands a drift faster than light
                raise::ErrorIf((double)beta_max >= 0.99,
                            "Flux tube drift >= 0.99c: the field requires more current than the\n"
                            "injected density can carry. beta_peak = AMP * alpha_t / tube_radius,\n"
                            "so raise tube_radius, raise the injected density, or lower sigma\n"
                            "(raise larmor0).", HERE);

                //! (b) the analytic current must be force-free (catches sign / geometry errors
                //!     in the cylindrical -> Cartesian conversion)
                raise::ErrorIf((double)ff_residual > 1.0e-4,
                            "Analytic current is not force-free: |J x B| != 0.\n"
                            "Check InitFields::current().", HERE);

                //! (c) Ampere invariant: the analytic current must BE the curl of the field
                //!     that is actually written to the grid. This is the flux-tube analogue of
                //!     the Harris J_ratio check and is what catches a missing AMP or a missing
                //!     1/dx in any future grid-based current.
                raise::ErrorIf((double)amp_dev_max > 1.0e-2,
                           "Ampere mismatch: the analytic J differs from curl(B) of the\n"
                           "initialised field. The drifting population will not support its own\n"
                           "field and the tubes will not be in equilibrium.\n"
                           "Check InitFields::current() against InitFields::bx1/bx2/bx3.", HERE);
            }
        }                    

        //! =====================================================================
        //!  Initialise particles
        //!
        //!  One uniform injection; the per-species drift is carried entirely by
        //!  FluxTubeDist, so there is no post-injection particle loop and no grid
        //!  current is ever computed.
        //! =====================================================================
        void InitPrtls(Domain<S, M>& local_domain)
        {
            metadomain.CommunicateFields(local_domain, Comm::E | Comm::B);

            //* sign(q) per species; the injector's species 1 / 2 are domain.species[0] / [1]
            const real_t charge_sign_1 = (local_domain.species[0].charge() < ZERO) ? -ONE : ONE;
            const real_t charge_sign_2 = (local_domain.species[1].charge() < ZERO) ? -ONE : ONE;

            const auto edist_1 = FluxTubeDist<M>(local_domain.random_pool(),
                                                 background_temperature, amp,
                                                 charge_sign_1, init_flds);
            const auto edist_2 = FluxTubeDist<M>(local_domain.random_pool(),
                                                 background_temperature, amp,
                                                 charge_sign_2, init_flds);

            const auto sdist = UniformDist<M::Dim>();

            //! total injected density = ONE. The relation beta = AMP * J depends on this.
            arch::InjectNonUniform<S, M, decltype(edist_1), decltype(edist_2), decltype(sdist)>(
                params, local_domain, { 1, 2 }, { edist_1, edist_2 }, sdist, ONE);
        }

        //! =====================================================================
        //!  One-shot current calibration (step 0 only).
        //!
        //!  Prints, at a handful of cells along +x from the tube-1 axis:
        //!      Jz_analytic  = InitFields::current()           [curl(B) units]
        //!      cur_raw      = CUR(i,j,cur::jx3)               [whatever units Entity uses]
        //!      curl_z       = the monitor's finite-difference stencil on bx1/bx2
        //!
        //!  Read the ratios:
        //!      Jz_analytic / curl_z        -> 1 means the STENCIL is right
        //!      cur_raw * inv_amp / Jz_an   -> 1 means CUR's UNITS are as assumed
        //!  Whichever one is 0.5 is where the factor of 2 lives.
        //! =====================================================================
        void calibrate_current(Domain<S, M>& domain)
        {
            constexpr int NP = 8;                       //* probes at r = 0.1R .. 0.8R

            const auto&  EM      = domain.fields.em;
            const auto&  CUR     = domain.fields.cur;
            const auto   metric  = domain.mesh.metric;  //! by VALUE: device lambda
            const auto   flds    = init_flds;           //! by VALUE: device lambda
            const real_t dx      = init_flds.cell_width_x;
            const real_t dy      = init_flds.cell_width_y;
            const real_t inv_amp = ONE / amp;
            const real_t cx      = init_flds.tube1_x_centre;
            const real_t cy      = init_flds.tube1_y_centre;
            const real_t R       = tube_radius;

            //* probe radii, device-side
            array_t<real_t*> probe_r { "probe_r", NP };
            auto probe_r_h = Kokkos::create_mirror_view(probe_r);
            for (int p = 0; p < NP; ++p) {
                probe_r_h(p) = (static_cast<real_t>(0.1) * static_cast<real_t>(p + 1)) * R;
            }
            Kokkos::deep_copy(probe_r, probe_r_h);

            //* 5 accumulator slots per probe: Jz_an, cur_raw, curl_z, r/R, hits.
            //! Slots a rank does not own stay 0, so a plain MPI_SUM gathers them.
            array_t<double**> acc { "probe_acc", NP, 5 };
            Kokkos::deep_copy(acc, 0.0);

            Kokkos::parallel_for(
                "CurrentCalib",
                domain.mesh.rangeActiveCells(),
                Lambda(cellidx_t i, cellidx_t j)
                {
                    //! CHECK: rangeActiveCells() indices include the ghost offset, so the
                    //! coordinate index is i - N_GHOSTS. If your Entity rev's range already
                    //! excludes ghosts, drop the subtraction - a wrong offset here shifts
                    //! every probe and the ratios come out meaningless rather than wrong.
                    const real_t x_Cd = static_cast<real_t>(i) - static_cast<real_t>(N_GHOSTS);
                    const real_t y_Cd = static_cast<real_t>(j) - static_cast<real_t>(N_GHOSTS);
                    const real_t x = metric.template convert<1, Crd::Cd, Crd::XYZ>(x_Cd);
                    const real_t y = metric.template convert<2, Crd::Cd, Crd::XYZ>(y_Cd);

                    for (int p = 0; p < NP; ++p)
                    {
                        //* does this cell sit on probe p? (half-cell window, +x radius)
                        if (math::abs(x - (cx + probe_r(p))) >= HALF * dx) { continue; }
                        if (math::abs(y - cy)                >= HALF * dy) { continue; }

                        real_t Jx, Jy, Jz;
                        flds.current(x, y, Jx, Jy, Jz);

                        const real_t cz = (EM(i + 1, j, em::bx2) - EM(i, j, em::bx2)) / dx
                                        - (EM(i, j + 1, em::bx1) - EM(i, j, em::bx1)) / dy;

                        //! atomic: cheap here, and guards against two cells matching one probe
                        Kokkos::atomic_add(&acc(p, 0), (double)Jz);
                        Kokkos::atomic_add(&acc(p, 1), (double)CUR(i, j, cur::jx3));
                        Kokkos::atomic_add(&acc(p, 2), (double)cz);
                        Kokkos::atomic_add(&acc(p, 3),
                                           (double)(math::sqrt(SQR(x - cx) + SQR(y - cy)) / R));
                        Kokkos::atomic_add(&acc(p, 4), 1.0);
                    }
                });

            auto acc_h = Kokkos::create_mirror_view(acc);
            Kokkos::deep_copy(acc_h, acc);

            double flat[NP * 5];
            for (int p = 0; p < NP; ++p) {
                for (int k = 0; k < 5; ++k) { flat[p * 5 + k] = acc_h(p, k); }
            }

            int mpi_rank = 0;
            #if defined(MPI_ENABLED)
                MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
                double gathered[NP * 5];
                MPI_Reduce(flat, gathered, NP * 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                for (int k = 0; k < NP * 5; ++k) { flat[k] = gathered[k]; }
            #endif

            if (mpi_rank != 0) { return; }

            std::string path = monitor_path;
            const auto  pos  = path.rfind('/');
            path = (pos == std::string::npos) ? std::string("Current_calibration.txt")
                                              : path.substr(0, pos + 1) + "Current_calibration.txt";

            std::ofstream fout(path);
            if (!fout.is_open()) { return; }

            fout << "# One-shot current calibration at step 0, along +x from tube 1 axis.\n"
                 << "# AMP = " << amp << "   inv_amp = " << inv_amp << "\n"
                 << "#\n"
                 << "#  an_over_curl : Jz_analytic / curl_z          -> 1 if the STENCIL is right\n"
                 << "#  dep_over_an  : cur_raw*inv_amp / Jz_analytic -> 1 if CUR's UNITS are right\n"
                 << "#  Whichever is ~0.5 is where the factor of 2 lives.\n"
                 << "#  (dep_over_an is noisy per cell at 16 ppc - read the trend over probes,\n"
                 << "#   not any single row.)\n"
                 << "#\n"
                 << "#" << std::setw(7)  << "probe"
                 << std::setw(10) << "hits"
                 << std::setw(14) << "r/R"
                 << std::setw(16) << "Jz_analytic"
                 << std::setw(16) << "cur_raw"
                 << std::setw(16) << "cur*inv_amp"
                 << std::setw(16) << "curl_z"
                 << std::setw(16) << "an_over_curl"
                 << std::setw(16) << "dep_over_an"
                 << "\n";

            for (int p = 0; p < NP; ++p)
            {
                const double n = flat[p * 5 + 4];
                if (n < 0.5) {              //* no rank owned a cell on this probe
                    fout << std::setw(8) << p << std::setw(10) << 0 << "   (not found)\n";
                    continue;
                }
                const double Jz_an  = flat[p * 5 + 0] / n;
                const double raw    = flat[p * 5 + 1] / n;
                const double curl_z = flat[p * 5 + 2] / n;
                const double rR     = flat[p * 5 + 3] / n;
                const double dep    = raw * (double)inv_amp;

                fout << std::setw(8)  << p
                     << std::setw(10) << (long)n
                     << std::setw(14) << std::fixed      << std::setprecision(4) << rR
                     << std::setw(16) << std::scientific << std::setprecision(6) << Jz_an
                     << std::setw(16) << raw
                     << std::setw(16) << dep
                     << std::setw(16) << curl_z
                     << std::setw(16) << ((curl_z != 0.0) ? Jz_an / curl_z : 0.0)
                     << std::setw(16) << ((Jz_an  != 0.0) ? dep    / Jz_an : 0.0)
                     << "\n";
            }
            fout.close();
        }

        //! =====================================================================
        //!  Runtime force-free / Ampere monitor
        //! =====================================================================

        //!  Tests what print_setup() cannot: whether the current the PARTICLES
        //!  actually deposit is (a) force-free against the live field and
        //!  (b) still equal to curl(B) of the live field.
        //!
        //!  Both residuals are built from SUMS over the box, never from a
        //!  per-cell max. At 16 ppc/species the deposited current in any single
        //!  cell is dominated by shot noise, so a max-based residual measures
        //!  the noise floor and nothing else (it sits flat at ~0.9 all run).
        //!  Worse, outside the tubes B is purely z, so any cell with J_z == 0
        //!  and J_x,J_y != 0 gives |JxB|/(|J||B|) == 1 exactly, pinning a max
        //!  at 1.000000 forever. Sums let the zero-mean noise cancel.
        //!
        //!  Columns:
        //!    e2_over_b2  <|E|^2>/<|B|^2>. Exponential rise => instability;
        //!                plateau => noise at its thermal level.
        //!    ff_weighted sum|JxB|/|B| over sum|J|.  Force-free  <=> 0.
        //!    amp_proj    sum(J_z curl_z)/sum(curl_z^2). Ampere  <=> 1.
        //!
        //!  Writes one row per sample to <simulation.name>/Force_free_monitor.txt.
        //!  Nothing goes to stdout, so the job's .out file stays clean.
        //! =====================================================================
        void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain)
        {
            if (monitor_interval <= 0)                         { return; }
            if (static_cast<int>(step) % monitor_interval != 0) { return; }

            if (step == 0) { calibrate_current(domain); }

            const auto& EM  = domain.fields.em;
            const auto& CUR = domain.fields.cur;            //! deposited J, code units
            const real_t dx      = init_flds.cell_width_x;
            const real_t dy      = init_flds.cell_width_y;
            const real_t inv_amp = ONE / amp;               //* J_code = AMP * curl(B)

            //! All accumulators are double even though real_t is float: summing
            //! ~1.6e6 terms of order 1e-10 into a float accumulator loses most of
            //! the result to truncation. fp64 is full-rate on MI250X and this runs
            //! once every monitor_interval steps, so the cost is irrelevant.
            double sum_e2 = 0.0, sum_b2 = 0.0;
            double sum_jxb = 0.0, sum_j = 0.0;              //* force-free, |J|-weighted
            double sum_jz_cz = 0.0, sum_cz2 = 0.0;          //* Ampere, projection form

            //* ---- pass 1: the six sums ---------------------------------------
            Kokkos::parallel_reduce(
                "FFMonitor_sums",
                domain.mesh.rangeActiveCells(),             //! CHECK name in your Entity rev
                Lambda(cellidx_t i, cellidx_t j,
                       double& e2, double& b2, double& jxb, double& jj,
                       double& jzcz, double& cz2)
                {
                    const real_t Ex = EM(i, j, em::ex1);
                    const real_t Ey = EM(i, j, em::ex2);
                    const real_t Ez = EM(i, j, em::ex3);
                    const real_t Bx = EM(i, j, em::bx1);
                    const real_t By = EM(i, j, em::bx2);
                    const real_t Bz = EM(i, j, em::bx3);

                    e2 += (double)(SQR(Ex) + SQR(Ey) + SQR(Ez));
                    b2 += (double)(SQR(Bx) + SQR(By) + SQR(Bz));

                    //* deposited current, converted out of code units so that it
                    //* is directly comparable with curl(B)
                    const real_t Jx = CUR(i, j, cur::jx1) * inv_amp;
                    const real_t Jy = CUR(i, j, cur::jx2) * inv_amp;
                    const real_t Jz = CUR(i, j, cur::jx3) * inv_amp;

                    //* (a) force-free residual, |J|-WEIGHTED.
                    //!     sum|JxB|/|B| against sum|J| means a vacuum cell, where J is
                    //!     pure noise, contributes in proportion to its tiny |J| rather
                    //!     than dominating the statistic.
                    const real_t Bm = math::sqrt(SQR(Bx) + SQR(By) + SQR(Bz));
                    if (Bm > ZERO)
                    {
                        const real_t rx = Jy * Bz - Jz * By;
                        const real_t ry = Jz * Bx - Jx * Bz;
                        const real_t rz = Jx * By - Jy * Bx;
                        jxb += (double)(math::sqrt(SQR(rx) + SQR(ry) + SQR(rz)) / Bm);
                    }
                    jj += (double)math::sqrt(SQR(Jx) + SQR(Jy) + SQR(Jz));

                    //* (b) Ampere as a PROJECTION of J_z onto curl(B)_z.
                    const real_t cz = (EM(i + 1, j, em::bx2) - EM(i, j, em::bx2)) - (EM(i, j + 1, em::bx1) - EM(i, j, em::bx1));
                    
                    jzcz += (double)(Jz * cz);
                    cz2  += (double)SQR(cz);
                },
                sum_e2, sum_b2, sum_jxb, sum_j, sum_jz_cz, sum_cz2);

            //* ---- pass 2: peak |E| (the only max worth keeping) ---------------
            real_t max_E = ZERO;
            Kokkos::parallel_reduce(
                "FFMonitor_maxE",
                domain.mesh.rangeActiveCells(),
                Lambda(cellidx_t i, cellidx_t j, real_t& m_E)
                {
                    const real_t Em = math::sqrt(SQR(EM(i, j, em::ex1))
                                               + SQR(EM(i, j, em::ex2))
                                               + SQR(EM(i, j, em::ex3)));
                    if (Em > m_E) { m_E = Em; }
                },
                Kokkos::Max<real_t>(max_E));

            //* cell count: taken from the mesh, not reduced (one fewer reducer)
            long n_cells = (long)domain.mesh.n_active(in::x1)
                         * (long)domain.mesh.n_active(in::x2);

            //* ---- global reduction --------------------------------------------
            int mpi_rank = 0;
            #if defined(MPI_ENABLED)
                MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

                double snd[6] = { sum_e2, sum_b2, sum_jxb, sum_j, sum_jz_cz, sum_cz2 };
                double rcv[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
                MPI_Reduce(snd, rcv, 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                sum_e2 = rcv[0]; sum_b2    = rcv[1]; sum_jxb = rcv[2];
                sum_j  = rcv[3]; sum_jz_cz = rcv[4]; sum_cz2 = rcv[5];

                long snd_n = n_cells, rcv_n = 0;
                MPI_Reduce(&snd_n, &rcv_n, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
                n_cells = rcv_n;

                //! promoted to double so the printed peak is readable in a float build
                double snd_max = (double)max_E, rcv_max = 0.0;
                MPI_Reduce(&snd_max, &rcv_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                max_E = (real_t)rcv_max;
            #endif

            if (mpi_rank != 0) { return; }

            //* ---- derived diagnostics ------------------------------------------
            const double inv_n      = (n_cells > 0) ? 1.0 / (double)n_cells : 0.0;
            const double mean_e2    = sum_e2 * inv_n;
            const double mean_b2    = sum_b2 * inv_n;
            const double e2_over_b2 = (mean_b2  > 0.0) ? mean_e2   / mean_b2 : 0.0;
            const double ff_weighted = (sum_j   > 0.0) ? sum_jxb   / sum_j   : 0.0;  //* -> 0
            const double amp_proj    = (sum_cz2 > 0.0) ? sum_jz_cz / sum_cz2 : 0.0;  //* -> 1
            const double t_wpe       = (double)time;                 //* d0 = 1 => t c/d0 = t w_pe
            const double t_over_Rc   = (double)time / (double)tube_radius;

            //* ---- append one row -------------------------------------------------
            std::ofstream fout(monitor_path, std::ios::app);
            if (!fout.is_open()) { return; }    //! silent: never write to stdout mid-run

            if (!monitor_header_written)
            {
                fout << "# Force-free / Ampere runtime monitor\n"
                     << "# e2_over_b2  : <|E|^2>/<|B|^2>, box mean. Exponential growth => instability;\n"
                     << "#               plateau => shot noise at its thermal level.\n"
                     << "# ff_weighted : sum|J_dep x B|/|B|  /  sum|J_dep|.  Force-free <=> 0.\n"
                     << "# amp_proj    : sum(J_dep,z curl_z) / sum(curl_z^2). Ampere    <=> 1.\n"
                     << "#               Staggering offset between cur::jx3 and curl_z puts the\n"
                     << "#               baseline a few % off 1 - watch for DRIFT, not the value.\n"
                     << "#\n"
                     << "#" << std::setw(11) << "step"
                     << std::setw(16) << "t[d0/c]"
                     << std::setw(16) << "t[R/c]"
                     << std::setw(16) << "mean_E2"
                     << std::setw(16) << "mean_B2"
                     << std::setw(16) << "e2_over_b2"
                     << std::setw(16) << "max_E"
                     << std::setw(16) << "ff_weighted"
                     << std::setw(16) << "amp_proj"
                     << "\n";
                monitor_header_written = true;
            }

            fout << std::setw(12) << (long)step
                 << std::setw(16) << std::scientific << std::setprecision(6) << t_wpe
                 << std::setw(16) << t_over_Rc
                 << std::setw(16) << mean_e2
                 << std::setw(16) << mean_b2
                 << std::setw(16) << e2_over_b2
                 << std::setw(16) << (double)max_E
                 << std::setw(16) << ff_weighted
                 << std::setw(16) << amp_proj
                 << "\n";
            fout.close();           //! flush every sample: the file survives a crash
        }

    }; // struct PGen

} // namespace user

#endif // PROBLEM_GENERATOR_H