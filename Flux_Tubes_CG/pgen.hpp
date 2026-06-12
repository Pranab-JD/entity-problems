#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

/**
 * @file fluxtube.hpp
 * @brief Force-free Lundquist flux-tube collision — Entity port of Tristan user_sheet_new.F90
 *
 * Physics setup
 * -------------
 * Two Lundquist tubes (Bessel-function force-free equilibria) are initialised
 * side-by-side along X (tangent at the domain centre) and given opposite
 * motional E = -v x B kicks so that they approach each other along X (head-on)
 * and reconnect.
 *
 *   Tube 1 = LEFT  (kicked +x, toward centre)
 *   Tube 2 = RIGHT (kicked -x, toward centre)
 *
 * Field definitions (Tristan user_sheet_new.F90, userInitFields)
 * --------------------------------------------------------------
 *   Az(r) = (tube_radius / alpha_t) * J0(alpha_t * r/tube_radius)   r <  tube_radius
 *   Az(r) = (tube_radius / alpha_t) * J0(alpha_t)                   r >= tube_radius
 *
 *   Bx =  dAz/dy          (Tristan: discrete per-cell difference of Az)
 *   By = -dAz/dx
 *   Bz =  sqrt( J0(alpha_t * r/tube_radius)^2 + guide_field_floor )
 *
 *   Motional E for an X-directed kick, E = -v x B with v = v_kick * x_hat:
 *     x_hat x B = (0, Bz, -By)  =>  E = -v x B = (0, -v*Bz, +v*By)
 *     tube 1 (v = +kick*x_hat, toward centre):  Ex = 0,  Ey = +kick*Bz,  Ez = -kick*By
 *     tube 2 (v = -kick*x_hat, toward centre):  Ex = 0,  Ey = -kick*Bz,  Ez = +kick*By
 *     (E = -v x B; signs chosen so ExB drift CONVERGES the tubes.)
 *     E = 0 outside both tubes.
 *   (NOTE: this differs from the previous top-bottom/Y-kick layout, where the
 *    nonzero motional components were Ex and Ez with By replaced by Bx.)
 *
 *   Force-free current J = curl(B), written analytically (cylindrical):
 *     Jz   = (alpha_t / tube_radius) * J0(alpha_t * r/tube_radius)
 *     Jphi = (alpha_t / tube_radius) * J0 * J1 / Bz        ( = -dBz/dr )
 *   (J is radially symmetric and unchanged by the layout; only the centres
 *    move and only the kick direction changes.)
 *
 * Current/drift normalisation  (THE key cross-code conversion)
 * ------------------------------------------------------------
 *     drift_velocity = J * sqrt(sigma) * skindepth * sign(q)   [density param = 1]
 *   (No factor 1/2 — calibrated against Tristan, 1.hpp, 2.hpp. See manual #1.)
 *
 * Particle initialisation — mirrors Tristan userInitParticles
 * -----------------------------------------------------------
 *  1. Uniform thermal Maxwellian everywhere.
 *  2. drift = sign(q) * sqrt(sigma) * skindepth * J(analytic)
 *           + ExB term. For an X-kick the DRIVEN drift is along x:
 *             drift_x += (E x B)_x / (By^2 + Bz^2)
 *                      = (Ey*Bz - Ez*By) / (By^2 + Bz^2)
 *     (Tristan-style denominator: the two in-plane-perp + guide components,
 *      i.e. By^2 + Bz^2 here, mirroring Bx^2 + Bz^2 in the Y-kick layout.
 *      >>> MANUAL CHECK #10 — this denominator is by symmetry, not a Tristan
 *      reference that runs tubes along X. <<<)
 *  3. Lorentz boost of the thermal momentum into the drift frame.
 *     (Probabilistic reflection intentionally OMITTED — manual #4.)
 *  4. [charge_correction] multiplicative weight perturbation
 *     (user_sheet_new.F90 form):
 *        w_new = w * (1 + rho0 * sqrt(sigma) * skindepth * sign(q))
 *     For an X-kick Ex = 0 and Ey is the nonzero in-plane E, so
 *        rho0 = div(E) = dEy/dy   (per-cell difference of Ey in y).
 *     No-op while kick_velocity = 0.
 *
 * Input parameters (setup.*)
 * --------------------------
 *   background_temperature    thermal temperature (m_e c^2)        [default 1e-2]
 *   kick_velocity             tube approach velocity / c           [default 0.1]
 *   guide_field_floor         floor keeping Bz > 0 everywhere      [default 0.01]
 *   charge_correction         weight correction for Gauss's law    [default true]
 *   single_tube               initialise only tube 1               [default false]
 *   tube_radius               tube radius (code units)             [default 0.99*Lx/4]
 *
 * =====================  MANUAL CHECKS REQUIRED  =====================
 *  #1 Factor in drift normalisation (sqrt(sigma)*skindepth, no /2):
 *     single_tube=true, kick_velocity=0, output J at step 0-1, compare Jz
 *     along the tube diameter with (alpha_t/R)*J0(alpha_t*r/R).
 *     Ratio 1 -> correct; ratio 0.5 -> restore /2.
 *  #2 Charge-correction factor sqrt(sigma)*skindepth is Tristan-units verbatim;
 *     re-derive in Entity units before kick != 0 runs.
 *  #3 rho0 is a PER-CELL difference (Tristan convention, cell = 1).
 *     Valid only if 1 grid cell = 1 code-length unit in the toml.
 *  #4 Reflection removed: for |drift| >~ 0.3 the boosted Maxwellian is biased.
 *  #5 Uniform injection (no buffer zones).
 *  #6 No field smoothing implemented here.
 *  #7 Assumes species charge() = +1 / -1 exactly.
 *  #8 absorbing_bnd block of user_sheet_new.F90 NOT ported.
 *  #9 Compile check against your Entity version.
 *  #10 X-KICK GEOMETRY (new): verify at t=0 that tube 1 (left) drifts +x and
 *      tube 2 (right) drifts -x (head-on). The ExB denominator By^2 + Bz^2 is
 *      a symmetry argument, not Tristan-sourced — confirm with the same
 *      force-free / drift-direction check as #1.
 *  #11 Tube-overlap assumption: the ex3 / ExB / charge-correction "else =>
 *      tube 2" fall-through is valid only because the tube interiors are
 *      disjoint (centres 2*tube_radius apart). If you move them closer, add an
 *      explicit tube-2 membership test.
 * ====================================================================
 */

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/utils.h"
#include "archetypes/field_setter.h"
#include "framework/domain/metadomain.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"

#include <utility>
#include <algorithm>

//! ---------------------------------------------------------------------------
//! GPU-compatible Bessel functions.
//! CUDA/HIP provide device intrinsics j0/j1; on CPU we use std::cyl_bessel_j.
//! ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION real_t bessel_J0(real_t argument)
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    return j0(argument);
#else
    return std::cyl_bessel_j(0, argument);
#endif
}

KOKKOS_INLINE_FUNCTION real_t bessel_J1(real_t argument)
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
    //!  E and B are initialised analytically.
    //!  entity calls each component at its own Yee-stagger position
    //!
    //!    normalised_radius = |x - centre| / tube_radius
    //!
    //!    Az  = (tube_radius/alpha_t) * J0(alpha_t * min(normalised_radius, 1))
    //!    Bz  = sqrt( J0(alpha_t*normalised_radius)^2 + guide_field_floor )   inside
    //!    Bz  = sqrt( J0(alpha_t)^2                   + guide_field_floor )   outside
    //!
    //! =========================================================================
    template <Dimension D>
    struct InitFields
    {
        // alpha_t: FIRST ZERO OF J1 (= 3.8317...). The tube boundary condition
        // B_phi(tube_radius) = J1(alpha_t) = 0 (no surface current) fixes this.
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
            // Exterior Az: constant -> contributes zero in-plane field
            Az_exterior = (tube_radius / alpha_t) * bessel_J0(alpha_t);
        }

        //!  Bx = dAz/dy   (unchanged by layout — radially symmetric)
        KOKKOS_INLINE_FUNCTION
        real_t bx1(const coord_t<D>& x_Ph) const
        {
            const real_t x = x_Ph[0];
            const real_t y = x_Ph[1];

            // Az at (x, y + cell_width_y/2)
            real_t Az_upper;
            {
                const real_t y_q = y + HALF * cell_width_y;
                const real_t normalised_radius_tube1 = math::sqrt(SQR(x - tube1_x_centre) + SQR(y_q - tube1_y_centre)) / tube_radius;
                const real_t normalised_radius_tube2 = math::sqrt(SQR(x - tube2_x_centre) + SQR(y_q - tube2_y_centre)) / tube_radius;
                
                if (normalised_radius_tube1 < ONE)
                    Az_upper = (tube_radius / alpha_t) * bessel_J0(alpha_t * normalised_radius_tube1);
                else if (!single_tube && normalised_radius_tube2 < ONE)
                    Az_upper = (tube_radius / alpha_t) * bessel_J0(alpha_t * normalised_radius_tube2);
                else
                    Az_upper = Az_exterior;
            }

            // Az at (x, y - cell_width_y/2)
            real_t Az_lower;
            {
                const real_t y_q = y - HALF * cell_width_y;
                const real_t normalised_radius_tube1 = math::sqrt(SQR(x - tube1_x_centre) + SQR(y_q - tube1_y_centre)) / tube_radius;
                const real_t normalised_radius_tube2 = math::sqrt(SQR(x - tube2_x_centre) + SQR(y_q - tube2_y_centre)) / tube_radius;
                
                if (normalised_radius_tube1 < ONE)
                    Az_lower = (tube_radius / alpha_t) * bessel_J0(alpha_t * normalised_radius_tube1);
                else if (!single_tube && normalised_radius_tube2 < ONE)
                    Az_lower = (tube_radius / alpha_t) * bessel_J0(alpha_t * normalised_radius_tube2);
                else
                    Az_lower = Az_exterior;
            }

            return (Az_upper - Az_lower) / cell_width_y;
        }

        //!  By = -dAz/dx   (unchanged by layout — radially symmetric)
        KOKKOS_INLINE_FUNCTION
        real_t bx2(const coord_t<D>& x_Ph) const
        {
            const real_t x = x_Ph[0];
            const real_t y = x_Ph[1];

            //* Az(x - dx/2, y)
            real_t Az_left;
            {
                const real_t x_q = x - HALF * cell_width_x;
                const real_t normalised_radius_tube1 = math::sqrt(SQR(x_q - tube1_x_centre) + SQR(y - tube1_y_centre)) / tube_radius;
                const real_t normalised_radius_tube2 = math::sqrt(SQR(x_q - tube2_x_centre) + SQR(y - tube2_y_centre)) / tube_radius;
                
                if (normalised_radius_tube1 < ONE)
                    Az_left = (tube_radius / alpha_t) * bessel_J0(alpha_t * normalised_radius_tube1);
                else if (!single_tube && normalised_radius_tube2 < ONE)
                    Az_left = (tube_radius / alpha_t) * bessel_J0(alpha_t * normalised_radius_tube2);
                else
                    Az_left = Az_exterior;
            }

            //* Az(x + dx/2, y)
            real_t Az_right;
            {
                const real_t x_q = x + HALF * cell_width_x;
                const real_t normalised_radius_tube1 = math::sqrt(SQR(x_q - tube1_x_centre) + SQR(y - tube1_y_centre)) / tube_radius;
                const real_t normalised_radius_tube2 = math::sqrt(SQR(x_q - tube2_x_centre) + SQR(y - tube2_y_centre)) / tube_radius;
                
                if (normalised_radius_tube1 < ONE)
                    Az_right = (tube_radius / alpha_t) * bessel_J0(alpha_t * normalised_radius_tube1);
                else if (!single_tube && normalised_radius_tube2 < ONE)
                    Az_right = (tube_radius / alpha_t) * bessel_J0(alpha_t * normalised_radius_tube2);
                else
                    Az_right = Az_exterior;
            }

            return (Az_left - Az_right) / cell_width_x;
        }

        //! Bz (Guide field)   (unchanged by layout)
        //!  Inside tubes  : Bz = sqrt( J0(alpha_t*rn)^2 + guide_field_floor )
        //!  Outside tubes : Bz = sqrt( J0(alpha_t)^2    + guide_field_floor )
        KOKKOS_INLINE_FUNCTION
        real_t bx3(const coord_t<D>& x_Ph) const
        {
            const real_t normalised_radius_tube1 = math::sqrt(SQR(x_Ph[0] - tube1_x_centre) + SQR(x_Ph[1] - tube1_y_centre)) / tube_radius;
            
            if (normalised_radius_tube1 < ONE)
                return math::sqrt(SQR(bessel_J0(alpha_t * normalised_radius_tube1)) + guide_field_floor);

            if (!single_tube) 
            {
                const real_t normalised_radius_tube2 =
                    math::sqrt(SQR(x_Ph[0] - tube2_x_centre) + SQR(x_Ph[1] - tube2_y_centre)) / tube_radius;
                if (normalised_radius_tube2 < ONE)
                    return math::sqrt(SQR(bessel_J0(alpha_t * normalised_radius_tube2)) + guide_field_floor);
            }
            
            return math::sqrt(SQR(bessel_J0(alpha_t)) + guide_field_floor);
        }

        //! Ex = -v x B |_x = 0 for an X-directed kick.
        KOKKOS_INLINE_FUNCTION
        real_t ex1(const coord_t<D>&) const { return ZERO; }

        //! Ey = -v x B |_y.
        //? Tube 1 (left,  +x kick): Ey = -kick_velocity * Bz
        //? Tube 2 (right, -x kick): Ey = +kick_velocity * Bz
        //? Outside both tubes: 0
        KOKKOS_INLINE_FUNCTION
        real_t ex2(const coord_t<D>& x_Ph) const
        {
            if (kick_velocity == ZERO) return ZERO;

            const real_t normalised_radius_tube1 = math::sqrt(SQR(x_Ph[0] - tube1_x_centre) + SQR(x_Ph[1] - tube1_y_centre)) / tube_radius;
            
            if (normalised_radius_tube1 < ONE)
                return +kick_velocity * math::sqrt(SQR(bessel_J0(alpha_t * normalised_radius_tube1)) + guide_field_floor);

            if (!single_tube) 
            {
                const real_t normalised_radius_tube2 = math::sqrt(SQR(x_Ph[0] - tube2_x_centre) + SQR(x_Ph[1] - tube2_y_centre)) / tube_radius;
                
                if (normalised_radius_tube2 < ONE)
                    return -kick_velocity * math::sqrt(SQR(bessel_J0(alpha_t * normalised_radius_tube2)) + guide_field_floor);
            }

            return ZERO;    // Outside of flux tubes
        }

        //! Ez = -v x B |_z.
        //? Tube 1 (left): Ez = +kick_velocity * By;   tube 2 (right): Ez = -kick_velocity * By.
        //? By = -dAz/dx, reconstructed as (Az(x-dx/2) - Az(x+dx/2))/dx  [mirrors bx2].
        KOKKOS_INLINE_FUNCTION
        real_t ex3(const coord_t<D>& x_Ph) const
        {
            if (kick_velocity == ZERO) return ZERO;

            const real_t x = x_Ph[0];
            const real_t y = x_Ph[1];

            const real_t normalised_radius_tube1 = math::sqrt(SQR(x - tube1_x_centre) + SQR(y - tube1_y_centre)) / tube_radius;
            real_t normalised_radius_tube2 = static_cast<real_t>(2.0);   // >=1: tube 2 absent (single_tube mode)
            if (!single_tube)
                normalised_radius_tube2 = math::sqrt(SQR(x - tube2_x_centre) + SQR(y - tube2_y_centre)) / tube_radius;

            if (normalised_radius_tube1 >= ONE && normalised_radius_tube2 >= ONE)
                return ZERO;

            //* Az(x - dx/2, y)
            real_t Az_left;
            {
                const real_t x_q = x - HALF * cell_width_x;
                const real_t rn1 = math::sqrt(SQR(x_q - tube1_x_centre) + SQR(y - tube1_y_centre)) / tube_radius;
                const real_t rn2 = math::sqrt(SQR(x_q - tube2_x_centre) + SQR(y - tube2_y_centre)) / tube_radius;
                
                if (rn1 < ONE)
                    Az_left = (tube_radius / alpha_t) * bessel_J0(alpha_t * rn1);
                else if (!single_tube && rn2 < ONE)
                    Az_left = (tube_radius / alpha_t) * bessel_J0(alpha_t * rn2);
                else
                    Az_left = Az_exterior;
            }

            //* Az(x + dx/2, y)
            real_t Az_right;
            {
                const real_t x_q = x + HALF * cell_width_x;
                const real_t rn1 = math::sqrt(SQR(x_q - tube1_x_centre) + SQR(y - tube1_y_centre)) / tube_radius;
                const real_t rn2 = math::sqrt(SQR(x_q - tube2_x_centre) + SQR(y - tube2_y_centre)) / tube_radius;
                
                if (rn1 < ONE)
                    Az_right = (tube_radius / alpha_t) * bessel_J0(alpha_t * rn1);
                else if (!single_tube && rn2 < ONE)
                    Az_right = (tube_radius / alpha_t) * bessel_J0(alpha_t * rn2);
                else
                    Az_right = Az_exterior;
            }

            //* By = -dAz/dx = (Az(x - dx/2) - Az(x + dx/2)) / dx
            const real_t By = (Az_left - Az_right) / cell_width_x;

            if (normalised_radius_tube1 < ONE)
                return -kick_velocity * By;      // tube 1 (left)
            return +kick_velocity * By;          // tube 2 (right)
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
        real_t Az_exterior       { ZERO  };
        bool   single_tube       { false };
    };


    //! =========================================================================
    //!  PGen
    //! =========================================================================
    template <SimEngine::type S, class M>
    struct PGen : public arch::ProblemGenerator<S, M>
    {
        static constexpr auto engines    { traits::compatible_with<SimEngine::SRPIC>::value };
        static constexpr auto metrics    { traits::compatible_with<Metric::Minkowski>::value };
        static constexpr auto dimensions { traits::compatible_with<Dim::_2D, Dim::_3D>::value };

        using Base            = arch::ProblemGenerator<S, M>;
        using metadomain_type = Metadomain<S, M>;

        using Base::D;
        using Base::C;
        using Base::params;

        metadomain_type& global_domain;

    private:
        real_t background_temperature  { static_cast<real_t>(0.01) };
        real_t kick_velocity           { static_cast<real_t>(0.01) };
        real_t guide_field_floor       { static_cast<real_t>(0.01) };
        bool   charge_correction       { true };
        bool   single_tube             { false };

    public:
        InitFields<D> init_flds;

        //!  Constructor
        inline PGen(const SimulationParams& p, metadomain_type& md): Base { p }, global_domain { md }
        {
            background_temperature  = p.template get<real_t>("setup.background_temperature", static_cast<real_t>(0.01));
            kick_velocity           = p.template get<real_t>("setup.kick_velocity",          static_cast<real_t>(0.01));
            guide_field_floor       = p.template get<real_t>("setup.guide_field_floor",      static_cast<real_t>(0.01));
            charge_correction       = p.template get<bool>  ("setup.charge_correction",      true);
            single_tube             = p.template get<bool>  ("setup.single_tube",            false);

            const auto& mesh = md.mesh();
            const real_t global_x_min = mesh.extent(in::x1).first;
            const real_t global_x_max = mesh.extent(in::x1).second;
            const real_t global_y_min = mesh.extent(in::x2).first;
            const real_t global_y_max = mesh.extent(in::x2).second;
            const real_t Lx = global_x_max - global_x_min;
            const real_t Ly = global_y_max - global_y_min;

            // Tube radius default = 0.99 * Lx/4 (slightly less than 1/4 to move tube edges away from boundaries).
            const real_t tube_radius = p.template get<real_t>("setup.tube_radius", Lx * static_cast<real_t>(0.99 * 0.25));

            // Tubes SIDE BY SIDE ALONG X (same y centre, offset in x):
            //   tube 1 (left)  centre = (Lx/2 - tube_radius, Ly/2)
            //   tube 2 (right) centre = (Lx/2 + tube_radius, Ly/2)
            const real_t domain_x_centre = global_x_min + HALF * Lx;
            const real_t domain_y_centre = global_y_min + HALF * Ly;

            const real_t dx = Lx / static_cast<real_t>(mesh.n_active(in::x1));
            const real_t dy = Ly / static_cast<real_t>(mesh.n_active(in::x2));

            init_flds = InitFields<D>(tube_radius,
                                      domain_x_centre - tube_radius, domain_y_centre,   // tube 1 (left)
                                      domain_x_centre + tube_radius, domain_y_centre,   // tube 2 (right)
                                      dx, dy, guide_field_floor, kick_velocity, single_tube);
        }

        inline PGen() {}

        auto MatchFields(simtime_t) const -> InitFields<D>
        {
            return init_flds;
        }


        //  Stage 2 — per-particle drift boost so that the deposited particle
        //            current reproduces J = curl(B):
        //            (a) drift = sign(q) * sqrt(sigma)*skindepth * J(analytic)
        //            (b) drift_x += (Ey*Bz - Ez*By)/(By^2 + Bz^2)   [ExB, X-kick]
        //            (c) Lorentz boost of thermal momentum into drift frame
        //            (d) [charge_correction] multiplicative weight
        //                perturbation from div(E) = dEy/dy
        //

        inline void InitPrtls(Domain<S, M>& domain)
        {
            //! STAGE 1: Initialise uniform thermal plasma everywhere
            arch::InjectUniformMaxwellian<S, M>(params, domain, ONE, background_temperature, { 1, 2 });

            //!  STAGE 2: Drift boost
            const real_t skindepth = params.template get<real_t>("scales.skindepth0");
            const real_t larmor    = params.template get<real_t>("scales.larmor0");
            const real_t sigma     = SQR(skindepth / larmor);                       //*   sigma = (skindepth0 / larmor0)^2
            const auto& mesh       = domain.mesh;

            // Local copies for device capture (members of *this and init_flds cannot be captured into a KOKKOS_LAMBDA directly)
            const real_t tube_radius       = init_flds.tube_radius;
            const real_t tube1_x_centre    = init_flds.tube1_x_centre;
            const real_t tube1_y_centre    = init_flds.tube1_y_centre;
            const real_t tube2_x_centre    = init_flds.tube2_x_centre;
            const real_t tube2_y_centre    = init_flds.tube2_y_centre;
            const real_t cell_width_y      = init_flds.cell_width_y;
            const real_t cell_width_x      = init_flds.cell_width_x;
            const real_t floor_local       = guide_field_floor;
            const real_t kick_local        = kick_velocity;
            const bool   single_tube_local = single_tube;
            const bool   do_charge_correction = charge_correction;
            const real_t Az_exterior       = init_flds.Az_exterior;
            constexpr real_t alpha_t       = InitFields<D>::alpha_t;

            for (auto s = 0u; s < domain.species.size(); ++s)
            {
                auto& sp            = domain.species[s];
                const real_t charge = sp.charge();   // assumed +-1 (electron-positron plasma)

                // Extract Kokkos view handles before the lambda
                // (sp has a deleted copy constructor; never capture by [=]).
                const auto cell_x = sp.i1;
                const auto cell_y = sp.i2;
                const auto frac_x = sp.dx1;
                const auto frac_y = sp.dx2;
                const auto tag    = sp.tag;
                const auto ux1    = sp.ux1;
                const auto ux2    = sp.ux2;
                const auto ux3    = sp.ux3;
                const auto weight = sp.weight;

                Kokkos::parallel_for("FluxTubeDrift", sp.rangeActiveParticles(), KOKKOS_LAMBDA(index_t p)
                {
                    if (tag(p) == ParticleTag::dead) return;

                    // Physical particle position
                    const real_t x_Cd = static_cast<real_t>(cell_x(p)) + static_cast<real_t>(frac_x(p));
                    const real_t y_Cd = static_cast<real_t>(cell_y(p)) + static_cast<real_t>(frac_y(p));
                    const real_t x = mesh.metric.template convert<1, Crd::Cd, Crd::XYZ>(x_Cd);
                    const real_t y = mesh.metric.template convert<2, Crd::Cd, Crd::XYZ>(y_Cd);

                    // ================================================
                    //  (a) Current-driven drift:
                    //      drift = sign(q) * sqrt(sigma)*skindepth * J
                    //
                    //  Analytic force-free J = curl(B) per tube (layout-independent):
                    //    Jz         = (alpha_t/R) * J0(alpha_t*rn)
                    //    Jphi       = (alpha_t/R) * J0*J1 / Bz       (= -dBz/dr)
                    //    Jx = -Jphi*(y-yc)/r,  Jy = +Jphi*(x-xc)/r
                    //  Tube selection: if / else-if, as in Tristan.
                    // ================================================
                    real_t Jx = ZERO, Jy = ZERO, Jz = ZERO;
                    {
                        const real_t delta_x_1 = x - tube1_x_centre;
                        const real_t delta_y_1 = y - tube1_y_centre;
                        const real_t radial_distance_1 =
                            math::sqrt(SQR(delta_x_1) + SQR(delta_y_1));
                        const real_t rn1 = radial_distance_1 / tube_radius;

                        const real_t delta_x_2 = x - tube2_x_centre;
                        const real_t delta_y_2 = y - tube2_y_centre;
                        const real_t radial_distance_2 =
                            math::sqrt(SQR(delta_x_2) + SQR(delta_y_2));
                        const real_t rn2 = radial_distance_2 / tube_radius;

                        const real_t axis_epsilon =
                            static_cast<real_t>(1e-10) * tube_radius;

                        if (rn1 < ONE && radial_distance_1 > axis_epsilon)
                        {
                            const real_t J0_value = bessel_J0(alpha_t * rn1);
                            const real_t J1_value = bessel_J1(alpha_t * rn1);
                            const real_t Bz_local =
                                math::sqrt(SQR(J0_value) + floor_local);
                            Jz = (alpha_t / tube_radius) * J0_value;
                            const real_t Jphi_over_r =
                                (alpha_t / tube_radius) * J0_value * J1_value
                                / (Bz_local * radial_distance_1);
                            Jx = -Jphi_over_r * delta_y_1;
                            Jy = +Jphi_over_r * delta_x_1;
                        }
                        else if (!single_tube_local &&
                                 rn2 < ONE && radial_distance_2 > axis_epsilon)
                        {
                            const real_t J0_value = bessel_J0(alpha_t * rn2);
                            const real_t J1_value = bessel_J1(alpha_t * rn2);
                            const real_t Bz_local =
                                math::sqrt(SQR(J0_value) + floor_local);
                            Jz = (alpha_t / tube_radius) * J0_value;
                            const real_t Jphi_over_r =
                                (alpha_t / tube_radius) * J0_value * J1_value
                                / (Bz_local * radial_distance_2);
                            Jx = -Jphi_over_r * delta_y_2;
                            Jy = +Jphi_over_r * delta_x_2;
                        }
                    }

                    //!  drift_velocity = J * sqrt(sigma) * skindepth * sign(q)
                    //? This is the correct entity conversion factor!!
                    real_t drift_x = skindepth * math::sqrt(sigma) * charge * Jx;
                    real_t drift_y = skindepth * math::sqrt(sigma) * charge * Jy;
                    real_t drift_z = skindepth * math::sqrt(sigma) * charge * Jz;

                    // ================================================
                    //  (b) ExB drift — X-KICK layout.
                    //      The DRIVEN drift is along x:
                    //        drift_x += (E x B)_x / (By^2 + Bz^2)
                    //                 = (Ey*Bz - Ez*By) / (By^2 + Bz^2)
                    //      Fields evaluated ANALYTICALLY at the particle position.
                    //      Denominator By^2 + Bz^2 by symmetry — MANUAL CHECK #10.
                    // ================================================
                    if (kick_local != ZERO)
                    {
                        const real_t rn1 =
                            math::sqrt(SQR(x - tube1_x_centre) + SQR(y - tube1_y_centre)) / tube_radius;
                        real_t rn2 = static_cast<real_t>(2.0);   // "outside" sentinel
                        if (!single_tube_local)
                            rn2 = math::sqrt(SQR(x - tube2_x_centre) + SQR(y - tube2_y_centre)) / tube_radius;

                        if (rn1 < ONE || rn2 < ONE)
                        {
                            // Bz at particle
                            real_t Bz;
                            if (rn1 < ONE)
                                Bz = math::sqrt(SQR(bessel_J0(alpha_t * rn1)) + floor_local);
                            else
                                Bz = math::sqrt(SQR(bessel_J0(alpha_t * rn2)) + floor_local);

                            // By at particle: (Az(x-dx/2) - Az(x+dx/2)) / dx   [= -dAz/dx]
                            real_t Az_left;
                            {
                                const real_t x_q = x - HALF * cell_width_x;
                                const real_t rq1 = math::sqrt(SQR(x_q - tube1_x_centre)
                                                            + SQR(y - tube1_y_centre)) / tube_radius;
                                const real_t rq2 = math::sqrt(SQR(x_q - tube2_x_centre)
                                                            + SQR(y - tube2_y_centre)) / tube_radius;
                                if (rq1 < ONE)
                                    Az_left = (tube_radius / alpha_t) * bessel_J0(alpha_t * rq1);
                                else if (!single_tube_local && rq2 < ONE)
                                    Az_left = (tube_radius / alpha_t) * bessel_J0(alpha_t * rq2);
                                else
                                    Az_left = Az_exterior;
                            }
                            real_t Az_right;
                            {
                                const real_t x_q = x + HALF * cell_width_x;
                                const real_t rq1 = math::sqrt(SQR(x_q - tube1_x_centre)
                                                            + SQR(y - tube1_y_centre)) / tube_radius;
                                const real_t rq2 = math::sqrt(SQR(x_q - tube2_x_centre)
                                                            + SQR(y - tube2_y_centre)) / tube_radius;
                                if (rq1 < ONE)
                                    Az_right = (tube_radius / alpha_t) * bessel_J0(alpha_t * rq1);
                                else if (!single_tube_local && rq2 < ONE)
                                    Az_right = (tube_radius / alpha_t) * bessel_J0(alpha_t * rq2);
                                else
                                    Az_right = Az_exterior;
                            }
                            const real_t By = (Az_left - Az_right) / cell_width_x;

                            // Motional E at particle (sign per tube), X-kick:
                            //   tube 1 (left):  Ey = -kick*Bz, Ez = +kick*By
                            //   tube 2 (right): Ey = +kick*Bz, Ez = -kick*By
                            real_t Ey, Ez;
                            if (rn1 < ONE) { Ey = +kick_local * Bz;  Ez = -kick_local * By; }
                            else           { Ey = -kick_local * Bz;  Ez = +kick_local * By; }

                            const real_t ByBz_squared = By * By + Bz * Bz;
                            if (ByBz_squared > ZERO)
                                drift_x += (Ey * Bz - Ez * By) / ByBz_squared;
                        }
                    }

                    // ================================================
                    //  Safety clamp on |drift|.
                    //  (Tristan instead ABORTS when beta_sq >= 100;
                    //   we clamp at 0.99 to stay device-safe. If the
                    //   clamp ever triggers, the normalisation is
                    //   wrong — investigate, do not ignore.)
                    // ================================================
                    real_t drift_speed_squared = drift_x * drift_x
                                               + drift_y * drift_y
                                               + drift_z * drift_z;
                    constexpr real_t MAX_DRIFT_SPEED    = static_cast<real_t>(0.99);
                    constexpr real_t MAX_DRIFT_SPEED_SQ = MAX_DRIFT_SPEED * MAX_DRIFT_SPEED;
                    if (drift_speed_squared >= MAX_DRIFT_SPEED_SQ)
                    {
                        const real_t rescale = MAX_DRIFT_SPEED
                                             / math::sqrt(drift_speed_squared);
                        drift_x *= rescale;
                        drift_y *= rescale;
                        drift_z *= rescale;
                        drift_speed_squared = MAX_DRIFT_SPEED_SQ;
                    }

                    // ================================================
                    //  (c) Lorentz boost into the drift frame.
                    //  (NO probabilistic reflection — removed on
                    //   request; manual check #4.)
                    // ================================================
                    if (drift_speed_squared > ZERO)
                    {
                        const real_t ux = ux1(p);
                        const real_t uy = ux2(p);
                        const real_t uz = ux3(p);
                        const real_t lorentz_factor_thermal =
                            math::sqrt(ONE + ux * ux + uy * uy + uz * uz);

                        const real_t lorentz_factor_drift =
                            ONE / math::sqrt(ONE - drift_speed_squared);
                        const real_t boosted_ux = lorentz_factor_drift * drift_x;
                        const real_t boosted_uy = lorentz_factor_drift * drift_y;
                        const real_t boosted_uz = lorentz_factor_drift * drift_z;
                        const real_t boost =
                            (ux * boosted_ux + uy * boosted_uy + uz * boosted_uz)
                            / (lorentz_factor_drift + ONE)
                            + lorentz_factor_thermal;

                        ux1(p) = ux + boost * boosted_ux;
                        ux2(p) = uy + boost * boosted_uy;
                        ux3(p) = uz + boost * boosted_uz;
                    }

                    // ================================================
                    //  (d) Charge-density weight correction.
                    //  user_sheet_new.F90 (MULTIPLICATIVE form):
                    //    w_new = w * (1 + rho0 * sqrt(sigma)*skindepth * sign(q))
                    //  X-KICK layout: Ex = 0, Ey is the nonzero in-plane E,
                    //  d/dz = 0, so div(E) = dEy/dy. Tristan-style per-cell:
                    //    rho0 = Ey(y + dy/2) - Ey(y - dy/2)     [cell = 1]
                    //  analytic Ey, NOT divided by dy -> requires 1 cell = 1
                    //  code unit (manual check #3). No-op when kick = 0.
                    // ================================================
                    if (do_charge_correction && kick_local != ZERO)
                    {
                        // Ey at (x, y + dy/2)
                        real_t Ey_upper;
                        {
                            const real_t y_q = y + HALF * cell_width_y;
                            const real_t rq1 = math::sqrt(SQR(x - tube1_x_centre)
                                                        + SQR(y_q - tube1_y_centre)) / tube_radius;
                            const real_t rq2 = math::sqrt(SQR(x - tube2_x_centre)
                                                        + SQR(y_q - tube2_y_centre)) / tube_radius;
                            if (rq1 < ONE)
                                Ey_upper = -kick_local *
                                    math::sqrt(SQR(bessel_J0(alpha_t * rq1)) + floor_local);
                            else if (!single_tube_local && rq2 < ONE)
                                Ey_upper = +kick_local *
                                    math::sqrt(SQR(bessel_J0(alpha_t * rq2)) + floor_local);
                            else
                                Ey_upper = ZERO;
                        }
                        // Ey at (x, y - dy/2)
                        real_t Ey_lower;
                        {
                            const real_t y_q = y - HALF * cell_width_y;
                            const real_t rq1 = math::sqrt(SQR(x - tube1_x_centre)
                                                        + SQR(y_q - tube1_y_centre)) / tube_radius;
                            const real_t rq2 = math::sqrt(SQR(x - tube2_x_centre)
                                                        + SQR(y_q - tube2_y_centre)) / tube_radius;
                            if (rq1 < ONE)
                                Ey_lower = -kick_local *
                                    math::sqrt(SQR(bessel_J0(alpha_t * rq1)) + floor_local);
                            else if (!single_tube_local && rq2 < ONE)
                                Ey_lower = +kick_local *
                                    math::sqrt(SQR(bessel_J0(alpha_t * rq2)) + floor_local);
                            else
                                Ey_lower = ZERO;
                        }

                        const real_t rho0 = Ey_upper - Ey_lower;   // per-cell div(E) = dEy/dy

                        const real_t new_weight =
                            weight(p) * (ONE + rho0 * skindepth * math::sqrt(sigma) * charge);
                        if (new_weight > ZERO)
                            weight(p) = new_weight;
                    }

                }); // parallel_for FluxTubeDrift

            } // species loop

        } // InitPrtls

    }; // struct PGen

} // namespace user

#endif // PROBLEM_GENERATOR_H