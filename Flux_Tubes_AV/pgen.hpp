#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

/**
 * @file Flux_Tubes_AV/pgen.hpp
 * @brief Fadeev force-free island coalescence (Velberg et al. 2026 reconnection setup)
 *
 *~ References:
 *      1. Velberg et al. (2026), VPIC reference deck (Velberg_2D.cc)
 *      2. Fadeev et al. (1965), Nucl. Fusion, 5, 202
 *
 *~ Coordinate mapping (VPIC -> Entity)
 * -------------------------------------------------------------
 *   x (along sheet)     -> x1   (X)
 *   y (normal to sheet) -> x2   (Y)
 *   z (out of plane)    -> x3   (Z)   <- guide-field / primary-current direction
 *
 *~ Field normalisation
 * -------------------------------------------------------------
 *   The asymptotic field amplitude is FIXED to unity in code units.
 *   In Entity's Minkowski normalisation a code-unit field B = 1 corresponds
 *   to the physical reference field B0 = 1 / larmor0. The magnetisation is
 *   therefore set ENTIRELY through the [scales] block, NOT through a field
 *   amplitude parameter:
 *
 *       sigma = (skindepth0 / larmor0)^2
 *
 *   VPIC's b0 (= sqrt(sigma) in its natural units) is a magnetisation knob,
 *   so a given VPIC b0 maps to Entity via:
 *
 *       sigma = b0_VPIC^2   <=>   skindepth0 / larmor0 = b0_VPIC
 *
 *   e.g. VPIC b0 = 5  ->  sigma = 25  ->  set skindepth0/larmor0 = 5 in the .toml.
 *   No field-amplitude parameter is needed (or allowed) here.
 *
 *~ Physics setup
 * -------------
 * Fadeev force-free equilibrium (two coalescing magnetic islands) for a
 * relativistic pair plasma (m_e = m_i = 1, sigma = 25) without a guide field
 * (guide_field_ratio = 0 by default). A small divergence-free perturbation
 * seeds the island coalescence.
 *
 *~ Field initialisation  (unit amplitude; physical B0 = 1/larmor0)
 * --------------------------------------------------------------
 *   fadeev_denom = cosh(y/sheet_half_thickness) + island_param * cos(x/sheet_half_thickness)
 *
 *   Bx =  sinh(y/sheet_half_thickness) / fadeev_denom                 [reversing / reconnecting, X]
 *   By =  island_param * sin(x/sheet_half_thickness) / fadeev_denom   [connecting / streaming,   Y]
 *   Bz =  sqrt( (1 - island_param^2)/fadeev_denom^2 + guide_field_ratio^2 )   [out-of-plane guide, Z]
 *   Ex =  Ey = Ez = 0   (electric field zero at t = 0)
 *
 *~ Symmetry-breaking perturbation (divergence-free)
 * --------------------------------------------------------------
 *   pert_by_amplitude = perturbation_fraction                      (fraction of unit field)
 *   pert_bx_amplitude = -pert_by_amplitude * Lx / (2 * Ly)         (from div(B) = 0)
 *
 *   dBx = pert_bx_amplitude * cos(2*pi*x/Lx) * sin(pi*y/Ly)
 *   dBy = pert_by_amplitude * cos(pi*y/Ly)  * sin(2*pi*x/Lx)
 *
 *~ Force-free current (J parallel to B, pair plasma, uniform density n0)
 * --------------------------------------------------------------
 *   force_free_norm = sqrt( (1 - island_param^2) + guide_field_ratio^2 * fadeev_denom^2 )
 *
 *   Jz = (1/sheet_half_thickness) * (1 - island_param^2) / fadeev_denom^2   [primary, out-of-plane]
 *   Jx = Jz * sinh(y/sheet_half_thickness) / force_free_norm                [in-plane]
 *   Jy = Jz * island_param * sin(x/sheet_half_thickness) / force_free_norm
 *
 *~ Drift / current normalisation
 * --------------------------------------------------------------
 *   In Entity normalisation the drift speed (beta = v/c) for each species:
 *     drift_x = (1/2) * sqrt(sigma) * skindepth * Jx
 *     drift_y = (1/2) * sqrt(sigma) * skindepth * Jy
 *     drift_z = (1/2) * sqrt(sigma) * skindepth * Jz
 *                ^ factor 1/2: pair plasma, each species carries half the current
 *                  (analogous to VPIC's VDY = -JY/2 for both-species current)
 *   Opposite species receive opposite kicks (net charge density = 0).
 *
 *~ Particle initialisation
 * -----------------------------------------------------------
 *  1. Uniform relativistic Maxwellian everywhere (uniform density: pressure
 *     balance is magnetic, so no sech^2 density profile is needed).
 *  2. Momentum kick  u += sign(q) * drift * gamma_drift  applied per species.
 *     Because drift << v_th for fiducial parameters, the simple momentum-kick
 *     approximation is accurate to O(beta^2/vth^2); the VPIC deck performs a
 *     full Maxwell-Juttner boost, which coincides with this in the small-drift
 *     limit.
 *
 *~ Boundaries
 * -----------------------------------------------------------
 *   x1 : PERIODIC                (fields + particles)
 *   x2 : CONDUCTING / REFLECTING (fields conduct, particles reflect)
 *
 *~ Parameters (set in the .toml)
 * -----------------------------------------------------------
 *   guide_field_ratio       B_guide (in units of the unit field)        [default 0.0]
 *   island_param            Fadeev island parameter (0 < eps < 1)       [default 0.4]
 *   sheet_half_thickness    current-layer half-thickness [code units]   [default 64.0]
 *   background_temperature  kT/(m c^2) (relativistic temperature)       [default 0.01]
 *   perturbation_fraction   |dBy| (fraction of unit field)              [default -0.1]
 *   skindepth0              electron skin depth (physical units)        [from scales]
 *   larmor0                 reference Larmor radius; B0 = 1/larmor0     [from scales]
 *
 **/

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/utils.h"
#include "archetypes/field_setter.h"
#include "framework/domain/metadomain.h"
#include "archetypes/problem_generator.h"

#include <utility>
#include <vector>

namespace user
{
    using namespace ntt;

    //! =========================================================================
    //!  E and B are initialised analytically (unit field amplitude in code units;
    //!  physical reference field B0 = 1/larmor0).
    //!  entity calls each component at its own Yee-stagger position.
    //!
    //!    fadeev_denom = cosh((y - sheet_y_centre)/sheet_half_thickness)
    //!                       + island_param * cos((x - sheet_x_centre)/sheet_half_thickness)
    //!
    //!    Bx = sinh(y/sheet_half_thickness) / fadeev_denom + perturbation
    //!    By = island_param * sin(x/sheet_half_thickness) / fadeev_denom + perturbation
    //!    Bz = sqrt( (1 - island_param^2)/fadeev_denom^2 + guide_field_ratio^2 )
    //!
    //! =========================================================================
    template <Dimension D>
    struct InitFields
    {
        InitFields() = default;

        InitFields( real_t guide_field_ratio_,
                    real_t island_param_,
                    real_t sheet_half_thickness_,
                    real_t Lx_,
                    real_t Ly_,
                    real_t perturbation_fraction_,
                    real_t sheet_x_centre_,
                    real_t sheet_y_centre_):
                    guide_field_ratio    { guide_field_ratio_    },
                    island_param         { island_param_         },
                    sheet_half_thickness { sheet_half_thickness_ },
                    Lx                   { Lx_                   },
                    Ly                   { Ly_                   },
                    sheet_x_centre       { sheet_x_centre_       },
                    sheet_y_centre       { sheet_y_centre_       },
                    pert_by_amplitude    { perturbation_fraction_ },
                    pert_bx_amplitude    { -perturbation_fraction_ * Lx_ / (TWO * Ly_) }   // from div(B) = 0
        {}

        //! fadeev_denom = cosh(dy/L) + island_param * cos(dx/L)
        Inline auto fadeev_denominator(const coord_t<D>& x_Ph) const -> real_t
        {
            const real_t delta_x = x_Ph[0] - sheet_x_centre;
            const real_t delta_y = x_Ph[1] - sheet_y_centre;
            return math::cosh(delta_y / sheet_half_thickness) + island_param * math::cos(delta_x / sheet_half_thickness);
        }

        //! Bx: reversing / reconnecting field
        //?  Bx_background = sinh(dy/L) / fadeev_denom
        //?  dBx          = pert_bx_amplitude * cos(2*pi*dx/Lx) * sin(pi*dy/Ly)
        Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t
        {
            const real_t fadeev_denom = fadeev_denominator(x_Ph);
            const real_t delta_x      = x_Ph[0] - sheet_x_centre;
            const real_t delta_y      = x_Ph[1] - sheet_y_centre;

            const real_t perturbation = pert_bx_amplitude * math::cos(TWO * static_cast<real_t>(constant::PI) * delta_x / Lx)
                                                          * math::sin(static_cast<real_t>(constant::PI) * delta_y / Ly);

            return math::sinh(delta_y / sheet_half_thickness) / fadeev_denom + perturbation;
        }

        //! By: connecting / streaming field
        //?  By_background = island_param * sin(dx/L) / fadeev_denom
        //?  dBy          = pert_by_amplitude * cos(pi*dy/Ly) * sin(2*pi*dx/Lx)
        Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t
        {
            const real_t fadeev_denom = fadeev_denominator(x_Ph);
            const real_t delta_x      = x_Ph[0] - sheet_x_centre;
            const real_t delta_y      = x_Ph[1] - sheet_y_centre;

            const real_t perturbation = pert_by_amplitude * math::cos(static_cast<real_t>(constant::PI) * delta_y / Ly)
                                                          * math::sin(TWO * static_cast<real_t>(constant::PI) * delta_x / Lx);

            return island_param * math::sin(delta_x / sheet_half_thickness) / fadeev_denom + perturbation;
        }

        //! Bz: out-of-plane guide field
        //?  Bz = sqrt( (1 - island_param^2)/fadeev_denom^2 + guide_field_ratio^2 )
        //?  guide_field_ratio = 0 by default  ->  reduces to force-free Bz = sqrt(1 - island_param^2)/fadeev_denom
        Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t
        {
            const real_t fadeev_denom        = fadeev_denominator(x_Ph);
            const real_t one_minus_island_sq = ONE - island_param * island_param;   // (1 - island_param^2)

            return math::sqrt(one_minus_island_sq / (fadeev_denom * fadeev_denom) + guide_field_ratio * guide_field_ratio);
        }

        //! Electric field: zero at t = 0
        Inline auto ex1(const coord_t<D>&) const -> real_t { return ZERO; }
        Inline auto ex2(const coord_t<D>&) const -> real_t { return ZERO; }
        Inline auto ex3(const coord_t<D>&) const -> real_t { return ZERO; }

        //*  Data members
        real_t guide_field_ratio    { ZERO };   // B_guide in units of the unit field; 0 -> no guide field
        real_t island_param         { ZERO };   // Fadeev island parameter eps in (0,1); 0 -> Harris sheet
        real_t sheet_half_thickness { ONE  };   // current-layer half-thickness [code units = skindepth]
        real_t Lx                   { ONE  };   // box dimension along sheet (x1)
        real_t Ly                   { ONE  };   // box dimension normal to sheet (x2)
        real_t sheet_x_centre       { ZERO };   // x-centre of the box = (xmax + xmin)/2
        real_t sheet_y_centre       { ZERO };   // y-centre of the box = (ymax + ymin)/2
        real_t pert_by_amplitude    { ZERO };   // dBy = perturbation_fraction (fraction of unit field)
        real_t pert_bx_amplitude    { ZERO };   // dBx = -dBy * Lx/(2*Ly), from div(B) = 0
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
        real_t guide_field_ratio      { ZERO };
        real_t island_param           { static_cast<real_t>(0.4)  };
        real_t sheet_half_thickness   { static_cast<real_t>(64.0) };
        real_t background_temperature { static_cast<real_t>(0.01) };
        real_t perturbation_fraction  { static_cast<real_t>(-0.1) };

    public:
        InitFields<D> init_flds;

        //!  Constructor
        inline PGen(const SimulationParams& p, metadomain_type& md): Base { p }, global_domain { md }
        {
            guide_field_ratio      = p.template get<real_t>("setup.bg",          ZERO);
            island_param           = p.template get<real_t>("setup.eps",         static_cast<real_t>(0.4));
            sheet_half_thickness   = p.template get<real_t>("setup.sheet_L",     static_cast<real_t>(64.0));
            background_temperature = p.template get<real_t>("setup.temperature", static_cast<real_t>(0.01));
            perturbation_fraction  = p.template get<real_t>("setup.dby_frac",    static_cast<real_t>(-0.1));

            const auto& mesh = md.mesh();
            const real_t global_x_min = mesh.extent(in::x1).first;
            const real_t global_x_max = mesh.extent(in::x1).second;
            const real_t global_y_min = mesh.extent(in::x2).first;
            const real_t global_y_max = mesh.extent(in::x2).second;
            const real_t Lx = global_x_max - global_x_min;
            const real_t Ly = global_y_max - global_y_min;

            const real_t sheet_x_centre = HALF * (global_x_max + global_x_min);
            const real_t sheet_y_centre = HALF * (global_y_max + global_y_min);

            init_flds = InitFields<D>(guide_field_ratio,
                                      island_param,
                                      sheet_half_thickness,
                                      Lx, Ly,
                                      perturbation_fraction,
                                      sheet_x_centre, sheet_y_centre);
        }

        inline PGen() {}

        auto MatchFields(simtime_t) const -> InitFields<D>
        {
            return init_flds;
        }

        inline void InitPrtls(Domain<S, M>& domain)
        {
            //! STAGE 1: Inject a uniform relativistic Maxwellian plasma everywhere.
            arch::InjectUniformMaxwellian<S, M>(params, domain, ONE, background_temperature, { 1, 2 });

            //!  STAGE 2: Drift boost
            const real_t skindepth = params.template get<real_t>("scales.skindepth0");
            const real_t larmor    = params.template get<real_t>("scales.larmor0");
            const real_t sigma     = SQR(skindepth / larmor);                       //*   sigma = (skindepth0 / larmor0)^2
            const auto& mesh       = domain.mesh;

            // Local copies for device capture (members of *this and init_flds cannot be captured into a KOKKOS_LAMBDA directly)
            const real_t guide_field_ratio_local    = init_flds.guide_field_ratio;
            const real_t island_param_local         = init_flds.island_param;
            const real_t sheet_half_thickness_local = init_flds.sheet_half_thickness;
            const real_t sheet_x_centre             = init_flds.sheet_x_centre;
            const real_t sheet_y_centre             = init_flds.sheet_y_centre;

            for (auto s = 0u; s < domain.species.size(); ++s)
            {
                auto& sp            = domain.species[s];
                const real_t charge = sp.charge();      // assumed +-1 (electron-positron plasma)

                // Extract Kokkos view handles before the lambda: the species object can't be
                // copied into a GPU kernel, but its array handles (i1, ux1, ...) can
                const auto cell_x = sp.i1;
                const auto cell_y = sp.i2;
                const auto frac_x = sp.dx1;
                const auto frac_y = sp.dx2;
                const auto tag    = sp.tag;
                const auto ux1    = sp.ux1;
                const auto ux2    = sp.ux2;
                const auto ux3    = sp.ux3;

                Kokkos::parallel_for("FadeevCurrentDrift", sp.rangeActiveParticles(), KOKKOS_LAMBDA(index_t p)
                {
                    if (tag(p) == ParticleTag::dead) return;

                    // Physical position of particle
                    const real_t x_Cd = static_cast<real_t>(cell_x(p)) + static_cast<real_t>(frac_x(p));
                    const real_t y_Cd = static_cast<real_t>(cell_y(p)) + static_cast<real_t>(frac_y(p));
                    const real_t x = mesh.metric.template convert<1, Crd::Cd, Crd::XYZ>(x_Cd);
                    const real_t y = mesh.metric.template convert<2, Crd::Cd, Crd::XYZ>(y_Cd);

                    const real_t delta_x = x - sheet_x_centre;    // offset from box centre (x1, X)
                    const real_t delta_y = y - sheet_y_centre;    // offset from box centre (x2, Y)

                    //! ==================== Fadeev current J = ∇×B ==================== !//

                    //? There is no motional ExB drift as the flux tubes are assumed to be
                    //? stationary. They are "perturbed" to merge and coalesce.

                    //* Fadeev denominator and force-free normalisation
                    const real_t fadeev_denom = math::cosh(delta_y / sheet_half_thickness_local) + island_param_local * math::cos(delta_x / sheet_half_thickness_local);    // D = cosh(dy/L) + eps*cos(dx/L)
                    const real_t one_minus_island_sq = ONE - SQR(island_param_local);                                                   // 1 - eps^2
                    const real_t force_free_norm = math::sqrt(one_minus_island_sq + SQR(guide_field_ratio_local) * SQR(fadeev_denom));  // F = sqrt[ (1 - eps^2) + bg^2 * D^2 ]

                    //* Normalised analytic current (unit field amplitude in code units)
                    const real_t Jz = one_minus_island_sq / (sheet_half_thickness_local * fadeev_denom * fadeev_denom);                 // Jz = (1/L)·(1−ε²)/D²
                    const real_t Jx = Jz * math::sinh(delta_y / sheet_half_thickness_local) / force_free_norm;                          // Jx = Jz · sinh(y/L) / F,   F = √[(1−ε²)+bg²·D²]
                    const real_t Jy = Jz * island_param_local * math::sin(delta_x / sheet_half_thickness_local) / force_free_norm;      // Jy = Jz · ε·sin(x/L) / F

                    //! Current-driven drift: β = (1/2) · √σ · skindepth · J
                    //TODO: Check factor 1/2 with dby_frac=0 : pair plasma, each species carries half the current
                    //?   sign(q) (charge) is applied in the momentum kick below, not here
                    const real_t drift_x = HALF * skindepth * math::sqrt(sigma) * Jx;
                    const real_t drift_y = HALF * skindepth * math::sqrt(sigma) * Jy;
                    const real_t drift_z = HALF * skindepth * math::sqrt(sigma) * Jz;

                    //* Drift Lorentz factor;  |beta_d|^2 = beta_x^2 + beta_y^2 + beta_z^2
                    const real_t drift_speed_squared = SQR(drift_x) + SQR(drift_y) + SQR(drift_z);

                    //? Numerical safety: skip if the drift is somehow superluminal
                    if (drift_speed_squared >= ONE) return;

                    //! Momentum kick:  u += sign(q) * drift * gamma_drift
                    const real_t lorentz_factor_drift = ONE / math::sqrt(ONE - drift_speed_squared);   //* gamma_d = 1/sqrt(1 - |beta_d|^2)
                    ux1(p) += charge * drift_x * lorentz_factor_drift;
                    ux2(p) += charge * drift_y * lorentz_factor_drift;
                    ux3(p) += charge * drift_z * lorentz_factor_drift;

                }); // parallel_for FadeevCurrentDrift

            } // species loop

        } // InitPrtls

    }; // struct PGen

} // namespace user

#endif // PROBLEM_GENERATOR_H

//! ============================================================================
//!  OUTSTANDING CHECKS (TODO)
//! ============================================================================
//
//TODO [factor 1/2]: Verify the both-species current split (HALF in drift_x/y/z).
//      Entity does NOT apply any species current-sharing factor (the injector is
//      called with zero drift; the drift is added here manually). Since Jx/Jy/Jz
//      is the TOTAL force-free current (∇×B), each of the two species should carry
//      half -> HALF is expected. BUT Camille's calibrated deck used NO 1/2 and gave
//      correct results, which can only be reconciled if her J was already per-species.
//      DECISIVE TEST: run with dby_frac = 0. A true force-free IC must stay static
//      (B frozen, E ~ 0). Compare runs with and without the HALF; the one giving a
//      static B is correct. (See drift block above.)
//
//TODO [kick vs boost]: The drift is applied as a first-order momentum kick
//      (u += q*drift*gamma_d), NOT the full Maxwell-Juttner frame boost that VPIC
//      performs. These agree only to O(beta_d^2 / vth^2). For fiducial parameters
//      beta_d ~ 1e-3 and vth ~ 0.6, so the error is ~1e-5 (negligible). If higher
//      fidelity is ever needed, replace the kick with the coordinate-free boost
//      u' = u + [ (u . u_d)/(gamma_d + 1) + gamma_th ] * u_d,  u_d = gamma_d * beta_d
//      (equivalent to VPIC's triad-decomposed boost, but degeneracy-free).
//
//TODO [Yee staggering]: Fields are evaluated as analytic POINT values at x_Ph.
//      VPIC's set_region_field evaluates each B component at its own Yee-staggered
//      location; Camille's reference deck reconstructs B from a finite-differenced
//      vector potential Az for exact discrete div(B) = 0. Confirm whether Entity's
//      MatchFields/field-setter passes each component its own staggered coordinate
//      (then point-eval is fine) or a single cell-centred x_Ph (then either stagger
//      manually or switch to the Az approach). Discrete div(B) is otherwise cleaned
//      by the solver, but check the initial transient.
