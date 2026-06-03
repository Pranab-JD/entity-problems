#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

/**
 * @file pgen.hpp
 * @brief Entity problem generator for Velberg 2-D island-coalescence reconnection.
 *
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * COORDINATE MAPPING  (VPIC → Entity)
 *   x  (along sheet)      → x1
 *   y  (normal to sheet)  → x2
 *   z  (out of plane)     → x3   ← guide-field / primary-current direction
 *
 * PHYSICS
 *   Fadeev force-free equilibrium (two coalescing islands) for a relativistic
 *   pair plasma (me = mi = 1, σ = 25) without a guide field (bg = 0 by default).
 *   A small symmetry-breaking perturbation seeds island coalescence.
 *
 *   Field structure (D ≡ cosh(x2/L) + ε·cos(x1/L)):
 *     bx1 = b0·sinh(x2/L)/D          [reversing / reconnecting component, X]
 *     bx2 = b0·ε·sin(x1/L)/D         [connecting / streaming component,   Y]
 *     bx3 = b0·√[(1−ε²)/D² + bg²]   [out-of-plane, force-free guide,     Z]
 *     ex1 = ex2 = ex3 = 0
 *
 *   Symmetry-breaking perturbation (div-free):
 *     dby = dby_frac · b0
 *     dbx = −dby · Lx / (2·Ly)
 *     δbx1 += dbx · cos(2π·x1/Lx) · sin(π·x2/Ly)
 *     δbx2 += dby · cos(π·x2/Ly)  · sin(2π·x1/Lx)
 *
 *   Force-free current (J ∥ B, pair-plasma, uniform density n0):
 *     J_x3 = (b0/L)·(1−ε²)/D²                            [primary]
 *     J_x1 = J_x3·sinh(x2/L) / √[(1−ε²)+bg²·D²]         [in-plane]
 *     J_x2 = J_x3·ε·sin(x1/L) / √[(1−ε²)+bg²·D²]
 *
 *   In Entity normalisation the drift velocity β̂ for each species is
 *     β_ref = √σ₀ · dₑ · (1−ε²) / [2 · L · D²]
 *     β_x   = β_ref · sinh(x2/L) / F,   F = √[(1−ε²)+bg²·D²]
 *     β_y   = β_ref · ε · sin(x1/L) / F
 *     β_z   = β_ref
 *   Opposite species receive opposite kicks (net charge density = 0).
 *
 * BOUNDARIES
 *   x1 : PERIODIC  (fields + particles)
 *   x2 : CONDUCTOR / REFLECT  (PEC walls, particles reflect)
 *
 * PARAMETERS (all in setup.* or scales.* in the .toml)
 *   b0          normalised asymptotic field strength              [default 1.0]
 *   bg          guide-field ratio B_guide/b0                     [default 0.0]
 *   eps         Fadeev island parameter (0 < ε < 1)              [default 0.4]
 *   sheet_L     current-layer half-thickness in code units       [required]
 *   temperature kT/(mₑ c²)  (relativistic temperature)          [required]
 *   dby_frac    perturbation amplitude |dBy|/b0                  [default -0.1]
 *   sigma0      magnetisation σ = (ωce/ωpe)²                     [from scales]
 *   skindepth0  electron skin depth dₑ in physical units         [from scales]
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/field_setter.h"
#include "archetypes/problem_generator.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"

#include <utility>

namespace user
{
    using namespace ntt;

    //! ==========================================================
    //! Fadeev equilibrium + perturbation field initialiser
    //! ==========================================================
    template <Dimension D>
    struct InitFields
    {

        InitFields(
            real_t b0,
            real_t bg,
            real_t eps,
            real_t sheet_L,
            real_t Lx,
            real_t Ly,
            real_t dby_frac,
            real_t cs_x,
            real_t cs_y) :
            b0      { b0 },
            bg      { bg },
            eps     { eps },
            L       { sheet_L },
            Lx      { Lx },
            Ly      { Ly },
            dby     { dby_frac * b0 },
            dbx     { -dby_frac * b0 * Lx / (TWO * Ly) },   // from ∇·B = 0
            cs_x    { cs_x },
            cs_y    { cs_y }
        {}

        Inline auto Fadeev_denominator(const coord_t<D>& x) const -> real_t
        {
            return math::cosh((x[1] - cs_y) / L) + eps * math::cos((x[0] - cs_x) / L);
        }

        //! Bx: reversing field
        Inline auto bx1(const coord_t<D>& x) const -> real_t
        {
            const real_t Denom = Fadeev_denominator(x);
            const real_t dx   = x[0] - cs_x;
            const real_t dy   = x[1] - cs_y;
            const real_t pert = dbx * math::cos(TWO * static_cast<real_t>(constant::PI) * dx / Lx)      //* δbx1 = dbx·cos(2π·dx/Lx)·sin(π·dy/Ly)
                                    * math::sin(static_cast<real_t>(constant::PI) * dy / Ly);

            return b0 * math::sinh(dy / L) / Denom + pert;          //* bx1_bg = b0·sinh(x2/L)/D
        }

        //! By: Reconnecting field
        Inline auto bx2(const coord_t<D>& x) const -> real_t
        {
            const real_t Denom = Fadeev_denominator(x);
            const real_t dx   = x[0] - cs_x;
            const real_t dy   = x[1] - cs_y;
            const real_t pert = dby * math::cos(static_cast<real_t>(constant::PI) * dy / Ly)            //* δbx2 = dby·cos(π·dy/Ly)·sin(2π·dx/Lx)
                                    * math::sin(TWO * static_cast<real_t>(constant::PI) * dx / Lx);

            return b0 * eps * math::sin(dx / L) / Denom + pert;     //* bx2_bg = b0·ε·sin(x1/L)/D
        }

        //! Bz: out-of-plane guide field
        Inline auto bx3(const coord_t<D>& x) const -> real_t
        {
            const real_t Denom = Fadeev_denominator(x);
            const real_t ome = ONE - eps * eps;   // (1 − ε²)

            return b0 * math::sqrt(ome / (Denom * Denom) + bg * bg); //* bx3 = b0·√[(1−ε²)/D² + bg²]; = 0 by default → reduces to force-free bx3 = b0·√(1−ε²)/D
        }

        //!Electric field: zero at t = 0
        Inline auto ex1(const coord_t<D>&) const -> real_t { return ZERO; }
        Inline auto ex2(const coord_t<D>&) const -> real_t { return ZERO; }
        Inline auto ex3(const coord_t<D>&) const -> real_t { return ZERO; }

    private:
        const real_t b0, bg, eps, L, Lx, Ly, dby, dbx, cs_x, cs_y;
    };

    //! ==========================================================
    //! Problem generator
    //! ==========================================================
    template <SimEngine::type S, class M>
    struct PGen : public arch::ProblemGenerator<S, M>
    {

        // Compatibility flags
        static constexpr auto engines
        {
            traits::compatible_with<SimEngine::SRPIC>::value
        };
        static constexpr auto metrics
        {
            traits::compatible_with<Metric::Minkowski>::value
        };

        // Island coalescence requires 2-D; 3-D could be added but is untested.
        static constexpr auto dimensions
        {
            traits::compatible_with<Dim::_2D>::value
        };

        using arch::ProblemGenerator<S, M>::D;
        using arch::ProblemGenerator<S, M>::C;
        using arch::ProblemGenerator<S, M>::params;

        Metadomain<S, M>& global_domain;

        //* Domain geometry
        const real_t global_xmin, global_xmax;   // x1 range
        const real_t global_ymin, global_ymax;   // x2 range
        const real_t Lx, Ly;                     // box dimensions
        const real_t cs_x, cs_y;                 // box centre

        //* Physics parameters
        const real_t b0;           // normalised asymptotic field strength
        const real_t bg;           // guide-field ratio B_guide / b0
        const real_t eps;          // Fadeev island parameter  (0 < ε < 1)
        const real_t sheet_L;      // current-layer half-thickness  [code units = dₑ]
        const real_t temperature;  // kT / (mₑ c²) — relativistic temperature
        const real_t dby_frac;     // perturbation amplitude  |δBy| / b0

        //* Normalisation scales
        const real_t sigma0;       // σ₀ = (ωce/ωpe)²
        const real_t skindepth0;   // dₑ in physical (code) units

        InitFields<D> init_flds;

        //! Constructor
        inline PGen(const SimulationParams& p, Metadomain<S, M>& global_domain) :
            arch::ProblemGenerator<S, M> { p },
            global_domain { global_domain },
            global_xmin   { global_domain.mesh().extent(in::x1).first  },
            global_xmax   { global_domain.mesh().extent(in::x1).second },
            global_ymin   { global_domain.mesh().extent(in::x2).first  },
            global_ymax   { global_domain.mesh().extent(in::x2).second },
            Lx            { global_xmax - global_xmin },
            Ly            { global_ymax - global_ymin },
            cs_x          { HALF * (global_xmax + global_xmin) },
            cs_y          { HALF * (global_ymax + global_ymin) },
            b0            { p.template get<real_t>("setup.b0",       ONE)  },
            bg            { p.template get<real_t>("setup.bg",       ZERO) },
            eps           { p.template get<real_t>("setup.eps",      static_cast<real_t>(0.4)) },
            sheet_L       { p.template get<real_t>("setup.sheet_L")        },
            temperature   { p.template get<real_t>("setup.temperature")    },
            dby_frac      { p.template get<real_t>("setup.dby_frac", static_cast<real_t>(-0.1)) },
            sigma0        { p.template get<real_t>("scales.sigma0")        },
            skindepth0    { p.template get<real_t>("scales.skindepth0")    },
            init_flds     { b0, bg, eps, sheet_L, Lx, Ly, dby_frac, cs_x, cs_y }
        {}

        inline PGen() {}

        // ── Field initialisation (called at t = 0) ────────────────────────────
        auto MatchFields(real_t) const -> InitFields<D> { return init_flds; }

        // ── Particle initialisation ───────────────────────────────────────────
        inline void InitPrtls(Domain<S, M>& domain)
        {

            // ──────────────────────────────────────────────────────────────────
            // Step 1 — Inject a uniform relativistic Maxwellian plasma.
            //
            // For a force-free current sheet the pressure balance is magnetic,
            // so the particle density can be uniform (no sech² profile needed),
            // exactly as in the VPIC Velberg_2D.cc deck.
            // Both species start with zero bulk drift; drifts are added in Step 2.
            // ──────────────────────────────────────────────────────────────────
            const auto temperatures = std::make_pair(temperature, temperature);
            const auto zero_drifts  = std::make_pair(
                std::vector<real_t> { ZERO, ZERO, ZERO },
                std::vector<real_t> { ZERO, ZERO, ZERO });

            boundaries_t<real_t> full_box;
            for (auto d { 0u }; d < (unsigned int)M::Dim; ++d)
                full_box.push_back(Range::All);

            arch::InjectUniformMaxwellians<S, M>(params, domain, ONE, temperatures, { 1, 2 }, zero_drifts, false, full_box);

            // ──────────────────────────────────────────────────────────────────
            // Step 2 — Apply the force-free current drift to each species.
            //
            // The Fadeev equilibrium requires J ∥ B everywhere (force-free
            // condition), giving three non-zero current components
            // (↔ VPIC JX, JY, JZ macros):
            //
            //   J_x3 = (b0/L)(1−ε²)/D²                     [primary, out-of-plane Z]
            //   J_x1 = J_x3 · sinh(x2/L) / F,   F = √[(1−ε²)+bg²D²]
            //   J_x2 = J_x3 · ε·sin(x1/L) / F
            //
            // In Entity normalisations (β = drift speed / c):
            //   β_ref = √σ₀ · dₑ · (1−ε²) / [2·L·D²]
            //           ↑ factor 2: pair plasma, each species carries half the current
            //           (analogous to VPIC's  VDY = −JY/2  for both-species current)
            //   β_x   = β_ref · sinh(x2/L) / F
            //   β_y   = β_ref · ε·sin(x1/L) / F
            //   β_z   = β_ref
            //
            // Because VD ≪ vth (β_ref ~ 0.003 for fiducial parameters), the
            // simple momentum-kick approximation  u += q₀·β·γ  is accurate to
            // O(β²/vth²), which is identical to the boost employed in the VPIC
            // deck.
            // ──────────────────────────────────────────────────────────────────
            const auto& mesh = domain.mesh;

            // Capture scalars for GPU kernels
            const real_t b0_     = b0;
            const real_t bg_     = bg;
            const real_t eps_    = eps;
            const real_t L_      = sheet_L;
            const real_t cs_x_   = cs_x;
            const real_t cs_y_   = cs_y;
            const real_t sigma0_ = sigma0;
            const real_t d0_     = skindepth0;   // dₑ in code units

            for (auto s { 0u }; s < 2; ++s) 
            {
                auto& species = domain.species[s];
                auto  i1      = species.i1;
                auto  i2      = species.i2;
                auto  dx1     = species.dx1;
                auto  dx2     = species.dx2;
                auto  tag     = species.tag;
                auto  ux1     = species.ux1;
                auto  ux2     = species.ux2;
                auto  ux3     = species.ux3;
                const real_t q0 = species.charge();   // −1 for e⁻, +1 for e⁺

                Kokkos::parallel_for("FadeevCurrentDrift", species.rangeActiveParticles(), Lambda(index_t p)
                {
                    if (tag(p) == ParticleTag::dead) { return; }

                    //? Physical position
                    coord_t<D> x_Ph { ZERO };
                    {
                        const real_t c1 = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
                        const real_t c2 = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
                        x_Ph[0] = mesh.metric.template convert<1, Crd::Cd, Crd::XYZ>(c1);
                        x_Ph[1] = mesh.metric.template convert<2, Crd::Cd, Crd::XYZ>(c2);
                    }

                    const real_t dx = x_Ph[0] - cs_x_;   // offset from box centre (x1, X)
                    const real_t dy = x_Ph[1] - cs_y_;   // offset from box centre (x2, Y)

                    //? Fadeev denominator and related quantities
                    const real_t Denom = math::cosh(dy / L_) + eps_ * math::cos(dx / L_);
                    const real_t ome = ONE - eps_ * eps_;                    // 1 − ε²
                    const real_t F = math::sqrt(ome + bg_ * bg_ * Denom * Denom); // force-free norm

                    //? Drift amplitude
                    const real_t beta_ref = math::sqrt(sigma0_) * d0_ * ome / (TWO * L_ * Denom * Denom);   //* β_ref = √σ₀ · dₑ · (1−ε²) / (2·L·D²)

                    //? Component decomposition along J ∥ B direction
                    const real_t beta_x = beta_ref * math::sinh(dy / L_) / F;
                    const real_t beta_y = beta_ref * eps_ * math::sin(dx / L_) / F;
                    const real_t beta_z = beta_ref;   // dominant (out-of-plane Z) component

                    //? Lorentz factor of the drift
                    const real_t beta_sq = beta_x * beta_x + beta_y * beta_y + beta_z * beta_z;
                    
                    //? Numerical safety: skip if drift is somehow superluminal
                    if (beta_sq >= ONE) { return; }
                    
                    //? Apply momentum kick
                    // q0 = −1 for electrons → receives −β kick
                    // q0 = +1 for positrons → receives +β kick
                    const real_t gd = ONE / math::sqrt(ONE - beta_sq);
                    ux1(p) += q0 * beta_x * gd;
                    ux2(p) += q0 * beta_y * gd;
                    ux3(p) += q0 * beta_z * gd;
                });
            }
        }

        // ── No post-step replenishment: closed periodic/PEC box ──────────────
        // void CustomPostStep(timestep_t, simtime_t, Domain<S, M>&) {}
    };

} // namespace user
#endif // PROBLEM_GENERATOR_H