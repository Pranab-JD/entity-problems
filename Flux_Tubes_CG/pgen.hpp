#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

/**
 * @file fluxtube.hpp
 * @brief Force-free Lundquist flux-tube collision — Entity port of Tristan user_sheet.F90
 *
 * Physics setup
 * -------------
 * Two coaxial Lundquist tubes (Bessel-function force-free equilibria) are
 * initialised side-by-side and given opposite motional kicks so that they
 * approach each other and reconnect.
 *
 * The tubes lie in the xy-plane in both 2D and 3D. In 3D the configuration
 * is uniform along z (the guide-field direction); only the xy cross-section
 * matters for the force-free equilibrium.
 *
 * Tristan normalisation (needed to understand drift_normalisation below)
 * -----------------------------------------------------------------------
 * In Tristan, grid current is stored with an implicit CC factor (speed of light
 * in code units). Conversion to physical drift velocity:
 *
 *   drift_velocity = J_grid * sqrt(sigma) * skindepth * sign(q) / CC
 *
 * In Entity, curl(B) = J directly (no CC factor), so:
 *
 *   drift_velocity = J_grid * sqrt(sigma) * skindepth * sign(q)
 *
 * Vector potential (Tristan userInitFields lines 402-408)
 * -------------------------------------------------------
 *   Az(r) = (tube_radius / alpha_t) * J0(alpha_t * r/tube_radius)   r < tube_radius
 *   Az(r) = (tube_radius / alpha_t) * J0(alpha_t)                   r >= tube_radius
 *
 * In-plane B from discrete curl of Az:
 *   Bx(i,j) =  Az(i,j+1) - Az(i,j)     [Tristan line 429, = dAz/dy]
 *   By(i,j) =  Az(i,j)   - Az(i+1,j)   [Tristan line 430, = -dAz/dx]
 *
 * Guide field:
 *   Bz = sqrt( J0(alpha_t * r/tube_radius)^2 + guide_field_floor )
 *
 * Force-free current J = curl(B) [Tristan one-sided stencil, lines 479-481]:
 *   Jx(i,j) = Bz(i,j)   - Bz(i,j-1)      [/cell_width_y]
 *   Jy(i,j) = Bz(i-1,j) - Bz(i,j)        [/cell_width_x]
 *   Jz(i,j) = Bx(i,j-1) - Bx(i,j)        [/cell_width_y]
 *           - By(i-1,j) + By(i,j)         [/cell_width_x]
 *
 * Motional electric field (tubes kicked toward each other along y):
 *   Tube 1: Ex = -kick_velocity * Bz,  Ez = +kick_velocity * Bx
 *   Tube 2: Ex = +kick_velocity * Bz,  Ez = -kick_velocity * Bx
 *
 * Particle initialisation — mirrors Tristan's userInitParticles exactly
 * -----------------------------------------------------------------------
 *  1. Build drift_velocity from J (current-driven) + E×B contributions.
 *     E×B uses Tristan denominator (Bx^2 + Bz^2), not full |B|^2.
 *  2. Probabilistic reflection: particles moving counter to drift are reflected
 *     with probability -drift.u/gamma before the Lorentz boost.
 *  3. Lorentz boost into the drift frame.
 *  4. [apply_charge_correction=true] Perturb particle weights by div(E) = rho
 *     to seed the charge-density perturbation satisfying Gauss's law.
 *
 * 2D vs 3D
 * --------
 *  - coord_t<D> always has exactly D components; all initialisations use {}.
 *  - z cell index and mesh.extent(in::x3) are guarded with if constexpr (D==Dim::_3D).
 *  - In 2D the guide field (bx3 / Bz) is the out-of-plane component.
 *  - In 3D the equilibrium is uniform in z.
 *
 * Input parameters (setup.*)
 * --------------------------
 *   background_temperature    thermal temperature (m_e c^2)     [default 1e-2]
 *   kick_velocity             tube approach velocity / c        [default 0.1]
 *   guide_field_floor         floor keeping Bz > 0 everywhere   [default 0.01]
 *   apply_charge_correction   weight correction for Gauss's law  [default true]
 *   single_tube               initialise only tube 1            [default false]
 *   n_smooth_passes           field smoothing passes            [default 32]
 *   buffer_zone_ppc           target ppc in x-buffer zones      [default 8.0]
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

// ---------------------------------------------------------------------------
// GPU-compatible Bessel functions.
// CUDA/HIP provide device intrinsics j0/j1; on CPU we use std::cyl_bessel_j.
// ---------------------------------------------------------------------------
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

    template <Dimension D>
    struct InitFields
    {
        // =====================================================================
        // Lundquist flux-tube field initialisation.
        // Follows Ripperda 2019 eq. 20-21 and Tristan user_sheet.F90.
        //
        // Magnetic field inside each tube (cylindrical, r <= tube_radius):
        //   B_phi(r) = J1(alpha_t * r/tube_radius)
        //   B_z(r)   = sqrt( J0(alpha_t * r/tube_radius)^2 + guide_field_floor )
        //
        // alpha_t = 3.8317... is the first zero of J1:
        //   - B_phi vanishes at r = tube_radius (no surface current)
        //   - total current per tube is zero (force-free, zero net current)
        //
        // guide_field_floor > 0 keeps B_z strictly positive everywhere,
        // preventing the sign reversal of the unmodified Lundquist tube.
        //
        // In-plane fields from vector potential:
        //   Az(r) = (tube_radius / alpha_t) * J0(alpha_t * r/tube_radius)
        //   Bx = dAz/dy,   By = -dAz/dx
        //
        // Force-free current (verified: J_phi/B_phi = J_z/B_z):
        //   J_z   = (alpha_t / tube_radius) * J0(alpha_t * r/tube_radius)
        //   J_phi = J_z * B_phi / B_z
        // =====================================================================

        // alpha_t: first zero of J1.
        // Both papers write "first root of J0" but give 3.8317, which is the
        // first zero of J1. The tube boundary condition (no surface current)
        // requires J1 = 0, confirming alpha_t is the first zero of J1.
        static constexpr real_t alpha_t { static_cast<real_t>(3.8317059702075125) };

        InitFields() = default;

        InitFields(real_t tube_radius_,
                   real_t tube1_x_centre_,  real_t tube1_y_centre_,
                   real_t tube2_x_centre_,  real_t tube2_y_centre_,
                   real_t cell_width_x_,    real_t cell_width_y_,
                   real_t guide_field_floor_,
                   real_t kick_velocity_,
                   bool   single_tube_,
                   int    n_smooth_passes_ = 32)
            : tube_radius        { tube_radius_        }
            , tube1_x_centre     { tube1_x_centre_     }
            , tube1_y_centre     { tube1_y_centre_     }
            , tube2_x_centre     { tube2_x_centre_     }
            , tube2_y_centre     { tube2_y_centre_     }
            , cell_width_x       { cell_width_x_       }
            , cell_width_y       { cell_width_y_       }
            , guide_field_floor  { guide_field_floor_  }
            , kick_velocity      { kick_velocity_      }
            , single_tube        { single_tube_        }
            , n_smooth_passes    { n_smooth_passes_    }
        {
            // Pre-compute exterior Bz = sqrt( J0(alpha_t)^2 + guide_field_floor ).
            // J0(alpha_t) ~ -0.4028 (NOT zero — alpha_t is the zero of J1, not J0).
            // This is the uniform guide field outside both tubes.
            const real_t J0_at_tube_boundary = bessel_J0(alpha_t);
            Bz_exterior = math::sqrt(SQR(J0_at_tube_boundary) + guide_field_floor);
        }

        // ------------------------------------------------------------------
        //  vector_potential_Az: Az for a single tube centred at (x_centre, y_centre).
        //
        //  Az(r) = (tube_radius / alpha_t) * J0(alpha_t * r/tube_radius)   inside
        //  Az(r) = (tube_radius / alpha_t) * J0(alpha_t)                   outside
        //  (constant outside → curl = 0 → Bx = By = 0 outside)
        // ------------------------------------------------------------------
        KOKKOS_INLINE_FUNCTION
        real_t vector_potential_Az(real_t x_position, real_t y_position,
                                   real_t x_centre,   real_t y_centre) const
        {
            const real_t radial_distance   = math::sqrt(SQR(x_position - x_centre)
                                                      + SQR(y_position - y_centre));
            const real_t normalised_radius = radial_distance / tube_radius;
            const real_t J0_argument       = (normalised_radius < ONE)
                                           ? alpha_t * normalised_radius
                                           : alpha_t;
            return (tube_radius / alpha_t) * bessel_J0(J0_argument);
        }

        // ------------------------------------------------------------------
        //  Bz_inside_tube: guide field at normalised_radius inside a tube.
        //  Bz = sqrt( J0(alpha_t * normalised_radius)^2 + guide_field_floor )
        //  Strictly positive for all radii when guide_field_floor > 0.
        //  Peak at centre: sqrt(1 + guide_field_floor) ~ 1.005 for floor=0.01.
        //  Minimum at J0=0 (normalised_radius ~ 0.628): sqrt(guide_field_floor) = 0.1.
        // ------------------------------------------------------------------
        KOKKOS_INLINE_FUNCTION
        real_t Bz_inside_tube(real_t normalised_radius) const
        {
            const real_t J0_value = bessel_J0(alpha_t * normalised_radius);
            return math::sqrt(SQR(J0_value) + guide_field_floor);
        }

        // ------------------------------------------------------------------
        //  bx1 = Bx = dAz/dy
        //  Discrete finite difference of Az, matching Tristan line 429:
        //    Bx(i,j) = Az(i,j+1) - Az(i,j)
        //  Called at Bx stagger position (x, y + 1/2*cell_width_y).
        // ------------------------------------------------------------------
        KOKKOS_INLINE_FUNCTION
        real_t bx1(const coord_t<D>& x_Ph) const
        {
            const real_t x_stagger = x_Ph[0];
            const real_t y_stagger = x_Ph[1];

            real_t Az_j_plus_1 = vector_potential_Az(x_stagger,
                                                     y_stagger + HALF * cell_width_y,
                                                     tube1_x_centre, tube1_y_centre);
            real_t Az_j        = vector_potential_Az(x_stagger,
                                                     y_stagger - HALF * cell_width_y,
                                                     tube1_x_centre, tube1_y_centre);
            if (!single_tube) {
                Az_j_plus_1 += vector_potential_Az(x_stagger,
                                                   y_stagger + HALF * cell_width_y,
                                                   tube2_x_centre, tube2_y_centre);
                Az_j        += vector_potential_Az(x_stagger,
                                                   y_stagger - HALF * cell_width_y,
                                                   tube2_x_centre, tube2_y_centre);
            }
            return Az_j_plus_1 - Az_j;
        }

        // ------------------------------------------------------------------
        //  bx2 = By = -dAz/dx
        //  Discrete finite difference of Az, matching Tristan line 430:
        //    By(i,j) = Az(i,j) - Az(i+1,j)
        //  Called at By stagger position (x + 1/2*cell_width_x, y).
        // ------------------------------------------------------------------
        KOKKOS_INLINE_FUNCTION
        real_t bx2(const coord_t<D>& x_Ph) const
        {
            const real_t x_stagger = x_Ph[0];
            const real_t y_stagger = x_Ph[1];

            real_t Az_i        = vector_potential_Az(x_stagger - HALF * cell_width_x,
                                                     y_stagger,
                                                     tube1_x_centre, tube1_y_centre);
            real_t Az_i_plus_1 = vector_potential_Az(x_stagger + HALF * cell_width_x,
                                                     y_stagger,
                                                     tube1_x_centre, tube1_y_centre);
            if (!single_tube) {
                Az_i        += vector_potential_Az(x_stagger - HALF * cell_width_x,
                                                   y_stagger,
                                                   tube2_x_centre, tube2_y_centre);
                Az_i_plus_1 += vector_potential_Az(x_stagger + HALF * cell_width_x,
                                                   y_stagger,
                                                   tube2_x_centre, tube2_y_centre);
            }
            return Az_i - Az_i_plus_1;
        }

        // ------------------------------------------------------------------
        //  bx3 = Bz: guide field.
        //  Inside each tube: Bz_inside_tube(normalised_radius).
        //  Outside both tubes: Bz_exterior (uniform, continuous at boundary).
        // ------------------------------------------------------------------
        KOKKOS_INLINE_FUNCTION
        real_t bx3(const coord_t<D>& x_Ph) const
        {
            const real_t normalised_radius_tube1 =
                math::sqrt(SQR(x_Ph[0] - tube1_x_centre)
                         + SQR(x_Ph[1] - tube1_y_centre)) / tube_radius;
            if (normalised_radius_tube1 < ONE)
                return Bz_inside_tube(normalised_radius_tube1);

            if (!single_tube) {
                const real_t normalised_radius_tube2 =
                    math::sqrt(SQR(x_Ph[0] - tube2_x_centre)
                             + SQR(x_Ph[1] - tube2_y_centre)) / tube_radius;
                if (normalised_radius_tube2 < ONE)
                    return Bz_inside_tube(normalised_radius_tube2);
            }
            return Bz_exterior;
        }

        // ------------------------------------------------------------------
        //  ex1/ex2/ex3: motional E = -kick_velocity x B inside each tube.
        //
        //  Tube 1 kicked +y (v = +kick_velocity * y_hat):
        //    Ex = -kick_velocity * Bz,   Ey = 0,   Ez = +kick_velocity * Bx
        //  Tube 2 kicked -y (v = -kick_velocity * y_hat):
        //    Ex = +kick_velocity * Bz,   Ey = 0,   Ez = -kick_velocity * Bx
        //  Outside both tubes: E = 0.
        // ------------------------------------------------------------------
        KOKKOS_INLINE_FUNCTION
        real_t ex1(const coord_t<D>& x_Ph) const
        {
            if (kick_velocity == ZERO) return ZERO;

            const real_t normalised_radius_tube1 =
                math::sqrt(SQR(x_Ph[0] - tube1_x_centre)
                         + SQR(x_Ph[1] - tube1_y_centre)) / tube_radius;
            if (normalised_radius_tube1 < ONE)
                return -kick_velocity * Bz_inside_tube(normalised_radius_tube1);

            if (!single_tube) {
                const real_t normalised_radius_tube2 =
                    math::sqrt(SQR(x_Ph[0] - tube2_x_centre)
                             + SQR(x_Ph[1] - tube2_y_centre)) / tube_radius;
                if (normalised_radius_tube2 < ONE)
                    return +kick_velocity * Bz_inside_tube(normalised_radius_tube2);
            }
            return ZERO;
        }

        KOKKOS_INLINE_FUNCTION
        real_t ex2(const coord_t<D>&) const { return ZERO; }

        KOKKOS_INLINE_FUNCTION
        real_t ex3(const coord_t<D>& x_Ph) const
        {
            if (kick_velocity == ZERO) return ZERO;

            const real_t x_stagger = x_Ph[0];
            const real_t y_stagger = x_Ph[1];

            const real_t normalised_radius_tube1 =
                math::sqrt(SQR(x_stagger - tube1_x_centre)
                         + SQR(y_stagger - tube1_y_centre)) / tube_radius;
            if (normalised_radius_tube1 < ONE) {
                // Bx at Ez stagger position (same as Ez: no half-cell offset needed)
                const real_t Az_j_plus_1 = vector_potential_Az(x_stagger,
                                                                y_stagger + HALF * cell_width_y,
                                                                tube1_x_centre, tube1_y_centre);
                const real_t Az_j        = vector_potential_Az(x_stagger,
                                                                y_stagger - HALF * cell_width_y,
                                                                tube1_x_centre, tube1_y_centre);
                const real_t Bx_at_Ez    = Az_j_plus_1 - Az_j;
                return +kick_velocity * Bx_at_Ez;
            }

            if (!single_tube) {
                const real_t normalised_radius_tube2 =
                    math::sqrt(SQR(x_stagger - tube2_x_centre)
                             + SQR(y_stagger - tube2_y_centre)) / tube_radius;
                if (normalised_radius_tube2 < ONE) {
                    const real_t Az_j_plus_1 = vector_potential_Az(x_stagger,
                                                                    y_stagger + HALF * cell_width_y,
                                                                    tube2_x_centre, tube2_y_centre);
                    const real_t Az_j        = vector_potential_Az(x_stagger,
                                                                    y_stagger - HALF * cell_width_y,
                                                                    tube2_x_centre, tube2_y_centre);
                    const real_t Bx_at_Ez    = Az_j_plus_1 - Az_j;
                    return -kick_velocity * Bx_at_Ez;
                }
            }
            return ZERO;
        }

        // ------------------------------------------------------------------
        //  J_inside_tube: analytic force-free current J = curl(B) for one tube.
        //
        //  Derivation (cylindrical, J || B condition):
        //    J_z   = (alpha_t / tube_radius) * J0(alpha_t * normalised_radius)
        //    J_phi = J_z * B_phi / B_z
        //          = (alpha_t / tube_radius) * J0 * J1 / B_z
        //    J_phi / r = (alpha_t / tube_radius) * J0 * J1 / (B_z * radial_distance)
        //
        //  Verification: J_phi/B_phi = J_z/B_z = alpha_t*J0/(tube_radius*B_z) ✓
        //
        //  In Cartesian:
        //    Jx = -(J_phi / radial_distance) * (y - y_centre)
        //    Jy = +(J_phi / radial_distance) * (x - x_centre)
        //    Jz =  (alpha_t / tube_radius)   * J0
        //
        //  Returns (0, 0, 0) outside tube or within epsilon of tube axis.
        // ------------------------------------------------------------------
        KOKKOS_INLINE_FUNCTION
        void J_inside_tube(real_t x_position, real_t y_position,
                           real_t x_centre,   real_t y_centre,
                           real_t& Jx, real_t& Jy, real_t& Jz) const
        {
            const real_t delta_x         = x_position - x_centre;
            const real_t delta_y         = y_position - y_centre;
            const real_t radial_distance = math::sqrt(SQR(delta_x) + SQR(delta_y));
            const real_t normalised_radius = radial_distance / tube_radius;

            Jx = ZERO; Jy = ZERO; Jz = ZERO;
            if (normalised_radius >= ONE ||
                radial_distance < static_cast<real_t>(1e-10) * tube_radius)
                return;

            const real_t J0_value  = bessel_J0(alpha_t * normalised_radius);
            const real_t J1_value  = bessel_J1(alpha_t * normalised_radius);
            const real_t Bz_local  = Bz_inside_tube(normalised_radius);

            Jz = (alpha_t / tube_radius) * J0_value;

            const real_t Jphi_over_r = (alpha_t / tube_radius) * J0_value * J1_value
                                     / (Bz_local * radial_distance);
            Jx = -Jphi_over_r * delta_y;
            Jy = +Jphi_over_r * delta_x;
        }

        KOKKOS_INLINE_FUNCTION
        real_t jx1(const coord_t<D>& x_Ph) const
        {
            real_t Jx = ZERO, Jy = ZERO, Jz = ZERO;
            J_inside_tube(x_Ph[0], x_Ph[1], tube1_x_centre, tube1_y_centre, Jx, Jy, Jz);
            if (!single_tube) {
                real_t Jx2 = ZERO, Jy2 = ZERO, Jz2 = ZERO;
                J_inside_tube(x_Ph[0], x_Ph[1], tube2_x_centre, tube2_y_centre, Jx2, Jy2, Jz2);
                Jx += Jx2;
            }
            return Jx;
        }

        KOKKOS_INLINE_FUNCTION
        real_t jx2(const coord_t<D>& x_Ph) const
        {
            real_t Jx = ZERO, Jy = ZERO, Jz = ZERO;
            J_inside_tube(x_Ph[0], x_Ph[1], tube1_x_centre, tube1_y_centre, Jx, Jy, Jz);
            if (!single_tube) {
                real_t Jx2 = ZERO, Jy2 = ZERO, Jz2 = ZERO;
                J_inside_tube(x_Ph[0], x_Ph[1], tube2_x_centre, tube2_y_centre, Jx2, Jy2, Jz2);
                Jy += Jy2;
            }
            return Jy;
        }

        KOKKOS_INLINE_FUNCTION
        real_t jx3(const coord_t<D>& x_Ph) const
        {
            real_t Jx = ZERO, Jy = ZERO, Jz = ZERO;
            J_inside_tube(x_Ph[0], x_Ph[1], tube1_x_centre, tube1_y_centre, Jx, Jy, Jz);
            if (!single_tube) {
                real_t Jx2 = ZERO, Jy2 = ZERO, Jz2 = ZERO;
                J_inside_tube(x_Ph[0], x_Ph[1], tube2_x_centre, tube2_y_centre, Jx2, Jy2, Jz2);
                Jz += Jz2;
            }
            return Jz;
        }

        // ------------------------------------------------------------------
        //  Data members
        // ------------------------------------------------------------------
        real_t tube_radius        { ZERO  };
        real_t tube1_x_centre     { ZERO  };
        real_t tube1_y_centre     { ZERO  };
        real_t tube2_x_centre     { ZERO  };
        real_t tube2_y_centre     { ZERO  };
        real_t cell_width_x       { ONE   };
        real_t cell_width_y       { ONE   };
        int    n_smooth_passes    { 32    };
        real_t guide_field_floor  { ZERO  };
        real_t kick_velocity      { ZERO  };
        real_t Bz_exterior        { ZERO  };
        bool   single_tube        { false };
    };


    // =========================================================================
    //  PGen
    // =========================================================================
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
        real_t background_temperature  { static_cast<real_t>(1e-2) };
        real_t kick_velocity           { static_cast<real_t>(0.1)  };
        real_t guide_field_floor       { static_cast<real_t>(0.01) };
        bool   apply_charge_correction { true  };
        bool   single_tube             { false };
        int    n_smooth_passes         { 32    };
        real_t buffer_zone_ppc         { static_cast<real_t>(8.0)  };

        // Global x extents cached for buffer-zone boundaries in InitPrtls.
        // InitPrtls receives a local domain tile; we need the global box here.
        real_t global_x_min { ZERO }, global_x_max { ZERO };

    public:
        InitFields<D> init_flds;

        // --------------------------------------------------------------------
        //  Constructor
        // --------------------------------------------------------------------
        inline PGen(const SimulationParams& p, metadomain_type& md)
            : Base { p }
            , global_domain { md }
        {
            background_temperature  = p.template get<real_t>("setup.background_temperature",
                                                              static_cast<real_t>(1e-2));
            kick_velocity           = p.template get<real_t>("setup.kick_velocity",
                                                              static_cast<real_t>(0.1));
            guide_field_floor       = p.template get<real_t>("setup.guide_field_floor",
                                                              static_cast<real_t>(0.01));
            apply_charge_correction = p.template get<bool>  ("setup.apply_charge_correction", true);
            single_tube             = p.template get<bool>  ("setup.single_tube", false);
            n_smooth_passes         = p.template get<int>   ("setup.n_smooth_passes", 32);
            buffer_zone_ppc         = p.template get<real_t>("setup.buffer_zone_ppc",
                                                              static_cast<real_t>(8.0));

            const auto& mesh   = md.mesh();
            global_x_min = mesh.extent(in::x1).first;
            global_x_max = mesh.extent(in::x1).second;
            const real_t global_y_min = mesh.extent(in::x2).first;
            const real_t global_y_max = mesh.extent(in::x2).second;
            const real_t Lx = global_x_max - global_x_min;
            const real_t Ly = global_y_max - global_y_min;

            // Tube radius default = 0.99 * Lx/4, matching Tristan line 380.
            // The 0.99 factor keeps the tube edge slightly inside the domain.
            // Override with setup.tube_radius in the input file.
            const real_t tube_radius = p.template get<real_t>(
                "setup.tube_radius",
                Lx * static_cast<real_t>(0.99 * 0.25));

            // Tube centres: both on x midline, offset by ±tube_radius in y.
            // Matching Tristan: x1=x2=0.5*sx, y1=0.5*sy-r_j, y2=0.5*sy+r_j.
            const real_t domain_x_centre = global_x_min + HALF * Lx;
            const real_t domain_y_centre = global_y_min + HALF * Ly;

            const real_t dx = Lx / static_cast<real_t>(mesh.n_active(in::x1));
            const real_t dy = Ly / static_cast<real_t>(mesh.n_active(in::x2));

            init_flds = InitFields<D>(
                tube_radius,
                domain_x_centre, domain_y_centre - tube_radius,   // tube 1 centre
                domain_x_centre, domain_y_centre + tube_radius,   // tube 2 centre
                dx, dy,
                guide_field_floor, kick_velocity, single_tube, n_smooth_passes);
        }

        inline PGen() {}

        // --------------------------------------------------------------------
        //  MatchFields
        // --------------------------------------------------------------------
        auto MatchFields(simtime_t) const -> InitFields<D>
        {
            return init_flds;
        }

        // --------------------------------------------------------------------
        //  InitPrtls
        //  Mirrors Tristan's userInitParticles() precisely.
        //
        //  Stage 1 — inject thermal Maxwellian plasma in three regions:
        //            inner (full ppc) and two x-buffer zones (reduced ppc).
        //  Stage 2 — apply current + E×B drift with probabilistic reflection
        //            and Lorentz boost.
        //  Stage 3 — [apply_charge_correction] weight perturbation from div(E).
        // --------------------------------------------------------------------
        inline void InitPrtls(Domain<S, M>& domain)
        {
            const auto& mesh = domain.mesh;
            const real_t Lx = global_x_max - global_x_min;

            // Drift normalisation: beta = J * sqrt(sigma) * skindepth * sign(q)
            // sigma = (skindepth / larmor)^2 from Entity parameters.cpp
            const real_t skindepth         = params.template get<real_t>("scales.skindepth0");
            const real_t larmor            = params.template get<real_t>("scales.larmor0");
            const real_t sigma             = SQR(skindepth / larmor);
            const real_t drift_normalisation = skindepth * math::sqrt(sigma)
                                             / static_cast<real_t>(2.0);

            const real_t ppc0             = params.template get<real_t>("particles.ppc0",
                                                                         static_cast<real_t>(4));
            const real_t buffer_zone_frac = buffer_zone_ppc / ppc0;

            // Buffer zone x boundaries (Tristan: 0.2*sx to 0.8*sx)
            const real_t buffer_x_inner_lo = global_x_min
                                           + static_cast<real_t>(0.2) * Lx;
            const real_t buffer_x_inner_hi = global_x_min
                                           + static_cast<real_t>(0.8) * Lx;

            // ================================================================
            //  STAGE 1: Inject thermal Maxwellian plasma.
            //
            //  Three injection calls match Tristan's three fillRegionWithThermal-
            //  Plasma calls (inner + left buffer + right buffer).
            //  make_injection_box is dimension-agnostic: loops over M::Dim and
            //  pushes Range::All for every dimension that is not x (d != 0).
            // ================================================================
            {
                const auto thermal_distribution = std::make_pair(background_temperature,
                                                                  background_temperature);
                const auto zero_drift = std::make_pair(
                    std::vector<real_t>{ ZERO, ZERO, ZERO },
                    std::vector<real_t>{ ZERO, ZERO, ZERO });

                auto make_injection_box = [&](real_t x_lo,
                                              real_t x_hi) -> boundaries_t<real_t>
                {
                    boundaries_t<real_t> injection_box;
                    for (auto d = 0u; d < M::Dim; ++d)
                    {
                        if (d == 0u) injection_box.push_back({ x_lo, x_hi });
                        else         injection_box.push_back(Range::All);
                    }
                    return injection_box;
                };

                // (a) Inner region [0.2 Lx, 0.8 Lx] at full ppc
                arch::InjectUniformMaxwellians<S, M>(
                    params, domain, ONE, thermal_distribution, { 1, 2 },
                    zero_drift, false,
                    make_injection_box(buffer_x_inner_lo, buffer_x_inner_hi));

                // (b) Left buffer [0, 0.2 Lx] at reduced ppc
                arch::InjectUniformMaxwellians<S, M>(
                    params, domain, buffer_zone_frac, thermal_distribution, { 1, 2 },
                    zero_drift, false,
                    make_injection_box(global_x_min, buffer_x_inner_lo));

                // (c) Right buffer [0.8 Lx, Lx] at reduced ppc
                arch::InjectUniformMaxwellians<S, M>(
                    params, domain, buffer_zone_frac, thermal_distribution, { 1, 2 },
                    zero_drift, false,
                    make_injection_box(buffer_x_inner_hi, global_x_max));
            }

            // ================================================================
            //  STAGE 2: Current-driven + E×B drift boost.
            //
            //  Mirrors Tristan userInitParticles lines 128-166 and 171-180.
            //
            //  For each particle:
            //    (a) Compute drift_velocity = sign(q) * drift_normalisation * J_analytic
            //                               + (E×B) / (Bx^2 + Bz^2)
            //    (b) Probabilistic reflection (Tristan lines 153-158)
            //    (c) Lorentz boost into drift frame (Tristan lines 159-166)
            //    (d) [apply_charge_correction] weight perturbation from div(E)
            //
            //  J is evaluated analytically (not from grid) because Entity does
            //  not store J in em_field — that array holds only E and B.
            //  Reading em_field at jx1/jx2/jx3 indices would silently return Ex.
            // ================================================================
            {
                const auto em_field = domain.fields.em;
                const auto rng_pool = domain.random_pool();

                // z-extent: queried only in 3D
                real_t z_origin { ZERO };
                real_t cell_width_z { ONE };
                if constexpr (D == Dim::_3D)
                {
                    z_origin     = mesh.extent(in::x3).first;
                    cell_width_z = (mesh.extent(in::x3).second - z_origin)
                                 / static_cast<real_t>(mesh.n_active(in::x3));
                }

                const bool   do_charge_correction = apply_charge_correction;
                const real_t drift_norm            = drift_normalisation;
                const real_t sigma_local           = sigma;
                const real_t ppc0_local            = ppc0;
                const auto   init                  = init_flds;

                // Physical coordinate origins and cell sizes of this MPI tile.
                const real_t x_origin = mesh.extent(in::x1).first;
                const real_t y_origin = mesh.extent(in::x2).first;
                const real_t dx = (mesh.extent(in::x1).second - x_origin)
                                / static_cast<real_t>(mesh.n_active(in::x1));
                const real_t dy = (mesh.extent(in::x2).second - y_origin)
                                / static_cast<real_t>(mesh.n_active(in::x2));

                for (auto s = 0u; s < domain.species.size(); ++s)
                {
                    auto& sp             = domain.species[s];
                    const real_t charge  = sp.charge();

                    // Extract Kokkos view handles before the lambda.
                    // sp has a deleted copy constructor; never capture by [=].
                    const auto cell_x    = sp.i1;
                    const auto cell_y    = sp.i2;
                    const auto frac_x    = sp.dx1;
                    const auto frac_y    = sp.dx2;
                    const auto tag       = sp.tag;
                    const auto ux1   = sp.ux1;
                    const auto ux2   = sp.ux2;
                    const auto ux3   = sp.ux3;
                    const auto weight    = sp.weight;

                    // cell_z / frac_z: placeholder = cell_x / frac_x in 2D
                    // (never used), reassigned to sp.i3/sp.dx3 in 3D.
                    [[maybe_unused]] auto cell_z = sp.i1;
                    [[maybe_unused]] auto frac_z = sp.dx1;
                    if constexpr (D == Dim::_3D)
                    {
                        cell_z = sp.i3;
                        frac_z = sp.dx3;
                    }

                    Kokkos::parallel_for(
                        "FluxTubeDrift",
                        sp.rangeActiveParticles(),
                        KOKKOS_LAMBDA(index_t p)
                    {
                        if (tag(p) == ParticleTag::dead) return;

                        // Integer cell indices and sub-cell fractions
                        const auto   ix = cell_x(p);
                        const auto   iy = cell_y(p);
                        const real_t sx = static_cast<real_t>(frac_x(p));   // x sub-cell fraction
                        const real_t sy = static_cast<real_t>(frac_y(p));   // y sub-cell fraction

                        int iz = 0;
                        if constexpr (D == Dim::_3D)
                            iz = static_cast<int>(cell_z(p));

                        // Physical position of the particle
                        coord_t<D> particle_position {};
                        particle_position[0] = x_origin
                                             + (static_cast<real_t>(ix) + sx) * dx;
                        particle_position[1] = y_origin
                                             + (static_cast<real_t>(iy) + sy) * dy;
                        if constexpr (D == Dim::_3D)
                        {
                            const real_t sz = static_cast<real_t>(frac_z(p));
                            particle_position[2] = z_origin
                                                 + (static_cast<real_t>(iz) + sz)
                                                 * cell_width_z;
                        }

                        // em_field reader — rank-3 in 2D: (ix, iy, comp)
                        //                   rank-4 in 3D: (ix, iy, iz, comp)
                        auto read_em_field = [&](auto I, auto J, auto COMP) -> real_t {
                            if constexpr (D == Dim::_2D)
                                return em_field(I, J, COMP);
                            else
                                return em_field(I, J, iz, COMP);
                        };

                        // Bilinear interpolation weights (Tristan interpFromEdges)
                        // f(particle) = f(ix,  iy  )*(1-sx)*(1-sy)
                        //             + f(ix+1,iy  )*sx    *(1-sy)
                        //             + f(ix,  iy+1)*(1-sx)*sy
                        //             + f(ix+1,iy+1)*sx    *sy
                        const real_t w_00 = (ONE - sx) * (ONE - sy);
                        const real_t w_10 = sx          * (ONE - sy);
                        const real_t w_01 = (ONE - sx)  * sy;
                        const real_t w_11 = sx           * sy;

                        // (a) Current-driven drift
                        // drift_velocity = sign(q) * drift_normalisation * J_analytic
                        // J evaluated analytically from the force-free Bessel formula.
                        real_t drift_x = drift_norm * charge * init.jx1(particle_position);
                        real_t drift_y = drift_norm * charge * init.jx2(particle_position);
                        real_t drift_z = drift_norm * charge * init.jx3(particle_position);

                        // (b) E×B drift — Tristan lines 141-146
                        // drift_y += (Ez*Bx - Ex*Bz) / (Bx^2 + Bz^2)
                        {
                            const real_t Ex = w_00 * read_em_field(ix,   iy,   em::ex1)
                                            + w_10 * read_em_field(ix+1, iy,   em::ex1)
                                            + w_01 * read_em_field(ix,   iy+1, em::ex1)
                                            + w_11 * read_em_field(ix+1, iy+1, em::ex1);
                            const real_t Ez = w_00 * read_em_field(ix,   iy,   em::ex3)
                                            + w_10 * read_em_field(ix+1, iy,   em::ex3)
                                            + w_01 * read_em_field(ix,   iy+1, em::ex3)
                                            + w_11 * read_em_field(ix+1, iy+1, em::ex3);
                            const real_t Bx = w_00 * read_em_field(ix,   iy,   em::bx1)
                                            + w_10 * read_em_field(ix+1, iy,   em::bx1)
                                            + w_01 * read_em_field(ix,   iy+1, em::bx1)
                                            + w_11 * read_em_field(ix+1, iy+1, em::bx1);
                            const real_t Bz = w_00 * read_em_field(ix,   iy,   em::bx3)
                                            + w_10 * read_em_field(ix+1, iy,   em::bx3)
                                            + w_01 * read_em_field(ix,   iy+1, em::bx3)
                                            + w_11 * read_em_field(ix+1, iy+1, em::bx3);

                            // Tristan uses Bx^2 + Bz^2 as denominator (not full |B|^2)
                            const real_t BxBz_squared = Bx*Bx + Bz*Bz;
                            if (BxBz_squared > ZERO)
                                drift_y += (Ez*Bx - Ex*Bz) / BxBz_squared;
                        }

                        real_t drift_speed_squared = drift_x*drift_x
                                                   + drift_y*drift_y
                                                   + drift_z*drift_z;

                        // Safety clamp — mirrors Tristan's beta_sq >= 20 check
                        constexpr real_t MAX_DRIFT_SPEED    = static_cast<real_t>(0.99);
                        constexpr real_t MAX_DRIFT_SPEED_SQ = MAX_DRIFT_SPEED * MAX_DRIFT_SPEED;
                        if (drift_speed_squared >= MAX_DRIFT_SPEED_SQ)
                        {
                            const real_t rescale = MAX_DRIFT_SPEED
                                                 / math::sqrt(drift_speed_squared);
                            drift_x            *= rescale;
                            drift_y            *= rescale;
                            drift_z            *= rescale;
                            drift_speed_squared = MAX_DRIFT_SPEED_SQ;
                        }

                        if (drift_speed_squared <= ZERO) return;

                        real_t ux  = ux1(p);
                        real_t uy  = ux2(p);
                        real_t uz  = ux3(p);
                        real_t lorentz_factor_thermal = math::sqrt(ONE + ux*ux
                                                                       + uy*uy
                                                                       + uz*uz);

                        // (c) Probabilistic reflection — Tristan lines 153-158.
                        // Particles whose thermal velocity opposes the drift are
                        // reflected about the drift direction with probability
                        // -drift.u/gamma, sharpening the initial distribution.
                        const real_t drift_dot_momentum = ux*drift_x
                                                        + uy*drift_y
                                                        + uz*drift_z;
                        auto         rng_state          = rng_pool.get_state();
                        const real_t random_number      = Random<real_t>(rng_state);
                        rng_pool.free_state(rng_state);

                        if (-drift_dot_momentum / lorentz_factor_thermal > random_number)
                        {
                            const real_t projection = drift_dot_momentum
                                                    / drift_speed_squared;
                            ux -= TWO * projection * drift_x;
                            uy -= TWO * projection * drift_y;
                            uz -= TWO * projection * drift_z;
                        }

                        // (d) Lorentz boost into drift frame — Tristan lines 159-166.
                        const real_t lorentz_factor_drift = ONE
                                                          / math::sqrt(ONE - drift_speed_squared);
                        const real_t boosted_ux = lorentz_factor_drift * drift_x;
                        const real_t boosted_uy = lorentz_factor_drift * drift_y;
                        const real_t boosted_uz = lorentz_factor_drift * drift_z;
                        const real_t boost  = (ux*boosted_ux + uy*boosted_uy + uz*boosted_uz)
                                             / (lorentz_factor_drift + ONE)
                                             + lorentz_factor_thermal;
                        ux1(p) = ux + boost * boosted_ux;
                        ux2(p) = uy + boost * boosted_uy;
                        ux3(p) = uz + boost * boosted_uz;

                        // (e) Charge-density weight correction — Tristan lines 171-180.
                        // Seeds div(E) = rho / n0 to satisfy Gauss's law.
                        // For kick_velocity=0: E=0 so div(E)=0 and this is a no-op.
                        if (do_charge_correction)
                        {
                            const real_t divE_00 = read_em_field(ix,   iy,   em::ex1)
                                                 - read_em_field(ix-1, iy,   em::ex1);
                            const real_t divE_10 = read_em_field(ix+1, iy,   em::ex1)
                                                 - read_em_field(ix,   iy,   em::ex1);
                            const real_t divE_01 = read_em_field(ix,   iy+1, em::ex1)
                                                 - read_em_field(ix-1, iy+1, em::ex1);
                            const real_t divE_11 = read_em_field(ix+1, iy+1, em::ex1)
                                                 - read_em_field(ix,   iy+1, em::ex1);

                            const real_t interpolated_divE = w_00 * divE_00
                                                           + w_10 * divE_10
                                                           + w_01 * divE_01
                                                           + w_11 * divE_11;

                            // delta_weight = div(E) * ppc0 * charge
                            // (V0 cancels: n0 * V0 = ppc0)
                            const real_t new_weight = weight(p)
                                                    + interpolated_divE * ppc0_local * charge;
                            if (new_weight > ZERO)
                                weight(p) = new_weight;
                        }

                    }); // parallel_for FluxTubeDrift

                } // species loop

            } // stage 2 block

        } // InitPrtls

    }; // struct PGen

} // namespace user

#endif // PROBLEM_GENERATOR_H