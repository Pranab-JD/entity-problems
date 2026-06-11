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
 * The tubes lie in the xy-plane in both 2D and 3D.  In 3D the configuration
 * is uniform along z (the guide-field direction); only the xy cross-section
 * matters for the force-free equilibrium.
 *
 * Tristan normalisation reminder (needed to understand jfac below)
 * ----------------------------------------------------------------
 * In Tristan the current on the grid is deposited with an implicit factor of CC
 * (the speed of light in code units, CC = dx/dt ~ 0.45 typically).  The
 * conversion from grid current J_grid to a physical drift velocity is therefore:
 *
 *   beta = J_grid * sqrt(sigma) * skindepth * sign(q) / CC
 *
 * The CORR*CC prefactor in userInitFields and the /CC in userInitParticles cancel
 * to give exactly beta = (curl B) * sqrt(sigma) * skindepth * sign(q), which is the
 * correct force-free drift.
 *
 * In Entity, J is stored in normalised units where curl(B) = J directly
 * (no implicit CC factor).  The force-free drift velocity is therefore:
 *
 *   beta = J_grid * sqrt(sigma) * skindepth * sign(q)
 *
 * with NO division by CC.  Tristan needed /CC because its grid stored
 * J = CORR*CC*curl(B); Entity does not apply that prefactor.
 *
 * Vector potential (Tristan userInitFields, lines 402-408)
 * ---------------------------------------------------------
 *   Az(r) = 2 * r_j / alpha * J0(alpha * r/r_j)   r < r_j
 *   Az(r) = 2 * r_j / alpha * J0(alpha)            r >= r_j  [exterior: constant]
 *
 * In-plane B from discrete curl of Az:
 *   Bx(i,j) =  Az(i,j+1) - Az(i,j)     [Tristan line 429, = dAz/dy]
 *   By(i,j) =  Az(i,j)   - Az(i+1,j)   [Tristan line 430, = -dAz/dx]
 *
 * Guide field:
 *   Bz = sqrt( J0(alpha*r/r_j)^2 + c_param )
 *
 * Force-free current J = curl(B) [Tristan one-sided stencil, lines 479-481]:
 *   Jx(i,j) = Bz(i,j)   - Bz(i,j-1)          [/dy]
 *   Jy(i,j) = Bz(i-1,j) - Bz(i,j)            [/dx]
 *   Jz(i,j) = Bx(i,j-1) - Bx(i,j)            [/dy]
 *           - By(i-1,j) + By(i,j)             [/dx]
 *
 * Motional electric field (tubes kicked toward each other along y):
 *   Tube 1: Ex = -beta_kick * Bz,  Ez = +beta_kick * Bx
 *   Tube 2: Ex = +beta_kick * Bz,  Ez = -beta_kick * Bx
 *
 * Particle initialisation — follows Camille's force-free formulation
 * -------------------------------------------------------------------
 *  1. Build beta from J (current-driven) + full 3-component E×B.
 *     beta = sign(q)*J_analytic + (E×B)/|B|^2  (full |B|^2, no approximation)
 *  2. Lorentz boost thermal momentum into the drift frame.
 *     No probabilistic reflection (Tristan legacy; omitted for cleaner J deposit).
 *  3. [init_rho=true] Perturb particle weights by div(E) to satisfy Gauss's law.
 *
 * 2D vs 3D
 * --------
 *  - coord_t<D> always has exactly D components.  All initialisations use {}
 *    (value-init) rather than a fixed-length brace list.
 *  - z cell index (i3/dx3) and mesh.extent(in::x3) are guarded with
 *    if constexpr (D == Dim::_3D) throughout.
 *  - In 2D the guide field (bx3 / Bz) is the out-of-plane component; the
 *    physics is identical to Tristan's 2D setup.
 *  - In 3D the equilibrium is uniform in z; the same xy analytic fields are
 *    used everywhere along z.
 *
 * Input parameters (setup.*)
 * --------------------------
 *   background_T   thermal temperature (m_e c^2 units)   [default 1e-2]
 *   beta_kick      tube approach velocity / c            [default 0.1]
 *   c_param        guide-field floor (keeps Bz > 0)      [default 0.01]
 *   init_rho       apply charge-density weight correction [default true]
 *   single_tube    initialise only tube 1               [default false]
 *   nsmooth        smoothing passes                      [default 32]
 *   ppc_buff       target ppc in x-buffer zones          [default 8.0]
 *   seed           RNG seed (0 = random)                 [default 0]
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
KOKKOS_INLINE_FUNCTION real_t ft_j0(real_t x)
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    return j0(x);
#else
    return std::cyl_bessel_j(0, x);
#endif
}

KOKKOS_INLINE_FUNCTION real_t ft_j1(real_t x)
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    return j1(x);
#else
    return std::cyl_bessel_j(1, x);
#endif
}

namespace user
{
    using namespace ntt;

    // =========================================================================
    //  InitFields
    //
    //  Purely analytic (continuum) field functions.
    //
    //  Entity's SetEMFields_kernel calls bx1/bx2/bx3/ex1/ex2/ex3 with
    //  x_Ph already set to the correct Yee-stagger position for each
    //  component.  The functions must return the analytic field value at
    //  that coordinate — no discrete curl or stagger offsets here.
    //  (Contrast with Tristan, which builds B via a discrete curl of Az
    //  on its own grid; Entity handles the stagger internally.)
    //
    //  Lundquist tube fields (cylindrical coords, axis at (xc, yc)):
    //    Az(r)   = (2*r_j/alpha) * J0(alpha*r/r_j)    r < r_j
    //    Az(r)   = (2*r_j/alpha) * J0(alpha)           r >= r_j  [constant]
    //    Bx      = dAz/dy  =  (2/alpha)*J0'(alpha*r/r_j)*alpha/r_j*(y-yc)/r
    //            = -2 * J1(alpha*r/r_j) * (y-yc)/r
    //    By      = -dAz/dx = +2 * J1(alpha*r/r_j) * (x-xc)/r
    //    Bz      = sqrt( J0(alpha*r/r_j)^2 + c_param )
    //
    //  Outside the tube Az is constant so Bx=By=0 exactly.
    //
    //  Analytic J = curl(B):
    //    Jx = dBz/dy
    //    Jy = -dBz/dx
    //    Jz = dBy/dx - dBx/dy
    //
    //  For the Lundquist tube, Jz = alpha/r_j * Bz_phi component
    //  = 2 * alpha/r_j * J0(alpha*r/r_j)  (the force-free condition).
    //  We compute these analytically using J0/J1 and their derivatives.
    // =========================================================================
    template <Dimension D>
    struct InitFields
    {
        // =========================================================================
        // Lundquist flux-tube field initialisation
        // Follows Ripperda 2019 eq. 20-21 and Tristan user_sheet_new.F90
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
        // preventing the sign reversal that occurs in the unmodified Lundquist tube.
        //
        // In-plane fields derived from vector potential:
        //   Az(r) = (tube_radius / alpha_t) * J0(alpha_t * r/tube_radius)
        //   Bx = dAz/dy,   By = -dAz/dx   (discrete, matching Tristan)
        //
        // Force-free current J || B (verified: J_phi/B_phi = J_z/B_z):
        //   J_z   = (alpha_t / tube_radius) * J0(alpha_t * r/tube_radius)
        //   J_phi = J_z * B_phi / B_z
        // =========================================================================

        // alpha_t: first zero of J1 — tube terminated where B_phi = 0.
        // Despite both papers writing "first root of J0", 3.8317 is the first
        // zero of J1. The tube boundary condition (no surface current) requires J1 = 0.
        static constexpr real_t alpha_t { static_cast<real_t>(3.8317059702075125) };

        InitFields() = default;

        InitFields(real_t tube_radius_,
                   real_t tube1_x_centre_, real_t tube1_y_centre_,
                   real_t tube2_x_centre_, real_t tube2_y_centre_,
                   real_t cell_width_x_,  real_t cell_width_y_,
                   real_t guide_field_floor_,
                   real_t kick_velocity_,
                   bool   single_tube_,
                   int    n_smooth_passes_ = 32)
            : tube_radius         { tube_radius_        }
            , tube1_x_centre      { tube1_x_centre_     }
            , tube1_y_centre      { tube1_y_centre_     }
            , tube2_x_centre      { tube2_x_centre_     }
            , tube2_y_centre      { tube2_y_centre_     }
            , cell_width_x        { cell_width_x_       }
            , cell_width_y        { cell_width_y_       }
            , guide_field_floor   { guide_field_floor_  }
            , kick_velocity       { kick_velocity_      }
            , single_tube         { single_tube_        }
            , n_smooth_passes     { n_smooth_passes_    }
        {
            // Pre-compute exterior B_z = sqrt( J0(alpha_t)^2 + guide_field_floor ).
            // J0(alpha_t) ~ -0.4028 (NOT zero — alpha_t is the zero of J1, not J0).
            // This is the uniform guide field outside both tubes.
            const real_t J0_at_tube_boundary = lm_j0(alpha_t);
            Bz_exterior = math::sqrt(SQR(J0_at_tube_boundary) + guide_field_floor);
        }

        // ------------------------------------------------------------------
        //  vector_potential_Az: Az at physical position (x_position, y_position)
        //  for a single tube centred at (x_centre, y_centre).
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
            return (tube_radius / alpha_t) * lm_j0(J0_argument);
        }

        // ------------------------------------------------------------------
        //  B_z_inside_tube: guide field at normalised radius inside a tube.
        //  B_z = sqrt( J0(alpha_t * normalised_radius)^2 + guide_field_floor )
        //  Strictly positive for all radii when guide_field_floor > 0.
        //  Peak at centre: sqrt(1 + guide_field_floor) ~ 1.005 for floor=0.01.
        //  Minimum at J0=0 (normalised_radius ~ 0.628): sqrt(guide_field_floor) = 0.1.
        // ------------------------------------------------------------------
        KOKKOS_INLINE_FUNCTION
        real_t Bz_inside_tube(real_t normalised_radius) const
        {
            const real_t J0_value = lm_j0(alpha_t * normalised_radius);
            return math::sqrt(SQR(J0_value) + guide_field_floor);
        }

        // ------------------------------------------------------------------
        //  bx1 = Bx = dAz/dy
        //  Discrete finite difference of Az, matching Tristan line 430:
        //    Bx(i,j) = Az(i,j+1) - Az(i,j)
        //  Called by SetEMFields_kernel at Bx stagger position (x, y + 1/2*dy).
        //  We shift ±1/2*dy from the stagger point to get the two Az values.
        // ------------------------------------------------------------------
        KOKKOS_INLINE_FUNCTION
        real_t bx1(const coord_t<D>& x_Ph) const
        {
            const real_t x_stagger = x_Ph[0];
            const real_t y_stagger = x_Ph[1];

            real_t Az_at_j_plus_1  = vector_potential_Az(x_stagger,
                                                         y_stagger + HALF * cell_width_y,
                                                         tube1_x_centre, tube1_y_centre);
            real_t Az_at_j         = vector_potential_Az(x_stagger,
                                                         y_stagger - HALF * cell_width_y,
                                                         tube1_x_centre, tube1_y_centre);

            if (!single_tube) {
                Az_at_j_plus_1 += vector_potential_Az(x_stagger,
                                                      y_stagger + HALF * cell_width_y,
                                                      tube2_x_centre, tube2_y_centre);
                Az_at_j        += vector_potential_Az(x_stagger,
                                                      y_stagger - HALF * cell_width_y,
                                                      tube2_x_centre, tube2_y_centre);
            }

            return Az_at_j_plus_1 - Az_at_j;
        }

        // ------------------------------------------------------------------
        //  bx2 = By = -dAz/dx
        //  Discrete finite difference of Az, matching Tristan line 431:
        //    By(i,j) = Az(i,j) - Az(i+1,j)
        //  Called at By stagger position (x + 1/2*dx, y).
        // ------------------------------------------------------------------
        KOKKOS_INLINE_FUNCTION
        real_t bx2(const coord_t<D>& x_Ph) const
        {
            const real_t x_stagger = x_Ph[0];
            const real_t y_stagger = x_Ph[1];

            real_t Az_at_i         = vector_potential_Az(x_stagger - HALF * cell_width_x,
                                                         y_stagger,
                                                         tube1_x_centre, tube1_y_centre);
            real_t Az_at_i_plus_1  = vector_potential_Az(x_stagger + HALF * cell_width_x,
                                                         y_stagger,
                                                         tube1_x_centre, tube1_y_centre);

            if (!single_tube) {
                Az_at_i        += vector_potential_Az(x_stagger - HALF * cell_width_x,
                                                      y_stagger,
                                                      tube2_x_centre, tube2_y_centre);
                Az_at_i_plus_1 += vector_potential_Az(x_stagger + HALF * cell_width_x,
                                                      y_stagger,
                                                      tube2_x_centre, tube2_y_centre);
            }

            // By = Az(i,j) - Az(i+1,j)
            return Az_at_i - Az_at_i_plus_1;
        }

        // ------------------------------------------------------------------
        //  bx3 = Bz: guide field.
        //  Inside each tube: Bz_inside_tube(normalised_radius).
        //  Outside both tubes: Bz_exterior (uniform, continuous at boundary).
        // ------------------------------------------------------------------
        KOKKOS_INLINE_FUNCTION
        real_t bx3(const coord_t<D>& x_Ph) const
        {
            const real_t radial_distance_tube1 = math::sqrt(SQR(x_Ph[0] - tube1_x_centre)
                                                           + SQR(x_Ph[1] - tube1_y_centre));
            const real_t normalised_radius_tube1 = radial_distance_tube1 / tube_radius;
            if (normalised_radius_tube1 < ONE)
                return Bz_inside_tube(normalised_radius_tube1);

            if (!single_tube) {
                const real_t radial_distance_tube2 = math::sqrt(SQR(x_Ph[0] - tube2_x_centre)
                                                               + SQR(x_Ph[1] - tube2_y_centre));
                const real_t normalised_radius_tube2 = radial_distance_tube2 / tube_radius;
                if (normalised_radius_tube2 < ONE)
                    return Bz_inside_tube(normalised_radius_tube2);
            }

            return Bz_exterior;
        }

        // E field: set to zero here; overwritten in InitPrtls from the
        // smoothed grid B using stagger-correct averaging (Tristan lines 450-466).
        KOKKOS_INLINE_FUNCTION real_t ex1(const coord_t<D>&) const { return ZERO; }
        KOKKOS_INLINE_FUNCTION real_t ex2(const coord_t<D>&) const { return ZERO; }
        KOKKOS_INLINE_FUNCTION real_t ex3(const coord_t<D>&) const { return ZERO; }

        // ------------------------------------------------------------------
        //  J_inside_tube: analytic force-free current density J = curl(B)
        //  for a single tube centred at (x_centre, y_centre).
        //
        //  Derivation (cylindrical, J || B condition):
        //    J_z   = (alpha_t / tube_radius) * J0(alpha_t * normalised_radius)
        //    J_phi = J_z * B_phi / B_z
        //          = (alpha_t / tube_radius) * J0 * J1(alpha_t*rn) / B_z
        //    J_phi / r = (alpha_t / tube_radius) * J0 * J1 / (B_z * radial_distance)
        //
        //  Verification: J_phi/B_phi = J_z/B_z = alpha_t*J0 / (tube_radius * B_z) ✓
        //
        //  In Cartesian:
        //    Jx = -(J_phi / radial_distance) * (y - y_centre)
        //    Jy = +(J_phi / radial_distance) * (x - x_centre)
        //    Jz = (alpha_t / tube_radius) * J0
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
            if (normalised_radius >= ONE || radial_distance < static_cast<real_t>(1e-10) * tube_radius)
                return;

            const real_t J0_value = lm_j0(alpha_t * normalised_radius);
            const real_t J1_value = lm_j1(alpha_t * normalised_radius);
            const real_t Bz_local = Bz_inside_tube(normalised_radius);

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
        real_t tube_radius       { ZERO  };
        real_t tube1_x_centre    { ZERO  };
        real_t tube1_y_centre    { ZERO  };
        real_t tube2_x_centre    { ZERO  };
        real_t tube2_y_centre    { ZERO  };
        real_t cell_width_x      { ONE   };
        real_t cell_width_y      { ONE   };
        int    n_smooth_passes   { 32    };
        real_t guide_field_floor { ZERO  };
        real_t kick_velocity     { ZERO  };
        real_t Bz_exterior       { ZERO  };
        bool   single_tube       { false };
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
        real_t background_T { static_cast<real_t>(1e-2) };
        real_t beta_kick    { static_cast<real_t>(0.1)  };
        real_t c_param      { static_cast<real_t>(0.01) };
        bool   init_rho     { true  };
        bool   single_tube  { false };
        int    nsmooth      { 32    };
        real_t ppc_buff        { static_cast<real_t>(8.0)  };
        real_t tube_separation { ZERO };

        // Global x extents needed to define buffer-zone boundaries in InitPrtls.
        // InitPrtls receives a *local* domain tile so we cache the global box here.
        real_t xmin_g { ZERO }, xmax_g { ZERO };


    public:
        InitFields<D> init_flds;

        // --------------------------------------------------------------------
        //  Constructor
        // --------------------------------------------------------------------
        inline PGen(const SimulationParams& p, metadomain_type& md)
            : Base { p }
            , global_domain { md }
        {
            background_T = p.template get<real_t>("setup.background_T",
                                                   static_cast<real_t>(1e-2));
            beta_kick    = p.template get<real_t>("setup.beta_kick",
                                                   static_cast<real_t>(0.1));
            c_param      = p.template get<real_t>("setup.c_param",
                                                   static_cast<real_t>(0.01));
            init_rho     = p.template get<bool>  ("setup.init_rho",    true);
            single_tube  = p.template get<bool>  ("setup.single_tube", false);
            nsmooth      = p.template get<int>   ("setup.nsmooth",     32);
            ppc_buff     = p.template get<real_t>("setup.ppc_buff",
                                                   static_cast<real_t>(8.0));
            // jfac is computed fresh in InitPrtls from scales.* each call.

            const auto& mesh = md.mesh();
            xmin_g = mesh.extent(in::x1).first;
            xmax_g = mesh.extent(in::x1).second;
            const real_t ymin_g = mesh.extent(in::x2).first;
            const real_t ymax_g = mesh.extent(in::x2).second;
            const real_t Lx = xmax_g - xmin_g;
            const real_t Ly = ymax_g - ymin_g;

            // Tube radius.
            // Matches user_sheet_new.F90 line 380: r_j = 0.99 * sx/4
            // The 0.99 factor moves the tube edge slightly away from the
            // y-periodic boundaries when Ly = 4*r_j.
            // Override with setup.r_j to set an explicit physical radius.
            const real_t r_j = p.template get<real_t>(
                "setup.r_j",
                Lx * static_cast<real_t>(0.99 * 0.25));   // default = 0.99*Lx/4

            // Centre-to-centre separation between the two tubes.
            // Default = 2*r_j (tubes touching, matching Tristan exactly).
            // Set setup.tube_separation > 2*r_j in the input to introduce
            // a gap and prevent spontaneous reconnection at t=0.
            tube_separation = p.template get<real_t>(
                "setup.tube_separation",
                TWO * r_j);   // default: touching

            // Tube centres: both on x midline, offset by ±tube_separation/2 in y.
            // (Tristan: x1=x2=0.5*sx, y1=0.5*sy-r_j, y2=0.5*sy+r_j)
            const real_t cx   = xmin_g + HALF * Lx;
            const real_t cy   = ymin_g + HALF * Ly;
            const real_t half_sep = HALF * tube_separation;

            const real_t cell_dx = Lx / static_cast<real_t>(mesh.n_active(in::x1));
            const real_t cell_dy = Ly / static_cast<real_t>(mesh.n_active(in::x2));

            init_flds = InitFields<D>(
                r_j,
                cx, cy - half_sep,  // tube 1 centre
                cx, cy + half_sep,  // tube 2 centre
                cell_dx, cell_dy,
                c_param, beta_kick, single_tube, nsmooth);
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
        //  Stage 1: inject Maxwellian background with buffer-zone thinning.
        //  Stage 2: current + E×B drift with probabilistic reflection + boost.
        //  Stage 3: [init_rho] weight perturbation from div(E).
        //
        //  2D / 3D differences handled here
        //  ----------------------------------
        //  * make_box: pushes Range::All for all non-x dimensions (d=1 in 2D,
        //    d=1 and d=2 in 3D) — correct for both cases via the loop.
        //  * Particle physical position: xp[2] set only when D == Dim::_3D
        //    inside if constexpr.
        //  * i3 / dx3 cell indices: accessed only when D == Dim::_3D inside
        //    if constexpr.  In 2D these members may not exist on the species.
        //  * mesh.extent(in::x3) / mesh.n_active(in::x3): called only when
        //    D == Dim::_3D inside if constexpr.
        // --------------------------------------------------------------------
        inline void InitPrtls(Domain<S, M>& domain)
        {
            const auto& mesh = domain.mesh;
            const real_t Lx  = xmax_g - xmin_g;

            // Read skindepth0 and larmor0 directly from [scales].
            // Compute sigma locally: sigma = (skindepth0/larmor0)^2.
            // Avoids depending on Entity-internal scales.sigma0.
            const real_t skindepth = params.template get<real_t>("scales.skindepth0");
            const real_t larmor    = params.template get<real_t>("scales.larmor0");
            const real_t sigma     = SQR(skindepth / larmor);

            // jfac: beta_s = skindepth0 * sqrt(sigma) / 2 * curl(B)
            // (image derivation + Tristan: beta = j_grid * sqrt(sigma) * skindepth / CC)
            // Override via setup.jfac in the input file.
            const real_t jfac = params.template get<real_t>(
                "setup.jfac",
                skindepth * math::sqrt(sigma) / static_cast<real_t>(2.0));

            const real_t ppc0_val = params.template get<real_t>("particles.ppc0",
                                                                  static_cast<real_t>(4));
            const real_t dwn_ppc_buff = ppc_buff / ppc0_val;

            // Buffer zone x boundaries (Tristan: 0.2*sx to 0.8*sx)
            const real_t x_buf_lo = xmin_g + static_cast<real_t>(0.2) * Lx;
            const real_t x_buf_hi = xmin_g + static_cast<real_t>(0.8) * Lx;

            // ================================================================
            //  STAGE 1: Inject thermal Maxwellian plasma
            //
            //  Three injection calls match Tristan's three fillRegionWithThermal-
            //  Plasma calls (inner + left buffer + right buffer).
            //
            //  make_box is dimension-agnostic: it loops over M::Dim and pushes
            //  Range::All for every dimension that isn't x (d != 0).  This
            //  correctly covers y only in 2D, and y+z in 3D.
            // ================================================================
            {
                const auto temps    = std::make_pair(background_T, background_T);
                const auto no_drift = std::make_pair(
                    std::vector<real_t>{ ZERO, ZERO, ZERO },
                    std::vector<real_t>{ ZERO, ZERO, ZERO });

                auto make_box = [&](real_t xlo, real_t xhi) -> boundaries_t<real_t>
                {
                    boundaries_t<real_t> box;
                    for (auto d = 0u; d < M::Dim; ++d)
                    {
                        if (d == 0u) box.push_back({ xlo, xhi });
                        else         box.push_back(Range::All);
                    }
                    return box;
                };

                // (a) Inner region [0.2 Lx, 0.8 Lx] — full ppc
                arch::InjectUniformMaxwellians<S, M>(
                    params, domain, ONE, temps, { 1, 2 }, no_drift, false,
                    make_box(x_buf_lo, x_buf_hi));

                // (b) Left buffer [0, 0.2 Lx] — reduced density
                // InjectUniformMaxwellians has no weight argument; Tristan's
                // buffer effect is reproduced by injecting at lower density.
                arch::InjectUniformMaxwellians<S, M>(
                    params, domain, dwn_ppc_buff, temps, { 1, 2 }, no_drift, false,
                    make_box(xmin_g, x_buf_lo));

                // (c) Right buffer [0.8 Lx, Lx] — reduced density
                arch::InjectUniformMaxwellians<S, M>(
                    params, domain, dwn_ppc_buff, temps, { 1, 2 }, no_drift, false,
                    make_box(x_buf_hi, xmax_g));
            }


            // ================================================================
            //  STAGES 1.5a → 1.2 → 1.5b:  Smooth B → Set E → Smooth E
            //
            //  Correct order for full beta_kick consistency:
            //
            //    Step A (1.5a): smooth B only  (nsmooth passes)
            //    Step B (1.2):  SetEfromB — compute E = -v×B from SMOOTHED B
            //    Step C (1.5b): smooth E only  (nsmooth passes)
            //
            //  This guarantees:
            //    • E is computed from the same smoothed B the field solver uses.
            //    • E and B are independently smoothed to the same width.
            //    • For beta_kick=0: Step B sets E=0, Step C is a no-op.
            //
            //  smooth_XY(pass, components) — separable (1/4,1/2,1/4) kernel
            //  applied to a caller-specified subset of EM components.
            //  run_smooth(comps, comm) — batches nsmooth passes with MPI syncs.
            // ================================================================

            // ----------------------------------------------------------------
            //  smooth_XY: one pass of (1/4,1/2,1/4) on the given components.
            //  `pass` (1-indexed within current batch) controls ghost depth.
            // ----------------------------------------------------------------
            auto smooth_XY = [&](int pass,
                                 std::initializer_list<em> components)
            {
                const auto& loc = domain.mesh;
                const int   nx  = static_cast<int>(loc.n_active(in::x1));
                const int   ny  = static_cast<int>(loc.n_active(in::x2));
                const int   ng  = static_cast<int>(N_GHOSTS);
                const int imin  = std::max(1,            ng - pass);
                const int imax  = std::min(nx + 2*ng - 2, nx + ng + pass - 1);
                const int jmin  = std::max(1,            ng - pass);
                const int jmax  = std::min(ny + 2*ng - 2, ny + ng + pass - 1);
                int kmin = 0, kmax = 0;
                if constexpr (D == Dim::_3D) {
                    const int nz = static_cast<int>(loc.n_active(in::x3));
                    kmin = std::max(1,            ng - pass);
                    kmax = std::min(nz + 2*ng - 2, nz + ng + pass - 1);
                }
                auto& F = domain.fields.em;
                for (auto c : components) {
                    const auto ci = static_cast<unsigned>(c);
                    // x-pass
                    if constexpr (D == Dim::_2D) {
                        Kokkos::parallel_for("SmoothX",
                            Kokkos::RangePolicy<>(jmin, jmax+1),
                            KOKKOS_LAMBDA(int j) {
                                real_t tmp = F(imin-1,j,ci);
                                for (int i=imin; i<=imax; ++i) {
                                    const real_t s = static_cast<real_t>(0.25)*F(i-1,j,ci)
                                                   + static_cast<real_t>(0.5) *F(i,  j,ci)
                                                   + static_cast<real_t>(0.25)*F(i+1,j,ci);
                                    F(i-1,j,ci)=tmp; tmp=s;
                                }
                                F(imax,j,ci)=tmp;
                        });
                    } else if constexpr (D == Dim::_3D) {
                        Kokkos::parallel_for("SmoothX",
                            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({jmin,kmin},{jmax+1,kmax+1}),
                            KOKKOS_LAMBDA(int j, int k) {
                                real_t tmp = F(imin-1,j,k,ci);
                                for (int i=imin; i<=imax; ++i) {
                                    const real_t s = static_cast<real_t>(0.25)*F(i-1,j,k,ci)
                                                   + static_cast<real_t>(0.5) *F(i,  j,k,ci)
                                                   + static_cast<real_t>(0.25)*F(i+1,j,k,ci);
                                    F(i-1,j,k,ci)=tmp; tmp=s;
                                }
                                F(imax,j,k,ci)=tmp;
                        });
                    }
                    // y-pass
                    if constexpr (D == Dim::_2D) {
                        Kokkos::parallel_for("SmoothY",
                            Kokkos::RangePolicy<>(imin, imax+1),
                            KOKKOS_LAMBDA(int i) {
                                real_t tmp = F(i,jmin-1,ci);
                                for (int j=jmin; j<=jmax; ++j) {
                                    const real_t s = static_cast<real_t>(0.25)*F(i,j-1,ci)
                                                   + static_cast<real_t>(0.5) *F(i,j,  ci)
                                                   + static_cast<real_t>(0.25)*F(i,j+1,ci);
                                    F(i,j-1,ci)=tmp; tmp=s;
                                }
                                F(i,jmax,ci)=tmp;
                        });
                    } else if constexpr (D == Dim::_3D) {
                        Kokkos::parallel_for("SmoothY",
                            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({imin,kmin},{imax+1,kmax+1}),
                            KOKKOS_LAMBDA(int i, int k) {
                                real_t tmp = F(i,jmin-1,k,ci);
                                for (int j=jmin; j<=jmax; ++j) {
                                    const real_t s = static_cast<real_t>(0.25)*F(i,j-1,k,ci)
                                                   + static_cast<real_t>(0.5) *F(i,j,  k,ci)
                                                   + static_cast<real_t>(0.25)*F(i,j+1,k,ci);
                                    F(i,j-1,k,ci)=tmp; tmp=s;
                                }
                                F(i,jmax,k,ci)=tmp;
                        });
                        // z-pass (3D only) — completes full isotropic 3×3×3 kernel
                        Kokkos::parallel_for("SmoothZ",
                            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({imin,jmin},{imax+1,jmax+1}),
                            KOKKOS_LAMBDA(int i, int j) {
                                real_t tmp = F(i,j,kmin-1,ci);
                                for (int k=kmin; k<=kmax; ++k) {
                                    const real_t s = static_cast<real_t>(0.25)*F(i,j,k-1,ci)
                                                   + static_cast<real_t>(0.5) *F(i,j,k,  ci)
                                                   + static_cast<real_t>(0.25)*F(i,j,k+1,ci);
                                    F(i,j,k-1,ci)=tmp; tmp=s;
                                }
                                F(i,j,kmax,ci)=tmp;
                        });
                    }
                }
            }; // smooth_XY

            // ----------------------------------------------------------------
            //  run_smooth_B: nsmooth passes on B components only.
            //  run_smooth_E: nsmooth passes on E components only.
            //  Batched in groups of N_GHOSTS with MPI syncs between batches.
            // ----------------------------------------------------------------
            auto run_smooth_B = [&]()
            {
                if (nsmooth <= 0) return;
                int remaining = nsmooth;
                while (remaining > 0) {
                    global_domain.CommunicateFields(domain, Comm::E | Comm::B);
                    const int batch = std::min(remaining, static_cast<int>(N_GHOSTS));
                    for (int pass = 1; pass <= batch; ++pass)
                        smooth_XY(pass, { em::bx1, em::bx2, em::bx3 });
                    remaining -= batch;
                }
                global_domain.CommunicateFields(domain, Comm::E | Comm::B);
            };

            auto run_smooth_E = [&]()
            {
                if (nsmooth <= 0) return;
                int remaining = nsmooth;
                while (remaining > 0) {
                    global_domain.CommunicateFields(domain, Comm::E | Comm::B);
                    const int batch = std::min(remaining, static_cast<int>(N_GHOSTS));
                    for (int pass = 1; pass <= batch; ++pass)
                        smooth_XY(pass, { em::ex1, em::ex2, em::ex3 });
                    remaining -= batch;
                }
                global_domain.CommunicateFields(domain, Comm::E | Comm::B);
            };

            // ================================================================
            //  STEP A: Smooth B only
            // ================================================================
            run_smooth_B();

            // ================================================================
            //  STEP B: Set E = -v×B from SMOOTHED grid B  (Tristan lines 435-469)
            //
            //  Stagger mapping (Entity i1=x, i2=y):
            //    Ex at (i+1/2,j): Bz avg of (i+1/2,j±1/2) → (i+1/2,j) ✓
            //    Ez at (i,    j): Bx avg of (i,    j±1/2) → (i,    j) ✓
            //    Ey = 0
            //  Tube membership checked at stagger position of each component.
            // ================================================================
            {
                const auto& loc  = domain.mesh;
                const int   nx   = static_cast<int>(loc.n_active(in::x1));
                const int   ny   = static_cast<int>(loc.n_active(in::x2));
                const int   ng   = static_cast<int>(N_GHOSTS);
                const real_t bk  = beta_kick;
                const real_t x1_ = init_flds.x1, y1_ = init_flds.y1;
                const real_t x2_ = init_flds.x2, y2_ = init_flds.y2;
                const real_t rj_ = init_flds.r_j;
                const bool   sgl = single_tube;
                const real_t x0e = loc.extent(in::x1).first;
                const real_t y0e = loc.extent(in::x2).first;
                const real_t dxe = (loc.extent(in::x1).second - x0e)
                                 / static_cast<real_t>(nx);
                const real_t dye = (loc.extent(in::x2).second - y0e)
                                 / static_cast<real_t>(ny);
                auto& F = domain.fields.em;

                // In 3D, iterate over all z-slices; the xy physics is identical.
                // em_field is rank-3 in 2D: F(i1,i2,comp)
                //           and rank-4 in 3D: F(i1,i2,i3,comp)
                if constexpr (D == Dim::_2D)
                {
                    Kokkos::parallel_for("SetEfromB",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ng,ng},{nx+ng,ny+ng}),
                        KOKKOS_LAMBDA(int i1, int i2)
                    {
                        // Ex = -beta_kick * Bz_at_Ex
                        const real_t Bz_at_Ex = HALF*(F(i1,i2,  em::bx3)
                                                    + F(i1,i2-1,em::bx3));
                        const real_t x_ex  = x0e+(static_cast<real_t>(i1-ng)+HALF)*dxe;
                        const real_t y_ex  = y0e+ static_cast<real_t>(i2-ng)      *dye;
                        const real_t r1_ex = math::sqrt(SQR(x_ex-x1_)+SQR(y_ex-y1_))/rj_;
                        const real_t r2_ex = math::sqrt(SQR(x_ex-x2_)+SQR(y_ex-y2_))/rj_;
                        real_t ex_val = ZERO;
                        if      (r1_ex < ONE)        ex_val = -bk*Bz_at_Ex;
                        else if (!sgl&&r2_ex < ONE)  ex_val = +bk*Bz_at_Ex;
                        F(i1,i2,em::ex1) = ex_val;
                        // Ey = 0
                        F(i1,i2,em::ex2) = ZERO;
                        // Ez = +beta_kick * Bx_at_Ez
                        const real_t Bx_at_Ez = HALF*(F(i1,i2,  em::bx1)
                                                    + F(i1,i2-1,em::bx1));
                        const real_t x_ez  = x0e+ static_cast<real_t>(i1-ng)*dxe;
                        const real_t y_ez  = y0e+ static_cast<real_t>(i2-ng)*dye;
                        const real_t r1_ez = math::sqrt(SQR(x_ez-x1_)+SQR(y_ez-y1_))/rj_;
                        const real_t r2_ez = math::sqrt(SQR(x_ez-x2_)+SQR(y_ez-y2_))/rj_;
                        real_t ez_val = ZERO;
                        if      (r1_ez < ONE)        ez_val = +bk*Bx_at_Ez;
                        else if (!sgl&&r2_ez < ONE)  ez_val = -bk*Bx_at_Ez;
                        F(i1,i2,em::ex3) = ez_val;
                    });
                }
                else if constexpr (D == Dim::_3D)
                {
                    // In 3D the tubes are uniform in z; the xy E-field profile
                    // is replicated across all z-slices.
                    const int nz = static_cast<int>(loc.n_active(in::x3));
                    Kokkos::parallel_for("SetEfromB",
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                            {ng,ng,ng},{nx+ng,ny+ng,nz+ng}),
                        KOKKOS_LAMBDA(int i1, int i2, int i3)
                    {
                        // Ex = -beta_kick * Bz_at_Ex
                        const real_t Bz_at_Ex = HALF*(F(i1,i2,  i3,em::bx3)
                                                    + F(i1,i2-1,i3,em::bx3));
                        const real_t x_ex  = x0e+(static_cast<real_t>(i1-ng)+HALF)*dxe;
                        const real_t y_ex  = y0e+ static_cast<real_t>(i2-ng)      *dye;
                        const real_t r1_ex = math::sqrt(SQR(x_ex-x1_)+SQR(y_ex-y1_))/rj_;
                        const real_t r2_ex = math::sqrt(SQR(x_ex-x2_)+SQR(y_ex-y2_))/rj_;
                        real_t ex_val = ZERO;
                        if      (r1_ex < ONE)        ex_val = -bk*Bz_at_Ex;
                        else if (!sgl&&r2_ex < ONE)  ex_val = +bk*Bz_at_Ex;
                        F(i1,i2,i3,em::ex1) = ex_val;
                        // Ey = 0
                        F(i1,i2,i3,em::ex2) = ZERO;
                        // Ez = +beta_kick * Bx_at_Ez
                        const real_t Bx_at_Ez = HALF*(F(i1,i2,  i3,em::bx1)
                                                    + F(i1,i2-1,i3,em::bx1));
                        const real_t x_ez  = x0e+ static_cast<real_t>(i1-ng)*dxe;
                        const real_t y_ez  = y0e+ static_cast<real_t>(i2-ng)*dye;
                        const real_t r1_ez = math::sqrt(SQR(x_ez-x1_)+SQR(y_ez-y1_))/rj_;
                        const real_t r2_ez = math::sqrt(SQR(x_ez-x2_)+SQR(y_ez-y2_))/rj_;
                        real_t ez_val = ZERO;
                        if      (r1_ez < ONE)        ez_val = +bk*Bx_at_Ez;
                        else if (!sgl&&r2_ez < ONE)  ez_val = -bk*Bx_at_Ez;
                        F(i1,i2,i3,em::ex3) = ez_val;
                    });
                }

                global_domain.CommunicateFields(domain, Comm::E | Comm::B);
            }

            // ================================================================
            //  STEP C: Smooth E only
            //  For beta_kick=0: E=0 everywhere → no-op.
            // ================================================================
            run_smooth_E();

            // Final sync: E and B ghost cells consistent for Stage 2.
            global_domain.CommunicateFields(domain, Comm::E | Comm::B);


            // ================================================================
            //  STAGE 2: Current-driven + E×B Lorentz boost.
            //
            //  J is now computed as discrete curl(B_smoothed) directly from the
            //  grid.  This ensures J_deposited = curl(B_smoothed) to machine
            //  precision, making the initial state force-free at the discrete
            //  level for the smoothed field configuration.
            //
            //  E and B are also read from the smoothed grid for the E×B term.
            //
            //  2D/3D compatibility
            //  --------------------
            //  * em_field is rank-3 in 2D: (i,j,component)
            //            and rank-4 in 3D: (i,j,k,component)
            //  * i3_v/dx3_v are extracted before the lambda.
            //  * All [2] array accesses are inside if constexpr (D==Dim::_3D).
            // ================================================================
            {
                const auto em_field = domain.fields.em;

                // z extent: only defined and queried in 3D
                real_t z0_loc { ZERO };
                real_t dz_loc { ONE  };
                if constexpr (D == Dim::_3D)
                {
                    z0_loc = mesh.extent(in::x3).first;
                    dz_loc = (mesh.extent(in::x3).second - z0_loc)
                           / static_cast<real_t>(mesh.n_active(in::x3));
                }

                const bool   do_rho    = init_rho;
                const real_t sigma_rho = sigma;    // for init_rho weight correction
                const real_t skindepth_rho = skindepth;   // (must be plain reals for lambda capture)
                const real_t jfac   = jfac;   // = 1 (n0/ppc0 in Entity normalisation)

                // Capture InitFields and mesh geometry for use inside the lambda.
                // KOKKOS_LAMBDA = [=] so all captured variables must be copyable.
                // InitFields is a plain struct with no heap members — safe to copy.
                const auto init = init_flds;

                // Physical coordinate origins and cell sizes.
                // Confirmed: i1→x1→x, i2→x2→y (from field_setter.h and sr.hpp).
                // x0_loc is the x-start of this MPI tile's local domain.
                const real_t x0_loc = mesh.extent(in::x1).first;  // x origin
                const real_t y0_loc = mesh.extent(in::x2).first;  // y origin
                const real_t dx_loc = (mesh.extent(in::x1).second - x0_loc)
                                    / static_cast<real_t>(mesh.n_active(in::x1));
                const real_t dy_loc = (mesh.extent(in::x2).second - y0_loc)
                                    / static_cast<real_t>(mesh.n_active(in::x2));

                for (auto s = 0u; s < domain.species.size(); ++s)
                {
                    auto& sp       = domain.species[s];
                    const real_t q = sp.charge();

                    // Extract all Kokkos view handles before the lambda.
                    // Particles has a deleted copy constructor; [=] must never
                    // capture sp directly.
                    const auto i1  = sp.i1;
                    const auto i2  = sp.i2;
                    const auto dx1 = sp.dx1;
                    const auto dx2 = sp.dx2;
                    const auto tag = sp.tag;
                    const auto ux1 = sp.ux1;
                    const auto ux2 = sp.ux2;
                    const auto ux3 = sp.ux3;
                    const auto wgt = sp.weight;

                    // i3/dx3: placeholder = i1/dx1 in 2D (never called),
                    // reassigned to sp.i3/sp.dx3 in 3D.
                    [[maybe_unused]] auto i3_v  = sp.i1;
                    [[maybe_unused]] auto dx3_v = sp.dx1;
                    if constexpr (D == Dim::_3D)
                    {
                        i3_v  = sp.i3;
                        dx3_v = sp.dx3;
                    }

                    Kokkos::parallel_for(
                        "FluxTubeDrift",
                        sp.rangeActiveParticles(),
                        KOKKOS_LAMBDA(index_t p)
                    {
                        if (tag(p) == ParticleTag::dead)
                            return;

                        const auto   ii = i1(p);
                        const auto   jj = i2(p);
                        const real_t fx = static_cast<real_t>(dx1(p));  // x1 sub-cell fraction (x)
                        const real_t fy = static_cast<real_t>(dx2(p));  // x2 sub-cell fraction (y)

                        // kk: z cell index — 0 in 2D, i3_v(p) in 3D
                        int kk = 0;
                        if constexpr (D == Dim::_3D)
                            kk = static_cast<int>(i3_v(p));

                        // Physical position of the particle.
                        //
                        // Confirmed from field_setter.h and sr.hpp:
                        //   i1(p) = x1-index = x-index,  dx1(p) = x sub-cell fraction
                        //   i2(p) = x2-index = y-index,  dx2(p) = y sub-cell fraction
                        //
                        // The field array EM(i1, i2, comp) has i1=x, i2=y.
                        // SetEMFields_kernel calls finit.bx1({i1_+stagger, i2_+stagger})
                        // where x_Phys[0] comes from i1 (x) and x_Phys[1] from i2 (y).
                        // Pusher getParticlePosition: xp[0]=i1+dx1 (x), xp[1]=i2+dx2 (y).
                        coord_t<D> xp {};
                        xp[0] = x0_loc + (static_cast<real_t>(ii) + fx) * dx_loc;
                        xp[1] = y0_loc + (static_cast<real_t>(jj) + fy) * dy_loc;
                        if constexpr (D == Dim::_3D)
                        {
                            // dz_loc and z0_loc were computed on the host above
                            // and are captured by [=] — no device calls needed.
                            const real_t fz = static_cast<real_t>(dx3_v(p));
                            xp[2] = z0_loc + (static_cast<real_t>(kk) + fz) * dz_loc;
                        }

                        // ---- Helper: read one em component from the grid ----
                        // em_field is rank-3 in 2D: (i,j,comp)
                        //           and rank-4 in 3D: (i,j,k,comp)
                        //
                        // CRITICAL: particle indices i1(p)/i2(p) run from 0 to
                        // n_active-1 (active domain only).  The em_field array
                        // has ghost cells prepended, so active cell (0,0) lives
                        // at array index (N_GHOSTS, N_GHOSTS).  All reads must
                        // add N_GHOSTS to both I and J (and K in 3D) to address
                        // the correct physical cell.
                        //
                        // Example: particle at i1=0 (leftmost active cell).
                        //   Without offset: em_field(0, ...) → first ghost cell.
                        //   With offset:    em_field(N_GHOSTS, ...) → correct.
                        //
                        // The curl stencil reads up to I+2, J+2.  With offset:
                        //   max index = (n_active-1) + N_GHOSTS + 2
                        //             = n_active + N_GHOSTS + 1
                        // For N_GHOSTS >= 3 this is safely within the allocated
                        // range of n_active + 2*N_GHOSTS - 1.
                        const int ng_off = static_cast<int>(N_GHOSTS);
                        auto em_rd = [&](auto I, auto J, auto COMP) -> real_t {
                            if constexpr (D == Dim::_2D)
                                return em_field(I + ng_off, J + ng_off, COMP);
                            else
                                return em_field(I + ng_off, J + ng_off,
                                               kk + ng_off, COMP);
                        };

                        // ---- Bilinear interpolation --------------------------
                        // Standard Yee bilinear, consistent with Entity's pusher.
                        // ii=i1=x-index, jj=i2=y-index, fx=dx1=x-frac, fy=dx2=y-frac.
                        //   w00 = (1-fx)*(1-fy)  →  em_rd(ii,   jj  )
                        //   w10 = fx    *(1-fy)  →  em_rd(ii+1, jj  )  [step x]
                        //   w01 = (1-fx)*fy      →  em_rd(ii,   jj+1)  [step y]
                        //   w11 = fx    *fy      →  em_rd(ii+1, jj+1)
                        // This exactly matches Entity's higher-order shape function
                        // interpolation in sr.hpp (S1p/S1d indexed by i1, S2p/S2d by i2).
                        const real_t w00 = (ONE - fx) * (ONE - fy);
                        const real_t w10 = fx          * (ONE - fy);
                        const real_t w01 = (ONE - fx)  * fy;
                        const real_t w11 = fx           * fy;

                        // ---- J from analytic force-free formula ----------------
                        // Uses Camille's J||B formulation:
                        //   Jphi/r = alpha^3*c_t^2*J0*J1 / (Bz*r_phys)
                        //   Jz     = alpha^2*c_t*J0
                        // This satisfies J_phi/B_phi = Jz/Bz exactly (J||B).
                        // jfac=0.5 accounts for both species contributing
                        // simultaneously (see jfac derivation above).
                        //
                        // Stagger correction
                        // ------------------
                        // Each J component lives at the same Yee position as
                        // the corresponding E component (Ampere's law is local):
                        //   Jx (= Ex stagger): (i+1/2, j    ) → +1/2*dx in x
                        //   Jy (= Ey stagger): (i,     j+1/2) → +1/2*dy in y
                        //   Jz (= Ez stagger): (i,     j    ) → no shift
                        //
                        // The particle position xp has no stagger offset, so we
                        // evaluate each J component at its correct stagger point.
                        // Without this, analytic J is half a cell out of phase
                        // with E, causing phase mismatch in the Ampere update.
                        // (The discrete curl had this correct implicitly because
                        // em_rd(ii,jj,bx3) reads Bz at its natural grid position.)
                        real_t beta_x = ZERO, beta_y = ZERO, beta_z = ZERO;
                        {
                            // Jx at Ex stagger: (i+1/2, j) → shift +1/2 cell in x
                            // coord_t<D> is a plain array — copy element by element
                            // (array copy-init `= xp` is not allowed in device lambdas)
                            coord_t<D> xp_Jx {};
                            for (int d = 0; d < static_cast<int>(D); ++d)
                                xp_Jx[d] = xp[d];
                            xp_Jx[0] += HALF * dx_loc;

                            // Jy at Ey stagger: (i, j+1/2) → shift +1/2 cell in y
                            coord_t<D> xp_Jy {};
                            for (int d = 0; d < static_cast<int>(D); ++d)
                                xp_Jy[d] = xp[d];
                            xp_Jy[1] += HALF * dy_loc;

                            // Jz at Ez stagger: (i, j) = particle position, no shift
                            // xp_Jz = xp (no copy needed)

                            const real_t jx0 = init.jx1(xp_Jx);
                            const real_t jy0 = init.jx2(xp_Jy);
                            const real_t jz0 = init.jx3(xp);
                            beta_x = jfac * q * jx0;
                            beta_y = jfac * q * jy0;
                            beta_z = jfac * q * jz0;
                        }

                        // ---- Full 3-component E×B drift --------------------
                        // beta += (E × B) / |B|^2
                        // Uses all three E and B components.
                        // For beta_kick=0: E=0 so this term is zero.
                        // For beta_kick>0: all three drift components matter.
                        {
                            const real_t ex0 = w00 * em_rd(ii,   jj,   em::ex1)
                                             + w10 * em_rd(ii+1, jj,   em::ex1)
                                             + w01 * em_rd(ii,   jj+1, em::ex1)
                                             + w11 * em_rd(ii+1, jj+1, em::ex1);
                            const real_t ey0 = w00 * em_rd(ii,   jj,   em::ex2)
                                             + w10 * em_rd(ii+1, jj,   em::ex2)
                                             + w01 * em_rd(ii,   jj+1, em::ex2)
                                             + w11 * em_rd(ii+1, jj+1, em::ex2);
                            const real_t ez0 = w00 * em_rd(ii,   jj,   em::ex3)
                                             + w10 * em_rd(ii+1, jj,   em::ex3)
                                             + w01 * em_rd(ii,   jj+1, em::ex3)
                                             + w11 * em_rd(ii+1, jj+1, em::ex3);
                            const real_t bx0 = w00 * em_rd(ii,   jj,   em::bx1)
                                             + w10 * em_rd(ii+1, jj,   em::bx1)
                                             + w01 * em_rd(ii,   jj+1, em::bx1)
                                             + w11 * em_rd(ii+1, jj+1, em::bx1);
                            const real_t by0 = w00 * em_rd(ii,   jj,   em::bx2)
                                             + w10 * em_rd(ii+1, jj,   em::bx2)
                                             + w01 * em_rd(ii,   jj+1, em::bx2)
                                             + w11 * em_rd(ii+1, jj+1, em::bx2);
                            const real_t bz0 = w00 * em_rd(ii,   jj,   em::bx3)
                                             + w10 * em_rd(ii+1, jj,   em::bx3)
                                             + w01 * em_rd(ii,   jj+1, em::bx3)
                                             + w11 * em_rd(ii+1, jj+1, em::bx3);

                            const real_t Bsq = bx0*bx0 + by0*by0 + bz0*bz0;
                            if (Bsq > ZERO) {
                                beta_x += (ey0*bz0 - ez0*by0) / Bsq;
                                beta_y += (ez0*bx0 - ex0*bz0) / Bsq;
                                beta_z += (ex0*by0 - ey0*bx0) / Bsq;
                            }
                        }

                        real_t beta_sq = beta_x*beta_x + beta_y*beta_y + beta_z*beta_z;

                        // Safety clamp — mirrors Tristan's beta_sq >= 20 error
                        constexpr real_t BETA_MAX    = static_cast<real_t>(0.99);
                        constexpr real_t BETA_MAX_SQ = BETA_MAX * BETA_MAX;
                        if (beta_sq >= BETA_MAX_SQ)
                        {
                            const real_t fac = BETA_MAX / math::sqrt(beta_sq);
                            beta_x *= fac; beta_y *= fac; beta_z *= fac;
                            beta_sq = BETA_MAX_SQ;
                        }

                        if (beta_sq <= ZERO) return;

                        real_t ux  = ux1(p);
                        real_t uy  = ux2(p);
                        real_t uz  = ux3(p);
                        real_t gam = math::sqrt(ONE + ux*ux + uy*uy + uz*uz);

                        // No probabilistic reflection — Lorentz boost directly.
                        // Reflection is a Tristan legacy that creates asymmetric
                        // J deposition; Camille's pgen omits it.

                        // ---- Lorentz boost (Tristan lines 159-166) ----------
                        const real_t gam_b = ONE / math::sqrt(ONE - beta_sq);
                        const real_t ux_b  = gam_b * beta_x;
                        const real_t uy_b  = gam_b * beta_y;
                        const real_t uz_b  = gam_b * beta_z;
                        const real_t boost = (ux*ux_b + uy*uy_b + uz*uz_b)
                                           / (gam_b + ONE) + gam;
                        ux1(p) = ux + boost * ux_b;
                        ux2(p) = uy + boost * uy_b;
                        ux3(p) = uz + boost * uz_b;

                        // ---- init_rho: weight correction ------------------
                        // Seeds the charge-density perturbation that satisfies
                        // Gauss's law: div(E) = rho / n0.
                        //
                        // In Entity (from parameters.cpp):
                        //   n0 = ppc0 / V0,  particle weight = 1/V0
                        //   div(E) [code units] = rho / n0
                        //   delta_weight = div(E) * n0 * V0 * q
                        //                = div(E) * ppc0 * q
                        //
                        // For beta_kick=0: E=0 everywhere so div(E)=0
                        // and this correction is identically zero.
                        if (do_rho)
                        {
                            // 4-point bilinear interpolation of div(E) to
                            // particle position (same stencil as Tristan lg_arr)
                            const real_t divE_00 = em_rd(ii,   jj,   em::ex1)
                                                 - em_rd(ii-1, jj,   em::ex1);
                            const real_t divE_10 = em_rd(ii+1, jj,   em::ex1)
                                                 - em_rd(ii,   jj,   em::ex1);
                            const real_t divE_01 = em_rd(ii,   jj+1, em::ex1)
                                                 - em_rd(ii-1, jj+1, em::ex1);
                            const real_t divE_11 = em_rd(ii+1, jj+1, em::ex1)
                                                 - em_rd(ii,   jj+1, em::ex1);

                            const real_t rho0 = w00 * divE_00 + w10 * divE_10
                                              + w01 * divE_01 + w11 * divE_11;

                            // Weight correction: Tristan line 179:
                            // wei_new = wei * (1 + rho0 * sqrt(sigma) * skindepth * sign(q))
                            // div(E) = rho_charge in code units; multiply by
                            // sqrt(sigma)*skindepth to convert to weight perturbation.
                            const real_t wei_new = wgt(p)
                                * (ONE + rho0 * math::sqrt(sigma_rho)
                                   * skindepth_rho * q);
                            if (wei_new > ZERO)
                                wgt(p) = wei_new;
                        }

                    }); // parallel_for FluxTubeDrift

                } // species loop

            } // stage 2 block

        } // InitPrtls

    }; // struct PGen

} // namespace user

#endif // PROBLEM_GENERATOR_H