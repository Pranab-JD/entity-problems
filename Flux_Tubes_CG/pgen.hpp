#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

/**
 * @file Flux_Tubes_CG/pgen.hpp
 * @brief Force-free Lundquist flux-tube collision, 2D (XY) and 3D (XYZ)
 *
 *~ References:
 *      1. Lyutikov et al. (2017), JPP, 83, 635830602
 *      2. Ripperda et al. (2019), MNRAS, 485, 299
 *      3. Granier  et al. (2025), ApJ, 992, 193
 *
 *~ Physics setup
 * -------------
 * Two Lundquist tubes (Bessel-function force-free equilibria) are initialised
 * side-by-side along X (tangent at the domain centre). With kick_velocity != 0,
 * they are given opposite motional E = -v x B kicks so that they approach
 * each other along X (head-on) and reconnect.
 *
 *   Tube 1 = LEFT  (kicked +x, toward centre)
 *   Tube 2 = RIGHT (kicked -x, toward centre)
 *
 *
 *~ Field initialisation (ANALYTIC, both A and B)
 * --------------------------------------------------------------
 *   Az(r) = (1/k_t) * J0(k_t r)                    r <  tube_radius
 *   Az(r) = (1/k_t) * J0(alpha_t)                  r >= tube_radius   [constant]
 *
 *   Bx   =  dAz/dy
 *   By   = -dAz/dx           ( => B_phi(r) = J1(k_t r) )
 *   Bz   =  sqrt( J0(k_t r)^2 + guide_field_floor )
 *
 * with r measured from the tube axis in the XY plane, for every z.
 *
 *~ Current (ANALYTIC)
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
 * Because d/dz = 0, these are unchanged in 3D.
 *
 *~ Normalisation (see entity_normalisations.md)
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
 *~ Particle initialisation
 * -----------------------------------------------------------
 *  1. Maxwellian at setup.background_temperature, uniform density.
 *  2. Position-dependent drift beta_s (above), plus the E x B drift of the
 *     motional field when kick_velocity != 0.
 *  3. Lorentz boost of the thermal four-momentum into the drift frame.
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
#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>
#include <algorithm>

//! =========================================================================
//! GPU-compatible Bessel functions.
//! CUDA/HIP provide device intrinsics j0/j1; on CPU we use std::cyl_bessel_j
//! =========================================================================
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

    template <Dimension D>
    Inline void set_xy(coord_t<D>& c, real_t x, real_t y)
    {
        c[0] = x;
        c[1] = y;
        if constexpr (D == Dim::_3D) { c[2] = ZERO; }
    }

    //! =========================================================================
    //! FIELD INITIALISATION
    //!
    //!  E and B are initialised analytically; entity calls each component at its
    //!  own Yee-stagger position. Field and current share the same parameters, 
    //!  so they cannot be set inconsistently - but whether the formulas actually
    //!  agree is a separate question, checked by the Ampere test in print_setup.
    //! =========================================================================
    template <Dimension D>
    struct InitFields
    {
        //* alpha_t: FIRST ZERO OF J1 (= 3.8317...). The tube boundary condition
        //* B_phi(tube_radius) = J1(alpha_t) = 0 (no surface current) fixes this.
        static constexpr real_t alpha_t { static_cast<real_t>(3.8317059702075125) };

        InitFields() = default;

        InitFields(real_t tube_radius_, real_t tube1_x_centre_, real_t tube1_y_centre_,
                   real_t tube2_x_centre_, real_t tube2_y_centre_, real_t cell_width_x_,
                   real_t cell_width_y_, real_t cell_width_z_, real_t guide_field_floor_,
                   real_t kick_velocity_, bool single_tube_):
                   tube_radius { tube_radius_ }, tube1_x_centre { tube1_x_centre_ },
                   tube1_y_centre { tube1_y_centre_ }, tube2_x_centre { tube2_x_centre_ },
                   tube2_y_centre { tube2_y_centre_ }, cell_width_x { cell_width_x_ },
                   cell_width_y { cell_width_y_ }, cell_width_z { cell_width_z_ },
                   guide_field_floor { guide_field_floor_ }, kick_velocity { kick_velocity_ },
                   single_tube { single_tube_ }
        {
            k_t = alpha_t / tube_radius;                //* radial wavenumber
            Az_exterior = bessel_J0(alpha_t) / k_t;
            Bz_exterior = math::sqrt(SQR(bessel_J0(alpha_t)) + guide_field_floor);
        }

        //! Which tube (if any) contains the point, and the local radius.
        //!   returns 0 = outside both, 1 = inside tube 1, 2 = inside tube 2
        //!   on return, radius / offset_x / offset_y refer to that tube's centre
        //! The test is on the XY distance only -> infinite cylinders in 3D.
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
        //~ UNUSED: defined only for reference
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
            if (radius <= ZERO) { return ZERO; }        //* J1(0) = 0
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

        //! Bz = sqrt( J0(k_t r)^2 + guide_field_floor )
        Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t
        {
            real_t radius, offset_x, offset_y;
            if (locate(x_Ph[0], x_Ph[1], radius, offset_x, offset_y) == 0) { return Bz_exterior; }
            return math::sqrt(SQR(bessel_J0(k_t * radius)) + guide_field_floor);
        }

        //! Ex = -v x B |_x = 0 for an X-directed kick.
        Inline auto ex1(const coord_t<D>&) const -> real_t { return ZERO; }

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
        //! Takes (x, y) only: d/dz = 0, so the 3D current is identical.
        //! ---------------------------------------------------------------------
        Inline void current(real_t x, real_t y, real_t& Jx, real_t& Jy, real_t& Jz) const
        {
            Jx = ZERO; Jy = ZERO; Jz = ZERO;

            real_t radius, offset_x, offset_y;
            if (locate(x, y, radius, offset_x, offset_y) == 0) { return; }   //* B uniform outside -> J = 0

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
        real_t cell_width_z      { ONE   };   //* 3D only; = 1 and unused in 2D
        real_t guide_field_floor { ZERO  };
        real_t kick_velocity     { ZERO  };
        real_t k_t               { ONE   };
        real_t Az_exterior       { ZERO  };
        real_t Bz_exterior       { ZERO  };
        bool   single_tube       { false };
    };


    //! =========================================================================
    //!  Energy distribution: thermal Maxwellian boosted into the local
    //!  current-carrying drift frame
    //!
    //!  beta_s(x) = AMP * J_analytic(x) * sign(q_s)   [+ E x B when kicked]
    //!
    //! =========================================================================
    template <class M>
    struct FluxTubeDist
    {
        static constexpr auto D = M::Dim;

        FluxTubeDist(random_number_pool_t& pool, real_t temperature, real_t amp, real_t charge_sign, const InitFields<M::Dim>& flds):
                     maxwellian { pool, temperature }, amp { amp }, charge_sign { charge_sign }, flds { flds } {}

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
                set_xy<D>(xy, x, y);

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
                    drift_y += (Ez * Bx) / B2;          //* (E x B)_y = Ez Bx - Ex Bz, Ex = 0
                    drift_z += (-Ey * Bx) / B2;         //* (E x B)_z = Ex By - Ey Bx, Ex = 0
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
        const real_t amp;           //* d0^2 / r0
        const real_t charge_sign;   //* +-1
        const InitFields<M::Dim> flds;
    };

    //! Uniform spatial distribution (density is set by the injector's ndens argument).
    template <Dimension D>
    struct UniformDist
    {
        Inline auto operator()(const coord_t<D>&) const -> real_t { return ONE; }
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
        static constexpr auto dimensions { ::traits::pgen::compatible_with<Dim::_2D, Dim::_3D> {} };

        const SimulationParams& params;
        Metadomain<S, M>&       metadomain;

        const real_t background_temperature;
        const real_t kick_velocity;
        const real_t guide_field_floor;
        const bool   single_tube;
        const real_t skindepth0;
        const real_t larmor0;
        const real_t xmin, xmax, ymin, ymax, Lx, Ly;
        const real_t zmin, zmax, Lz;    //* 3D only (zmin = 0, zmax = Lz = 1 in 2D)
        const real_t tube_radius;

        real_t sigma0 { ONE };          //* (d0/r0)^2
        real_t amp    { ONE };          //* d0^2/r0 = d0 sqrt(sigma0): J_code = AMP curl(B)

        InitFields<D> init_flds;

        //!  Z extent, guarded for 2D
        static auto z_extent(Metadomain<S, M>& m) -> std::pair<real_t, real_t>
        {
            if constexpr (D == Dim::_3D)
            {
                return { m.mesh().extent(in::x3).first, m.mesh().extent(in::x3).second };
            }
            else
            {
                (void)m;
                return { ZERO, ONE };   //* dummy: Lz = 1, never used in 2D
            }
        }

        static void check_pair_plasma(Metadomain<S, M>& m)
        {
            const auto& sp = m.species_params();
            raise::ErrorIf(sp.size() < 2, "Flux tube pgen needs >= 2 species (an electron-positron pair).", HERE);

            const double m1 = (double)sp[0].mass(),   m2 = (double)sp[1].mass();
            const double q1 = (double)sp[0].charge(), q2 = (double)sp[1].charge();
            constexpr double TOL = 1.0e-5;

            raise::ErrorIf(std::fabs(m1 - m2) > TOL, "Species 1 and 2 have different masses: this pgen is pair-only. Every normalisation (sigma, skin depth, Debye length, P_th) assumes m_s = 1 for both species.", HERE);
            raise::ErrorIf(std::fabs(m1 - 1.0) > TOL, "Species mass != 1: the diagnostics and the drift normalisation are written for m_s = 1 (fiducial mass). Set both species masses to 1.", HERE);
            raise::ErrorIf(std::fabs(std::fabs(q1) - 1.0) > TOL || std::fabs(std::fabs(q2) - 1.0) > TOL, "Species charge magnitude != 1: beta_s = AMP * J * sign(q_s) assumes |q_s| = 1, so the tubes would carry the wrong current.", HERE);
            raise::ErrorIf(q1 * q2 >= 0.0, "Species 1 and 2 have the same charge sign: the two populations must counter-drift to carry J at zero net charge.", HERE);
        }

        static auto make_init(PGen& g) -> InitFields<D>
        {
            check_pair_plasma(g.metadomain);

            //! The analytic J_phi = k_t J0 J1 / Bz divides by Bz = sqrt(J0^2 + floor),
            //! which vanishes at the zero of J0 (r/R = 0.6276) when floor = 0.
            //! A zero floor is a valid force-free solution but it is not usable with an
            //! analytic current: it is 0/0 there, and B_phi reverses discontinuously.
            raise::ErrorIf(g.guide_field_floor <= ZERO, "setup.guide_field_floor must be > 0: the analytic J_phi divides by Bz = sqrt(J0^2 + guide_field_floor), which is zero at the first zero of J0 when the floor is zero. Use e.g. 0.01.", HERE);

            g.sigma0 = SQR(g.skindepth0 / g.larmor0);
            g.amp    = SQR(g.skindepth0) / g.larmor0;       //* == d0 * sqrt(sigma0)

            const auto&  mesh            = g.metadomain.mesh();
            const real_t domain_x_centre = g.xmin + HALF * g.Lx;
            const real_t domain_y_centre = g.ymin + HALF * g.Ly;
            const real_t dx = g.Lx / static_cast<real_t>(mesh.n_active(in::x1));
            const real_t dy = g.Ly / static_cast<real_t>(mesh.n_active(in::x2));

            real_t dz = ONE;
            if constexpr (D == Dim::_3D)
            {
                dz = g.Lz / static_cast<real_t>(mesh.n_active(in::x3));
                raise::ErrorIf(math::abs(dz - dx) > static_cast<real_t>(1.0e-5) * dx || math::abs(dy - dx) > static_cast<real_t>(1.0e-5) * dx, "Non-cubic cells: the Minkowski FDTD kernels assume dx = dy = dz. Adjust grid.extent or grid.resolution so the cells are cubic.", HERE);
            }

            return InitFields<D> { g.tube_radius,
                                   domain_x_centre - g.tube_radius, domain_y_centre,  //* Tube 1 (left)
                                   domain_x_centre + g.tube_radius, domain_y_centre,  //* Tube 2 (right)
                                   dx, dy, dz, g.guide_field_floor, g.kick_velocity, g.single_tube };
        }

        //!  Constructor
        PGen(const SimulationParams& p, Metadomain<S, M>& m): params { p }, metadomain { m },
        background_temperature { p.template get<real_t>("setup.background_temperature", static_cast<real_t>(0.01)) },
        kick_velocity          { p.template get<real_t>("setup.kick_velocity",          ZERO) },
        guide_field_floor      { p.template get<real_t>("setup.guide_field_floor",      static_cast<real_t>(0.01)) },
        single_tube            { p.template get<bool>  ("setup.single_tube",            false) },
        skindepth0             { p.template get<real_t>("scales.skindepth0") },
        larmor0                { p.template get<real_t>("scales.larmor0") },
        xmin { m.mesh().extent(in::x1).first  }, xmax { m.mesh().extent(in::x1).second },
        ymin { m.mesh().extent(in::x2).first  }, ymax { m.mesh().extent(in::x2).second },
        Lx   { xmax - xmin }, Ly { ymax - ymin },
        zmin { z_extent(m).first }, zmax { z_extent(m).second }, Lz { zmax - zmin },
        //* Tube radius default = 0.99 * Lx/4 (slightly less than 1/4 to move tube edges away from boundaries)
        tube_radius { p.template get<real_t>("setup.tube_radius", Lx * static_cast<real_t>(0.99 * 0.25)) },
        init_flds   { make_init(*this) }
        {
            print_setup();
        }

        auto MatchFields(simtime_t) const -> InitFields<D> { return init_flds; }

        //! =====================================================================
        //!  Startup diagnostics and consistency checks.
        //!  All lengths are normalised to skindepth0.
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

            const real_t n_total  = ONE;
            const real_t n_per_sp = HALF * n_total;

            constexpr int NS = 256;
            const real_t  r_scan_max = static_cast<real_t>(0.90) * tube_radius;

            long   n_samples = 0;
            double J_sum = 0.0, curl_sum = 0.0;
            real_t J_max = ZERO, curl_max = ZERO;
            real_t beta_max       = ZERO;   //* max |AMP * J|
            real_t ff_residual    = ZERO;   //* max |J x B| / (|J||B|)
            real_t amp_dev_absmax = ZERO;   //* max |J - curl(B)|, un-normalised
            real_t amp_dev_max    = ZERO;   //* = amp_dev_absmax / curl_max, filled after the loop

            //* finite-difference curl of the field as it is actually initialised.
            //! d/dz == 0 by construction, so the 3D curl has no extra terms:
            //!   (curl B)_x = dBz/dy - dBy/dz = dBz/dy
            //!   (curl B)_y = dBx/dz - dBz/dx = -dBz/dx
            auto curl_B = [&](real_t x, real_t y, real_t& cx, real_t& cy, real_t& cz)
            {
                //! coord_t<D> is a raw array: declare, then fill (see set_xy).
                coord_t<D> xp { ZERO }, xm { ZERO }, yp { ZERO }, ym { ZERO };
                set_xy<D>(xp, x + HALF * dx, y);
                set_xy<D>(xm, x - HALF * dx, y);
                set_xy<D>(yp, x, y + HALF * dy);
                set_xy<D>(ym, x, y - HALF * dy);
                cx =  (init_flds.bx3(yp) - init_flds.bx3(ym)) / dy;     //*  dBz/dy
                cy = -(init_flds.bx3(xp) - init_flds.bx3(xm)) / dx;     //* -dBz/dx
                cz =  (init_flds.bx2(xp) - init_flds.bx2(xm)) / dx - (init_flds.bx1(yp) - init_flds.bx1(ym)) / dy;
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
                coord_t<D> xy { ZERO };
                set_xy<D>(xy, x, y);
                ++n_samples;

                //* --- analytic current ---
                real_t Jx, Jy, Jz;
                init_flds.current(x, y, Jx, Jy, Jz);
                const real_t Jmag = (real_t)std::sqrt((double)(SQR(Jx) + SQR(Jy) + SQR(Jz)));

                J_sum += (double)Jmag;
                if (Jmag > J_max) J_max = Jmag;

                //* --- field at the same point ---
                const real_t Bx = init_flds.bx1(xy);
                const real_t By = init_flds.bx2(xy);
                const real_t Bz = init_flds.bx3(xy);
                const real_t Bmag = (real_t)std::sqrt((double)(SQR(Bx) + SQR(By) + SQR(Bz)));

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
            }

            const real_t inv_n      = (n_samples > 0) ? ONE / (real_t)n_samples : ZERO;
            const real_t J_mean     = (real_t)J_sum    * inv_n;
            const real_t curl_mean  = (real_t)curl_sum * inv_n;
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
            const real_t Jz_axis   = k_t;   //* k_t J0(0) = k_t
            const real_t beta_axis = amp * Jz_axis;
            const real_t Bz_axis   = (real_t)std::sqrt(1.0 + (double)guide_field_floor);

            //* ---- thermal factors ------------------------------------------------
            //* <gamma> for a Maxwell-Juttner distribution (entity_normalisations.md, section 3)
            const real_t gamma_mean = ONE + theta * ((real_t)6.0 + (real_t)15.0 * theta) / ((real_t)4.0 + (real_t)5.0 * theta);
            const real_t u_thermal  = (real_t)std::sqrt(std::max(0.0, (double)(SQR(gamma_mean) - ONE)));

            //* ---- characteristic scales -------------------------------------------
            const real_t d_pair_cold = d0;                              //* n_total = 1
            const real_t d_pair      = d_pair_cold * (real_t)std::sqrt((double)gamma_mean);
            const real_t d_sp_cold   = d0 * (real_t)std::sqrt(2.0);     //* one species, n_s = 0.5
            const real_t d_sp        = d_sp_cold * (real_t)std::sqrt((double)gamma_mean);
            const real_t lambda_D    = (real_t)std::sqrt((double)theta) * d_pair;
            const real_t rho_th      = r0 * u_thermal / Bz_axis;
            const real_t rho_drift   = r0 * (beta_axis * ONE / (real_t)std::sqrt(std::max(1.0e-12, 1.0 - (double)SQR(beta_axis)))) / Bz_axis;

            //* ---- magnetisation ---------------------------------------------------
            //*   sigma_s = B^2 sigma0 / (n_s m_s);  sigma_tot = B^2 sigma0 / sum_s n_s m_s
            const real_t B2_axis        = SQR(Bz_axis);                 //* B_phi = 0 on the axis
            const real_t sigma_tot_cold = B2_axis * sigma0 / n_total;
            const real_t sigma_sp_cold  = B2_axis * sigma0 / n_per_sp;
            //! gamma_d_max is the PEAK drift Lorentz factor over the tube, not a mean, so
            //! this is a lower bound on the hot magnetisation rather than a typical value.
            const real_t sigma_tot_hot  = sigma_tot_cold / (gamma_mean * gamma_d_max);

            //* ---- energetics -------------------------------------------------------
            const real_t U_B         = sigma0 * B2_axis / (real_t)2.0;  //* units of n0 m0 c^2
            const real_t P_th        = n_total * theta;                 //* sum_s n_s theta_s m_s
            const real_t plasma_beta = P_th / U_B;

            //* ---- timing -------------------------------------------------------------
            const real_t runtime = (real_t)params.template get<simtime_t>("simulation.runtime");
            const real_t dt      = params.template get<real_t>("algorithms.timestep.dt");
            const real_t n_steps = (dt > ZERO) ? runtime / dt : ZERO;

            constexpr int LW1 = 36;
            constexpr int LW2 = 30;
            auto kv1 = [&](const std::string& label) -> std::string { std::ostringstream t; t << "\t"   << std::left << std::setw(LW1) << label << "= "; return t.str(); };
            auto kv2 = [&](const std::string& label) -> std::string { std::ostringstream t; t << "\t\t" << std::left << std::setw(LW2) << label << "= "; return t.str(); };

            std::ostringstream oss;
            oss << "\n"
                << "===========================================================\n"
                << "Force-free merging Lundquist flux tubes"                << "\n"
                << "===========================================================\n\n"

                << "GEOMETRY\n"
                << kv1("Dimensionality")                   << (int)D << "D\n";

            if constexpr (D == Dim::_3D)
            {
                const auto   nx3 = metadomain.mesh().n_active(in::x3);
                oss << kv1("Resolution")                   << nx1 << " x " << nx2 << " x " << nx3 << "\n"
                    << kv1("Box (Lx x Ly x Lz) [d0]")      << Lx << " x " << Ly << " x " << Lz << "\n"
                    << kv1("Grid size (dx) [d0]")          << dx << "\n";
            }
            else
            {
                oss << kv1("Resolution")                   << nx1 << " x " << nx2 << "\n"
                    << kv1("Box (Lx x Ly) [d0]")           << Lx << " x " << Ly << "\n"
                    << kv1("Grid size (dx) [d0]")          << dx << "\n";
            }

            oss << kv1("Tube radius (R) [d0]")             << tube_radius << "\n"
                << kv1("Tube 1 centre")                    << "(" << init_flds.tube1_x_centre << ", " << init_flds.tube1_y_centre << ")\n";
            if (!single_tube)
                oss << kv1("Tube 2 centre")                << "(" << init_flds.tube2_x_centre << ", " << init_flds.tube2_y_centre << ")\n";
            oss << kv1("Timestep (dt)")                    << dt << "\n"
                << kv1("Runtime [L/c]")                    << runtime << "   (" << n_steps << " steps)\n"
                << kv1("Runtime [R/c]")                    << runtime / tube_radius << "\n\n"

                << "NORMALISATIONS \n"
                << kv1("skindepth0 (d0)")                  << d0 << "\n"
                << kv1("larmor0 (r0)")                     << r0 << "\n"
                << kv1("sigma0 [(d0/r0)^2]")               << sigma0 << "\n"
                << kv1("AMP [d0^2/r0] = J_code/curl(B)")   << amp << "\n"
                << kv1("ppc per species")                  << HALF * ppc0 << "\n\n"

                << "FIELD and CURRENT \n"
                << kv1("alpha_t [1st zero of J1]")         << InitFields<D>::alpha_t << "\n"
                << kv1("k_t = alpha_t/R [1/d0]")           << k_t << "\n"
                << kv1("Guide field floor ")               << guide_field_floor << "\n"
                << kv1("Kick velocity [v/c]")              << kick_velocity << "\n\n"

                << "DRIFT \n"
                << kv1("beta (v/c) on axis (max)")          << beta_axis << "\n"
                << kv1("Lorentz factor at max |beta (v/c)|") << gamma_d_max << "\n\n"

                << "J vs CURL(B) \n"
                << "\tCode current  J_code = AMP * J   (what the drift supplies)\n"
                << kv2("max |J_code|")                     << amp * J_max << "\n"
                << kv2("mean |J_code|")                    << amp * J_mean << "\n"
                << "\tCode current  AMP * curl(B)      (what the field demands)\n"
                << kv2("max |AMP curl B|")                 << amp * curl_max << "\n"
                << kv2("mean |AMP curl B|")                << amp * curl_mean << "\n\n"

                << "MAGNETISATION (on axis)\n"
                << kv1("total (cold)")                     << sigma_tot_cold << "\n"
                << kv1("per species (cold)")               << sigma_sp_cold << "\n"
                << kv1("total (hot + PEAK drift)")         << sigma_tot_hot << "\n"
                << kv1("Mean Lorentz factor")              << gamma_mean << "\n\n"

                << "CHARACTERISTIC SCALES \n"
                << "\tPair plasma \n"
                << kv2("Skin depth (cold)")                << d_pair_cold << "  (resolved with " << d_pair_cold / dx << " cells)\n"
                << kv2("Skin depth (hot)")                 << d_pair      << "  (resolved with " << d_pair / dx      << " cells)\n"
                << kv2("Debye length")                     << lambda_D    << "  (resolved with " << lambda_D / dx    << " cells)\n"
                << "\tOne species alone (supplementary info) \n"
                << kv2("Skin depth (cold)")                << d_sp_cold   << "  (resolved with " << d_sp_cold / dx   << " cells)\n"
                << kv2("Skin depth (hot)")                 << d_sp        << "  (resolved with " << d_sp / dx        << " cells)\n"
                << "\tLarmor radius (on axis)\n"
                << kv2("thermal plasma")                   << rho_th      << "  (resolved with " << rho_th / dx      << " cells)\n"
                << kv2("drifting beam (inside tubes)")     << rho_drift   << "  (resolved with " << rho_drift / dx   << " cells)\n\n"

                << "ENERGETICS (units of [n0 m0 c^2]) \n"
                << kv1("Temperature (kT/mc^2)")            << theta << "\n"
                << kv1("Magnetic energy (axis)")           << U_B << "\n"
                << kv1("Thermal pressure")                 << P_th << "\n"
                << kv1("Plasma beta")                      << plasma_beta << "\n\n"

                << "CONSISTENCY CHECKS \n"
                << kv1("max |beta (v/c)| (must be < 1)")   << beta_max << "\n"
                << kv1("max |JxB|/(|J||B|) (must be 0)")   << ff_residual << "\n"
                << kv1("max|J|/max|curl B| (must be 1)")   << ratio_max << "\n"
                << kv1("mean|J|/mean|curl B| (must be 1)") << ratio_mean << "\n"
                << kv1("J/curl(B) on axis (must be 1)")    << amp_ratio_axis << "\n"
                << kv1("max|J-curl(B)|/max|curl B| (-> 0)") << amp_dev_max << "\n"
                << kv1("lambda_D / dx (should be >~ 1)")   << lambda_D / dx << "\n\n";

            if (lambda_D / dx < ONE)
                oss << "!! WARNING: the pair Debye length is under-resolved (" << lambda_D / dx << " cells).\n"
                    << "!! Expect numerical heating; refine the grid or raise the temperature.\n\n";

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
                raise::ErrorIf((double)beta_max >= 0.99, "Flux tube drift >= 0.99c: the field requires more current than the what the density can carry. \n"
                                                         "Either increase tube_radius or raise the density (ppc) or lower sigma (raise larmor0).", HERE);

                //! (b) the analytic current must be force-free (catches sign / geometry errors  in the cylindrical -> Cartesian conversion)
                raise::ErrorIf((double)ff_residual > 1.0e-4, "Analytic current is not force-free: |J x B| != 0. Check InitFields::current().", HERE);

                //! (c) Ampere invariant: the analytic current must BE the curl of the field that is actually written to the grid
                //! (catches a missing AMP or a missing 1/dx in any future grid-based current)
                raise::ErrorIf((double)amp_dev_max > 1.0e-2, "Ampere mismatch: the analytic J differs from curl(B) of the initialised field. \n"
                                                             "The drifting particles can not support its own field and the tubes will not be in equilibrium. \n"
                                                             "Check InitFields::current() against InitFields::bx1/bx2/bx3.", HERE);
            }
        }

        //! =====================================================================
        //!  INITIALISE PARTICLES
        //!
        //!  The per-species drift is carried entirely by FluxTubeDist, 
        //!  so there is no post-injection particle loop.
        //! =====================================================================
        void InitPrtls(Domain<S, M>& local_domain)
        {
            metadomain.CommunicateFields(local_domain, Comm::E | Comm::B);

            //* sign(q) per species; the injector's species 1 / 2 are domain.species[0] / [1]
            const real_t charge_sign_1 = (local_domain.species[0].charge() < ZERO) ? -ONE : ONE;
            const real_t charge_sign_2 = (local_domain.species[1].charge() < ZERO) ? -ONE : ONE;

            const auto edist_1 = FluxTubeDist<M>(local_domain.random_pool(), background_temperature, amp, charge_sign_1, init_flds);
            const auto edist_2 = FluxTubeDist<M>(local_domain.random_pool(), background_temperature, amp, charge_sign_2, init_flds);
            const auto sdist   = UniformDist<M::Dim>();

            //! total injected density = ONE. The relation beta = AMP * J depends on this.
            arch::InjectNonUniform<S, M, decltype(edist_1), decltype(edist_2), decltype(sdist)>(params, local_domain, { 1, 2 }, { edist_1, edist_2 }, sdist, ONE);
        }

    };      //* struct PGen

}           //* namespace user

#endif      //* PROBLEM_GENERATOR_H