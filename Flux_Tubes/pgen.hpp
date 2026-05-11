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
 * Vector potential:
 *   A_z(r) = (2 r_j / α) J₀(α r/r_j)   r < r_j
 *   A_z(r) = (2 r_j / α) J₀(α)          r ≥ r_j   [constant → no exterior field]
 *
 * In-plane B from the same discrete curl of A_z used in Tristan:
 *   B_x(i,j) = A_z(i,j+1) − A_z(i,j)
 *   B_y(i,j) = A_z(i,j) − A_z(i+1,j)
 *
 * Guide field:
 *   B_z = sqrt( J₀(α r/r_j)² + c_param )
 *
 * Force-free current density J = curl(B) [Tristan one-sided stencil]:
 *   J_x(i,j) = B_z(i,j) − B_z(i,j−1)
 *   J_y(i,j) = B_z(i−1,j) − B_z(i,j)
 *   J_z(i,j) = B_x(i,j−1) − B_x(i,j) − B_y(i−1,j) + B_y(i,j)
 *
 * Motional electric field (tubes kicked toward each other along y):
 *   Tube 1: E_x = −β_kick B_z,  E_z = +β_kick B_x
 *   Tube 2: E_x = +β_kick B_z,  E_z = −β_kick B_x
 *
 * Particle initialisation (mirrors Tristan's userInitParticles)
 * -------------------------------------------------------------
 * Each particle receives a Lorentz-boosted drift chosen so that the macroscopic
 * particle current matches J = curl(B):
 *   1. Build drift velocity β from J (current-driven) + E×B contributions.
 *   2. Probabilistically reflect particles moving strongly counter to β
 *      (Tristan "reflection trick": preserves the thermal distribution while
 *       building the net drift).
 *   3. Apply a Lorentz boost into the drift frame.
 *   4. [Optional, init_rho=true] Perturb particle weights by div(E) to seed
 *      the tiny charge-density perturbation that balances the electric field.
 *
 * Input parameters (under [problem] / setup.*)
 * --------------------------------------------
 *   background_T  — thermal temperature (in units of m_e c²)     [default 1e-2]
 *   beta_kick     — tube approach velocity / c                    [default 0.1]
 *   c_param       — guide-field background (prevents B_z → 0)    [default 0.01]
 *   init_rho      — apply charge-density weight correction        [default true]
 *   single_tube   — initialise only one tube                     [default false]
 *   nsmooth       — smoothing passes matching Tristan default          [default 32]
 *   ppc_buff      — target ppc in x-buffer zones (0.2–0.8 Lx)   [default 8]
 *   seed          — RNG seed (0 = random)                        [default 0]
 */

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/field_setter.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"

#include <utility>
#include <algorithm>

//? ---------------------------------------------------------------------------
//? GPU-compatible Bessel functions
//? CUDA/HIP provide device intrinsics j0/j1; on CPU we use std::cyl_bessel_j.
//? ---------------------------------------------------------------------------
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

    //! =========================================================================
    //!  InitFields
    //!
    //!  Analytical, device-callable description of the initial field state.
    //!  Provides bx1/bx2/bx3, ex1/ex2/ex3 (required by MatchFields / field BCs)
    //!  and jx1/jx2/jx3 (used during particle initialisation only).
    //! =========================================================================
    template <Dimension D>
    struct InitFields 
    {
        // First zero of J₁ — the eigenvalue of the Lundquist-tube problem
        static constexpr real_t ALPHA { static_cast<real_t>(3.8317059702075) };

        // Default-constructible so PGen can declare it before the ctor body runs
        InitFields() = default;

        /**
        * @param r_j_       Tube radius (code units).
        * @param x1_,y1_    Centre of tube 1.
        * @param x2_,y2_    Centre of tube 2.
        * @param dx_,dy_    Physical cell sizes (for finite-diff current evaluation).
        * @param c_param_   Guide-field floor parameter (keeps B_z > 0).
        * @param beta_kick_ Initial approach speed / c.
        * @param single_    If true, only tube 1 is initialised.
        */

        InitFields( real_t r_j_, real_t x1_, real_t y1_, real_t x2_, real_t y2_,
                    real_t dx_, real_t dy_, real_t c_param_, real_t beta_kick_, bool single_, int nsmooth_ = 32) : 
                    r_j { r_j_ }, 
                    x1 { x1_ }, y1 { y1_ }, 
                    x2 { x2_ }, y2 { y2_ }, 
                    dx { dx_ }, dy { dy_ }, 
                    nsmooth { nsmooth_ }, 
                    c_param { c_param_ }, 
                    beta_kick { beta_kick_ }, 
                    single_tube { single_ } {}

        KOKKOS_INLINE_FUNCTION
        auto az_local(real_t rn) const -> real_t 
        {
            const real_t rnc = rn < ONE ? rn : ONE;
            const real_t j0 = ft_j0(ALPHA * rnc);

            return r_j * j0 / ALPHA;
        }

        KOKKOS_INLINE_FUNCTION
        auto bz_local(real_t rn) const -> real_t 
        {
            const real_t rnc = rn < ONE ? rn : ONE;
            const real_t j0 = ft_j0(ALPHA * rnc);

            return math::sqrt(SQR(j0) + c_param);
        }

        KOKKOS_INLINE_FUNCTION
        auto az(const coord_t<D>& x_Ph) const -> real_t 
        {
            const real_t xp = x_Ph[0];
            const real_t yp = x_Ph[1];

            const real_t r1n = math::sqrt(SQR(xp - x1) + SQR(yp - y1)) / r_j;

            if (r1n < ONE)
                return az_local(r1n);

            if (!single_tube) 
            {
                const real_t r2n = math::sqrt(SQR(xp - x2) + SQR(yp - y2)) / r_j;

                if (r2n < ONE)
                return az_local(r2n);
            }

            return az_local(ONE);
        }

        KOKKOS_INLINE_FUNCTION
        auto bz_at(const coord_t<D>& x_Ph) const -> real_t 
        {
            const real_t xp = x_Ph[0], yp = x_Ph[1];

            const real_t r1n = math::sqrt(SQR(xp - x1) + SQR(yp - y1)) / r_j;
            if (r1n < ONE)
                return bz_local(r1n);

            if (!single_tube) 
            {
                const real_t r2n = math::sqrt(SQR(xp - x2) + SQR(yp - y2)) / r_j;
                if (r2n < ONE)
                    return bz_local(r2n);
            }
            return bz_local(ONE);   // = ft_j0(ALPHA) everywhere outside
        }

        KOKKOS_INLINE_FUNCTION
        void shifted(const coord_t<D>& x_Ph, real_t sx, real_t sy, coord_t<D>& xs) const 
        {
            xs[0] = x_Ph[0] + sx;
            xs[1] = x_Ph[1] + sy;
            xs[2] = x_Ph[2];
        }

        KOKKOS_INLINE_FUNCTION
        auto bx1_raw(const coord_t<D>& x_Ph) const -> real_t 
        {
            coord_t<D> yp { ZERO, ZERO, ZERO };
            coord_t<D> ym { ZERO, ZERO, ZERO };

            shifted(x_Ph, ZERO, HALF * dy, yp);
            shifted(x_Ph, ZERO, -HALF * dy, ym);

            return (az(yp) - az(ym)) / dy;
        }

        KOKKOS_INLINE_FUNCTION
        auto bx2_raw(const coord_t<D>& x_Ph) const -> real_t 
        {
            coord_t<D> xp { ZERO, ZERO, ZERO };
            coord_t<D> xm { ZERO, ZERO, ZERO };

            shifted(x_Ph, HALF * dx, ZERO, xp);
            shifted(x_Ph, -HALF * dx, ZERO, xm);

            return -(az(xp) - az(xm)) / dx;
        }

        KOKKOS_INLINE_FUNCTION
        auto bx3_raw(const coord_t<D>& x_Ph) const -> real_t 
        {
            return bz_at(x_Ph);
        }

        KOKKOS_INLINE_FUNCTION
        auto bx1(const coord_t<D>& x_Ph) const -> real_t 
        {
            return bx1_raw(x_Ph);
        }

        KOKKOS_INLINE_FUNCTION
        auto bx2(const coord_t<D>& x_Ph) const -> real_t 
        {
            return bx2_raw(x_Ph);
        }

        KOKKOS_INLINE_FUNCTION
        auto bx3(const coord_t<D>& x_Ph) const -> real_t 
        {
            return bx3_raw(x_Ph);
        }

        KOKKOS_INLINE_FUNCTION
        auto ex1(const coord_t<D>& x_Ph) const -> real_t 
        {
            if (beta_kick == ZERO)
                return ZERO;

            const real_t xp = x_Ph[0];
            const real_t yp = x_Ph[1];

            const real_t r1n = math::sqrt(SQR(xp - x1) + SQR(yp - y1)) / r_j;

            if (r1n < ONE)
                return -beta_kick * bx3(x_Ph);

            if (!single_tube) {
                const real_t r2n = math::sqrt(SQR(xp - x2) + SQR(yp - y2)) / r_j;

                if (r2n < ONE)
                    return beta_kick * bx3(x_Ph);
            }

            return ZERO;
        }

        KOKKOS_INLINE_FUNCTION
        auto ex2(const coord_t<D>&) const -> real_t 
        {
            return ZERO;
        }

        KOKKOS_INLINE_FUNCTION
        auto ex3(const coord_t<D>& x_Ph) const -> real_t 
        {
            if (beta_kick == ZERO)
                return ZERO;

            const real_t xp = x_Ph[0];
            const real_t yp = x_Ph[1];

            const real_t r1n = math::sqrt(SQR(xp - x1) + SQR(yp - y1)) / r_j;

            if (r1n < ONE)
                return beta_kick * bx1(x_Ph);

            if (!single_tube) {
                const real_t r2n = math::sqrt(SQR(xp - x2) + SQR(yp - y2)) / r_j;

                if (r2n < ONE)
                    return -beta_kick * bx1(x_Ph);
            }

            return ZERO;
        }

        KOKKOS_INLINE_FUNCTION
        auto jx1(const coord_t<D>& x_Ph) const -> real_t 
        {
            coord_t<D> yp { ZERO, ZERO, ZERO };
            coord_t<D> ym { ZERO, ZERO, ZERO };

            shifted(x_Ph, ZERO, HALF * dy, yp);
            shifted(x_Ph, ZERO, -HALF * dy, ym);

            return (bx3(yp) - bx3(ym)) / dy;
        }

        KOKKOS_INLINE_FUNCTION
        auto jx2(const coord_t<D>& x_Ph) const -> real_t 
        {
            coord_t<D> xp { ZERO, ZERO, ZERO };
            coord_t<D> xm { ZERO, ZERO, ZERO };

            shifted(x_Ph, HALF * dx, ZERO, xp);
            shifted(x_Ph, -HALF * dx, ZERO, xm);

            return -(bx3(xp) - bx3(xm)) / dx;
        }

        KOKKOS_INLINE_FUNCTION
        auto jx3(const coord_t<D>& x_Ph) const -> real_t 
        {
            coord_t<D> yp { ZERO, ZERO, ZERO };
            coord_t<D> ym { ZERO, ZERO, ZERO };
            coord_t<D> xp { ZERO, ZERO, ZERO };
            coord_t<D> xm { ZERO, ZERO, ZERO };

            shifted(x_Ph, ZERO, HALF * dy, yp);
            shifted(x_Ph, ZERO, -HALF * dy, ym);
            shifted(x_Ph, HALF * dx, ZERO, xp);
            shifted(x_Ph, -HALF * dx, ZERO, xm);

            const real_t dby_dx = (bx2(xp) - bx2(xm)) / dx;
            const real_t dbx_dy = (bx1(yp) - bx1(ym)) / dy;

            return dby_dx - dbx_dy;
        }

        real_t r_j { ZERO }, x1 { ZERO }, y1 { ZERO }, x2 { ZERO }, y2 { ZERO };
        real_t dx { ONE }, dy { ONE };
        int nsmooth { 32 };
        real_t c_param { ZERO }, beta_kick { ZERO };
        bool single_tube { false };
    };

    //! =========================================================================
    //!  PGen
    //! =========================================================================
    template <SimEngine::type S, class M>
    struct PGen : public arch::ProblemGenerator<S, M> 
    {

        static constexpr auto engines    { traits::compatible_with<SimEngine::SRPIC>::value };
        static constexpr auto metrics    { traits::compatible_with<Metric::Minkowski>::value };
        static constexpr auto dimensions { traits::compatible_with<Dim::_3D>::value };

        using Base            = arch::ProblemGenerator<S, M>;
        using metadomain_type = Metadomain<S, M>;

        using Base::D;
        using Base::C;
        using Base::params;

        // Reference to the global domain — needed for MPI field exchange
        metadomain_type& global_domain;

    private:
        // ------- runtime parameters (read in constructor) -------
        real_t background_T { static_cast<real_t>(1e-2) };
        real_t beta_kick    { static_cast<real_t>(0.1)  };
        real_t c_param      { static_cast<real_t>(0.01) };
        bool   init_rho     { true  };
        bool   single_tube  { false };
        int    nsmooth      { 32    };
        real_t ppc_buff     { static_cast<real_t>(8.0)  };

        // ------- global domain extents (set in constructor, used in InitPrtls) -------
        // InitPrtls receives a *local* Domain whose mesh covers only this MPI tile;
        // the buffer-zone boundaries must be computed from the global box size.
        real_t xmin_g { ZERO }, xmax_g { ZERO };

    public:
        InitFields<D> init_flds;

        //* -----------------------------------------------------------------------
        //*  Constructor
        //* -----------------------------------------------------------------------
        inline PGen(const SimulationParams& p, metadomain_type& md) : Base { p }, global_domain { md } 
        {

            // ---- Read input ----
            background_T = p.template get<real_t>("setup.background_T", static_cast<real_t>(1e-2));
            beta_kick    = p.template get<real_t>("setup.beta_kick",    static_cast<real_t>(0.1));
            c_param      = p.template get<real_t>("setup.c_param",      static_cast<real_t>(0.01));
            init_rho     = p.template get<bool>  ("setup.init_rho",     true);
            single_tube  = p.template get<bool>  ("setup.single_tube",  false);
            nsmooth      = p.template get<int>   ("setup.nsmooth",      32);
            ppc_buff     = p.template get<real_t>("setup.ppc_buff",     static_cast<real_t>(8.0));

            const auto& mesh = md.mesh();

            xmin_g = mesh.extent(in::x1).first;
            xmax_g = mesh.extent(in::x1).second;
            const real_t ymin_g = mesh.extent(in::x2).first;
            const real_t ymax_g = mesh.extent(in::x2).second;
            const real_t Lx = xmax_g - xmin_g;
            const real_t Ly = ymax_g - ymin_g;

            const real_t r_j = Lx * static_cast<real_t>(0.25);
            const real_t cx  = xmin_g + HALF * Lx;
            const real_t cy  = ymin_g + HALF * Ly;

            // Cell sizes (used by the finite-diff current evaluation in InitFields)
            const real_t cell_dx = Lx / static_cast<real_t>(mesh.n_active(in::x1));
            const real_t cell_dy = Ly / static_cast<real_t>(mesh.n_active(in::x2));

            init_flds = InitFields<D>(r_j,
                                    cx,  cy - r_j,   // tube 1 centre
                                    cx,  cy + r_j,   // tube 2 centre
                                    cell_dx, cell_dy,
                                    c_param, beta_kick, single_tube, nsmooth);
        }

        inline PGen() {}

        //* -----------------------------------------------------------------------
        //*  MatchFields
        //*  Returns the analytical field solution so the framework can set B and E
        //*  everywhere (and enforce field boundary conditions at each step).
        //* -----------------------------------------------------------------------
        auto MatchFields(simtime_t) const -> InitFields<D> 
        {
            return init_flds;
        }

        //* -----------------------------------------------------------------------
        //*  InitPrtls
        //*  Three stages, mirroring userInitParticles() in user_sheet.F90:
        //*    1. Inject thermal Maxwellian background (with buffer-zone thinning).
        //*    2. Apply current + E×B drift via a Lorentz boost to every particle.
        //*    3. [init_rho] Perturb particle weights by the charge density from div(E).
        //* -----------------------------------------------------------------------
        inline void InitPrtls(Domain<S, M>& domain) 
        {

            const auto& mesh = domain.mesh;

            // Buffer-zone boundaries derived from the *global* box
            // (xmin_g / xmax_g are stored in the constructor from the metadomain).
            const real_t Lx   = xmax_g - xmin_g;
            const real_t x_lo = xmin_g + static_cast<real_t>(0.2) * Lx;
            const real_t x_hi = xmin_g + static_cast<real_t>(0.8) * Lx;

            // Density fraction for buffer particles.
            // Tristan injects ppc_buff particles/cell in the buffer, each with
            // weight = ppc0/ppc_buff, preserving the physical density.
            const real_t ppc0     = params.template get<real_t>("particles.ppc0", static_cast<real_t>(4));
            const real_t buf_frac = ppc_buff / ppc0;   // <1 ↔ fewer macro-particles

            // ---- Stage 1: inject thermal plasma ----
            //
            // InjectUniformMaxwellians (plural) is the box-aware overload; signature:
            //   (params, domain, density, {T_e,T_i}, species, {drift_e,drift_i},
            //    use_weights, box)
            // Both species get the same temperature and zero bulk drift here;
            // the current-carrying drift is built and Lorentz-boosted in Stage 2.
            {
                const auto temps = std::make_pair(background_T, background_T);
                const auto no_drift = std::make_pair(
                std::vector<real_t>{ ZERO, ZERO, ZERO },
                std::vector<real_t>{ ZERO, ZERO, ZERO });

                boundaries_t<real_t> box_all;
                for (auto d = 0u; d < M::Dim; ++d)
                    box_all.push_back(Range::All);

                arch::InjectUniformMaxwellians<S, M>(
                params, domain, ONE, temps, { 1, 2 }, no_drift, false, box_all);
            }

            // ---- Stage 2: current-driven + E×B Lorentz boost ----
            //
            // Tristan uses the already-initialised grid fields, smooths them in
            // userInitFields(), computes J = curl(B) on the grid, and then
            // interpolates that grid J to the particles.  Therefore Entity should
            // not use init.jx1/2/3 here.  It should use domain.fields.em after
            // smoothing/communication.
            //   smoothFieldsInit(domain);

            const real_t sigma = params.template get<real_t>("scales.sigma0");
            const real_t c_omp = params.template get<real_t>("scales.skindepth0");
            const real_t Jfac  = math::sqrt(sigma) * c_omp;   // J → β conversion factor

            const auto em_field = domain.fields.em;
            auto       rpool    = domain.random_pool();

            const real_t x0_loc = mesh.extent(in::x1).first;
            const real_t x1_loc = mesh.extent(in::x1).second;
            const real_t y0_loc = mesh.extent(in::x2).first;
            const real_t y1_loc = mesh.extent(in::x2).second;

            const real_t dx_loc = (x1_loc - x0_loc) / static_cast<real_t>(mesh.n_active(in::x1));
            const real_t dy_loc = (y1_loc - y0_loc) / static_cast<real_t>(mesh.n_active(in::x2));

            real_t z0_loc = ZERO;
            real_t dz_loc = ONE;

            //TODO: check this with shock code
            if constexpr (D == 3) 
            {
                const real_t z1_loc = mesh.extent(in::x3).second;
                z0_loc = mesh.extent(in::x3).first;
                dz_loc = (z1_loc - z0_loc) / static_cast<real_t>(mesh.n_active(in::x3));
            }

            const auto init = init_flds;

            for (auto s = 0u; s < domain.species.size(); ++s) 
            {
                auto& sp      = domain.species[s];
                const real_t q = sp.charge();   // +1 (positron) or −1 (electron)

                const auto& i1  = sp.i1;
                const auto& i2  = sp.i2;
                const auto& i3  = sp.i3;
                const auto& dx1 = sp.dx1;
                const auto& dx2 = sp.dx2;
                const auto& dx3 = sp.dx3;
                const auto& tag = sp.tag;
                auto        ux1 = sp.ux1;
                auto        ux2 = sp.ux2;
                auto        ux3 = sp.ux3;

                Kokkos::parallel_for("FluxTubeDrift", sp.rangeActiveParticles(), KOKKOS_LAMBDA(index_t p) 
                {
                    if (tag(p) == ParticleTag::dead)
                    return;

                    const auto ii = i1(p);
                    const auto jj = i2(p);
                    const auto kk = i3(p);

                    const real_t fx = static_cast<real_t>(dx1(p));
                    const real_t fy = static_cast<real_t>(dx2(p));
                    const real_t fz = static_cast<real_t>(dx3(p));

                    coord_t<D> xp { ZERO, ZERO, ZERO };

                    xp[0] = x0_loc + (static_cast<real_t>(ii) + fx) * dx_loc;
                    xp[1] = y0_loc + (static_cast<real_t>(jj) + fy) * dy_loc;

                    if constexpr (D == 3)
                        xp[2] = z0_loc + (static_cast<real_t>(kk) + fz) * dz_loc;

                    real_t beta_x = q * Jfac * init.jx1(xp);
                    real_t beta_y = q * Jfac * init.jx2(xp);
                    real_t beta_z = q * Jfac * init.jx3(xp);

                    const real_t Ex = init.ex1(xp);
                    const real_t Ez = init.ex3(xp);
                    const real_t Bx = init.bx1(xp);
                    const real_t Bz = init.bx3(xp);
                    const real_t Bsq = Bx * Bx + Bz * Bz;

                    if (Bsq > ZERO)
                        beta_y += (Ez * Bx - Ex * Bz) / Bsq;

                    real_t beta_sq = beta_x*beta_x + beta_y*beta_y + beta_z*beta_z;

                    if (beta_sq <= ZERO)
                    return;

                    constexpr real_t BETA_MAX    = static_cast<real_t>(0.99);
                    constexpr real_t BETA_MAX_SQ = BETA_MAX * BETA_MAX;
                    if (beta_sq >= BETA_MAX_SQ) {
                    const real_t fac = BETA_MAX / math::sqrt(beta_sq);
                    beta_x *= fac;
                    beta_y *= fac;
                    beta_z *= fac;
                    beta_sq = BETA_MAX_SQ;
                    }

                    // ---- Probabilistic reflection (Tristan trick) ----
                    real_t ux  = ux1(p);
                    real_t uy  = ux2(p);
                    real_t uz  = ux3(p);
                    real_t gam = math::sqrt(ONE + ux*ux + uy*uy + uz*uz);

                    const real_t beta_dot_u = ux*beta_x + uy*beta_y + uz*beta_z;

                    auto gen = rpool.get_state();
                    const real_t rnd = Kokkos::rand<decltype(gen), real_t>::draw(gen);
                    rpool.free_state(gen);

                    if (-beta_dot_u / gam > rnd) {
                    const real_t inv_bsq = ONE / beta_sq;
                    ux -= static_cast<real_t>(2) * beta_dot_u * beta_x * inv_bsq;
                    uy -= static_cast<real_t>(2) * beta_dot_u * beta_y * inv_bsq;
                    uz -= static_cast<real_t>(2) * beta_dot_u * beta_z * inv_bsq;
                    gam = math::sqrt(ONE + ux*ux + uy*uy + uz*uz);
                    }

                    // ---- Lorentz boost into the drift frame ----
                    const real_t gam_b = ONE / math::sqrt(ONE - beta_sq);
                    const real_t ux_b  = gam_b * beta_x;
                    const real_t uy_b  = gam_b * beta_y;
                    const real_t uz_b  = gam_b * beta_z;

                    const real_t boost = (ux*ux_b + uy*uy_b + uz*uz_b) / (gam_b + ONE) + gam;

                    ux1(p) = ux + boost * ux_b;
                    ux2(p) = uy + boost * uy_b;
                    ux3(p) = uz + boost * uz_b;
                });
            }

      // ---- Stage 3: charge-density weight correction ----
      //
      // Tristan builds lg_arr from grid Ex using lg_arr(i,j) = ex(i,j) - ex(i-1,j),
      // then bilinearly interpolates lg_arr to the particle position before modifying weights.
    //   if (init_rho) {
    //     const real_t rho_fac = math::sqrt(sigma) * c_omp;
    //     // const auto em_field_rho = domain.fields.em;

    //     for (auto s = 0u; s < domain.species.size(); ++s) {
    //       auto& sp      = domain.species[s];
    //       const real_t q = sp.charge();

    //       const auto& i1  = sp.i1;
    //       const auto& i2  = sp.i2;
    //       const auto& i3  = sp.i3;
    //       const auto& dx1 = sp.dx1;
    //       const auto& dx2 = sp.dx2;
    //       const auto& tag = sp.tag;
    //       auto        wgt = sp.weight;

    //       Kokkos::parallel_for(
    //         "FluxTubeRho",
    //         sp.rangeActiveParticles(),
    //         KOKKOS_LAMBDA(index_t p) {
    //           if (tag(p) == ParticleTag::dead)
    //             return;

    //           const auto ii = i1(p);
    //           const auto jj = i2(p);
    //           const auto kk = i3(p);

    //           const real_t fx = static_cast<real_t>(dx1(p));
    //           const real_t fy = static_cast<real_t>(dx2(p));
    //           const real_t w00 = (ONE - fx) * (ONE - fy);
    //           const real_t w10 = fx * (ONE - fy);
    //           const real_t w01 = (ONE - fx) * fy;
    //           const real_t w11 = fx * fy;

    //           const real_t rho00 = em_field_rho(ii, jj, kk, em::ex1) - em_field_rho(ii - 1, jj, kk, em::ex1);
    //           const real_t rho10 = em_field_rho(ii + 1, jj, kk, em::ex1) - em_field_rho(ii, jj, kk, em::ex1);
    //           const real_t rho01 = em_field_rho(ii, jj + 1, kk, em::ex1) - em_field_rho(ii - 1, jj + 1, kk, em::ex1);
    //           const real_t rho11 = em_field_rho(ii + 1, jj + 1, kk, em::ex1) - em_field_rho(ii, jj + 1, kk, em::ex1);

    //           const real_t rho0 = w00 * rho00 + w10 * rho10 + w01 * rho01 + w11 * rho11;

    //           const real_t w_old = wgt(p);
    //           const real_t dw    = rho0 * rho_fac * q;
    //           const real_t w_new = w_old + dw;

    //           if (math::abs(dw) > static_cast<real_t>(1000))
    //             return;

    //           if (w_new <= ZERO)
    //             return;

    //           wgt(p) = w_new;
    //         });
    //     }
    //   }
    // }

      // ---- Stage 3 [optional]: charge-density weight correction (init_rho) ----
      //
      // In Tristan, the tiny charge density ρ₀ = div(E) ≈ ∂Ex/∂x is computed
      // from the (smoothed) field arrays and used to perturb particle weights:
      //   w_new = w + ρ₀ · √σ · d_e · sign(q)
      //
      // This seeds the electrostatic charge perturbation that must balance the
      // curl-free part of the electric field, preventing a transient burst of
      // plasma oscillations at t = 0.
      //
      // Here we compute ρ₀ analytically from InitFields.ex1 using a centred
      // finite difference ∂Ex/∂x ≈ [ex1(x+dx/2) − ex1(x−dx/2)] / dx.
    //   if (init_rho) {
    //     const real_t rho_fac = math::sqrt(sigma) * c_omp;  // ρ → Δw factor
    //     const real_t hdx     = HALF * init_flds.dx;

    //     for (auto s = 0u; s < domain.species.size(); ++s) {
    //       auto& sp       = domain.species[s];
    //       const real_t q = sp.charge();

    //       const auto& i1  = sp.i1;
    //       const auto& i2  = sp.i2;
    //       const auto& i3  = sp.i3;
    //       const auto& dx1 = sp.dx1;
    //       const auto& dx2 = sp.dx2;
    //       const auto& dx3 = sp.dx3;
    //       const auto& tag = sp.tag;
    //       auto        wgt = sp.weight;

    //       Kokkos::parallel_for(
    //         "FluxTubeRho",
    //         sp.rangeActiveParticles(),
    //         KOKKOS_LAMBDA(index_t p) {
    //           if (tag(p) == ParticleTag::dead)
    //             return;

    //           coord_t<M::Dim> x_Ph { ZERO };
    //           x_Ph[0] = mesh.metric.template convert<1, Crd::Cd, Crd::XYZ>(
    //             static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p)));
    //           x_Ph[1] = mesh.metric.template convert<2, Crd::Cd, Crd::XYZ>(
    //             static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p)));
    //           x_Ph[2] = mesh.metric.template convert<3, Crd::Cd, Crd::XYZ>(
    //             static_cast<real_t>(i3(p)) + static_cast<real_t>(dx3(p)));

    //           // ∂Ex/∂x centred finite difference
    //           coord_t<M::Dim> xp { ZERO, ZERO, ZERO }, xm { ZERO, ZERO, ZERO };
    //           xp[0] = x_Ph[0];
    //           xp[1] = x_Ph[1];
    //           xp[2] = x_Ph[2];
    //           xm[0] = x_Ph[0];
    //           xm[1] = x_Ph[1];
    //           xm[2] = x_Ph[2];
    //           xp[0] += hdx;
    //           xm[0] -= hdx;
    //           const real_t rho0 = (init.ex1(xp) - init.ex1(xm)) / init.dx;

    //           // Perturb weight.  Safety bounds matching Tristan's 1000-threshold.
    //           const real_t w_old = wgt(p);
    //           const real_t dw    = rho0 * rho_fac * q;
    //           const real_t w_new = w_old + dw;

    //           if (math::abs(dw) > static_cast<real_t>(1000))
    //             return;   // ignore pathological cells (e.g. at a sharp boundary)
    //           if (w_new <= ZERO)
    //             return;   // never allow negative weights

    //           wgt(p) = w_new;
    //         });
        // }
    //   }
        }   //! InitPrtls

    // -----------------------------------------------------------------------
    //  (Optional) CustomPostStep — nothing to do here; evolution is fully PIC.
    // -----------------------------------------------------------------------

    }; //! struct PGen

} //! namespace user

#endif // PROBLEM_GENERATOR_H