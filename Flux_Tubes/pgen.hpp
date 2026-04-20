#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/field_setter.h"
#include "archetypes/problem_generator.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <type_traits>

namespace user
{
using namespace ntt;

template <Dimension D>
struct InitFields
{
    InitFields(real_t B0, real_t r_j, real_t x1, real_t y1, real_t x2, real_t y2, bool single)
    : B0{B0}, r_j{r_j}, x1{x1}, y1{y1}, x2{x2}, y2{y2}, single{single} {}

    Inline auto profile(real_t r) const -> real_t
    {
        if (r >= r_j) return ZERO;
        return static_cast<real_t>(0.5)*(ONE + math::cos(M_PI * r / r_j));
    }

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t
    {
        const real_t x = x_Ph[0];
        const real_t y = x_Ph[1];

        const real_t dx1 = x - x1;
        const real_t dy1 = y - y1;
        const real_t r1  = math::sqrt(SQR(dx1)+SQR(dy1));

        const real_t dx2 = x - x2;
        const real_t dy2 = y - y2;
        const real_t r2  = math::sqrt(SQR(dx2)+SQR(dy2));

        real_t Bx = ZERO;

        if (r1 > ZERO)
            Bx += -(dy1 / r1) * B0 * profile(r1);

        if (!single && r2 > ZERO)
            Bx += +(dy2 / r2) * B0 * profile(r2);

        return Bx;
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t
    {
        const real_t x = x_Ph[0];
        const real_t y = x_Ph[1];

        const real_t dx1 = x - x1;
        const real_t dy1 = y - y1;
        const real_t r1  = math::sqrt(SQR(dx1)+SQR(dy1));

        const real_t dx2 = x - x2;
        const real_t dy2 = y - y2;
        const real_t r2  = math::sqrt(SQR(dx2)+SQR(dy2));

        real_t By = ZERO;

        if (r1 > ZERO)
            By += +(dx1 / r1) * B0 * profile(r1);

        if (!single && r2 > ZERO)
            By += -(dx2 / r2) * B0 * profile(r2);

        return By;
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t
    {
        const real_t x = x_Ph[0];
        const real_t y = x_Ph[1];

        const real_t r1 = math::sqrt(SQR(x-x1)+SQR(y-y1));
        const real_t r2 = math::sqrt(SQR(x-x2)+SQR(y-y2));

        real_t Bz = ZERO;

        Bz += B0 * profile(r1);

        if (!single)
            Bz += B0 * profile(r2);

        return Bz;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t
    {
        return -beta * bx3(x_Ph);
    }

    Inline auto ex2(const coord_t<D>&) const -> real_t
    {
        return ZERO;
    }

    Inline auto ex3(const coord_t<D>& x_Ph) const -> real_t
    {
        return beta * bx1(x_Ph);
    }

private:
    const real_t B0, r_j, x1, y1, x2, y2;
    const bool single;
    const real_t beta{0.1};
};

template <SimEngine::type S, class M>
struct PGen : public arch::ProblemGenerator<S, M>
{
public:
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics { traits::compatible_with<Metric::Minkowski>::value };
    static constexpr auto dimensions { traits::compatible_with<Dim::_3D>::value };

    using Base = arch::ProblemGenerator<S, M>;

    using Base::D;
    using Base::C;
    using Base::params;

    using metadomain_type = Metadomain<S, M>;
    using real_t = std::remove_cv_t<decltype(ZERO)>;

    metadomain_type& global_domain;

private:
    real_t background_T { static_cast<real_t>(1.0e-2) };
    real_t beta_kick    { static_cast<real_t>(1.0e-1) };
    real_t c_param      { static_cast<real_t>(1.0e-4) };

    bool init_J         { true };
    bool init_rho       { true };
    bool init_ufl       { true };
    bool smooth_flds    { true };
    bool single_tube    { false };

    int   nsmooth       { 32 };
    real_t ppc_buff     { static_cast<real_t>(8.0) };

    // geometry
    real_t r_j { ZERO };
    real_t x1  { ZERO };
    real_t y1  { ZERO };
    real_t x2  { ZERO };
    real_t y2  { ZERO };

public:

    InitFields<M::Dim> init_flds;

    inline PGen(const SimulationParams& p, metadomain_type& md): Base{p}, global_domain(md), init_flds
    {
        p.template get<real_t>("setup.B0", static_cast<real_t>(1.0)),

        static_cast<real_t>((md.mesh().extent(in::x2).second - md.mesh().extent(in::x2).first) /
            p.template get<real_t>("setup.radius_fraction", static_cast<real_t>(4.0))),

        static_cast<real_t>(0.5) * static_cast<real_t>(md.mesh().extent(in::x1).first + md.mesh().extent(in::x1).second),

        static_cast<real_t>(0.5) * static_cast<real_t>(md.mesh().extent(in::x2).first + md.mesh().extent(in::x2).second) 
        - static_cast<real_t>((md.mesh().extent(in::x2).second - md.mesh().extent(in::x2).first) / p.template get<real_t>("setup.radius_fraction", static_cast<real_t>(4.0))),

        static_cast<real_t>(0.5) * static_cast<real_t>(md.mesh().extent(in::x1).first + md.mesh().extent(in::x1).second),

        static_cast<real_t>(0.5) * static_cast<real_t>(md.mesh().extent(in::x2).first + md.mesh().extent(in::x2).second) 
        + static_cast<real_t>((md.mesh().extent(in::x2).second - md.mesh().extent(in::x2).first) / p.template get<real_t>("setup.radius_fraction", static_cast<real_t>(4.0))),

        p.template get<bool>("setup.single_tube", false)
    }
    
    {
        ReadInput();
    }

    inline PGen() {}

    // ============================================================
    // INPUT
    // ============================================================
    inline void ReadInput()
    {
        background_T = params.template get<real_t>("setup.background_T", static_cast<real_t>(1.0e-2));
        beta_kick    = params.template get<real_t>("setup.beta_kick",    static_cast<real_t>(1.0e-1));
        c_param      = params.template get<real_t>("setup.c_param",      static_cast<real_t>(1.0e-4));

        init_J       = params.template get<bool>("setup.init_J", true);
        init_rho     = params.template get<bool>("setup.init_rho", true);
        init_ufl     = params.template get<bool>("setup.init_ufl", true);
        smooth_flds  = params.template get<bool>("setup.smooth_flds", true);
        single_tube  = params.template get<bool>("setup.single_tube", false);

        nsmooth      = params.template get<int>("setup.nsmooth", 32);
        ppc_buff     = params.template get<real_t>("setup.ppc_buff", static_cast<real_t>(8.0));
    }

    // ============================================================
    // FILTERS (unchanged except FIX)
    // ============================================================
    template <class Arr>
    inline void filterInX(Arr& arr, int do_n_times)
    {
        if (do_n_times <= 0) return;

        for (int n_pass = 1; n_pass <= do_n_times; ++n_pass)
        {
            int imin = arr.i_min() + n_pass;
            int imax = arr.i_max() - n_pass;

            int jmin = arr.j_min() + n_pass;
            int jmax = arr.j_max() - n_pass;

            int kmin = arr.k_min() + n_pass;
            int kmax = arr.k_max() - n_pass;

            for (int k = kmin; k <= kmax; ++k)
            for (int j = jmin; j <= jmax; ++j)
            for (int i = imin; i <= imax; ++i)
                arr(i,j,k) = 0.25*arr(i-1,j,k) + 0.5*arr(i,j,k) + 0.25*arr(i+1,j,k);
        }
    }

    template <class Arr>
    inline void filterInY(Arr& arr, int do_n_times)
    {
        if (do_n_times <= 0) return;

        for (int n_pass = 1; n_pass <= do_n_times; ++n_pass)
        {
            int imin = arr.i_min() + n_pass;
            int imax = arr.i_max() - n_pass;

            int jmin = arr.j_min() + n_pass;
            int jmax = arr.j_max() - n_pass;

            int kmin = arr.k_min() + n_pass;
            int kmax = arr.k_max() - n_pass;

            for (int k = kmin; k <= kmax; ++k)
            for (int j = jmin; j <= jmax; ++j)
            for (int i = imin; i <= imax; ++i)
                arr(i,j,k) = 0.25*arr(i,j-1,k) + 0.5*arr(i,j,k) + 0.25*arr(i,j+1,k);
        }
    }

    template <class ExArr, class EzArr, class BxArr, class ByArr, class BzArr>
    inline void filterFields(ExArr& ex, EzArr& ez, BxArr& bx, ByArr& by, BzArr& bz)
    {
        for (int n = 0; n < nsmooth; ++n)
        {
            filterInX(ex,1); filterInX(ez,1);
            filterInX(bx,1); filterInX(by,1); filterInX(bz,1);

            filterInY(ex,1); filterInY(ez,1);
            filterInY(bx,1); filterInY(by,1); filterInY(bz,1);

            global_domain.exchangeFields();
        }
    }

    // ============================================================
    // PARTICLES
    // ============================================================
    inline void InitPrtls(Domain<S, M>& domain)
    {
        arch::InjectUniformMaxwellian<S, M>(params, domain, ONE, background_T, {1,2});
    }
};

    //! ============================================================
    //! PARTICLE INITIALISATION
    //! ============================================================
    // inline void InitPrtls(Domain<S, M>& domain)
    // {
    //     auto& fields = domain.fields;

    //     auto& ex = fields.ex; auto& ey = fields.ey; auto& ez = fields.ez;
    //     auto& bx = fields.bx; auto& by = fields.by; auto& bz = fields.bz;
    //     auto& jx = fields.jx; auto& jy = fields.jy; auto& jz = fields.jz;
    //     auto& lg = fields.user_lg_arr;

    //     auto& species_vec = domain.species;

    //     const real_t sigma = params.template get<real_t>("setup.sigma", static_cast<real_t>(1.0));
    //     const real_t c_omp = params.template get<real_t>("setup.c_omp", static_cast<real_t>(1.0));
    //     const real_t cc    = params.template get<real_t>("algorithm.c", static_cast<real_t>(1.0));

    //     //* Initialise Maxwellian
    //     arch::InjectUniformMaxwellian<S, M>(params, domain, ONE, background_T, {1,2});

    //     Kokkos::fence();  // ensure particles exist before modifying

    //     //* Apply drif boosts
    //     for (auto& sp : species_vec)
    //     {
    //         const real_t qsign = (sp.q > ZERO) ? ONE : -ONE;

    //         auto& ux1 = sp.ux1;
    //         auto& ux2 = sp.ux2;
    //         auto& ux3 = sp.ux3;
    //         auto& w   = sp.weight;

    //         const auto& i1 = sp.i1;
    //         const auto& i2 = sp.i2;
    //         const auto& i3 = sp.i3;

    //         const auto& dx1 = sp.dx1;
    //         const auto& dx2 = sp.dx2;
    //         const auto& dx3 = sp.dx3;

    //         Kokkos::parallel_for("InitParticles", sp.npart(), Lambda(index_t p)
    //         {
    //             int i = i1(p);
    //             int j = i2(p);
    //             int k = i3(p);

    //             real_t dx = dx1(p);
    //             real_t dy = dx2(p);
    //             real_t dz = dx3(p);

    //             real_t ux = ux1(p);
    //             real_t uy = ux2(p);
    //             real_t uz = ux3(p);

    //             real_t wei = w(p);
    //             real_t gam = math::sqrt(ONE + ux*ux + uy*uy + uz*uz);

    //             //* Current-driven drift
    //             real_t beta_x = ZERO;
    //             real_t beta_y = ZERO;
    //             real_t beta_z = ZERO;

    //             if (init_J)
    //             {
    //                 real_t jx0 = jx(i,j,k);
    //                 real_t jy0 = jy(i,j,k);
    //                 real_t jz0 = jz(i,j,k);

    //                 beta_x = jx0 * math::sqrt(sigma) * c_omp * qsign / cc;
    //                 beta_y = jy0 * math::sqrt(sigma) * c_omp * qsign / cc;
    //                 beta_z = jz0 * math::sqrt(sigma) * c_omp * qsign / cc;
    //             }

    //             //* E x B drift
    //             if (init_ufl)
    //             {
    //                 real_t ex0 = ex(i,j,k);
    //                 real_t ey0 = ey(i,j,k);
    //                 real_t ez0 = ez(i,j,k);

    //                 real_t bx0 = bx(i,j,k);
    //                 real_t by0 = by(i,j,k);
    //                 real_t bz0 = bz(i,j,k);

    //                 real_t denom = bx0*bx0 + bz0*bz0;

    //                 if (denom > ZERO)
    //                     beta_y += (ez0 * bx0 - ex0 * bz0) / denom;
    //             }

    //             real_t beta_sq = beta_x*beta_x + beta_y*beta_y + beta_z*beta_z;

    //             //* Lorentz boost
    //             if (beta_sq < ONE && beta_sq > ZERO)
    //             {
    //                 real_t beta_dot_u = ux*beta_x + uy*beta_y + uz*beta_z;

    //                 real_t gam_boost = ONE / math::sqrt(ONE - beta_sq);

    //                 real_t ux_boost = gam_boost * beta_x;
    //                 real_t uy_boost = gam_boost * beta_y;
    //                 real_t uz_boost = gam_boost * beta_z;

    //                 real_t boost = (ux*ux_boost + uy*uy_boost + uz*uz_boost) / (gam_boost + ONE) + gam;

    //                 ux += boost * ux_boost;
    //                 uy += boost * uy_boost;
    //                 uz += boost * uz_boost;
    //             }

    //             ux1(p) = ux;
    //             ux2(p) = uy;
    //             ux3(p) = uz;

    //             //* Charge perturbation
    //             if (init_rho)
    //             {
    //                 real_t rho0 =
    //                     lg(i,   j,   0) * (ONE - dx)*(ONE - dy) +
    //                     lg(i+1, j,   0) * dx*(ONE - dy) +
    //                     lg(i,   j+1, 0) * (ONE - dx)*dy +
    //                     lg(i+1, j+1, 0) * dx*dy;

    //                 real_t wei_new = wei + rho0 * math::sqrt(sigma) * c_omp * qsign;

    //                 if (wei_new > ZERO)
    //                     w(p) = wei_new;
    //             }
    //         });
    //     }

    //     Kokkos::fence();

    //     //* DEBUG
    //     for (auto& sp : domain.species)
    //         printf("Species %s npart = %ld\n", sp.label().c_str(), sp.npart());
    //     fflush(stdout);
    // }

    // ============================================================
    // OPTIONAL CUSTOM HOOKS
    // ============================================================
    // inline void CustomPostStep(const metadomain_type&) {}

    // inline void CustomFieldOutput(const metadomain_type&) const {}

    // inline void CustomParticleOutput(const metadomain_type&) const {}
// };

} // namespace user

#endif
