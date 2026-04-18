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

template <SimEngine::type S, class M>
class PGen : public arch::ProblemGenerator<S, M>
{
public:
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics { traits::compatible_with<Metric::Minkowski>::value };
    static constexpr auto dimensions { traits::compatible_with<Dim::_3D>::value };

    using Base = arch::ProblemGenerator<S, M>;

    // for easy access to variables in the child class
    using Base::D;
    using Base::C;
    using Base::params;

    using metadomain_type = Metadomain<S, M>;
    using real_t = std::remove_cv_t<decltype(ZERO)>;

    metadomain_type& global_domain;

private:
    //? User parameters from input
    real_t background_T { static_cast<real_t>(1.0e-2) };    //* Background thermal temperature (kT/mc^2); initial Maxwellian spread of particles
    real_t beta_kick    { static_cast<real_t>(1.0e-1) };    //* Amplitude of motional E-field via E = -v × B; controls initial bulk flow / shear strength
    real_t c_param      { static_cast<real_t>(1.0e-4) };    //* Prevents Bz → 0 and stabilizes force balance; acts like a weak guide-field / pressure support term

    bool init_J         { true };                           //* Enforce J = curl(B) via particle velocity boost
    bool init_rho       { true };                           //* Apply charge correction using lg = ∇·E
    bool init_ufl       { true };                           //* Include E×B drift contribution in particle velocities
    bool smooth_flds    { true };                           //* Apply smoothing to E and B fields
    bool single_tube    { false };                          //* True = single flux tube OR False = two tubes 

    int   nsmooth       { 32 };
    real_t ppc_buff     { static_cast<real_t>(8.0) };

    //? Flux Tube params
    real_t r_j   { ZERO };
    real_t alpha { ZERO };
    real_t x1    { ZERO };
    real_t y1    { ZERO };
    real_t x2    { ZERO };
    real_t y2    { ZERO };

public:
    inline PGen(const SimulationParams& p, metadomain_type& md) : Base { p }, global_domain(md)
    {
        ReadInput();
    }

    //! ============================================================
    //! INPUT
    //! ============================================================
    inline void ReadInput()
    {
        background_T = params.template get<real_t>("setup.background_T", static_cast<real_t>(1.0e-2));
        beta_kick    = params.template get<real_t>("setup.beta_kick",    static_cast<real_t>(1.0e-1));
        c_param      = params.template get<real_t>("setup.c_param",      static_cast<real_t>(1.0e-2));

        init_J       = params.template get<bool>("setup.init_J",         true);
        init_rho     = params.template get<bool>("setup.init_rho",       true);
        init_ufl     = params.template get<bool>("setup.init_ufl",       true);
        single_tube  = params.template get<bool>("setup.single_tube",    false);
        smooth_flds  = params.template get<bool>("setup.smooth_flds",    true);

        nsmooth      = params.template get<int>("setup.nsmooth",         32);
        ppc_buff     = params.template get<real_t>("setup.ppc_buff",     static_cast<real_t>(8.0));
    }

    //? Helpers
    static inline real_t sqr(real_t x) { return x * x; }

    static inline real_t bessel_j0(real_t x)
    {
    #if __cplusplus >= 201703L
        return std::cyl_bessel_j(static_cast<real_t>(0), x);
    #else
        return std::cyl_bessel_j(0, x);
    #endif
    }

    //? Smoothing filters
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

            if (((imax - imin) % 2) == 0)
            {
                for (int k = kmin; k <= kmax; ++k)
                    for (int j = jmin; j <= jmax; ++j)
                    {
                        real_t tmp2 = arr(imin - 1, j, k);
                        for (int i = imin; i <= imax - 1; i += 2)
                        {
                            real_t tmp1 = 0.25 * arr(i-1, j, k) + 0.5  * arr(i, j, k) + 0.25 * arr(i+1, j, k);

                            arr(i-1, j, k) = tmp2;

                            tmp2 = 0.25 * arr(i, j, k) + 0.5 * arr(i+1, j, k) + 0.25 * arr(i+2, j, k);

                            arr(i, j, k) = tmp1;
                        }

                        real_t tmp1 = 0.25 * arr(imax-1, j, k) + 0.5  * arr(imax, j, k) + 0.25 * arr(imax+1, j, k);

                        arr(imax-1, j, k) = tmp2;
                        arr(imax, j, k)   = tmp1;
                    }
            }
            else
            {
                for (int k = kmin; k <= kmax; ++k)
                    for (int j = jmin; j <= jmax; ++j)
                    {
                        real_t tmp2 = arr(imin - 1, j, k);
                        for (int i = imin; i <= imax; i += 2)
                        {
                            real_t tmp1 = 0.25 * arr(i - 1, j, k) + 0.5  * arr(i, j, k) + 0.25 * arr(i + 1, j, k);

                            arr(i - 1, j, k) = tmp2;

                            tmp2 = 0.25 * arr(i, j, k) + 0.5  * arr(i + 1, j, k) + 0.25 * arr(i + 2, j, k);

                            arr(i, j, k) = tmp1;
                        }
                        arr(imax, j, k) = tmp2;
                    }
            }
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

            if (((jmax - jmin) % 2) == 0)
            {
                for (int k = kmin; k <= kmax; ++k)
                    for (int i = imin; i <= imax; ++i)
                    {
                        real_t tmp2 = arr(i, jmin - 1, k);
                        for (int j = jmin; j <= jmax - 1; j += 2)
                        {
                            real_t tmp1 = 0.25 * arr(i, j-1, k) + 0.5 * arr(i, j, k) + 0.25 * arr(i, j+1, k);

                            arr(i, j - 1, k) = tmp2;

                            tmp2 = 0.25 * arr(i, j, k) + 0.5  * arr(i, j+1, k) + 0.25 * arr(i, j+2, k);

                            arr(i, j, k) = tmp1;
                        }

                        real_t tmp1 = 0.25 * arr(i, jmax-1, k) + 0.5 * arr(i, jmax, k) + 0.25 * arr(i, jmax+1, k);

                        arr(i, jmax - 1, k) = tmp2;
                        arr(i, jmax,     k) = tmp1;
                    }
            }
            else
            {
                for (int k = kmin; k <= kmax; ++k)
                    for (int i = imin; i <= imax; ++i)
                    {
                        real_t tmp2 = arr(i, jmin - 1, k);
                        for (int j = jmin; j <= jmax; j += 2)
                        {
                            real_t tmp1 = 0.25 * arr(i, j - 1, k) + 0.5  * arr(i, j, k) + 0.25 * arr(i, j + 1, k);

                            arr(i, j - 1, k) = tmp2;

                            tmp2 = 0.25 * arr(i, j, k) + 0.5  * arr(i, j + 1, k) + 0.25 * arr(i, j + 2, k);

                            arr(i, j, k) = tmp1;
                        }
                        arr(i, jmax, k) = tmp2;
                    }
            }
        }
    }

    template <class ExArr, class EzArr, class BxArr, class ByArr, class BzArr>
    inline void filterFields(ExArr& ex, EzArr& ez, BxArr& bx, ByArr& by, BzArr& bz)
    {
        int n_pass = 3;
        int iter   = 0;

        while (true)
        {
            int i = (n_pass >= nsmooth) ? (nsmooth - iter * 3) : 3;

            if (nsmooth > 0)
            {
                filterInX(ex, i); filterInX(ez, i);
                filterInX(bx, i); filterInX(by, i); filterInX(bz, i);

                filterInY(ex, i); filterInY(ez, i);
                filterInY(bx, i); filterInY(by, i); filterInY(bz, i);

                global_domain.exchangeFields();   // FIXED
            }

            if (n_pass >= nsmooth) break;

            n_pass += 3;
            ++iter;
        }
    }

public:
    //! ============================================================
    //! FIELD INITIALISATION
    //! ============================================================
    inline void InitFields(const metadomain_type& /*md*/)
    {
        auto& fields = global_domain.fields;

        auto& ex = fields.ex;
        auto& ey = fields.ey;
        auto& ez = fields.ez;
        auto& bx = fields.bx;
        auto& by = fields.by;
        auto& bz = fields.bz;
        auto& jx = fields.jx;
        auto& jy = fields.jy;
        auto& jz = fields.jz;
        auto& lg = fields.user_lg_arr;

        ex.fill(ZERO); ey.fill(ZERO); ez.fill(ZERO);
        bx.fill(ZERO); by.fill(ZERO); bz.fill(ZERO);
        jx.fill(ZERO); jy.fill(ZERO); jz.fill(ZERO);
        lg.fill(ZERO);

        // physical box size
        const real_t Lx = global_domain.mesh().extent(in::x1).second - global_domain.mesh().extent(in::x1).first;
        const real_t Ly = global_domain.mesh().extent(in::x2).second - global_domain.mesh().extent(in::x2).first;

        // flux tube geometry
        r_j   = static_cast<real_t>(0.25) * Lx;
        alpha = static_cast<real_t>(3.8317059702075);

        x1 = static_cast<real_t>(0.5) * Lx;
        x2 = x1;
        y1 = static_cast<real_t>(0.5) * Ly - r_j;
        y2 = static_cast<real_t>(0.5) * Ly + r_j;

        // Ay (stored temporarily in ey) and Bz
        for (int j = ey.j_min(); j <= ey.j_max(); ++j)
            for (int i = ey.i_min(); i <= ey.i_max(); ++i)
            {
                const real_t x_ = global_domain.mesh().coord(in::x1, i);
                const real_t y_ = global_domain.mesh().coord(in::x2, j);

                const real_t x_shift = x_ + HALF * global_domain.mesh().dx(in::x1);
                const real_t y_shift = y_ + HALF * global_domain.mesh().dx(in::x2);

                const real_t r1       = math::sqrt(sqr(x_ - x1)      + sqr(y_ - y1))      / r_j;
                const real_t r1_shift = math::sqrt(sqr(x_shift - x1) + sqr(y_shift - y1)) / r_j;
                const real_t r2       = math::sqrt(sqr(x_ - x2)      + sqr(y_ - y2))      / r_j;
                const real_t r2_shift = math::sqrt(sqr(x_shift - x2) + sqr(y_shift - y2)) / r_j;

                if (r1 < ONE)
                    ey(i,j,0) = TWO * bessel_j0(r1 * alpha) * r_j / alpha;
                else if ((r2 < ONE) && (!single_tube))
                    ey(i,j,0) = -TWO * bessel_j0(r2 * alpha) * r_j / alpha;
                else
                    ey(i,j,0) = TWO * bessel_j0(alpha) * r_j / alpha;

                if (r1_shift < ONE)
                    bz(i,j,0) = math::sqrt(sqr(bessel_j0(r1_shift * alpha)) + c_param);
                else if ((r2_shift < ONE) && (!single_tube))
                    bz(i,j,0) = math::sqrt(sqr(bessel_j0(r2_shift * alpha)) + c_param);
                else
                    bz(i,j,0) = math::sqrt(sqr(bessel_j0(alpha)) + c_param);
            }

        //* Bx, By = curl(Ay)
        for (int j = bx.j_min(); j < bx.j_max(); ++j)
            for (int i = bx.i_min(); i < bx.i_max(); ++i)
            {
                bx(i,j,0) = -ey(i,j,0) + ey(i,j+1,0);
                by(i,j,0) =  ey(i,j,0) - ey(i+1,j,0);
            }

        ey.fill(ZERO);

        //* Motional electric field
        for (int j = ez.j_min() + 1; j <= ez.j_max(); ++j)
            for (int i = ez.i_min() + 1; i <= ez.i_max(); ++i)
            {
                const real_t x_ = global_domain.mesh().coord(in::x1, i);
                const real_t y_ = global_domain.mesh().coord(in::x2, j);

                const real_t x_shift = x_ + HALF * global_domain.mesh().dx(in::x1);

                const real_t r1       = math::sqrt(sqr(x_ - x1)      + sqr(y_ - y1)) / r_j;
                const real_t r1_shift = math::sqrt(sqr(x_shift - x1) + sqr(y_ - y1)) / r_j;
                const real_t r2       = math::sqrt(sqr(x_ - x2)      + sqr(y_ - y2)) / r_j;
                const real_t r2_shift = math::sqrt(sqr(x_shift - x2) + sqr(y_ - y2)) / r_j;

                const real_t bx0 = HALF * (bx(i,j,0) + bx(i,j-1,0));
                const real_t bz0 = HALF * (bz(i,j,0) + bz(i,j-1,0));

                if (r1 < ONE)
                    ez(i,j,0) =  beta_kick * bx0;
                else if ((r2 < ONE) && (!single_tube))
                    ez(i,j,0) = -beta_kick * bx0;
                else
                    ez(i,j,0) = ZERO;

                if (r1_shift < ONE)
                    ex(i,j,0) = -beta_kick * bz0;
                else if ((r2_shift < ONE) && (!single_tube))
                    ex(i,j,0) =  beta_kick * bz0;
                else
                    ex(i,j,0) = ZERO;
            }

        global_domain.exchangeFields();

        if (smooth_flds)
            filterFields(ex, ez, bx, by, bz);

        //? J = curl(B)
        const real_t corr = params.template get<real_t>("algorithm.corr", static_cast<real_t>(1.0));
        const real_t cc   = params.template get<real_t>("algorithm.c",    static_cast<real_t>(1.0));

        for (int j = jx.j_min(); j <= jx.j_max(); ++j)
            for (int i = jx.i_min(); i <= jx.i_max(); ++i)
            {
                jx(i,j,0) = corr * cc * (-bz(i,j-1,0) + bz(i,j,0));
                jy(i,j,0) = corr * cc * ( bz(i-1,j,0) - bz(i,j,0));
                jz(i,j,0) = corr * cc * ( bx(i,j-1,0) - bx(i,j,0) - by(i-1,j,0) + by(i,j,0));
            }

        global_domain.exchangeFields();

        // small charge perturbation
        for (int j = lg.j_min(); j <= lg.j_max(); ++j)
            for (int i = lg.i_min() + 1; i <= lg.i_max(); ++i)
                lg(i,j,0) = ex(i,j,0) - ex(i-1,j,0);
    }

    //! ============================================================
    //! PARTICLE INITIALISATION
    //! ============================================================
    template <class DomainType>
    inline void InitPrtls(DomainType& domain)
    {
        auto& fields = domain.fields;

        auto& ex = fields.ex; auto& ey = fields.ey; auto& ez = fields.ez;
        auto& bx = fields.bx; auto& by = fields.by; auto& bz = fields.bz;
        auto& jx = fields.jx; auto& jy = fields.jy; auto& jz = fields.jz;
        auto& lg = fields.user_lg_arr;

        auto& species_vec = domain.species;

        const real_t sigma = params.template get<real_t>("plasma.sigma", static_cast<real_t>(1.0));
        const real_t c_omp = params.template get<real_t>("plasma.c_omp", static_cast<real_t>(1.0));
        const real_t cc    = params.template get<real_t>("algorithm.c",  static_cast<real_t>(1.0));

        //? Background Thermal Plasma
        arch::InjectUniformMaxwellian<S, M>(params, domain, ONE, background_T, {1, 2});

        //? Current (via particle boost)
        for (auto& sp : species_vec)
        {
            const real_t qsign = (sp.q > ZERO) ? ONE : -ONE;

            auto& ux1 = sp.ux1;
            auto& ux2 = sp.ux2;
            auto& ux3 = sp.ux3;
            auto& w   = sp.weight;

            const auto& i1 = sp.i1;
            const auto& i2 = sp.i2;
            const auto& i3 = sp.i3;

            const auto& dx1 = sp.dx1;
            const auto& dx2 = sp.dx2;
            const auto& dx3 = sp.dx3;

            Kokkos::parallel_for("InitParticles", sp.npart(), Lambda(index_t p)
            {
                int i = i1(p);
                int j = i2(p);
                int k = i3(p);

                real_t dx = dx1(p);
                real_t dy = dx2(p);
                real_t dz = dx3(p);

                real_t ux = ux1(p);
                real_t uy = ux2(p);
                real_t uz = ux3(p);

                real_t wei = w(p);
                real_t gam = math::sqrt(ONE + ux*ux + uy*uy + uz*uz);

                //* Current-driven drift
                real_t beta_x = ZERO;
                real_t beta_y = ZERO;
                real_t beta_z = ZERO;

                if (init_J)
                {
                    // cell-centered approx
                    real_t jx0 = jx(i,j,k);
                    real_t jy0 = jy(i,j,k);
                    real_t jz0 = jz(i,j,k);

                    beta_x = jx0 * math::sqrt(sigma) * c_omp * qsign / cc;
                    beta_y = jy0 * math::sqrt(sigma) * c_omp * qsign / cc;
                    beta_z = jz0 * math::sqrt(sigma) * c_omp * qsign / cc;
                }

                real_t beta_sq = beta_x*beta_x + beta_y*beta_y + beta_z*beta_z;

                //* E x B drift
                if (init_ufl)
                {
                    real_t ex0 = ex(i,j,k);
                    real_t ey0 = ey(i,j,k);
                    real_t ez0 = ez(i,j,k);

                    real_t bx0 = bx(i,j,k);
                    real_t by0 = by(i,j,k);
                    real_t bz0 = bz(i,j,k);

                    real_t denom = bx0*bx0 + bz0*bz0;

                    if (denom > ZERO)
                        beta_y += (ez0 * bx0 - ex0 * bz0) / denom;
                }

                beta_sq = beta_x*beta_x + beta_y*beta_y + beta_z*beta_z;

                //* Lorentz boost
                if (beta_sq < ONE)
                {
                    real_t beta_dot_u = ux*beta_x + uy*beta_y + uz*beta_z;

                    if ((-beta_dot_u / gam) > Random<real_t>(domain.random_pool()))
                    {
                        ux -= TWO * beta_dot_u * beta_x / beta_sq;
                        uy -= TWO * beta_dot_u * beta_y / beta_sq;
                        uz -= TWO * beta_dot_u * beta_z / beta_sq;
                    }

                    real_t gam_boost = ONE / math::sqrt(ONE - beta_sq);

                    real_t ux_boost = gam_boost * beta_x;
                    real_t uy_boost = gam_boost * beta_y;
                    real_t uz_boost = gam_boost * beta_z;

                    real_t boost = (ux*ux_boost + uy*uy_boost + uz*uz_boost) / (gam_boost + ONE) + gam;

                    ux += boost * ux_boost;
                    uy += boost * uy_boost;
                    uz += boost * uz_boost;
                }

                ux1(p) = ux;
                ux2(p) = uy;
                ux3(p) = uz;

                //* Charge Perturbation
                if (init_rho)
                {
                    real_t rho0 =
                        lg(i,   j,   0) * (ONE - dx)*(ONE - dy) +
                        lg(i+1, j,   0) * dx*(ONE - dy) +
                        lg(i,   j+1, 0) * (ONE - dx)*dy +
                        lg(i+1, j+1, 0) * dx*dy;

                    real_t wei_new = wei + rho0 * math::sqrt(sigma) * c_omp * qsign;

                    if (wei_new > ZERO)
                        w(p) = wei_new;
                }
            });
        }

        lg.fill(ZERO);
    }

    // ============================================================
    // OPTIONAL CUSTOM HOOKS
    // ============================================================
    // inline void CustomPostStep(const metadomain_type&) {}

    // inline void CustomFieldOutput(const metadomain_type&) const {}

    // inline void CustomParticleOutput(const metadomain_type&) const {}
};

} // namespace user

#endif
