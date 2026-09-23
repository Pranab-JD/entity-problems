#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

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

#include <array>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace user
{
    using namespace ntt;

    //! =====================================================================
    //!  sigma_ion = sigma0 / mass_ratio = (skindepth0/larmor0)^2 / mass_ratio
    //!  B_BG = sqrt(sigma_ion) = (skindepth0/larmor0) / sqrt(mass_ratio)
    //! =====================================================================

    //! sech^2 density profile for one sheet centred at Ly/2
    template <Dimension D>
    struct HarrisProfile 
    {
        HarrisProfile(real_t cs_width, real_t cs_y) : cs_width { cs_width }, cs_y { cs_y } {}
        Inline auto operator()(const coord_t<D>& x_Ph) const -> real_t { return ONE / SQR(math::cosh((x_Ph[1] - cs_y) / cs_width));}

    private:
        const real_t cs_width, cs_y;
    };

    struct TurbMode {real_t kx, ky, kz, phase, amp; };

    //! =====================================================================
    //! Construct the divergence-free turbulent mode on HOST from a
    //! single vector-potential component (plane-selected)  
    //! RMS of dB = turbulence_amplitude
    //! plane: 0 -> xy (A_z), 1 -> yz (A_x), 2 -> zx (A_y)
    //! =====================================================================
    inline auto initial_turbulence(real_t Lx, real_t Ly, real_t Lz, int plane, real_t turbulence_amplitude,
                                    int kmin, int kmax, long long seed, real_t spectral_index,
                                    std::vector<TurbMode>& modes_out) -> std::array<real_t, 3> 
    {
        modes_out.clear();
        bool vary_x = false, vary_y = false, vary_z = false;
        if      (plane == 0) { vary_x = true; vary_y = true; }
        else if (plane == 1) { vary_y = true; vary_z = true; }
        else if (plane == 2) { vary_z = true; vary_x = true; }

        //* If turbulence_amplitude <= 0.0, return
        if (turbulence_amplitude <= (real_t)0.0 || kmax < kmin) return { (real_t)0.0, (real_t)0.0, (real_t)0.0 };

        std::mt19937_64 rng(static_cast<uint64_t>(seed));
        std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * M_PI);
        std::normal_distribution<double>       gauss_dist(0.0, 1.0);

        const real_t k0x = (real_t)(2.0 * M_PI) / Lx;
        const real_t k0y = (real_t)(2.0 * M_PI) / Ly;
        const real_t k0z = (real_t)(2.0 * M_PI) / Lz;

        real_t k0_ref = (real_t)0.0;
        if (vary_x) k0_ref = (k0_ref == (real_t)0.0) ? k0x : std::min(k0_ref, k0x);
        if (vary_y) k0_ref = (k0_ref == (real_t)0.0) ? k0y : std::min(k0_ref, k0y);
        if (vary_z) k0_ref = (k0_ref == (real_t)0.0) ? k0z : std::min(k0_ref, k0z);

        const real_t weight_exp = (real_t)(-0.5) * (spectral_index + (real_t)3.0);

        const int range_x = vary_x ? kmax : 0;
        const int range_y = vary_y ? kmax : 0;
        const int range_z = vary_z ? kmax : 0;

        for (int nx = 0; nx <= range_x; ++nx)
            for (int ny = -range_y; ny <= range_y; ++ny)
                for (int nz = -range_z; nz <= range_z; ++nz) 
                {
                    //* Half-space mode selection: avoid double-counting +/-k, skip zero mode
                    if (nx < 0) continue;
                    if (nx == 0 && ny < 0) continue;
                    if (nx == 0 && ny == 0 && nz <= 0) continue;

                    const real_t n_mag = (real_t)std::sqrt((double)(nx * nx + ny * ny + nz * nz));
                    if (n_mag < (real_t)kmin) continue;
                    if (n_mag > (real_t)kmax) continue;

                    const real_t kx = vary_x ? (real_t)nx * k0x : (real_t)0.0;
                    const real_t ky = vary_y ? (real_t)ny * k0y : (real_t)0.0;
                    const real_t kz = vary_z ? (real_t)nz * k0z : (real_t)0.0;
                    const real_t k_mag = (real_t)std::sqrt((double)(kx * kx + ky * ky + kz * kz));
                    if (k_mag == (real_t)0.0) continue;

                    const real_t rand_fac = (spectral_index > (real_t)0.0) ? (real_t)1.0 : (real_t)gauss_dist(rng);
                    const real_t pot_wt   = (spectral_index > (real_t)0.0) ? (real_t)std::pow((double)(k_mag / k0_ref), (double)weight_exp) : (real_t)1.0 / k_mag;

                    modes_out.push_back(TurbMode { kx, ky, kz, (real_t)phase_dist(rng), rand_fac * pot_wt });
                }

        //* Per-component RMS on a coarse host grid (resolution-independent enough)
        const int NS = 64; long   cnt  = 0;
        double s_bx = 0.0, s_by = 0.0, s_bz = 0.0;

        for (int ix = 0; ix < NS; ++ix)
            for (int iy = 0; iy < NS; ++iy)
            {
                const double x = ((double)ix + 0.5) / NS * (double)Lx;
                const double y = ((double)iy + 0.5) / NS * (double)Ly;
                const double z = 0.0;
                double dbx = 0.0, dby = 0.0, dbz = 0.0;
                
                for (const auto& m : modes_out) 
                {
                    const double s = std::sin((double)m.kx * x + (double)m.ky * y + (double)m.kz * z + (double)m.phase);
                    if      (plane == 0) { dbx += -(double)m.ky * (double)m.amp * s; dby += (double)m.kx * (double)m.amp * s; }
                    else if (plane == 1) { dby += -(double)m.kz * (double)m.amp * s; dbz += (double)m.ky * (double)m.amp * s; }
                    else if (plane == 2) { dbz += -(double)m.kx * (double)m.amp * s; dbx += (double)m.kz * (double)m.amp * s; }
                }
                
                s_bx += dbx * dbx; s_by += dby * dby; s_bz += dbz * dbz; ++cnt;
            }

        std::array<real_t, 3> scale { (real_t)0.0, (real_t)0.0, (real_t)0.0 };
        if (cnt > 0) 
        {
            const double tgt = (double)turbulence_amplitude;
            if (s_bx > 0.0) scale[0] = (real_t)(tgt / std::sqrt(s_bx / cnt));
            if (s_by > 0.0) scale[1] = (real_t)(tgt / std::sqrt(s_by / cnt));
            if (s_bz > 0.0) scale[2] = (real_t)(tgt / std::sqrt(s_bz / cnt));
        }
        return scale;
    }

    //! =====================================================================
    //! Single Harris tanh reversal + guide field + turbulence
    //!  B_x(y) = B0 * tanh((y - cs_y) / cs_width)
    //! =====================================================================
    template <Dimension D>
    struct InitFields 
    {
        InitFields(real_t bg_B, real_t guide_field, real_t cs_width, real_t cs_y,
                    int plane, array_t<real_t*> kx, array_t<real_t*> ky,
                    array_t<real_t*> kz, array_t<real_t*> ph, array_t<real_t*> am,
                    std::size_t nmodes, real_t sx, real_t sy, real_t sz): 
                    bg_B { bg_B }, guide_field { guide_field }, cs_width { cs_width }, 
                    cs_y { cs_y }, plane { plane }, 
                    kx { kx }, ky { ky }, kz { kz }, ph { ph }, am { am }, 
                    nmodes { nmodes }, sx { sx }, sy { sy }, sz { sz } {}

        Inline auto turb(const coord_t<D>& x, int comp) const -> real_t 
        {
            real_t acc = ZERO;
            const real_t X = x[0];
            const real_t Y = x[1];
            const real_t Z = (D == Dim::_3D) ? x[2] : ZERO;

            for (std::size_t m = 0; m < nmodes; ++m) 
            {
                const real_t s = math::sin(kx(m) * X + ky(m) * Y + kz(m) * Z + ph(m));
                if (plane == 0) 
                {   //* xy: A_z -> dBx=dA_z/dy, dBy=-dA_z/dx
                    if (comp == 0) acc += -ky(m) * am(m) * s;
                    if (comp == 1) acc +=  kx(m) * am(m) * s;
                } 
                else if (plane == 1) 
                {   //* yz: A_x -> dBy=dA_x/dz, dBz=-dA_x/dy
                    if (comp == 1) acc += -kz(m) * am(m) * s;
                    if (comp == 2) acc +=  ky(m) * am(m) * s;
                } 
                else if (plane == 2) 
                {   //* zx: A_y -> dBz=dA_y/dx, dBx=-dA_y/dz
                    if (comp == 2) acc += -kx(m) * am(m) * s;
                    if (comp == 0) acc +=  kz(m) * am(m) * s;
                }
            }
            return acc;
        }

        Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t 
        {
            return bg_B * math::tanh((x_Ph[1] - cs_y) / cs_width) + sx * turb(x_Ph, 0);
        }

        Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t 
        {
            return sy * turb(x_Ph, 1);
        }

        Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t 
        {
            return guide_field + sz * turb(x_Ph, 2);
        }

    private:
        const real_t      bg_B, guide_field, cs_width, cs_y;
        const int         plane;
        array_t<real_t*>  kx, ky, kz, ph, am;
        const std::size_t nmodes;
        const real_t      sx, sy, sz;
    };

    //! =====================================================================
    template <SimEngine::type S, class M>
    struct PGen 
    {
        static constexpr auto D { M::Dim };

        static constexpr auto engines { ::traits::pgen::compatible_with<SimEngine::SRPIC> {} };
        static constexpr auto metrics { ::traits::pgen::compatible_with<Metric::Minkowski> {} };
        static constexpr auto dimensions { ::traits::pgen::compatible_with<Dim::_2D, Dim::_3D> {} };

        const SimulationParams& params;
        Metadomain<S, M>&       metadomain;

        const real_t    larmor0;
        const real_t    skindepth0;
        const real_t    mass_ratio;     //* DERIVED = species[1].mass / species[0].mass (ion/electron)
        real_t          sigma_ion;      //* DERIVED = (skindepth0/larmor0)^2 / mass_ratio

        const real_t    cs_density;     //* n_CS/n_BG, overdensity of CS
        const real_t    cs_width;       //* Half-thickness of CS
        const real_t    guide_field;    //* Bg / B_in-plane
        
        const long long turb_seed;
        const real_t    turb_amp, spectral_index;
        const int       kmin, kmax, turb_plane;

        const real_t xmin, xmax, ymin, ymax, zmin, zmax, Lx, Ly, Lz, cs_y;

        //! Injection (along Y) Boundary Conditions
        const bool   inject_y;          //! true: replenish upstream plasma at the y-walls
        const real_t inj_ypad;          //! injection standoff from the y-walls (inject_y only)

        real_t B_BG;                    //* in-plane field (code units)
        real_t drift_u;                 //* drift four-velocity along x3
        real_t T_bg_e, T_bg_i;          //* background theta = kT/(m_s c^2), per species
        real_t T_cs_e, T_cs_i;          //* current-sheet theta, per species

        std::vector<TurbMode> modes_host;
        array_t<real_t*>      kx_d, ky_d, kz_d, ph_d, am_d;
        std::array<real_t, 3> tscale;

        InitFields<D> init_flds;

        static auto make_init(PGen& g) -> InitFields<D> 
        {
            raise::ErrorIf(g.mass_ratio < (real_t)1.0, "ion/positron mass (species 2) must be >= electron mass (species 1); mass ratio=1 is pair plasma", HERE);

            //* sigma_ion = (skindepth0/larmor0)^2 / mass_ratio
            g.sigma_ion = (g.skindepth0 / g.larmor0 * g.skindepth0 / g.larmor0) / g.mass_ratio;
            
            //* B_BG = sqrt(sigma_ion)
            g.B_BG = (real_t)std::sqrt((double)g.sigma_ion);

            const real_t beta_d = g.B_BG / ((real_t)2.0 * g.cs_density * g.cs_width);
            const real_t gamma_d = ONE / (real_t)std::sqrt(1.0 - (double)(beta_d * beta_d));
            g.drift_u = beta_d * gamma_d;

            g.T_cs_i = (g.B_BG * g.B_BG) * gamma_d / ((real_t)4.0 * g.cs_density);          //* theta_i^CS
            g.T_cs_e = g.T_cs_i * g.mass_ratio;                                             //* theta_e^CS = theta_i^CS * mass ratio

            g.T_bg_i = g.params.template get<real_t>("setup.bg_theta_i", (real_t)0.01);     //* theta_i^BG
            g.T_bg_e = g.T_bg_i * g.mass_ratio;                                             //* theta_e^BG = theta_i^BG * mass ratio

            //? Initialise Trubulence
            g.tscale = initial_turbulence(g.Lx, g.Ly, g.Lz, g.turb_plane, g.turb_amp * g.B_BG, 
                                          g.kmin, g.kmax, g.turb_seed, g.spectral_index, g.modes_host);

            const std::size_t nm = g.modes_host.size();
            g.kx_d = array_t<real_t*> { "turb_kx", nm };
            g.ky_d = array_t<real_t*> { "turb_ky", nm };
            g.kz_d = array_t<real_t*> { "turb_kz", nm };
            g.ph_d = array_t<real_t*> { "turb_ph", nm };
            g.am_d = array_t<real_t*> { "turb_am", nm };
            auto h_kx = Kokkos::create_mirror_view(g.kx_d);
            auto h_ky = Kokkos::create_mirror_view(g.ky_d);
            auto h_kz = Kokkos::create_mirror_view(g.kz_d);
            auto h_ph = Kokkos::create_mirror_view(g.ph_d);
            auto h_am = Kokkos::create_mirror_view(g.am_d);
            for (std::size_t m = 0; m < nm; ++m) 
            {
                h_kx(m) = g.modes_host[m].kx; h_ky(m) = g.modes_host[m].ky;
                h_kz(m) = g.modes_host[m].kz; h_ph(m) = g.modes_host[m].phase;
                h_am(m) = g.modes_host[m].amp;
            }
            Kokkos::deep_copy(g.kx_d, h_kx); Kokkos::deep_copy(g.ky_d, h_ky);
            Kokkos::deep_copy(g.kz_d, h_kz); Kokkos::deep_copy(g.ph_d, h_ph);
            Kokkos::deep_copy(g.am_d, h_am);

            return InitFields<D> { g.B_BG, g.B_BG * g.guide_field, g.cs_width, g.cs_y,
                                    g.turb_plane, g.kx_d, g.ky_d, g.kz_d, g.ph_d, g.am_d,
                                    nm, g.tscale[0], g.tscale[1], g.tscale[2] };
        }

        static auto mass_ratio_from_species(Metadomain<S, M>& m) -> real_t 
        {
            const auto& species = m.species_params();
            raise::ErrorIf(species.size() < 2, "ERROR in mass_ratio_from_species: pgen needs >= 2 species (electron=1, ion=2)", HERE);
            const real_t m_e = (real_t)species[0].mass();
            const real_t m_i = (real_t)species[1].mass();
            raise::ErrorIf(m_e <= (real_t)0.0, "ERROR in mass_ratio_from_species: species 1 (electron) must have mass > 0", HERE);
            return m_i / m_e;   //! = 1 for pair plasma
        }

        PGen(const SimulationParams& p, Metadomain<S, M>& m): params { p }, metadomain { m },
        larmor0         { p.template get<real_t>("scales.larmor0") },
        skindepth0      { p.template get<real_t>("scales.skindepth0") },
        mass_ratio      { mass_ratio_from_species(m) },
        cs_density      { p.template get<real_t>("setup.cs_density") },
        cs_width        { p.template get<real_t>("setup.cs_width") },
        guide_field     { p.template get<real_t>("setup.guide_field", (real_t)0.0) },
        turb_seed       { static_cast<long long>(p.template get<int>("setup.turb_seed", 12345)) },
        turb_amp        { p.template get<real_t>("setup.turb_amp", (real_t)0.0) },
        kmin            { p.template get<int>("setup.kmin", 1) },
        kmax            { p.template get<int>("setup.kmax", 3) },
        turb_plane      { p.template get<int>("setup.turb_plane", 0) },
        spectral_index  { (real_t)std::max(0.0, (double)p.template get<real_t>("setup.spectral_index", (real_t)1.6667)) },
        xmin            { m.mesh().extent(in::x1).first },
        xmax            { m.mesh().extent(in::x1).second },
        ymin            { m.mesh().extent(in::x2).first },
        ymax            { m.mesh().extent(in::x2).second },
        zmin            { (D == Dim::_3D) ? m.mesh().extent(in::x3).first : (real_t)0.0 },
        zmax            { (D == Dim::_3D) ? m.mesh().extent(in::x3).second : (real_t)1.0 },
        Lx              { xmax - xmin },
        Ly              { ymax - ymin },
        Lz              { zmax - zmin },
        cs_y            { (real_t)0.5 * (ymin + ymax) },                     //* current sheet at Y centre
        inject_y        { p.template get<bool>("setup.inject_y", false) },   //* default: reflecting Y
        inj_ypad        { p.template get<real_t>("setup.inj_ypad", (real_t)0.05 * (ymax - ymin)) },
        init_flds       { make_init(*this) } { print_setup(); }


        //? Startup diagnostics; all lengths in code units.
        void print_setup() 
        {
            #if defined(MPI_ENABLED)
                int mpi_rank = 0;
                MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
                if (mpi_rank != 0) return;
            #endif

            const bool is_pair = (mass_ratio == (real_t)1.0);
            //* species-2 label and the "electron freq." vs "ion freq." wording
            const std::string sp2      = is_pair ? "positron" : "ion";
            const std::string title    = is_pair ? "Relativistic Harris sheet (pair plasma)"
                                                 : "Relativistic Harris sheet (ion-electron plasma)";

            const auto nx1 = metadomain.mesh().n_active(in::x1);
            const auto nx2 = metadomain.mesh().n_active(in::x2);
            const real_t dx = Lx / (real_t)nx1;

            //* mean Lorentz factor depends only on theta = kT/(m_s c^2), NOT on particle identity.
            //* gamma correction (skin depth ~ sqrt<g>, Larmor ~ <g>) is thermal
            auto gamma_mean = [](real_t theta) -> real_t 
            {
                return ONE + theta * ((real_t)6.0 + (real_t)15.0 * theta) / ((real_t)4.0 + (real_t)5.0 * theta);
            };
            const real_t gamma_e_mean = gamma_mean(T_bg_e);                                 //* electron <gamma> from theta_e
            const real_t gamma_i_mean = gamma_mean(T_bg_i);                                 //* sp2 (positron/ion) <gamma> from theta_i

            //* characteristic scales (code units = d0). Both species carry the relativistic
            //* correction from their own <gamma>: skin depth ~ sqrt(<g>), Larmor ~ <g>.
            const real_t d_e       = skindepth0 * (real_t)std::sqrt((double)gamma_e_mean);                  //* electron skin depth = sqrt(<g_e>) * d0
            const real_t d_i       = skindepth0 * (real_t)std::sqrt((double)(mass_ratio * gamma_i_mean));   //* sp2 skin depth = sqrt(mr * <g_sp2>) * d0  (mass + thermal)
            const real_t rho_e     = larmor0 * gamma_e_mean;                                                //* electron Larmor = <g_e> * rho_e,cold
            const real_t rho_i     = larmor0 * mass_ratio * gamma_i_mean;                                   //* sp2 Larmor = mr * <g_sp2> * larmor0  (mass + thermal)
            const real_t lambda_De = (real_t)std::sqrt((double)T_bg_e) * d_e;                               //* electron Debye = sqrt(theta_e) * d_e,cold (theta_e already hot)
            const real_t lambda_Di = (real_t)std::sqrt((double)T_bg_i) * d_i;                               //* sp2 Debye = sqrt(theta_i) * d_i
            const real_t beta_d    = drift_u / (real_t)std::sqrt(1.0 + (double)(drift_u * drift_u));        //* drift 3-velocity
            const real_t gamma_d   = (real_t)std::sqrt(1.0 + (double)(drift_u * drift_u));                  //* drift Lorentz factor (= u/beta)
            const real_t runtime   = params.template get<simtime_t>("simulation.runtime");
            const real_t t_wpe     = runtime;                                                               //* 1/omega_pe (code unit as-is)
            const real_t t_wpi     = runtime / (real_t)std::sqrt((double)mass_ratio);                       //* 1/omega_p(sp2)
            const real_t t_Lx      = runtime / Lx;                                                          //* light-crossing times of Lx (c=1)

            const real_t sigma_e_cold = sigma_ion * mass_ratio;                                             //* B^2/(4pi n m_e c^2)
            const real_t sigma_e_hot  = sigma_e_cold / gamma_e_mean;                                        //* B^2/(4pi n <gamma_e> m_e c^2)

            //* PRESSURE BALANCE across the sheet (Harris equilibrium), code units, n_bg=1.
            //*   upstream magnetic  P_mag = B_BG^2 / 2
            //*   upstream thermal   P_th  = 2 * n_bg * theta_b        (2 species)
            //*   CS thermal (peak)  P_cs  = 2 * n_CS * theta_d / gamma_d   (co-moving, both species)
            const real_t P_mag = B_BG * B_BG / (real_t)2.0;
            const real_t P_th  = (real_t)2.0 * T_bg_i;                                                      //* n_bg=1 per species (pair: both theta_b)
            const real_t P_cs  = (real_t)2.0 * cs_density * T_cs_i / gamma_d;                               //* n_CS=cs_density, both species
            const real_t plasma_beta = P_th / P_mag;                                                        //* upstream thermal / magnetic pressure

            constexpr int LW1 = 34; 
            constexpr int LW2 = 30; 
            auto kv1 = [&](const std::string& label) -> std::string { std::ostringstream t; t << "\t" << std::left << std::setw(LW1) << label << "= "; return t.str(); };
            auto kv2 = [&](const std::string& label) -> std::string { std::ostringstream t; t << "\t\t" << std::left << std::setw(LW2) << label << "= "; return t.str(); };
            const std::string SP2u = is_pair ? "Positrons" : "Ions";
            const std::string sp2s = is_pair ? "positrons" : "ions";

            std::ostringstream oss;
            oss << "\n"
                << "===========================================================\n"
                << title << "\n"
                << "===========================================================\n\n"
                << "GEOMETRY\n"
                << kv1("Dimensionality") << (int)D << "D\n";
            if constexpr (M::Dim == Dim::_3D)
            {
                const auto nx3 = metadomain.mesh().n_active(in::x3);
                oss << kv1("Resolution") << nx1 << " x " << nx2 << " x " << nx3 << "\n"
                    << kv1("Box (Lx x Ly x Lz) [d0]") << Lx << " x " << Ly << " x " << Lz << "\n";
            }
            else
            {
                oss << kv1("Resolution") << nx1 << " x " << nx2 << "\n"
                    << kv1("Box (Lx x Ly) [d0]") << Lx << " x " << Ly << "\n";
            }
            oss << kv1("Grid size [d0]") << dx << "\n"
                << kv1("Current sheet at y [d0]") << cs_y << "\n"
                << kv1("Y boundaries") << (inject_y ? "INJECTION" : "REFLECTION") << "\n"
                << kv1("Runtime [cold electron freq.] ") << t_wpe << "\n"
                << kv1("Runtime [cold " + sp2 + " freq.] ") << t_wpi << "\n"
                << kv1("Runtime [c/L]") << t_Lx << "\n\n"

                << "MASS / MAGNETISATION\n"
                << kv1("Mass ratio [m_i/m_e]") << mass_ratio << "\n"
                << kv1("Sigma (" + sp2 + ")") << sigma_ion << "\n"
                << kv1("Sigma (electron, cold)") << sigma_e_cold << "\n"
                << kv1("Sigma (electron, hot)") << sigma_e_hot << "\n"
                << kv1("Mean electron Lorentz factor") << gamma_e_mean << "\n"
                << kv1("B_BG [B0]") << B_BG << "\n"
                << kv1("Guide field [Bg/B0]") << guide_field << "\n\n"

                << "CHARACTERISTIC SCALES\n"
                << "\tElectrons\n"
                << kv2("Skin depth") << d_e       << "  (resolved with " << d_e / dx       << " cells)\n"
                << kv2("Larmor radius") << rho_e << "  (resolved with " << rho_e / dx     << " cells)\n"
                << kv2("Debye length") << lambda_De << "  (resolved with " << lambda_De / dx << " cells)\n"
                << "\t" << SP2u << "\n"
                << kv2("Skin depth") << d_i       << "  (resolved with " << d_i / dx       << " cells)\n"
                << kv2("Larmor radius") << rho_i << "  (resolved with " << rho_i / dx     << " cells)\n"
                << kv2("Debye length") << lambda_Di << "  (resolved with " << lambda_Di / dx << " cells)\n\n"

                << "TEMPERATURES (theta = kT/mc^2)\n"
                << "\tBackground\n"
                << kv2("electrons") << T_bg_e << "\n"
                << kv2(sp2s) << T_bg_i << "\n"
                << "\tCurrent Sheet\n"
                << kv2("electrons") << T_cs_e << "\n"
                << kv2(sp2s) << T_cs_i << "\n\n"

                << "TURBULENCE\n"
                << kv1("Amplitude [dB/B0] (per comp)") << turb_amp << "\n"
                << kv1("Mode band [kmin, kmax]") << "[" << kmin << ", " << kmax << "]\n"
                << kv1("Plane (0=xy,1=yz,2=zx)") << turb_plane << "\n"
                << kv1("Spectral index [p]") << spectral_index << "\n"
                << kv1("Number of modes") << modes_host.size() << "\n\n"

                << "CURRENT SHEET\n"
                << kv1("Overdensity [n_CS/n_BG]") << cs_density << "\n"
                << kv1("Half-thickness [d0]") << cs_width << "\n"
                << kv1("Thermal spread of " + sp2s) << T_cs_i << "\n"
                << kv1("Thermal spread of electrons") << T_cs_e << "\n"
                << kv1("Drift four-velocity [u]") << drift_u << "\n"
                << kv1("beta = v/c ") << beta_d << "\n"
                << kv1("Lorentz factor of particles") << gamma_d << "\n\n"
                
                << "PRESSURE BALANCE\n"
                << kv1("Thermal pressure (CS)") << P_cs << "\n"
                << kv1("Thermal pressure (upstream)") << P_th << "\n"
                << kv1("Magnetic pressure (upstream)") << P_mag << "\n"
                << kv1("Plasma beta (upstream)") << plasma_beta << "\n\n"
                << "===========================================================\n";

            std::cout << oss.str() << std::endl;

            //* Write to <simulation.name>/Simulation_params.txt
            const std::string sim_name = params.template get<std::string>("simulation.name");
            std::string path = sim_name;
            if (!path.empty() && path.back() != '/') path += '/';
            path += "Simulation_params.txt";
            std::ofstream fout(path);
            if (fout.is_open()) { fout << oss.str(); fout.close(); }
            else { std::cout << "WARNING: could not write " << path << std::endl; }
        }

        //! =========================================================
        //! Initialise particles
        //! =========================================================

        void InitPrtls(Domain<S, M>& local_domain) 
        {
            //* InjectUniformMaxwellians divides each temperature by the species mass, so it expects FIDUCIAL (m0 c^2) units
            arch::InjectUniformMaxwellians<S, M>(params, local_domain, ONE, { T_bg_e, T_bg_e }, { 1, 2 });

            auto e_cs = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(local_domain.random_pool(), T_cs_e, { ZERO, ZERO,  drift_u });      //* electron drift + Z
            auto i_cs = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(local_domain.random_pool(), T_cs_i, { ZERO, ZERO, -drift_u });      //* ion/positron drift - Z
            
            const auto prof_cs = HarrisProfile<M::Dim>(cs_width, cs_y);
            arch::InjectNonUniform<S, M, decltype(e_cs), decltype(i_cs), decltype(prof_cs)>(params, local_domain, { 1, 2 }, { e_cs, i_cs }, prof_cs, cs_density);
        }

        //! =========================================================
        //! Replenish plasma at Y-walls (ONLY if Y is injecting)
        //! =========================================================
        void CustomPostStep(timestep_t, simtime_t, Domain<S, M>& domain) 
        {
            if (not inject_y) { return; }   //! reflecting Y: no replenishment

            //! Replenish background (upstream) plasma in thin slabs at each y-wall
            const auto energy_dist_e = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>( domain.random_pool(), T_bg_e);
            const auto energy_dist_i = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>( domain.random_pool(), T_bg_i);

            const auto dy = domain.mesh.metric.template sqrt_h_<2, 2>({});

            boundaries_t<real_t> inj_box_up, inj_box_down;
            inj_box_up.push_back(Range::All);                                           //* Inject through all of X
            inj_box_down.push_back(Range::All);

            inj_box_up.push_back({ ymax - inj_ypad - 10 * dy, ymax - inj_ypad });       //* Inject a tiny region along Y (top) 
            inj_box_down.push_back({ ymin + inj_ypad, ymin + inj_ypad + 10 * dy });     //* Inject a tiny region along Y (bottom) 
            
            if constexpr (M::Dim == Dim::_3D) 
            {
                inj_box_up.push_back(Range::All);                                       //* Inject through all of Z (if 3D)
                inj_box_down.push_back(Range::All);
            }

            //* target density = background n0 (buffer holds current species 1+2 density)
            arch::ComputeMomentWithSpecies<S, M, FldsID::Rho, 3>(params, domain, { 1, 2 }, domain.fields.buff);
            const auto replenish_sdist = arch::spatial_dist::ReplenishUniform<M, 3>(domain.mesh.metric, domain.fields.buff, 0u, ONE);

            arch::InjectNonUniform<S, M, decltype(energy_dist_e), decltype(energy_dist_i), decltype(replenish_sdist)>(params, 
                domain, { 1, 2 }, { energy_dist_e, energy_dist_i }, replenish_sdist, ONE, params.template get<bool>("particles.use_weights"), inj_box_up);

            arch::InjectNonUniform<S, M, decltype(energy_dist_e), decltype(energy_dist_i), decltype(replenish_sdist)>(params, 
                domain, { 1, 2 }, { energy_dist_e, energy_dist_i }, replenish_sdist, ONE, params.template get<bool>("particles.use_weights"), inj_box_down);
        }
    };

} // namespace user

#endif