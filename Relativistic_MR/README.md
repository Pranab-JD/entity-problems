# Relativistic Harris Sheet

Everything here is built on the code normalisations in `entity_normalisations.md` and 3 results from there are used constantly:

$$\mathrm{AMP} = \frac{d_0^{2}}{r_0},\qquad
U_B = \frac{\sigma_0\hat B^{2}}{2},\qquad
P_{\rm th} = \sum_s \hat n_s\theta_s\hat m_s ,$$

with $\sigma_0 = (d_0/r_0)^2$, all in units of $n_0 m_0 c^2$, and $\hat{\mathbf{J}} = \mathrm{AMP}
(\hat\nabla\times\hat{\mathbf{B}})$ in equilibrium.

---

## 1. Geometry and profiles

Reversal across $y$, outflow along $x$, current and guide field along $z$:

$$\hat B_x(y) = \hat B\tanh\!\left(\frac{y-y_0}{\delta}\right),\qquad
\hat B_z = \hat B_g,\qquad
\hat n(y) = \hat n_{\rm bg} + \hat n_{\rm CS}\mathrm{sech}^2\!\left(\frac{y-y_0}{\delta}\right),$$

with $y_0 = \tfrac12(y_{\min}+y_{\max})$.

| TOML | symbol | meaning |
|---|---|---|
| `cs_density` | $\hat n_{\rm CS}$ | sheet overdensity, **total** over both species, in units of $n_0$ |
| `cs_width` | $\delta$ | **half**-thickness, in code length units |
| `guide_field` | $\hat B_g/\hat B$ | guide field, as a fraction of the in-plane field |
| `bg_theta_i` | $\theta_i^{\rm bg}$ | background temperature of species 2 |

The particles split the density, so each species peaks at $\hat n_{\rm CS}/2$ and the background is
$\hat n_{\rm bg}/2$ per species. $\hat n_{\rm bg} = 1$ with the standard uniform injection.

The useful identity for integrals, since $\int\mathrm{sech}^2(u)\mathrm{d}u = 2$:

$$\int \bigl(\hat n - \hat n_{\rm bg}\bigr)\mathrm{d}y = 2\hat n_{\rm CS}\delta,
\qquad \text{so } \langle \hat n\rangle_{\rm box} = \hat n_{\rm bg} + \frac{2\hat n_{\rm CS}\delta}{L_y}.$$

---

## 2. The field amplitude

$\hat B = 1$. The magnetisation is fixed by `[scales]`, not by the field:

$$\sigma_{\rm tot} = \frac{\hat B^{2}\sigma_0}{\sum_s \hat n_s\hat m_s}
\xrightarrow[\hat n_{\rm bg}=1]
\frac{2\hat B^{2}\sigma_0}{1+\mathrm{mr}} .$$

To hit a target, invert it: $\sigma_0 = \sigma_{\rm tot}^{\rm target}(1+\mathrm{mr})/2$ and then
$r_0 = d_0/\sqrt{\sigma_0}$. Setting $\hat B = \sqrt{\sigma_0}$ instead yields an actual magnetisation of
$\sigma_0^2$, and the field in the box is then $\sqrt{\sigma_0}$ times what you intended.

---

## 3. Drift: the Ampere condition

The two species counter-drift along $z$ at $\pm\beta_d$ with charges $\pm1$, so the sheet carries

$$\hat J_z = \sum_s \hat n_s\hat q_s\beta_s
= 2\left(\frac{\hat n_{\rm CS}}{2}\right)\beta_d = \hat n_{\rm CS}\beta_d
\qquad\text{at the peak.}$$

The field demands, from $\hat J = \mathrm{AMP} (\hat\nabla\times\hat B)$ with
$(\hat\nabla\times\hat B)_z = -\partial_y\hat B_x$ peaking at $\hat B/\delta$:

$$\hat J_z^{\rm req} = \mathrm{AMP}\frac{\hat B}{\delta}
\qquad\Longrightarrow\qquad
\boxed{\beta_d = \frac{\mathrm{AMP}\hat B}{\hat n_{\rm CS}\delta}}$$

Then $\gamma_d = (1-\beta_d^2)^{-1/2}$ and the four-velocity handed to the Maxwellian is
$u_d = \beta_d\gamma_d$.

**Feasibility.** $\beta_d < 1$ requires $\hat n_{\rm CS}\delta > \mathrm{AMP}\hat B$. A sheet that is
too thin or too dilute for its own field cannot be built; raise `cs_density`, raise `cs_width`, or lower
$\sigma$.

**The AMP factor is the trap.** Writing $\beta_d = \hat B/(\hat n_{\rm CS}\delta)$ gives a sheet carrying
$1/\mathrm{AMP}$ of the current its field requires. The residual $(\hat\nabla\times\hat B)_z - \hat J_z/
\mathrm{AMP}$ then drives $E_z$ from the first timestep, the layer pinches, and the whole box rings.

---

## 4. Temperature: pressure balance

Thermal pressure in the sheet must match the upstream magnetic pressure. With $\hat n_s = \hat n_{\rm CS}/2$
and equal $kT$ for both species ($\theta_i = \theta_e/\mathrm{mr}$, so $\theta_s\hat m_s = \theta_e$ for
each):

$$P_{\rm CS} = \frac{1}{\gamma_d}\sum_s\hat n_s\theta_s\hat m_s = \frac{\hat n_{\rm CS}\theta_e}{\gamma_d}
\overset{!}{=} U_B = \frac{\sigma_0\hat B^{2}}{2}$$

$$\boxed{\theta_e^{\rm CS} = \frac{\sigma_0\hat B^{2}\gamma_d}{2\hat n_{\rm CS}},
\qquad \theta_i^{\rm CS} = \frac{\theta_e^{\rm CS}}{\mathrm{mr}}}$$

The $1/\gamma_d$ converts the comoving pressure to the lab frame. Omitting $\sigma_0$ leaves the sheet
under-pressured by that factor and it expands from $t=0$ — a slow, monotonic broadening that is easy to
mistake for numerical diffusion.

**Upstream:**

$$P_{\rm up} = \frac{\hat n_{\rm bg}}{2}\bigl(\theta_e + \theta_i\mathrm{mr}\bigr) = \hat n_{\rm bg}\theta_e^{\rm bg},
\qquad
\beta_{\rm plasma} = \frac{P_{\rm up}}{U_B} = \frac{2\hat n_{\rm bg}\theta_e^{\rm bg}}{\sigma_0\hat B^{2}} .$$

Note the temperature convention: `bg_theta_i` sets species 2, and $\theta_e = \theta_i\mathrm{mr}$.

---

## 5. Startup invariants

Both must equal 1, otherwise the setup will fail:

$$\frac{\hat n_{\rm CS}\beta_d}{\mathrm{AMP}\hat B/\delta} = 1
\qquad\text{(Ampère)},
\qquad\qquad
\frac{P_{\rm CS}}{U_B} = 1
\qquad\text{(pressure)} .$$

They are independent: the first is wrong if AMP is missing from $\beta_d$, the second if $\sigma_0$ is
missing from $\theta^{\rm CS}$.

---

## 6. Derived quantities

With $\mathrm{mr} = \hat m_2/\hat m_1$:

| quantity | expression | note |
|---|---|---|
| in-plane field | $\hat B = 1$ | fixed, not a free parameter |
| drift | $\beta_d = \mathrm{AMP}\hat B/(\hat n_{\rm CS}\delta)$, $u_d = \beta_d\gamma_d$ | §3 |
| sheet temperature | $\theta_e^{\rm CS} = \sigma_0\hat B^2\gamma_d/(2\hat n_{\rm CS})$ | §4 |
| peak current | $\hat J_z = \hat n_{\rm CS}\beta_d = \mathrm{AMP}\hat B/\delta$ | §3 |
| current HWHM | $0.8814\delta$ | $\mathrm{sech}^2$ half-width, $\mathrm{arccosh}\sqrt2$ |
| Alfvén speed | $v_A/c = \sqrt{\sigma_{\rm tot}/(1+\sigma_{\rm tot})}$ | sets the outflow |
| reconnection rate | $\mathcal{R} = \lvert\hat E_z\rvert/(\hat B_{\rm up}v_A)$ | $\hat B_{\rm up}$ instantaneous, not initial |

---

## 7. Effects of current filtering

The written $\hat J$ is post-filter; the written $\hat n$ is not (`srpic.hpp` step ordering). For a
$\mathrm{sech}^2$ layer of half-thickness $\delta$ smoothed by $n$ binomial passes,
$\sigma_f = \sqrt{n/2}\mathrm{d}x$, the peak is reduced by roughly

| $\delta$ [cells] | 4 passes | peak deficit |
|---|---|---|
| 2.0 | $\sigma_f = 1.41$ | ~27 % |
| 2.5 | $\sigma_f = 1.41$ | ~21 % |
| 5.0 | $\sigma_f = 1.41$ | ~7 % |

while the integral $\int\hat J_z\mathrm{d}y$ is conserved. Consequences, all expected and none a bug:

- $\hat J_z^{\rm peak}$ is below $\mathrm{AMP}\hat B/\delta$ by that fraction at $t=0$;
- the written $\hat J$ is below $\hat n_{\rm CS}\beta_d$ by the same fraction;
- the pointwise ratio $\hat J/(\hat\nabla\times\hat B)$ is low at the centre and high on the flanks, so a
  fit over the layer shows a spread of order the deficit while its integral-weighted mean stays near AMP;
- $\int\hat J_z\mathrm{d}y = \mathrm{AMP}\Delta\hat B_x$ still holds to the discretisation error.

---

## 8. Failure modes specific to this setup

| symptom | cause |
|---|---|
| sheet pinches and the box rings from $t=0$ | $\beta_d$ omits AMP |
| sheet broadens monotonically from $t=0$ | $\theta^{\rm CS}$ omits $\sigma_0$ |
| field is $\sqrt{\sigma_0}$ times too strong, $\sigma$ is $\sigma_0^2$ | $\hat B$ set to $\sqrt{\sigma_0}$ |
| drift current a factor 2 off | `cs_density` treated as per species rather than total |
| $\beta_d \geq 1$ at startup | $\hat n_{\rm CS}\delta < \mathrm{AMP}\hat B$: too thin or too dilute for the field |
| coherent disturbance arriving at $t \approx L_y$ | start-up pulse reflected off conducting $y$ walls |
| written $\hat J$ peak below $\hat n_{\rm CS}\beta_d$ by a fixed fraction | current filter, see §7 |