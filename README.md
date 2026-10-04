# entity normalisations

Reference for the SRPIC path, independent of any particular problem generator. Hatted symbols are code
quantities (what the TOML sets, what ADIOS writes); unhatted symbols are physical.

Each relation is marked **[source]** if read directly off the code, **[derived]** if it follows
algebraically, or **[convention]** if it is an interpretation the source is consistent with but does not
state. There is exactly one of the third kind, in §1.

---

## 1. Unit system

Heaviside–Lorentz-like units with $c = 1$ and the $4\pi$ absorbed:

$$\omega_{ps}^2 = \frac{n_s q_s^2}{m_s}, \qquad \omega_{cs} = \frac{q_s B}{m_s}, \qquad
\nabla\times\mathbf{B} = \mathbf{J} + \frac{\partial \mathbf{E}}{\partial t} .$$

**[convention]** The fiducial charge and mass satisfy $q_0/m_0 = 1$. This is forced by
`set("scales.omegaB0", ONE/larmor0)` sitting alongside `set("scales.B0", ONE/larmor0)`: the two are equal
only if $\omega_{c0} = q_0 B_0/m_0 = B_0$.

### Fiducial quantities

| symbol | code name | definition | source |
|---|---|---|---|
| $L$ | — | the code length unit, i.e. the unit of `grid.extent` | implicit |
| $d_0$ | `skindepth0` | fiducial skin depth, in units of $L$ | `[scales]` **[source]** |
| $r_0$ | `larmor0` | fiducial Larmor radius, in units of $L$ | `[scales]` **[source]** |
| $\sigma_0$ | `scales.sigma0` | $(d_0/r_0)^2$ | `parameters.cpp` **[source]** |
| $B_0$ | `scales.B0` | $1/r_0$ | `parameters.cpp` **[source]** |
| $V_0$ | `scales.V0` | $\sqrt{\det h}$ at a cell corner; the cell volume, $=\mathrm{d}x^D$ in Minkowski | `grid.cpp` **[source]** |
| $n_0$ | `scales.n0` | $\mathrm{ppc}_0/V_0$ | `parameters.cpp` **[source]** |
| $q_0$ | `scales.q0` | $V_0/(\mathrm{ppc}_0 d_0^{\,2})$ | `parameters.cpp` **[source]** |

**[derived]** The last two give the relation everything else rests on:

$$q_0 n_0 = \frac{1}{d_0^{\,2}}
\quad\Longleftrightarrow\quad
\omega_{p0}^2 \equiv \frac{n_0 q_0^2}{m_0} = \frac{1}{d_0^{\,2}}
\quad\text{(using } q_0/m_0 = 1\text{)} .$$

So $d_0$ is the skin depth of a plasma of **total** density $n_0$ made of unit-mass, unit-charge
particles. It is not per species, and it is not the skin depth of any particular species unless that
species happens to carry all of $n_0$ at unit mass and charge.

### Code variables

$$\hat{\mathbf{B}} = \frac{\mathbf{B}}{B_0},\quad
\hat{\mathbf{E}} = \frac{\mathbf{E}}{B_0},\quad
\hat n_s = \frac{n_s}{n_0},\quad
\hat m_s = \frac{m_s}{m_0},\quad
\hat q_s = \frac{q_s}{q_0},\quad
\hat{\mathbf{J}} = \frac{\mathbf{J}}{n_0 q_0 c},$$

lengths in units of $L$, time in units of $L/c$. The `mass` and `charge` entries of
`[[particles.species]]` are $\hat m_s$ and $\hat q_s$.

---

## 2. Field equations as integrated

**[source]** `faraday_mink.hpp`:

$$\frac{\partial \hat{\mathbf{B}}}{\partial \hat t} = -\hat\nabla\times\hat{\mathbf{E}} .$$

**[source]** `ampere_mink.hpp`, 3D. The curl term carries $\mathrm{coeff}_1 = \mathrm{d}t/\mathrm{d}x$. The
current term carries $\mathrm{coeff} = -\mathrm{d}t\,q_0/(B_0 V_0)$ applied to the **raw** deposit, which
is $\mathrm{ppc}_0\hat{J}$ because `J /= ppc0` happens only after the field update. Therefore

$$\frac{\partial \hat{\mathbf{E}}}{\partial \hat t}
= \hat\nabla\times\hat{\mathbf{B}} \;-\; \frac{q_0 n_0}{B_0}\,\hat{\mathbf{J}}
= \hat\nabla\times\hat{\mathbf{B}} \;-\; \frac{r_0}{d_0^{\,2}}\,\hat{\mathbf{J}} .$$

**[derived]** Define

$$\boxed{\;\mathrm{AMP} \;\equiv\; \frac{d_0^{\,2}}{r_0}\;}
\qquad\Longrightarrow\qquad
\hat{\mathbf{J}} = \mathrm{AMP}\,\bigl(\hat\nabla\times\hat{\mathbf{B}}\bigr)\;\;\text{in equilibrium}.$$

For $d_0 = 1$ this is $\sqrt{\sigma_0}$, but only then: in general $\mathrm{AMP} = d_0\sqrt{\sigma_0}$.

A problem generator that assumes $\hat J = \hat\nabla\times\hat B$ initialises a current layer with
$1/\mathrm{AMP}$ of the current its own field requires.


**[source]** `srpic.hpp` orders a step as Faraday($\tfrac12$) → push → deposit → filter → Faraday($\tfrac12$)
→ Ampère(curl) → Ampère(currents). The written $\hat J$ is therefore **post-filter**, while $\hat n$ from
`particle_moments.hpp` is not: the two disagree at a sharp peak by the filter's peak deficit while
conserving their integrals.

**[source]** Timestep: $\mathrm{d}t = \mathrm{CFL}\cdot\mathrm{d}x/\sqrt{D}$.

---

## 3. Per-species quantities

**[derived]** $\hat n_s$, $\hat q_s$, $\hat m_s$

| quantity | expression |
|---|---|
| plasma frequency | $\omega_{ps} = \dfrac{1}{d_0}\,\lvert\hat q_s\rvert\sqrt{\dfrac{\hat n_s}{\hat m_s}}$ |
| skin depth | $d_s = \dfrac{d_0}{\lvert\hat q_s\rvert}\sqrt{\dfrac{\hat m_s}{\hat n_s}}$ |
| gyrofrequency | $\omega_{cs} = \dfrac{\lvert\hat q_s\rvert\,\hat B}{r_0\,\hat m_s\,\gamma}$ |
| Larmor radius | $\rho_s = \dfrac{r_0\,\hat m_s\,u_\perp}{\lvert\hat q_s\rvert\,\hat B}$,  $u = \gamma\beta$ |
| magnetisation | $\sigma_s = \left(\dfrac{\omega_{cs}}{\omega_{ps}}\right)^2_{\gamma=1} = \dfrac{\hat B^{2}\sigma_0}{\hat n_s \hat m_s}$ |
| Debye length | $\lambda_{Ds} = \sqrt{\theta_s}\,d_s$,  $\theta_s = kT_s/(m_s c^2)$ |


Two things worth noting:

- **$\sigma_s$ carries no charge factor.** $\omega_c \propto q$ and $\omega_p \propto q$, so $\hat q_s^2$
  cancels in the ratio: $\sigma_s = \hat B^2\sigma_0/(\hat n_s\hat m_s)$ exactly, for any charge.
  $\omega_{ps}$, $d_s$ and $\rho_s$ do each carry $\hat q_s$.
- **The Larmor radius uses $u_\perp$, not $\gamma$.** $\rho = \gamma m v_\perp/(qB) = m u_\perp/(qB)$.
  Replacing $u_\perp$ by $\gamma$ assumes $v_\perp = c$, which is only right for ultra-relativistic
  particles. For a thermal population use $u_{\rm th} = \sqrt{\langle\gamma\rangle^2 - 1}$, which is
  $\simeq\sqrt{3\theta}$ when $\theta\ll1$. At $\theta = 0.01$ the difference is a factor 5.8.

### Total magnetisation

$$\sigma_{\rm tot} = \frac{\hat B^{2}\sigma_0}{\sum_s \hat n_s \hat m_s} .$$

For a pair plasma ($\hat n_1 = \hat n_2 = \hat n/2$, $\hat m_1 = 1$,
$\hat m_2 = \mathrm{mr}$):

$$\sigma_s = \frac{2\hat B^2\sigma_0}{\hat n\,\hat m_s},
\qquad
\sigma_{\rm tot} = \frac{2\hat B^{2}\sigma_0}{\hat n\,(1+\mathrm{mr})}
\;\;\xrightarrow[\;\hat n = 1\;]{}\;\; \frac{2\hat B^{2}\sigma_0}{1+\mathrm{mr}} .$$

For pairs $\sigma_{\rm tot} = \hat B^2\sigma_0$ while each species individually has twice that; for
ion–electron $\sigma_{\rm tot}\simeq\sigma_i$ and the electron value $2\hat B^2\sigma_0$ is irrelevant
except as a bound on electron energisation.

### Setting $\sigma$ to a target

$\sigma$ is set by `[scales]`, **not** by the field amplitude. Inverting the $\sigma_{\rm tot}$ relation:

$$\sigma_0 = \frac{\sigma_{\rm tot}^{\rm target}}{\hat B^2}\sum_s \hat n_s\hat m_s,
\qquad r_0 = \frac{d_0}{\sqrt{\sigma_0}} .$$

With $\hat B = 1$ and the pair injectors at $\hat n = 1$ this is
$r_0 = d_0\sqrt{2/\bigl(\sigma_{\rm tot}(1+\mathrm{mr})\bigr)}$, which reduces to $d_0/\sqrt{\sigma_{\rm tot}}$
for pairs. Setting $\hat B = \sqrt{\sigma_0}$ instead of adjusting $r_0$ gives an actual magnetisation of
$\sigma_0^2$.

### Thermal corrections

For a Maxwell–Jüttner distribution,
$\langle\gamma\rangle \simeq 1 + \theta(6+15\theta)/(4+5\theta)$, and

$$d_s \to d_s\sqrt{\langle\gamma\rangle},\qquad
\rho_s \to \frac{r_0\hat m_s\sqrt{\langle\gamma\rangle^2-1}}{\lvert\hat q_s\rvert\hat B},\qquad
\sigma_s^{\rm hot} = \frac{\sigma_s}{\langle\gamma\rangle} .$$

Strictly $\sigma^{\rm hot}$ should divide by the **enthalpy** per particle rather than the energy:

$$\frac{w}{n m c^2} = \langle\gamma\rangle + \theta
\qquad\text{(energy }\langle\gamma\rangle\text{ plus pressure }\theta\text{)},$$

which reduces to the familiar $1+4\theta$ only in the ultra-relativistic limit, where
$\langle\gamma\rangle\to3\theta$. Do **not** write $\langle\gamma\rangle + 4\theta$: that counts the
thermal part twice. The enthalpy correction relative to $\langle\gamma\rangle$ alone is 1.0 % at
$\theta = 0.01$, 8.6 % at $\theta = 0.1$ and a factor 1.30 at $\theta = 1$.

---

## 4. Energy and pressure

**[derived]** In units of $n_0 m_0 c^2$, using $n_0 m_0 = 1/d_0^{\,2}$:

$$U_B = \frac{\sigma_0\hat B^{2}}{2},\qquad
U_E = \frac{\sigma_0\hat E^{2}}{2},\qquad
P_{\rm th} = \sum_s \hat n_s\,\theta_s\,\hat m_s,\qquad
U_{\rm kin} = \sum_s \hat n_s \hat m_s \langle\gamma_s\rangle .$$

Note $\theta_s\hat m_s = kT_s/(m_0c^2)$, so $P_{\rm th}$ is a sum of $\hat n_s kT_s$ as it should be.
`[output.stats]` writes `B^2`, `E^2` and `T00` in exactly these units, which is what makes the budget
check possible: field energy lost, $\sigma_0\Delta(\hat B^2)/2$, must reappear as $\Delta T^{00}$ plus
$\sigma_0\hat E^2/2$.

**[derived]** Alfvén speed, from the **total** magnetisation:

$$\frac{v_A}{c} = \sqrt{\frac{\sigma_{\rm tot}}{1+\sigma_{\rm tot}}} .$$

Reconnection rate $\mathcal{R} = \lvert\hat E_{\rm rec}\rvert/(\hat B_{\rm up}v_A)$ with $\hat B_{\rm up}$
the **instantaneous** upstream field; using the initial value understates $\mathcal{R}$ once the field
decays.

---


## 5. Common failure modes

| symptom | cause |
|---|---|
| a current layer carries a factor AMP too little current at $t=0$ | the pgen assumed $\hat J = \hat\nabla\times\hat B$ |
| actual $\sigma$ is $\sigma_0^2$ | $\hat B$ set to $\sqrt{\sigma_0}$ instead of adjusting $r_0$ |
| written $\hat J$ below the prediction from $\hat n$ by a fixed fraction | current filter: $\hat J$ filtered, $\hat n$ not |
| per-species $d_s$ or $\sigma_s$ off by $\sqrt2$ or 2 | treating $n_0$ as per species rather than total |
| $\rho_s$ too large by $\sim1/\beta$ for a cold species | using $\gamma$ instead of $u_\perp$ in the Larmor radius |
| energy budget fails to close | $U_B$ written as $\hat B^2/2$ without the $\sigma_0$ |