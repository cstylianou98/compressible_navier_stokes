# Entropy Viscosity - 1D Sod Shock Tube Problem

Idea is to build on the existing FEM code developed to add entropy viscosity to the two step taylor galerkin/RK4 method.

## What about temporal term?
Temporal term is a bit tricky - In this first implementation we will neglect it and approximate the residual to the convection term of the entropy equation. This method detects the shocks well but is a bit over diffusive. It should be fine for first implementation. 

So, computing the entropy residual which will be the advection term of the entropy, it will be then used to calculate the entropy viscosity. This will be done in the first time-step. A zero viscosity is to be expected everywhere but the shock. 

## What have I done?

So firstly I define the entropy at the element as:

$$
\eta_{el} = \frac{\rho_{el}}{(\gamma-1)} log(\frac{p_{el}}{\rho_{el}^{\gamma}})
$$

The entropy flux at the element is taken as:

$$
Q_{el} = u_{el} \eta_{el}
$$

Where:

$$
u_{el} = m/\rho
$$

Thereafter sending these to the gaussian points we get:

$$
\eta_{gp} = N \cdot \eta_{el}
$$

$$
Q_{gp} = N \cdot Q_{el}
$$

$$
\nabla Q_{gp} = N_x \cdot Q_{el}
$$


With this, we can calculate a viscosity given by:

$$
\nu_e = \frac{h^2 * \nabla Q}{|\max(\eta)-\min(\eta)|}    
$$

The entropy plots at t = 0.2s are as follows:

<br>
<div>
    <img src="./RK4_TG2_two_step_EV/RK4_TG2_two_step_EV_entropy_plots_t=0.2.png" alt="Entropy plots at t=0.2s" style="display: block; margin: 0 auto; width: 60%;">
</div>
<br>

Showing indeed that we are capturing the jump as entropy changes, thus a viscosity is being added to dampen it.

Now adding the viscosity term calculated to our FEM formulation:

$$
\int_{\Omega} w U_t dx - \int_{\Omega} w_x F(U) dx - \int_{\Omega} w_x F_{visc} (U) dx = 0
$$

Discretizing leads to:

$$
\sum \int N_A N_B U_t dx - \sum \int \frac{\partial N_B}{\partial x} F_{gp} dx - \sum \int \frac{\partial N_B}{\partial x} F^{visc}_{gp} dx = 0
$$

Since the viscosity is built after the MASS and FLUX matrices are built, another loop is created where the $$F^{visc}_{gp}$$ is calculated as follows: 

<br>
<div>
    <img src="f_visc_eq_gp.png" alt="F_visc eq gp" style="display: block; margin: 0 auto; width: 60%;">
</div>
<br>



Where specifically each term again is taken from the current U and viscosity and sent to the Gaussian points:



$$
\nu_{el} = \frac {\mu_{el}}{\rho_{el}}
$$

$$
\mu_{gp} = N \cdot \mu_{el}
$$

$$
\nu_{gp} = N \cdot \nu_{el}
$$

$$
\nabla \rho_{gp} = N_x \cdot \rho_{el}
$$

$$
\nabla u_{gp} = N_x \cdot u_{el}
$$

$$
u_{gp} = N \cdot u_{el}
$$

$$
\kappa_{el} = \frac{\mu_{el}}{(\gamma - 1)}
$$

$$
\kappa_{gp} = N \cdot \kappa_{el}
$$

$$
T_{el} = \frac {p_{el}}{\rho_{el}}
$$

$$
T_{gp} = N \cdot T_{el} 
$$

Thereafter, now that the $$F_{visc}$$ is built the system of equations is solved with the RHS taken as:

$$
RHS = F + F_{visc_i}
$$

and the LHS as the mass matrix. 


## LPS - Local Projection Stabilization

This is a different stabilization method that has some over-undershoots and therefore can be combined quite nicely with the Entropy Viscosity method.

Formulating the LPS stabilized term:

$$
s_h^{LPS,e} = v_h^{LPS,e} \int_{K^e} \nabla w_h \cdot(\nabla U_h - g_h) dx
$$

**Physically**: This term represents the contribution to stabilization that aims to reduce numerical oscillations by adjusting the gradient of the solution $U_h$. The term $g_h$ can be seen as a "projection" of the gradient $\nabla U_h$, designed to filter out higher-frequency components that may cause instability or oscillations.

Where:

$$
v^{LPS,e} = \frac{\omega h^e ||f'(u_h)||_{L^\infty(K^e)}}{2p}
$$


With $\omega = 1$ (default) $p = 1$ (degree of FEM polynomial). 

In essence the $v^{LPS,e}$ described above is none other than just the upwind viscosity in our 1D code:

$$
v^{UPW} = \frac{h^e \max (|u_{el}| + c_{el})}{2}
$$

**Physically**: This represents the strength of the stabilization. $||f'(u_h)||$ measures how fast the solution is changing. The bigger the element size $h_e$ the greater the stabilization added. Basically controls how much stabilization is applied based on how sharp variations are.


Following this I will outline the steps in which I took to apply it in my code:

- At first, I solve the system of equations $M g = rhs$. Where $M$ is the consistent mass matrix and $rhs$ is built as follows: 
$$rhs = \sum \int N_x U_{gp}$$
- **NOTE**: The system of equations being solved essentially solve for the projected gradient $g_h$ which is a smoother represenation of the actual gradient $U_h$. This filters out sharp oscillations and introduces smoothness.  
- Following this the upwind viscosity is calculated at the element, $v^{UPW}$:

- Retrieve the value of g at the current element by taking g calculated before, passing it to our function and the calculating g at the gaussian point as:

$$ g_{gp} = N \cdot g_{el}$$

- Calculate $U_x$ at the gaussian points by:

$$ U_{gpx} = N_x \cdot U_{el}$$

- Form the LPS matrix that is to be added to the FEM formulation:

$$ F_{LPS} = - \sum \int N_x * v_{UPW,e} * (U_{gpx} - g_{gp})dx $$


The overall system of equations is complete and solved:

$$
M U_t = F + F_{visc} + F_{LPS}
$$


**OVERALL**: The LPS method stabilizes the solution by filtering out oscillations in the gradient. It compares the true gradient $U_x$ with a smoothed gradient $g_h$. The difference between these two is controlled by the stabilization parameter $v_{LPS,e}$ which ensures that overly sharp changes that result in numerical issues are penalized and smoothed out.


## Evolution of numerical method used to solve the shock tube problem:

I first started off applying the standard galerkin for space and the RK4 time discretization scheme on the shock tube problem. All results shown are at a time = 0.2s. As you can see significant instabilities can be seen:

<br>
<div>
    <img src="./RK4_standard_galerkin/RK4_galerkin_t_end=0.2_shock.png" alt="RK4 standard galerkin plots at t=0.2s" style="display: block; margin: 0 auto; width: 60%;">
</div>
<br>

Thereafter a one-step Taylor Galerkin of order 2 was employed, which indeed didn't work great:

<br>
<div>
    <img src="./TG2_one_step/TG2_one_step_t_end=0.2_shock.png" alt="TG2 one-step plots at t=0.2s" style="display: block; margin: 0 auto; width: 60%;">
</div>
<br>

Following this a two-step Taylor Galerkin of order 2 was employed, significantly improving the stability, however not following the shocks at some points.

<br>
<div>
    <img src="./TG2_two_step/TG2_two_step_t_end=0.2_shock.png" alt="TG2 two-step t=0.2s" style="display: block; margin: 0 auto; width: 60%;">
</div>
<br>

The next idea was to combine two-step Taylor Galerkin order 2 with a RK4 time stepping which improved shock capture yet still had some instabilities, as you can see at some points:


<br>
<div>
    <img src="./RK4_TG2_two_step/RK4_TG2_two_step_t_end=0.2_shock.png" alt="RK4 TG2 two-step t=0.2s" style="display: block; margin: 0 auto; width: 60%;">
</div>
<br>

Following the next idea was to combine the RK4 TG2 two-step with Entropy viscosity! A greatly improved shock capture (tunable constant c_e = 0.4):

<br>
<div>
    <img src="./RK4_TG2_two_step_EV/RK4_TG2_two_step_EV_t_end=0.2_c_e=0.4_shock.png" alt="RK4 TG2 two-step t=0.2s" style="display: block; margin: 0 auto; width: 60%;">
</div>
<br>


Thereafter, leaving the TG2 two step method behind, the next idea is to combine the Entropy Viscosity with the LPS projection method. With a tunable constant we get the following promising results:

<br>
<div>
    <img src="./RK4_EV_LPS/RK4_EV_LPS_t_end=0.2_c_e=0.4_shock.png" alt="RK4 TG2 two-step t=0.2s" style="display: block; margin: 0 auto; width: 60%;">
</div>
<br>