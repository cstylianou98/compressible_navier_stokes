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

The entropy plots at t = 0.01 are as follows:

<br>
<div>
    <img src="./RK4_TG2_two_step_EV/RK4_TG2_two_step_EV_entropy_plots_t=0.01.png" alt="Viscosity Graph at Timestep 1" style="display: block; margin: 0 auto; width: 60%;">
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

$$ F^{visc}_{gp} = \left [\begin{array}{cc} \nu_{gp} \nabla \rho_{gp}\\ \mu_{gp} \nabla u_{gp} \\ \mu_{gp} \nabla u_{gp} \cdot u_{gp} - \kappa_{gp} \nabla T_{gp} \end{array}\right] $$


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

and the LHS as the mass matrix
