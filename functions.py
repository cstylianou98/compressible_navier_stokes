import numpy as np
import matplotlib.pyplot as plt 
import math

from scipy.optimize import newton


from matplotlib.animation import FuncAnimation

def U_init_shock_tube(xnode, numnp):
    '''Initial condition of Shock Wave Problem
    (input) xnode arr: Array with x values stored 
    (input) numnp int: Number of nodes

    (output) U tuple: U tuple stores initial conditions for each of the three variables investigated.
    '''

    U = (np.zeros(numnp), np.zeros(numnp), np.zeros(numnp))

    for i in range(numnp):
        if xnode[i] < 0.5:
            U[0][i] = 1.0
            U[1][i] = 0
            U[2][i] = 2.5
        elif xnode[i] >= 0.5:
            U[0][i] = 0.125
            U[1][i] = 0
            U[2][i] = 0.25
    return U


def U_init_explosion(xnode, numnp, gamma=1.4):
    '''
    Initial condition for a 1D explosion problem with high pressure in the middle.
    (input) xnode arr: Array with x values stored 
    (input) numnp int: Number of nodes
    (input) gamma float: Ratio of specific heats

    (output) U tuple: U tuple stores initial conditions for each of the three variables investigated.
    '''
    
    U = (np.zeros(numnp), np.zeros(numnp), np.zeros(numnp))  # rho, momentum, energy

    for i in range(numnp):
        if 0.6 <= xnode[i] <= 1.4:  # Middle region with high pressure
            U[0][i] = 1.0       # rho = 1.0
            U[1][i] = 0.0       # momentum = rho * u = 1.0 * 0 = 0
            U[2][i] = 1.0 / (gamma - 1)  # energy, using p=1.0
        else:                     # Outer regions with low pressure
            U[0][i] = 0.125     # rho = 0.125
            U[1][i] = 0.0       # momentum = rho * u = 0.125 * 0 = 0
            U[2][i] = 0.1 / (gamma - 1)  # energy, using p=0.1

    return U

def phi_init(xnode, numnp):
    '''
    Initial condition of scalar transport term
    (input) xnode arr: Array with x values stored
    (input) numnp int: Number of nodes

    (output) phi array: Phi array stores initial conditions for the variable to be transportede
    '''

    phi = np.zeros(numnp)

    for i in range(numnp):
        if 0<= xnode[i] <= 0.5:
            phi[i] = 25
        else:
            phi[i] = 12.5

    return phi



def calc_p (gamma, rho_E, m, rho):
    '''
    (input) gamma float: Ratio of Cv/Cp (7/5 for ideal gas)
    (input) rho_E float: Value of density*Energy 
    (input) m float: Value of momentum 
    (input) rho float: Value of density 

    (output) p float: Return output of calculated pressure
    ''' 

    p = (gamma - 1) * (rho_E - (m**2)/(2*rho))
    return p

def assemble_standard_galerkin(U_current, numel, xnode, N_mef, Nxi_mef, wpg, gamma):
    '''
    (input) numel int: Number of elements
    (input) xnode arr: Array with x values 
    (input) wpg array: Array with weights 
    (input) N_mef arr: Array with the shape function 

    (output) M: Return output with assembled M matrix
    ''' 
    numnp = numel + 1
    M, F, _, _ = initialize_matrices(numnp)

    for i in range(numel):
        h = xnode[i + 1] - xnode[i]
        weight = wpg * h / 2
        isp = [i, i + 1]  # Global number of the nodes of the current element

        # Get value of each variable at current element  
        rho_el, m_el, rho_E_el =  U_current[0][isp], U_current[1][isp], U_current[2][isp] 
        p_el = calc_p(gamma, rho_E_el, m_el, rho_el)

        ngaus = wpg.shape[0]
        for ig in range(ngaus):
            N = N_mef[ig, :]
            Nx = Nxi_mef[ig, :] * 2 / h
            w_ig = weight[ig]

            rho_gp, m_gp, rho_E_gp = gaussian_values(N, rho_el, m_el, rho_E_el)
            p_gp = np.dot(N, p_el)

            F_rho_gp = m_gp
            F_m_gp = m_gp**2/rho_gp + p_gp
            F_rho_E_gp = (m_gp * (rho_E_gp + p_gp)/ rho_gp)


            M[0][np.ix_(isp, isp)] += w_ig * (np.outer(N, N))
            M[1][np.ix_(isp, isp)] += w_ig * (np.outer(N, N))
            M[2][np.ix_(isp, isp)] += w_ig * (np.outer(N, N))

            F[0][isp] += w_ig * (Nx * F_rho_gp)
            F[1][isp] += w_ig * (Nx * F_m_gp)
            F[2][isp] += w_ig * (Nx * F_rho_E_gp)

    M[0][0,0] = 1
    M[1][0,0] = 1
    M[2][0,0] = 1

    M[0][-1, -1] = 1
    M[1][-1, -1] = 1
    M[2][-1, -1] = 1   

    return M, F

def apply_boundary_conditions(U_temp, numnp, initial_problem_choice):
    '''
    (input) U tuple: List of solution arrays at specific timestep
    (input) numnp int: Number of nodes

    (output) U_temp tuple: U_temp list with applied boundary conditions  
    '''
    if initial_problem_choice == 1:        
            U_temp[0][0] = 1.0  # Homogeneous inflow boundary condition at the first node of each variable
            U_temp[1][0] = 0.0
            U_temp[2][0] = 2.5

            U_temp[0][numnp-1] = 0.125  # Homogeneous outflow boundary condition at the last node of each variable
            U_temp[1][numnp-1] = 0
            U_temp[2][numnp-1] = 0.25

    return U_temp

def compute_jacobian (rho, m, rho_E, gamma):
    '''
    (input) rho float: Value of density 
    (input) m float: Value of momentum 
    (input) rho_E float: Value of density*Energy 
    (input) gamma float: Ratio of Cv/Cp (7/5 for ideal gas)

    (output) A arr: Array of Jacobian matrix
    ''' 

    A = np.array([
                 [0, 1, 0],
                 [(gamma-3)/2 * m**2/rho**2, (3-gamma)*(m/rho), (gamma-1)],
                 [-gamma * ((m * rho_E)/rho**2) + (gamma-1)*(m**3/rho**3), gamma*(rho_E / rho) - 1.5*(gamma-1)* (m**2/rho**2), gamma * m / rho]
                 ] 
                 )
    return A

def assemble_TG_one_step(U_current, numel, xnode, N_mef, Nxi_mef, wpg, gamma, dt):
    '''
    (input) U_current tuple: current solution tuple
    (input) numel int: number of elements
    (input) xnode arr: Array with x values stored
    (input) N_mef arr: Array with the shape function 
    (input) Nxi_mef arr: Array with shape function derivatives
    (input) wpg arr: Array with weights 
    (input) gamma float: Ratio of Cv/Cp (7/5 for ideal gas)
    (input) dt float: timestep


    (output) M: Mass matrix tuple returned by function 
    '''
    numnp = numel + 1

    M, F, _, _ = initialize_matrices(numnp)

    K_rho = np.zeros((numnp, numnp))
    K_m = np.zeros((numnp, numnp))
    K_rho_E = np.zeros((numnp, numnp))
    K = (K_rho, K_m, K_rho_E) # Tuple of stifness matrices for rho, m, rho_E



    for i in range(numel):
        h = xnode[i + 1] - xnode[i]
        weight = wpg * h / 2
        isp = [i, i + 1]  # Global number of the nodes of the current element

        # Get value of each variable at current element  
        rho_el, m_el, rho_E_el =  U_current[0][isp], U_current[1][isp], U_current[2][isp] 
        p_el, _, _, F_rho_el, F_m_el, F_rho_E_el = calculate_element_properties(gamma, rho_el, m_el, rho_E_el)

        ngaus = wpg.shape[0]
        for ig in range(ngaus):
            N = N_mef[ig, :]
            Nx = Nxi_mef[ig, :] * 2 / h
            w_ig = weight[ig]

            # # Calculate values of rho, m, rho_E and p and their derivatives at the gaussian points.
            rho_gp, m_gp, rho_E_gp = gaussian_values(N, rho_el, m_el, rho_E_el)
            p_gp = np.dot(N, p_el)
            u_gp = m_gp/rho_gp

            # Used to check if velocity is positive or negative. For compression regions dv/dx < 0 we take the linear approximation.
            if u_gp >= 0:
                F_rho_gp = m_gp
                F_m_gp = m_gp**2/rho_gp + p_gp
                F_rho_E_gp = (m_gp * (rho_E_gp + p_gp)/ rho_gp)

            elif u_gp <0:
                F_rho_gp, F_m_gp, F_rho_E_gp = gaussian_values(N, F_rho_el, F_m_el, F_rho_E_el)


            M[0][np.ix_(isp, isp)] += w_ig * np.outer(N, N)
            M[1][np.ix_(isp, isp)] += w_ig * np.outer(N, N)
            M[2][np.ix_(isp, isp)] += w_ig * np.outer(N, N)

            F[0][isp] += w_ig * ( F_rho_gp * Nx )
            F[1][isp] += w_ig * ( F_m_gp * Nx )
            F[2][isp] += w_ig * ( F_rho_E_gp * Nx )


            A = compute_jacobian(rho_gp, m_gp, rho_E_gp, gamma)
            A_squared = A**2

            K_rho[np.ix_(isp, isp)] +=  - 0.5 * dt * w_ig * (A_squared[0, 0] *(np.outer(Nx, Nx)) + 
                                        A_squared[1, 0] *(np.outer(Nx, Nx)) +  
                                        A_squared[2, 0] *(np.outer(Nx, Nx)))
            
            K_m[np.ix_(isp, isp)] +=  - 0.5 * dt * w_ig * (A_squared[0, 1] * (np.outer(Nx, Nx)) + 
                                      A_squared[1, 1] *(np.outer(Nx, Nx)) +  
                                      A_squared[2, 1] *(np.outer(Nx, Nx)))
            
            K_rho_E[np.ix_(isp, isp)] += - 0.5 * dt * w_ig * (A_squared[0, 2] *(np.outer(Nx, Nx)) + 
                                          A_squared[1, 2] *(np.outer(Nx, Nx)) +  
                                          A_squared[2, 2] *(np.outer(Nx, Nx)))
            
    M[0][0,0] = 1
    M[1][0,0] = 1
    M[2][0,0] = 1

    M[0][-1, -1] = 1
    M[1][-1, -1] = 1
    M[2][-1, -1] = 1  
            
    return M, F, K

def assemble_TG_two_step(U_current, numel, xnode, N_mef, Nxi_mef, wpg, gamma, dt):
    '''
    (input) U_current tuple: current solution tuple
    (input) numel int: number of elements
    (input) xnode arr: Array with x values stored
    (input) N_mef arr: Array with the shape function 
    (input) Nxi_mef arr: Array with shape function derivatives
    (input) wpg arr: Array with weights 
    (input) gamma float: Ratio of Cv/Cp (7/5 for ideal gas)
    (input) dt float: timestep

    (output) M: Mass matrix tuple returned by function 
    (output) F: Flux matrix tuple returned by function
    '''
    numnp = numel + 1

    M, F, _, _ = initialize_matrices(numnp)

    for i in range(numel):
        h = xnode[i + 1] - xnode[i]
        weight = wpg * h / 2
        isp = [i, i + 1]  # Global number of the nodes of the current element

        # Get value of each variable at current element  
        rho_el, m_el, rho_E_el =  U_current[0][isp], U_current[1][isp], U_current[2][isp]  

        p_el, _, _, F_rho_el, F_m_el, F_rho_E_el = calculate_element_properties(gamma, rho_el, m_el, rho_E_el)

        ngaus = wpg.shape[0]
        for ig in range(ngaus):
            N = N_mef[ig, :]
            Nx = Nxi_mef[ig, :] * 2 / h
            w_ig = weight[ig]

            # Intergration points (Gaussian):
            rho_gp, m_gp, rho_E_gp = gaussian_values(N, rho_el, m_el, rho_E_el)
            F_rho_gpx, F_m_gpx, F_rho_E_gpx = gaussian_values(Nx, F_rho_el, F_m_el, F_rho_E_el)

            rho_inter, m_inter, rho_E_inter, p_inter = inter_gaussian_values(gamma, rho_gp, m_gp, rho_E_gp, dt, F_rho_gpx, F_m_gpx, F_rho_E_gpx)
            F_rho_inter, F_m_inter, F_rho_E_inter = flux_inter_gaussian_values(rho_inter, m_inter, rho_E_inter, p_inter)

            M[0][np.ix_(isp, isp)] += w_ig * np.outer(N, N)
            M[1][np.ix_(isp, isp)] += w_ig * np.outer(N, N)
            M[2][np.ix_(isp, isp)] += w_ig * np.outer(N, N)

            F[0][isp] += w_ig * (Nx * F_rho_inter)
            F[1][isp] += w_ig * (Nx * F_m_inter)
            F[2][isp] += w_ig * (Nx * F_rho_E_inter)

    M[0][0,0] = 1
    M[1][0,0] = 1
    M[2][0,0] = 1

    M[0][-1, -1] = 1
    M[1][-1, -1] = 1
    M[2][-1, -1] = 1  

    return M, F

def assemble_TG_two_step_EV(U_current, numel, xnode, N_mef, Nxi_mef, wpg, gamma, dt, c_e):
    '''
    (input) U_current tuple: current solution tuple
    (input) numel int: number of elements
    (input) xnode arr: Array with x values stored
    (input) N_mef arr: Array with the shape function 
    (input) Nxi_mef arr: Array with shape function derivatives
    (input) wpg arr: Array with weights 
    (input) gamma float: Ratio of Cv/Cp (7/5 for ideal gas)
    (input) dt float: timestep

    (output) M: Mass matrix tuple returned by function 
    '''
    numnp = numel + 1

    M, F, F_visc, _ = initialize_matrices(numnp)

    entropy, entropy_flux, entropy_res, viscosity_e = np.zeros(numnp), np.zeros(numnp), np.zeros(numnp), np.zeros(numnp) 

    for i in range(numel):
        h = xnode[i + 1] - xnode[i]
        weight = wpg * h / 2
        isp = [i, i + 1]  # Global number of the nodes of the current element

        # Get value of each variable at current element  
        rho_el, m_el, rho_E_el =  U_current[0][isp], U_current[1][isp], U_current[2][isp] 

        p_el, u_el, _, F_rho_el, F_m_el, F_rho_E_el = calculate_element_properties(gamma, rho_el, m_el, rho_E_el)

        entropy_el = rho_el/(gamma-1) * np.log(p_el/rho_el**gamma)
        entropy_flux_el = entropy_el * u_el

        ngaus = wpg.shape[0]
        for ig in range(ngaus):
            N = N_mef[ig, :]
            Nx = Nxi_mef[ig, :] * 2 / h
            w_ig = weight[ig]

            # Intermediate value at integration(Gaussian) point:
            rho_gp, m_gp, rho_E_gp = gaussian_values(N, rho_el, m_el, rho_E_el)
            F_rho_gpx, F_m_gpx, F_rho_E_gpx = gaussian_values(Nx, F_rho_el, F_m_el, F_rho_E_el)

            rho_inter, m_inter, rho_E_inter, p_inter = inter_gaussian_values(gamma, rho_gp, m_gp, rho_E_gp, dt, F_rho_gpx, F_m_gpx, F_rho_E_gpx)
            F_rho_inter, F_m_inter, F_rho_E_inter = flux_inter_gaussian_values(rho_inter, m_inter, rho_E_inter, p_inter)

            entropy_gp = np.dot(N, entropy_el)
            entropy_flux_gp = np.dot(N, entropy_flux_el)
            entropy_flux_gpx = np.dot(Nx, entropy_flux_el)

            M[0][np.ix_(isp, isp)] += w_ig * np.outer(N, N)
            M[1][np.ix_(isp, isp)] += w_ig * np.outer(N, N)
            M[2][np.ix_(isp, isp)] += w_ig * np.outer(N, N)

            F[0][isp] += w_ig * (Nx * F_rho_inter)
            F[1][isp] += w_ig * (Nx * F_m_inter)
            F[2][isp] += w_ig * (Nx * F_rho_E_inter)

            entropy[isp] += w_ig * entropy_gp 
            entropy_flux[isp] += w_ig * entropy_flux_gp
            entropy_res[isp] += w_ig * entropy_flux_gpx
            
    M[0][0,0] = 1
    M[1][0,0] = 1
    M[2][0,0] = 1

    M[0][-1, -1] = 1
    M[1][-1, -1] = 1
    M[2][-1, -1] = 1  

    viscosity_e = c_e * np.abs((h**2 * entropy_res)/np.abs((np.max(entropy)-np.min(entropy))))

    for i in range(numel):
        h = xnode[i + 1] - xnode[i]
        weight = wpg * h / 2
        isp = [i, i + 1]  # Global number of the nodes of the current element

        # Get value of each variable at current element  
        rho_el, m_el, rho_E_el =  U_current[0][isp], U_current[1][isp], U_current[2][isp]

        p_el = calc_p(gamma, rho_E_el, m_el, rho_el)
        u_el = m_el/rho_el

        c_e = np.sqrt(gamma * (p_el/rho_el)) 

        viscosity_el_1 = 0.5 * h * ( np.max(np.abs(u_el) + c_e))
        viscosity_el_2 =  viscosity_e[isp]
        viscosity_el = np.minimum(viscosity_el_1, viscosity_el_2) # because at a shock viscosity_el_2 is huge so viscosity_el_1 must be taken 
        kinematic_visc_el = viscosity_el/rho_el

        kappa_el = viscosity_el/(gamma - 1)
        temp_el = p_el/rho_el

        for ig in range(ngaus):
            N = N_mef[ig, :]
            Nx = Nxi_mef[ig, :] * 2 / h
            w_ig = weight[ig]

            kinematic_visc_gp = np.dot(N, kinematic_visc_el)
            rho_gpx = np.dot(Nx, rho_el)

            viscosity_gp = np.dot(N, viscosity_el)
            kinematic_visc_gp = np.dot(N, kinematic_visc_el)
            u_gpx = np.dot(Nx, u_el)

            u_gp = np.dot(N, u_el)
            kappa_gp = np.dot(N, kappa_el)
            temp_gpx = np.dot(Nx, temp_el)

            F_visc[0][isp] +=   - w_ig * (Nx * kinematic_visc_gp * rho_gpx)
            F_visc[1][isp] +=  - w_ig * (Nx * viscosity_gp * u_gpx)
            F_visc[2][isp] += - w_ig * Nx * ((viscosity_gp * u_gpx * u_gp + kappa_gp * temp_gpx))

    return M, F, entropy, entropy_flux, entropy_res, viscosity_e, F_visc

def assemble_g_rhs_system(U_current, numel, xnode, N_mef, Nxi_mef, wpg):
    '''
    (input) U_current tuple: current solution tuple
    (input) numel int: number of elements
    (input) xnode arr: Array with x values stored
    (input) N_mef arr: Array with the shape function 
    (input) Nxi_mef arr: Array with shape function derivatives
    (input) wpg arr: Array with weights 
    (input) gamma float: Ratio of Cv/Cp (7/5 for ideal gas)
    (input) dt float: timestep

    (output) M: Mass matrix tuple returned by function 
    '''
    numnp = numel + 1

    M_g = np.zeros((numnp, numnp))

    rhs_rho = np.zeros(numnp)
    rhs_m = np.zeros(numnp)
    rhs_rho_E = np.zeros(numnp)
    rhs = (rhs_rho, rhs_m, rhs_rho_E)


    for i in range(numel):
        h = xnode[i + 1] - xnode[i]
        weight = wpg * h / 2
        isp = [i, i + 1]  # Global number of the nodes of the current element

        # Get value of each variable at current element  
        rho_el =  U_current[0][isp]
        m_el =  U_current[1][isp]
        rho_E_el = U_current[2][isp] 

        ngaus = wpg.shape[0]


        for ig in range(ngaus):
            N = N_mef[ig, :]
            Nx = Nxi_mef[ig, :] * 2 / h
            w_ig = weight[ig]

            rho_gpx = np.dot(Nx, rho_el)
            m_gpx = np.dot(Nx, m_el)
            rho_E_gpx = np.dot(Nx, rho_E_el)
        
            M_g[np.ix_(isp, isp)] += w_ig * np.outer(N, N)

            rhs_rho[isp] += w_ig * (N * rho_gpx)
            rhs_m[isp] += w_ig * (N  * m_gpx)
            rhs_rho_E[isp] += w_ig * (N * rho_E_gpx)

    M_g[0,0] = 1
    M_g[-1, -1] = 1

    return M_g, rhs

def assemble_EV_LPS(U_current, numel, xnode, N_mef, Nxi_mef, wpg, gamma, dt, c_e, g_tuple):
    '''
    (input) U_current tuple: current solution tuple
    (input) numel int: number of elements
    (input) xnode arr: Array with x values stored
    (input) N_mef arr: Array with the shape function 
    (input) Nxi_mef arr: Array with shape function derivatives
    (input) wpg arr: Array with weights 
    (input) gamma float: Ratio of Cv/Cp (7/5 for ideal gas)
    (input) dt float: timestep

    (output) M: Mass matrix tuple returned by function 
    '''
    numnp = numel + 1

    M, F, F_visc, F_lps = initialize_matrices(numnp)

    entropy, entropy_flux, entropy_res, viscosity_e = np.zeros(numnp), np.zeros(numnp), np.zeros(numnp), np.zeros(numnp)

    for i in range(numel):
        h = xnode[i + 1] - xnode[i]
        weight = wpg * h / 2
        isp = [i, i + 1]  # Global number of the nodes of the current element

        # Get value of each variable at current element  
        rho_el, m_el, rho_E_el =  U_current[0][isp], U_current[1][isp], U_current[2][isp]
        g_rho_el, g_m_el, g_rho_E_el = g_tuple[0][isp], g_tuple[1][isp], g_tuple[2][isp]
  
        # Compute pressure, flux and speed from element variable values
        p_el, u_el, c_el, F_rho_el, F_m_el, F_rho_E_el = calculate_element_properties(gamma, rho_el, m_el, rho_E_el)

        entropy_el = rho_el/(gamma-1) * np.log(p_el/rho_el**gamma)
        entropy_flux_el = entropy_el * u_el

        v_upw = (h*np.max(np.abs(u_el) + c_el))/2

        ngaus = wpg.shape[0]
        for ig in range(ngaus):
            N = N_mef[ig, :]
            Nx = Nxi_mef[ig, :] * 2 / h
            w_ig = weight[ig]

            # Intergration points (Gaussian):
            rho_gpx, m_gpx, rho_E_gpx = gaussian_values(Nx, rho_el, m_el, rho_E_el)
            F_rho_gp, F_m_gp, F_rho_E_gp = gaussian_values(N, F_rho_el, F_m_el, F_rho_E_el)            
            g_rho_gp, g_m_gp, g_rho_E_gp = gaussian_values(N, g_rho_el, g_m_el, g_rho_E_el)


            entropy_gp = np.dot(N, entropy_el)
            entropy_flux_gp = np.dot(N, entropy_flux_el)
            entropy_flux_gpx = np.dot(Nx, entropy_flux_el)

            entropy[isp] += w_ig * entropy_gp 
            entropy_flux[isp] += w_ig * entropy_flux_gp
            entropy_res[isp] += w_ig * entropy_flux_gpx

            M[0][np.ix_(isp, isp)] += w_ig * np.outer(N, N)
            M[1][np.ix_(isp, isp)] += w_ig * np.outer(N, N)
            M[2][np.ix_(isp, isp)] += w_ig * np.outer(N, N)

            F[0][isp] += w_ig * (Nx * F_rho_gp)
            F[1][isp] += w_ig * (Nx * F_m_gp)
            F[2][isp] += w_ig * (Nx * F_rho_E_gp)


            F_lps[0][isp] += - w_ig * v_upw * Nx * (rho_gpx - g_rho_gp)
            F_lps[1][isp] += - w_ig  * v_upw * Nx * (m_gpx - g_m_gp)
            F_lps[2][isp] += - w_ig * v_upw * Nx * (rho_E_gpx - g_rho_E_gp)

           
    M[0][0,0] = 1
    M[1][0,0] = 1
    M[2][0,0] = 1

    M[0][-1, -1] = 1
    M[1][-1, -1] = 1
    M[2][-1, -1] = 1  

    viscosity_e = c_e * np.abs((h**2 * entropy_res)/np.abs((np.max(entropy)-np.min(entropy))))

    for i in range(numel):
        h = xnode[i + 1] - xnode[i]
        weight = wpg * h / 2
        isp = [i, i + 1]  # Global number of the nodes of the current element

        # Get value of each variable at current element  
        rho_el, m_el, rho_E_el =  U_current[0][isp], U_current[1][isp], U_current[2][isp]
        p_el = calc_p(gamma, rho_E_el, m_el, rho_el)
        u_el = m_el/rho_el
        c_el = np.sqrt(gamma * (p_el/rho_el)) 

        viscosity_el_1 = 0.5 * h * ( np.max(np.abs(u_el) + c_el))
        viscosity_el_2 =  viscosity_e[isp]
        viscosity_el = np.minimum(viscosity_el_1, viscosity_el_2) # because at a shock viscosity_el_2 is huge so viscosity_el_1 must be taken 
        kinematic_visc_el = viscosity_el/rho_el

        kappa_el = viscosity_el/(gamma - 1)
        temp_el = p_el/rho_el

        for ig in range(ngaus):
            N = N_mef[ig, :]
            Nx = Nxi_mef[ig, :] * 2 / h
            w_ig = weight[ig]

            kinematic_visc_gp = np.dot(N, kinematic_visc_el)
            rho_gpx = np.dot(Nx, rho_el)

            viscosity_gp = np.dot(N, viscosity_el)
            kinematic_visc_gp = np.dot(N, kinematic_visc_el)
            u_gpx = np.dot(Nx, u_el)

            u_gp = np.dot(N, u_el)
            kappa_gp = np.dot(N, kappa_el)
            temp_gpx = np.dot(Nx, temp_el)

            F_visc[0][isp] +=   - w_ig * (Nx * kinematic_visc_gp * rho_gpx)
            F_visc[1][isp] +=  - w_ig * (Nx * viscosity_gp * u_gpx)
            F_visc[2][isp] += - w_ig * Nx * ((viscosity_gp * u_gpx * u_gp + kappa_gp * temp_gpx))

    return M, F, F_visc, F_lps

def assemble_transport(U_current, phi_current, numel, xnode, N_mef, Nxi_mef, wpg, gamma):
    '''
    (input) U_current tuple: current solution tuple
    (input) numel int: number of elements
    (input) xnode arr: Array with x values stored
    (input) N_mef arr: Array with the shape function 
    (input) Nxi_mef arr: Array with shape function derivatives
    (input) wpg arr: Array with weights 
    (input) gamma float: Ratio of Cv/Cp (7/5 for ideal gas)
    
    (output) F_phi arr: RHS matrix for solving scalar transport 
    '''
    numnp = numel + 1

    F_phi = np.zeros(numnp)

    for i in range(numel):
        h = xnode[i + 1] - xnode[i]
        weight = wpg * h / 2
        isp = [i, i + 1]  # Global number of the nodes of the current element

        # Get value of each variable at current element  
        rho_el, m_el =  U_current[0][isp], U_current[1][isp]
        phi_el = phi_current[isp]

        u_el = m_el/rho_el

        ngaus = wpg.shape[0]
        for ig in range(ngaus):
            N = N_mef[ig, :]
            Nx = Nxi_mef[ig, :] * 2 / h
            w_ig = weight[ig]

            rho_gp = np.dot(N, rho_el)
            phi_gp = np.dot(N, phi_el)
            u_gp = np.dot(N, u_el)

            F_phi[isp] += Nx * u_gp * rho_gp * phi_gp

    return F_phi            

def initialize_matrices(numnp):
    M_rho, M_m, M_rho_E = np.zeros((numnp, numnp)), np.zeros((numnp, numnp)), np.zeros((numnp, numnp))
    F_rho, F_m, F_rho_E = np.zeros(numnp), np.zeros(numnp), np.zeros(numnp)
    F_visc_rho, F_visc_m, F_visc_rho_E = np.zeros(numnp), np.zeros(numnp), np.zeros(numnp) 
    F_lps_rho, F_lps_m, F_lps_rho_E = np.zeros(numnp), np.zeros(numnp), np.zeros(numnp)
    return (M_rho, M_m, M_rho_E), (F_rho, F_m, F_rho_E), (F_visc_rho, F_visc_m, F_visc_rho_E), (F_lps_rho, F_lps_m, F_lps_rho_E)

def calculate_element_properties(gamma, rho_el, m_el, rho_E_el):
    p_el = calc_p(gamma, rho_E_el, m_el, rho_el)
    u_el = m_el/rho_el
    c_el = np.sqrt(gamma * (p_el/rho_el))

    F_rho_el = m_el
    F_m_el = m_el**2/rho_el + p_el
    F_rho_E_el = (m_el * (rho_E_el + p_el) / rho_el)

    return p_el, u_el, c_el, F_rho_el, F_m_el, F_rho_E_el

def gaussian_values(shape_func, rho, m, rho_E):
    '''
    Same function used for both gp and gpx values calculated, therefore defined generic function inputs
    '''
    rho_gaus = np.dot(shape_func, rho)
    m_gaus = np.dot(shape_func, m)
    rho_E_gaus = np.dot(shape_func, rho_E)
    return rho_gaus, m_gaus, rho_E_gaus

def inter_gaussian_values(gamma, rho_gp, m_gp, rho_E_gp,  dt, F_rho_gpx, F_m_gpx, F_rho_E_gpx):
    rho_inter = rho_gp - 0.5 * dt * F_rho_gpx
    m_inter = m_gp - 0.5 * dt * F_m_gpx
    rho_E_inter = rho_E_gp - 0.5 * dt * F_rho_E_gpx
    p_inter = calc_p(gamma, rho_E_inter, m_inter, rho_inter)
    return rho_inter, m_inter, rho_E_inter, p_inter

def flux_inter_gaussian_values(rho_inter, m_inter, rho_E_inter, p_inter):
    F_rho_inter = m_inter
    F_m_inter = m_inter**2/ rho_inter + p_inter
    F_rho_E_inter = (m_inter * (rho_E_inter + p_inter)/ rho_inter)
    return F_rho_inter, F_m_inter, F_rho_E_inter




## ANALYTIC CALCULATION FUNCTIONS
def f(P, pL, pR, cL, cR, gamma):
    a = (gamma-1)*(cR/cL)*(P-1) 
    b = np.sqrt( 2*gamma*(2*gamma + (gamma+1)*(P-1) ) )
    return P - pL/pR*( 1 - a/b )**(2.*gamma/(gamma-1.))

def SodShockAnalytic(config, t_end):

    h = config['xnode'][1]
    Nx = len(config['xnode'])
    v_analytic = np.zeros((3,Nx),dtype='float64')

    # compute speed of sound
    cL = np.sqrt(config['gamma']*config['U_init_analytical_left'][2]/config['U_init_analytical_left'][0]) 
    cR = np.sqrt(config['gamma']*config['U_init_analytical_right'][2]/config['U_init_analytical_right'][0])
    # compute P
    P = newton(f, 0.5, args=(config['U_init_analytical_left'][2], config['U_init_analytical_right'][2], cL, cR, config['gamma']), tol=1e-12)

    # compute region positions right to left
    # region R
    c_shock = config['U_init_analytical_right'][1] + cR*np.sqrt( (config['gamma']-1+P*(config['gamma']+1)) / (2*config['gamma']) )
    x_shock = config['x0_analytical'] + int(np.floor(c_shock*t_end/h))
    v_analytic[0,x_shock-1:] = config['U_init_analytical_right'][0]
    v_analytic[1,x_shock-1:] = config['U_init_analytical_right'][1]
    v_analytic[2,x_shock-1:] = config['U_init_analytical_right'][2]
    
    # region 2
    alpha = (config['gamma']+1)/(config['gamma']-1)
    c_contact = config['U_init_analytical_left'][1] + 2*cL/(config['gamma']-1)*( 1-(P*config['U_init_analytical_right'][2]/config['U_init_analytical_left'][2])**((config['gamma']-1.)/2/config['gamma']) )
    x_contact = config['x0_analytical'] + int(np.floor(c_contact*t_end/h))
    v_analytic[0,x_contact:x_shock-1] = (1 + alpha*P)/(alpha+P)*config['U_init_analytical_right'][0]
    v_analytic[1,x_contact:x_shock-1] = c_contact
    v_analytic[2,x_contact:x_shock-1] = P*config['U_init_analytical_right'][2]
    
    # region 3
    r3 = config['U_init_analytical_left'][0]*(P*config['U_init_analytical_right'][2]/config['U_init_analytical_left'][2])**(1/config['gamma'])
    p3 = P*config['U_init_analytical_right'][2]
    c_fanright = c_contact - np.sqrt(config['gamma']*p3/r3)
    x_fanright = config['x0_analytical'] + int(np.ceil(c_fanright*t_end/h))
    v_analytic[0,x_fanright:x_contact] = r3
    v_analytic[1,x_fanright:x_contact] = c_contact
    v_analytic[2,x_fanright:x_contact] = P*config['U_init_analytical_right'][2]
    
    # region 4
    c_fanleft = -cL
    x_fanleft = config['x0_analytical'] + int(np.ceil(c_fanleft*t_end/h))
    u4 = 2 / (config['gamma']+1) * (cL + (config['xnode'][x_fanleft:x_fanright]-config['xnode'][config['x0_analytical']])/t_end )
    v_analytic[0,x_fanleft:x_fanright] = config['U_init_analytical_left'][0]*(1 - (config['gamma']-1)/2.*u4/cL)**(2/(config['gamma']-1));
    v_analytic[1,x_fanleft:x_fanright] = u4
    v_analytic[2,x_fanleft:x_fanright] = config['U_init_analytical_left'][2]*(1 - (config['gamma']-1)/2.*u4/cL)**(2*config['gamma']/(config['gamma']-1));

    # region L
    v_analytic[0,:x_fanleft] = config['U_init_analytical_left'][0]
    v_analytic[1,:x_fanleft] = config['U_init_analytical_left'][1]
    v_analytic[2,:x_fanleft] = config['U_init_analytical_left'][2]

    rho_energy = v_analytic[2]/(config['gamma']-1) + (v_analytic[0] * v_analytic[1]**2)/2

    return v_analytic, rho_energy

## Plotting Functions
def plot_entropy_res(variables_tuple, config):
    # Create a figure with two subplots
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    # Entropy
    ax[0, 0].set_xlabel('x')
    ax[0, 0].set_xlim([0.0, 1.05])
    ax[0, 0].set_xticks([i * 0.1 for i in range(11)])
    ax[0, 0].set_ylabel(r'$\eta$')
    ax[0, 0].set_title(f"Entropy t={config['t_end']}s")
    ax[0, 0].plot(config['xnode'], variables_tuple[5])


    # Entropy Flux
    ax[0, 1].set_xlabel('x')
    ax[0, 1].set_xlim([0.0, 1.05])
    ax[0, 1].set_xticks([i * 0.1 for i in range(11)])
    ax[0, 1].set_ylabel(r'Q')
    ax[0, 1].set_title(f"Entropy Flux t={config['t_end']}s")
    ax[0, 1].plot(config['xnode'], variables_tuple[6])

    # Entropy Residual
    ax[1, 0].set_xlabel('x')
    ax[1, 0].set_xlim([0.0, 1.05])
    ax[1, 0].set_xticks([i * 0.1 for i in range(11)])
    ax[1, 0].set_ylabel(r'$\nabla Q$')
    ax[1, 0].set_title(f"Entropy Residual t={config['t_end']}s")
    ax[1, 0].plot(config['xnode'], variables_tuple[7])

    # Entropy Residual
    ax[1, 1].set_xlabel('x')
    ax[1, 1].set_xlim([0.0, 1.05])
    ax[1, 1].set_xticks([i * 0.1 for i in range(11)])
    ax[1, 1].set_ylabel(r'$\nu_e$')
    ax[1, 1].set_title(f"Viscosity t={config['t_end']}s")
    ax[1, 1].plot(config['xnode'], variables_tuple[8])


    # Save the figure with both plots
    plt.tight_layout()
    plt.savefig(f"./{config['folder_path']}/{config['method_file_name']}_entropy_plots_t={config['t_end']}.png")
    plt.close()

def plot_solution(t_end, variables_tuple , config, analytic, rho_energy_analytic):
    fig, ax = plt.subplots(2,2,figsize=(8,8), layout='constrained')
    if config['stabilization_choice'] == 5 or config['stabilization_choice'] == 6:
        graph_title_c_e = f"c_e = {config['c_e']}"
        file_c_e = f"_c_e={config['c_e']}"
    else:
        graph_title_c_e = ''
        file_c_e = ''

    if config['initial_problem_choice'] == 1:
            # First row
            ax[0, 0].set_title(f"Density - t = {t_end}s {graph_title_c_e}")
            ax[0, 0].plot(config['xnode'], variables_tuple[0][:, config['nstep']], linestyle ="", marker="x")
            ax[0, 0].plot(config['xnode'], analytic[0].T) 
            ax[0, 0].set_ylabel('rho', fontweight='bold')
            ax[0, 0].set_xlabel('x', fontweight='bold')
            ax[0, 0].set_xlim([0.0, 1.05])
            ax[0, 0].set_xticks([i * 0.1 for i in range(11)])
            ax[0, 0].set_ylim([-0.05,1.05])

            ax[0, 1].set_title(f"Velocity - t = {t_end}s {graph_title_c_e}")
            ax[0, 1].plot(config['xnode'], variables_tuple[1][:, config['nstep']], linestyle ="", marker="x")
            ax[0, 1].plot(config['xnode'], analytic[1].T)
            ax[0, 1].set_ylabel('v', fontweight='bold')
            ax[0, 1].set_xlabel('x', fontweight='bold')
            ax[0, 1].set_xlim([0.0, 1.05])
            ax[0, 1].set_xticks([i * 0.1 for i in range(11)])
            ax[0, 1].set_ylim([-0.05,1.05])

            # Second row
            ax[1, 0].set_title(f"Pressure - t = {t_end}s {graph_title_c_e}")
            ax[1, 0].plot(config['xnode'], variables_tuple[2][:, config['nstep']], linestyle ="", marker="x")
            ax[1, 0].plot(config['xnode'], analytic[2].T)
            ax[1, 0].set_ylabel('p', fontweight='bold')
            ax[1, 0].set_xlabel('x', fontweight='bold')
            ax[1, 0].set_xlim([0.0, 1.05])
            ax[1, 0].set_xticks([i * 0.1 for i in range(11)])
            ax[1, 0].set_ylim([-0.05,1.05])

            ax[1, 1].set_title(f"rho_Energy - t = {t_end}s {graph_title_c_e}")
            ax[1, 1].plot(config['xnode'], variables_tuple[3][:, config['nstep']], linestyle ="", marker="x")
            ax[1, 1].plot(config['xnode'], rho_energy_analytic.T)
            ax[1, 1].set_ylabel('rho_E', fontweight='bold')
            ax[1, 1].set_xlabel('x', fontweight='bold')
            ax[1, 1].set_xlim([0.0, 1.05])
            ax[1, 1].set_xticks([i * 0.1 for i in range(11)])
            ax[1, 1].set_ylim([-0.10, 3.05])
            plt.suptitle(f"{config['stabilization_graph_title']} - {config['init_problem_name'][0]}")
            plt.savefig(f"./{config['folder_path']}/{config['method_file_name']}_t_end={t_end}{file_c_e}_{config['init_problem_name'][0]}.png")

    elif config['initial_problem_choice'] == 2:
            # First row
            ax[0, 0].set_title(f"Density - t = {t_end}s {graph_title_c_e}")
            ax[0, 0].plot(config['xnode'], variables_tuple[0][:, config['nstep']], linestyle ="", marker="x")
            ax[0, 0].set_ylabel('rho', fontweight='bold')
            ax[0, 0].set_xlabel('x', fontweight='bold')
            ax[0, 0].set_xlim([1.0, 2.05])
            ax[0, 0].set_xticks([(i * 0.1) + 1 for i in range(11)])
            ax[0, 0].set_ylim([-0.05,1.05])

            ax[0, 1].set_title(f"Velocity - t = {t_end}s {graph_title_c_e}")
            ax[0, 1].plot(config['xnode'], variables_tuple[1][:, config['nstep']], linestyle ="", marker="x")
            ax[0, 1].set_ylabel('v', fontweight='bold')
            ax[0, 1].set_xlabel('x', fontweight='bold')
            ax[0, 1].set_xlim([1.0, 2.05])
            ax[0, 1].set_xticks([(i * 0.1) + 1 for i in range(11)])
            ax[0, 1].set_ylim([-0.05,1.05])

            # Second row
            ax[1, 0].set_title(f"Pressure - t = {t_end}s {graph_title_c_e}")
            ax[1, 0].plot(config['xnode'], variables_tuple[2][:, config['nstep']], linestyle ="", marker="x")
            ax[1, 0].set_ylabel('p', fontweight='bold')
            ax[1, 0].set_xlabel('x', fontweight='bold')
            ax[1, 0].set_xlim([1.0, 2.05])
            ax[1, 0].set_xticks([(i * 0.1) + 1 for i in range(11)])
            ax[1, 0].set_ylim([-0.05,1.05])

            ax[1, 1].set_title(f"Internal energy - t = {t_end}s {graph_title_c_e}")
            ax[1, 1].plot(config['xnode'], variables_tuple[4][:, config['nstep']], linestyle ="", marker="x")
            ax[1, 1].set_ylabel('int_e', fontweight='bold')
            ax[1, 1].set_xlabel('x', fontweight='bold')
            ax[1, 1].set_xlim([1.0, 2.05])
            ax[1, 1].set_xticks([(i * 0.1) + 1 for i in range(11)])
            ax[1, 1].set_ylim([-0.10, 3.25])
            plt.suptitle(f"{config['stabilization_graph_title']} - {config['init_problem_name'][1]}")
            plt.savefig(f"./{config['folder_path']}/{config['method_file_name']}_t_end={t_end}{file_c_e}_{config['init_problem_name'][1]}.png")

    plt.close()

