import numpy as np
from functions import *
import os
import matplotlib.pyplot as plt

from scipy.linalg import solve


def get_valid_choice(prompt, valid_choices):
    choice = int(input(prompt))
    while choice not in valid_choices:
        print(f"\n>>> Invalid choice. Please choose from {valid_choices}.")
        choice = int(input(prompt))
    return choice

def configure_simulation():
    initial_problem_prompt = (
        "\n>>> Please choose your initial condition problem. \n"
        "1. Shock tube initial conditions \n"
        "2. Explosion initial conditions \n"
        "\nType your choice here -----> "
    )
    
    stabilization_choice_prompt = (
        "\n>>> Please choose your numerical scheme. \n"
        "1. RK4 with Standard Galerkin \n"
        "2. Taylor Galerkin (TG2) One-Step \n"
        "3. Taylor Galerkin (TG2) Two-Step \n"
        "4. RK4 with Taylor Galerkin Two-Step \n"
        "5. RK4 with Taylor Galerkin Two-Step and EV stabilization \n"
        "6. RK4 with EV and LPS stabilization \n"
        "\nType your choice here -----> "
    )

    # Get valid initial problem choice
    initial_problem_choice = get_valid_choice(initial_problem_prompt, [1, 2])

    # Get the last timestep
    t_end = float(input("\n>>> Input the last timestep (s) -----> "))

    # Get valid stabilization choice
    stabilization_choice = get_valid_choice(stabilization_choice_prompt, [1, 2, 3, 4, 5, 6])

    return t_end, stabilization_choice, initial_problem_choice


# Setup and initialization
def setup_simulation(t_end, stabilization_choice, initial_problem_choice):
    variables_titles = ['- Density', '- Velocity', '- Pressure', '- Energy']
    y_axis_labels = ['rho', 'v', 'p', 'E']
    stabilization_graph_titles = ['1D Shock Tube (RK4-Standard Galerkin) ', '1D Shock Tube (TG2-One step)', '1D Shock Tube (TG2-Two step)', '1D Shock Tube (RK4-TG2 Two step)', 'RK4-TG2 Two-step and EV', 'RK4, EV and LPS stabilization']
    folder_paths = ['./RK4_standard_galerkin', './TG2_one_step', './TG2_two_step', './RK4_TG2_two_step', './RK4_TG2_two_step_EV', './RK4_EV_LPS']
    file_names = ['RK4_standard_galerkin', 'TG2_one_step', './TG2_two_step', './RK4_TG2_two_step', './RK4_TG2_two_step_EV', './RK4_EV_LPS']
    methods_file_name = ['RK4_galerkin', 'TG2_one_step', 'TG2_two_step', 'RK4_TG2_two_step', 'RK4_TG2_two_step_EV', 'RK4_EV_LPS']
    init_problem_name = ['shock', 'explosion']


    # Create a dictionary to map stabilization_choice to corresponding values
    stabilization_map = {
        1: {'title': stabilization_graph_titles[0], 'path': folder_paths[0], 'file': file_names[0], 'method': methods_file_name[0]},
        2: {'title': stabilization_graph_titles[1], 'path': folder_paths[1], 'file': file_names[1], 'method': methods_file_name[1]},
        3: {'title': stabilization_graph_titles[2], 'path': folder_paths[2], 'file': file_names[2], 'method': methods_file_name[2]},
        4: {'title': stabilization_graph_titles[3], 'path': folder_paths[3], 'file': file_names[3], 'method': methods_file_name[3]},
        5: {'title': stabilization_graph_titles[4], 'path': folder_paths[4], 'file': file_names[4], 'method': methods_file_name[4]},
        6: {'title': stabilization_graph_titles[5], 'path': folder_paths[5], 'file': file_names[5], 'method': methods_file_name[5]}
    }

    # Use the stabilization_choice to retrieve the necessary parameters
    if stabilization_choice in stabilization_map:
        selected_choice = stabilization_map[stabilization_choice]
        stabilization_graph_title = selected_choice['title']
        folder_path = selected_choice['path']
        file_name = selected_choice['file']
        method_file_name = selected_choice['method']
        dt = 1.5 * 10**(-3)
    else:
        raise ValueError("Invalid stabilization choice")


    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Parameters 
    if initial_problem_choice == 1:
        L = 1.0
        numel = 100
        numnp = numel + 1
        xnode = np.linspace(0, L, numnp)
        rho_init, m_init, rho_E_init = U_init_shock_tube(xnode, numnp)
        phi_init = phi_init_shock_tube(xnode, numnp)
    
    elif initial_problem_choice == 2:
        L = 2.0
        numel = 200
        numnp = numel + 1
        xnode = np.linspace(0, L, numnp)
        rho_init, m_init, rho_E_init = U_init_explosion(xnode, numnp)

    gamma = 1.4
    nstep = int(t_end / dt) 

    # Gauss points and weights for [-1, 1]
    xipg = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    wpg = np.array([1, 1])

    # Shape functions and their derivatives on reference element
    N_mef = np.array([(1 - xipg) / 2, (1 + xipg) / 2])
    Nxi_mef = np.array([[-1/2, 1/2], [-1/2, 1/2]])

    #initializing solution array
    U = (np.zeros((numnp, nstep + 1)), 
        np.zeros((numnp, nstep + 1)), 
        np.zeros((numnp, nstep + 1)))
    phi = np.zeros((numnp, nstep+1))

    U[0][:,0] = rho_init
    U[1][:,0] = m_init
    U[2][:,0] = rho_E_init

    phi[:,0] = phi_init

    # Get analytical solutions inputs
    U_init_analytical_left = np.array([1.0, 0.0, 1])
    U_init_analytical_right = np.array([0.125, 0.0, 0.1])
    x0_analytical = numel // 2

    # Entropy viscosity tunable constant
    c_e = 0.4

    return {
         't_end': t_end,
         'variables_titles': variables_titles,
         'y_axis_labels': y_axis_labels,
         'initial_problem_choice': initial_problem_choice,
         'stabilization_choice': stabilization_choice,
         'stabilization_graph_title': stabilization_graph_title,
         'folder_path': folder_path,
         'file_name': file_name,
         'method_file_name': method_file_name,
         'init_problem_name': init_problem_name,
         'L': L,
         'xnode': xnode,
         'numel': numel,
         'numnp': numnp,
         'gamma': gamma,
         'dt': dt,
         'nstep': nstep,
         'N_mef': N_mef,
         'Nxi_mef': Nxi_mef,
         'wpg': wpg,
         'U': U,
         'phi': phi,
         'U_init_analytical_left': U_init_analytical_left,
         'U_init_analytical_right': U_init_analytical_right,
         'x0_analytical': x0_analytical,
         'c_e': c_e

    }


# Main time-stepping loop
def run_simulation(config):

    # Butcher tableau for RK4 (f_rk == 4)
    a = [[0,   0,   0,   0],
        [0.5, 0,   0,   0],
        [0,   0.5, 0,   0],
        [0,   0,   1,   0]]

    c = [0, 0.5, 0.5, 1]
    w = [1/6, 1/3, 1/3, 1/6]
    n_stages = 4


    if config['stabilization_choice'] == 1:

        for n in range(config['nstep']):

            print('timestep is', n)

            # Initialize U_temp and k for the RK stages
            U_temp = [config['U'][0][:, n], config['U'][1][:, n], config['U'][2][:, n]]
            k = [None] * n_stages

            for s in range(n_stages):
                # update U_temp based on previous k values and Butcher tableau coefficients 'a'
                if s == 0:
                    U_temp_stage = U_temp  # k1 doesnt involve any manipulation to U_temp

                else:
                    U_temp_stage = [config['U'][var][:, n] + sum(a[s][j] * k[j][var] for j in range(s)) for var in range(len(U_temp))]

                U_temp_stage = apply_boundary_conditions(U_temp_stage, config['numnp'], config['initial_problem_choice'])
                M_tuple, F_tuple = assemble_standard_galerkin(U_temp_stage, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'])
                k[s] = tuple(config['dt'] * solve(M_tuple[var], F_tuple[var]) for var in range(len(U_temp)))

            for var in range(len(config['U'])):
                config['U'][var][:, n + 1] = config['U'][var][:, n] + sum(w[s] * k[s][var] for s in range(n_stages))

            if config['initial_problem_choice'] == 1:
                # Apply boundary conditions again
                config['U'][0][0, n+1] = 1.0
                config['U'][0][config['numnp']-1, n+1] = 0.125

                config['U'][1][0, n+1] = 0.0
                config['U'][1][config['numnp']-1, n+1] = 0.0

                config['U'][2][0, n+1] = 2.5
                config['U'][2][config['numnp']-1, n+1] = 0.25     

        rho = config['U'][0]
        vel = config['U'][1] / config['U'][0]
        final_p = calc_p(config['gamma'], config['U'][2], config['U'][1], config['U'][0])
        energy = config['U'][2]
        e_int = final_p/((config['gamma']-1)*rho)
        variables_tuple = (rho, vel, final_p, energy, e_int)

    elif config['stabilization_choice'] == 2:
        for n in range (config['nstep']):
            print ('timestep is', n)
            U_n = (config['U'][0][:, n], 
                      config['U'][1][:, n], 
                      config['U'][2][:, n])

            U_n = apply_boundary_conditions(U_n, config['numnp'], config['initial_problem_choice'])

            M_tuple, F_tuple, K_tuple = assemble_TG_one_step(U_n, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'], config['dt'])

            for var in range(len(config['U'])):
                M = M_tuple[var]
                F = F_tuple[var]
                K = K_tuple[var]
                
                b = config['dt'] * (F + K @ config['U'][var][:, n]) + M @ config['U'][var][:, n]
                U_next = np.linalg.solve(M, b)
                config['U'][var][:, n + 1] = U_next

            if config['initial_problem_choice'] == 1:
                # Apply boundary conditions again
                config['U'][0][0, n+1] = 1.0
                config['U'][0][config['numnp']-1, n+1] = 0.125

                config['U'][1][0, n+1] = 0.0
                config['U'][1][config['numnp']-1, n+1] = 0.0

                config['U'][2][0, n+1] = 2.5
                config['U'][2][config['numnp']-1, n+1] = 0.25          


        rho = config['U'][0]
        vel = config['U'][1] / config['U'][0]
        final_p = calc_p(config['gamma'], config['U'][2], config['U'][1], config['U'][0])
        energy = config['U'][2]
        e_int = final_p/((config['gamma']-1)*rho)
        variables_tuple = (rho, vel, final_p, energy, e_int)
 
    elif config['stabilization_choice'] == 3:
        for n in range (config['nstep']):
            U_n = (config['U'][0][:, n], 
                      config['U'][1][:, n], 
                      config['U'][2][:, n])

            U_n = apply_boundary_conditions(U_n, config['numnp'], config['initial_problem_choice'])

            M_tuple, F_tuple= assemble_TG_two_step(U_n, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'], config['dt'])

            for var in range(len(config['U'])):
                M = M_tuple[var]
                F = F_tuple[var]
                
                b = config['dt'] * (F) + M @ config['U'][var][:, n]
                U_next = np.linalg.solve(M, b)
                config['U'][var][:, n + 1] = U_next

            if config['initial_problem_choice'] == 1:
                # Apply boundary conditions again
                config['U'][0][0, n+1] = 1.0
                config['U'][0][config['numnp']-1, n+1] = 0.125

                config['U'][1][0, n+1] = 0.0
                config['U'][1][config['numnp']-1, n+1] = 0.0

                config['U'][2][0, n+1] = 2.5
                config['U'][2][config['numnp']-1, n+1] = 0.25     


        rho = config['U'][0]
        vel = config['U'][1] / config['U'][0]
        final_p = calc_p(config['gamma'], config['U'][2], config['U'][1], config['U'][0])
        energy = config['U'][2]
        e_int = final_p/((config['gamma']-1)*rho)
        variables_tuple = (rho, vel, final_p, energy, e_int)

    elif config['stabilization_choice'] == 4:
        for n in range(config['nstep']):
            print('timestep is:', n)
            # Initialize U_temp and k for the RK stages
            U_temp = [config['U'][0][:, n], config['U'][1][:, n], config['U'][2][:, n]]
            k = [None] * n_stages

            # Loop over RK stages
            for s in range(n_stages):
                # Update U_temp based on previous k values and Butcher tableau coefficients 'a'
                if s == 0:
                    U_temp_stage = U_temp  # k1 doesnt involve any manipulation to U_temp
                else:
                    U_temp_stage = [config['U'][var][:, n] + sum(a[s][j] * k[j][var] for j in range(s)) for var in range(len(U_temp))]

                U_temp_stage = apply_boundary_conditions(U_temp_stage, config['numnp'], config['initial_problem_choice'])
                M_tuple, F_tuple = assemble_TG_two_step(U_temp_stage, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'], config['dt'])
                k[s] = tuple(config['dt'] * solve(M_tuple[var], F_tuple[var]) for var in range(len(U_temp)))
            
            for var in range(len(config['U'])):
                config['U'][var][:, n + 1] = config['U'][var][:, n] + sum(w[s] * k[s][var] for s in range(n_stages))

            if config['initial_problem_choice'] == 1:
                # Apply boundary conditions again
                config['U'][0][0, n+1] = 1.0
                config['U'][0][config['numnp']-1, n+1] = 0.125

                config['U'][1][0, n+1] = 0.0
                config['U'][1][config['numnp']-1, n+1] = 0.0

                config['U'][2][0, n+1] = 2.5
                config['U'][2][config['numnp']-1, n+1] = 0.25     

        rho = config['U'][0]
        vel = config['U'][1] / config['U'][0]
        final_p = calc_p(config['gamma'], config['U'][2], config['U'][1], config['U'][0])
        energy = config['U'][2]
        e_int = final_p/((config['gamma']-1)*rho)
        variables_tuple = (rho, vel, final_p, energy, e_int)

    elif config['stabilization_choice'] == 5:
        
        viscosity_e = np.zeros((config['numnp'], config['numnp']))

        for n in range(config['nstep']):

            print('timestep is', n)
            
            # Initialize U_temp and k for the RK stages
            U_temp = [config['U'][0][:, n], config['U'][1][:, n], config['U'][2][:, n]]
            k = [None] * n_stages
            
            # Loop over RK stages
            for s in range(n_stages):
                # Update U_temp based on previous k values and Butcher tableau coefficients 'a'
                if s == 0:
                    U_temp_stage = U_temp  # k1 doesnt involve any manipulation to U_temp
                else:
                    U_temp_stage = [config['U'][var][:, n] + sum(a[s][j] * k[j][var] for j in range(s)) for var in range(len(U_temp))]

                U_temp_stage = apply_boundary_conditions(U_temp_stage, config['numnp'], config['initial_problem_choice'])

                M_tuple, F_tuple, entropy, entropy_flux, entropy_res, viscosity_e, F_visc = assemble_TG_two_step_EV(U_temp_stage, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'], config['dt'], config['c_e'])

                k[s] = tuple(config['dt'] * solve(M_tuple[var], (F_tuple[var] + F_visc[var])) for var in range(len(U_temp)))

            for var in range(len(config['U'])):
                config['U'][var][:, n + 1] = config['U'][var][:, n] + sum(w[s] * k[s][var] for s in range(n_stages))

            if config['initial_problem_choice'] == 1:
                # Apply boundary conditions again
                config['U'][0][0, n+1] = 1.0
                config['U'][0][config['numnp']-1, n+1] = 0.125

                config['U'][1][0, n+1] = 0.0
                config['U'][1][config['numnp']-1, n+1] = 0.0

                config['U'][2][0, n+1] = 2.5
                config['U'][2][config['numnp']-1, n+1] = 0.25     

        rho = config['U'][0]
        vel = config['U'][1] / config['U'][0]
        final_p = calc_p(config['gamma'], config['U'][2], config['U'][1], config['U'][0])
        energy = config['U'][2]
        e_int = final_p/((config['gamma']-1)*rho)
        variables_tuple = (rho, vel, final_p, energy, e_int, entropy, entropy_flux, entropy_res, viscosity_e)
  
    elif config['stabilization_choice'] == 6:
        
        viscosity_e = np.zeros((config['numnp'], config['numnp']))

        for n in range(config['nstep']):

            print('timestep is', n)
            
            # Initialize U_temp and k for the RK stages
            U_temp = [config['U'][0][:, n], config['U'][1][:, n], config['U'][2][:, n]]
            k = [None] * n_stages

            U_temp_stage = apply_boundary_conditions(U_temp, config['numnp'], config['initial_problem_choice'])

            # Solve linear system for g (LPS method). Independent of time so can be done here no need for within RK4 steps
            M_g, rhs_tuple = assemble_g_rhs_system(U_temp_stage, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'])
            g_tuple = tuple(solve(M_g, rhs_tuple[var]) for var in range (len(rhs_tuple)))

            # Loop over RK stages
            for s in range(n_stages):
                # Update U_temp based on previous k values and Butcher tableau coefficients 'a'
                if s == 0:
                    U_temp_stage = U_temp  # k1 doesnt involve any manipulation to U_temp
                else:
                    U_temp_stage = [config['U'][var][:, n] + sum(a[s][j] * k[j][var] for j in range(s)) for var in range(len(U_temp))]

                U_temp_stage = apply_boundary_conditions(U_temp_stage, config['numnp'], config['initial_problem_choice'])

                M_tuple, F_tuple, F_visc_tuple, F_lps_tuple = assemble_EV_LPS(U_temp_stage, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'], config['dt'], config['c_e'], g_tuple)

                k[s] = tuple(config['dt'] * solve(M_tuple[var], (F_tuple[var] + F_visc_tuple[var] + F_lps_tuple[var])) for var in range(len(U_temp)))


            for var in range(len(config['U'])):
                config['U'][var][:, n + 1] = config['U'][var][:, n] + sum(w[s] * k[s][var] for s in range(n_stages))

            if config['initial_problem_choice'] == 1:
                # Apply boundary conditions again
                config['U'][0][0, n+1] = 1.0
                config['U'][0][config['numnp']-1, n+1] = 0.125

                config['U'][1][0, n+1] = 0.0
                config['U'][1][config['numnp']-1, n+1] = 0.0

                config['U'][2][0, n+1] = 2.5
                config['U'][2][config['numnp']-1, n+1] = 0.25     

        rho = config['U'][0]
        vel = config['U'][1] / config['U'][0]
        final_p = calc_p(config['gamma'], config['U'][2], config['U'][1], config['U'][0])
        energy = config['U'][2]
        e_int = final_p/((config['gamma']-1)*rho)

        U_phi = (config['U'][0], config['U'][1], config['U'][2])
        for n in range (config['nstep']):
            # Initialize U_temp and k for the RK stages
            phi_temp = config['phi'][:, n]
            k = [None] * n_stages

             # Loop over RK stages
            for s in range(n_stages):
                # Update U_temp based on previous k values and Butcher tableau coefficients 'a'
                if s == 0:
                    phi_temp_stage = phi_temp  # k1 doesn't involve any manipulation to phi_temp
                else:
                    phi_temp_stage = phi_temp + sum(a[s][j] * k[j] for j in range(s))

                U_current_phi = (U_phi[0][:,n], U_phi[1][:,n], U_phi[2][:,n])

                phi_temp_stage = apply_boundary_conditions_phi(phi_temp_stage,  config['numnp'])

                M_phi, F_phi = assemble_transport(U_current_phi, phi_temp_stage, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'])

                k[s] = (config['dt'] * solve(M_phi, F_phi))

            config['phi'][:, n + 1] = config['phi'][:, n] + sum(w[s] * k[s] for s in range(n_stages))

            config['phi'][0, n+1] = 1.0
            config['phi'][config['numnp']-1, n+1] = 0.0

        phi_transported  = config['phi']
        variables_tuple = (rho, vel, final_p, energy, e_int, phi_transported)

    return variables_tuple     

    
# Main Execution
def main():
    t_end, stabilization_choice, initial_problem_choice = configure_simulation()
    config = setup_simulation(t_end, stabilization_choice, initial_problem_choice)
    variables_tuple = run_simulation(config)
    analytic, rho_energy_analytic = SodShockAnalytic(config, t_end)  
    plot_solution(t_end, variables_tuple, config, analytic, rho_energy_analytic)

    if config['stabilization_choice'] == 5:
        plot_entropy_res(variables_tuple, config)

    if config['stabilization_choice'] == 6:
        plot_phi_trans(variables_tuple, config)

if __name__ == "__main__":
    main()
