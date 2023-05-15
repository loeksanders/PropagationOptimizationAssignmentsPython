###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################


# General imports
import matplotlib.pyplot as plt
import numpy as np
import os

# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.math import interpolators

# Problem-specific imports
import LowThrustUtilities as Util

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()
# NOTE TO STUDENTS: INPUT YOUR PARAMETER SET HERE, FROM THE INPUT FILES
# ON BRIGHTSPACE, FOR YOUR SPECIFIC STUDENT NUMBER.
trajectory_parameters = [2616.6782407408,
                         780.09375,
                         1,
                         -6534.69,
                         1320.06,
                         -4655.8,
                         -2902.23,
                         -5091.87,
                         -9810.39]

# Choose whether benchmark is run
use_benchmark = True
run_integrator_analysis = True

# Choose whether output of the propagation is written to files
write_results_to_file = True
# Get path of current directory
current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Vehicle settings
vehicle_mass = 4.0E3
specific_impulse = 3000.0
# Fixed parameters
minimum_mars_distance = 5.0E7
# Time since 'departure from Earth CoM' at which propagation starts (and similar
# for arrival time)
time_buffer = 30.0 * constants.JULIAN_DAY
# Time at which to start propagation
initial_propagation_time = Util.get_trajectory_initial_time(trajectory_parameters,
                                                            time_buffer)
###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Define settings for celestial bodies
bodies_to_create = ['Earth',
                    'Mars',
                    'Sun']
# Define coordinate system
global_frame_origin = 'SSB'
global_frame_orientation = 'ECLIPJ2000'
# Create body settings
body_settings = environment_setup.get_default_body_settings(bodies_to_create,
                                                            global_frame_origin,
                                                            global_frame_orientation)
# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Create vehicle object and add it to the existing system of bodies
bodies.create_empty_body('Vehicle')
bodies.get_body('Vehicle').mass = vehicle_mass
thrust_magnitude_settings = (
    propagation_setup.thrust.custom_thrust_magnitude_fixed_isp( lambda time : 0.0, specific_impulse ) )
environment_setup.add_engine_model(
    'Vehicle', 'LowThrustEngine', thrust_magnitude_settings, bodies )
environment_setup.add_rotation_model(
    bodies, 'Vehicle', environment_setup.rotation_model.custom_inertial_direction_based(
        lambda time : np.array([1,0,0] ), global_frame_orientation, 'VehcleFixed' ) )

###########################################################################
# CREATE PROPAGATOR SETTINGS ##############################################
###########################################################################


# Retrieve termination settings
termination_settings = Util.get_termination_settings(trajectory_parameters,
                                                     minimum_mars_distance,
                                                     time_buffer)
# Retrieve dependent variables to save
dependent_variables_to_save = Util.get_dependent_variable_save_settings()
# Check whether there is any
are_dependent_variables_to_save = False if not dependent_variables_to_save else True

if use_benchmark:
    # Define benchmark interpolator settings to make a comparison between the two benchmarks
    benchmark_interpolator_settings = interpolators.lagrange_interpolation(
        8, boundary_interpolation = interpolators.extrapolate_at_boundary)

    # Create propagator settings for benchmark (Cowell)
    propagator_settings = Util.get_propagator_settings(
        trajectory_parameters,
        bodies,
        initial_propagation_time,
        vehicle_mass,
        termination_settings,
        dependent_variables_to_save)

    benchmark_output_path = current_dir + '/SimulationOutput/benchmarks/' if write_results_to_file else None

    # Generate benchmarks
    benchmark_step_size = 43200
    benchmark_list = Util.generate_benchmarks(benchmark_step_size,
                                              initial_propagation_time,
                                              bodies,
                                              propagator_settings,
                                              are_dependent_variables_to_save,
                                              benchmark_output_path)
    # Extract benchmark states
    first_benchmark_state_history = benchmark_list[0]
    second_benchmark_state_history = benchmark_list[1]
    # Create state interpolator for first benchmark
    benchmark_state_interpolator = interpolators.create_one_dimensional_vector_interpolator(second_benchmark_state_history,
                                                                                            benchmark_interpolator_settings)
    benchmark_state_difference = Util.compare_benchmarks(first_benchmark_state_history,
                                                         second_benchmark_state_history,
                                                         benchmark_output_path,
                                                         'benchmarks_state_difference.dat')
    if are_dependent_variables_to_save:
        first_benchmark_dependent_variable_history = benchmark_list[2]
        second_benchmark_dependent_variable_history = benchmark_list[3]
        # Create dependent variable interpolator for first benchmark
        benchmark_dependent_variable_interpolator = interpolators.create_one_dimensional_vector_interpolator(
            first_benchmark_dependent_variable_history,
            benchmark_interpolator_settings)

hodographic_shaping_object = Util.create_hodographic_trajectory(trajectory_parameters,
                                                                bodies)

# Define list of propagators
available_propagators = [propagation_setup.propagator.cowell,
                         propagation_setup.propagator.encke,
                         propagation_setup.propagator.gauss_keplerian,
                         propagation_setup.propagator.gauss_modified_equinoctial,
                         propagation_setup.propagator.unified_state_model_quaternions,
                         propagation_setup.propagator.unified_state_model_modified_rodrigues_parameters,
                         propagation_setup.propagator.unified_state_model_exponential_map
                         ]
# Define settings to loop over
number_of_propagators = len(available_propagators)
number_of_integrators = 13

# Loop over propagators
for propagator_index in [0,1,3,4]:
    # Get current propagator, and define translational state propagation settings
    current_propagator = available_propagators[propagator_index]

    initial_time = initial_propagation_time + 4*43200
    # Define propagation settings
    current_propagator_settings = Util.get_propagator_settings(
        trajectory_parameters,
        bodies,
        initial_propagation_time,
        vehicle_mass,
        termination_settings,
        dependent_variables_to_save,
        current_propagator)
    plt.figure(figsize=(4,3))
    for integrator_index in [3,5,8,11]:
        current_integrator_settings = Util.get_good_integrator_settings(propagator_index,
                                                                   integrator_index,
                                                                   initial_propagation_time)
        current_propagator_settings.integrator_settings = current_integrator_settings

        # Propagate dynamics
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            bodies, current_propagator_settings)

        state_history = dynamics_simulator.state_history
        unprocessed_state_history = dynamics_simulator.unprocessed_state_history
        dependent_variable_history = dynamics_simulator.dependent_variable_history

        # Get the number of function evaluations (for comparison of different integrators)
        function_evaluation_dict = dynamics_simulator.cumulative_number_of_function_evaluations
        number_of_function_evaluations = list(function_evaluation_dict.values())[-1]

        if use_benchmark:
            # Initialize containers
            state_difference = dict()
            # Loop over the propagated states and use the benchmark interpolators
            # NOTE TO STUDENTS: it can happen that the benchmark ends earlier than the regular simulation, due to
            # the shorter step size. Therefore, the following lines of code will be forced to extrapolate the
            # benchmark states (or dependent variables), producing a warning. Be aware of it!
            for epoch in state_history.keys():
                state_difference[epoch] = state_history[epoch] - benchmark_state_interpolator.interpolate(epoch)
            # Do the same for dependent variables, if present
            if are_dependent_variables_to_save:
                # Initialize containers
                dependent_difference = dict()
                # Loop over the propagated dependent variables and use the benchmark interpolators
                for epoch in dependent_variable_history.keys():
                    dependent_difference[epoch] = dependent_variable_history[epoch] - benchmark_dependent_variable_interpolator.interpolate(epoch)

        bench_time = np.array(list(benchmark_state_difference.keys()))
        new_bench_dict = {k: v for k, v in benchmark_state_difference.items() if k > bench_time[0] + 10 * benchmark_step_size and k < bench_time[-1] - 10 * benchmark_step_size}
        new_bench_time = np.array(list(new_bench_dict.keys()))
        bench_diff = np.linalg.norm(np.array(list(new_bench_dict.values()))[:, 0:3], axis=1)
        new_dict = {k: v for k, v in state_difference.items() if k > bench_time[0] + 10 * benchmark_step_size and k < bench_time[-1] - 10 * benchmark_step_size}
        time = np.array(list(new_dict.keys()))
        diff = np.linalg.norm(np.array(list(new_dict.values()))[:, 0:3], axis=1)
        plt.plot(time,diff,label=str(integrator_index))
    plt.plot(new_bench_time,bench_diff,linestyle='dashed', label='benchmark')
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel('Position error [m]')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

