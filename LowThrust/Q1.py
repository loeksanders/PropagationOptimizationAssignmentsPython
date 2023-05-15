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
benchmark_step_size_lst = [5400, 10800, 21600, 43200, 86400, 172800,345600,691200,1382400]
bench_diff = []
bench_times = []
pos_error = []
for benchmark_step_size in benchmark_step_size_lst:
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
    benchmark_state_interpolator = interpolators.create_one_dimensional_vector_interpolator(first_benchmark_state_history,
                                                                                            benchmark_interpolator_settings)


        # Compare benchmark states, returning interpolator of the first benchmark
    benchmark_state_difference = Util.compare_benchmarks(first_benchmark_state_history,
                                                         second_benchmark_state_history,
                                                         benchmark_output_path,
                                                         'benchmarks_state_difference.dat')
    diff = np.array(list(benchmark_state_difference.values()))
    time = np.array(list(benchmark_state_difference.keys()))
    #wow = np.linalg.norm(diff[:,0:3], axis=1)
    bench_diff.append(diff)
    pos_error.append(np.linalg.norm(diff[:,0:3], axis=1))
    bench_times.append(time)


    # Extract benchmark dependent variables, if present
    if are_dependent_variables_to_save:
        first_benchmark_dependent_variable_history = benchmark_list[2]
        second_benchmark_dependent_variable_history = benchmark_list[3]
        # Create dependent variable interpolator for first benchmark
        benchmark_dependent_variable_interpolator = interpolators.create_one_dimensional_vector_interpolator(
            first_benchmark_dependent_variable_history,
            benchmark_interpolator_settings)

        # Compare benchmark dependent variables, returning interpolator of the first benchmark, if present
        benchmark_dependent_difference = Util.compare_benchmarks(first_benchmark_dependent_variable_history,
                                                                 second_benchmark_dependent_variable_history,
                                                                 benchmark_output_path,
                                                                 'benchmarks_dependent_variable_difference.dat')

max1 = np.amax(pos_error[0][4:-4])
max2 = np.amax(pos_error[1][4:-4])
max3 = np.amax(pos_error[2][4:-4])
max4 = np.amax(pos_error[3][4:-4])
max5 = np.amax(pos_error[4][4:-4])
max6 = np.amax(pos_error[5][4:-4])
max7 = np.amax(pos_error[6][4:-4])
max8 = np.amax(pos_error[7][4:-4])
max9 = np.amax(pos_error[8][4:-4])
max_lst = np.array([max1,max2,max3,max4,max5,max6,max7,max8,max9])

plt.plot(bench_times[0][4:-4], np.linalg.norm(bench_diff[0][4:-4,0:3],axis=1), label='10800 s')
plt.plot(bench_times[1][4:-4], np.linalg.norm(bench_diff[1][4:-4, 0:3], axis=1), label='21600 s')
plt.plot(bench_times[2][4:-4], np.linalg.norm(bench_diff[2][4:-4, 0:3], axis=1), label='43200 s' )
plt.plot(bench_times[3][4:-4], np.linalg.norm(bench_diff[3][4:-4, 0:3], axis=1), label='86400 s')
plt.plot(bench_times[4][4:-4], np.linalg.norm(bench_diff[4][4:-4, 0:3], axis=1), label='172800 s')
plt.plot(bench_times[5][4:-4], np.linalg.norm(bench_diff[5][4:-4, 0:3], axis=1), label='345600 s')
plt.yscale('log')
plt.legend()
plt.grid()
plt.xlabel('time [s]')
plt.ylabel('position error [m]')
plt.show()


#default_x_ticks = range(len(benchmark_step_size_lst))
plt.plot(np.array(benchmark_step_size_lst)*2, max_lst,marker='o')
#plt.xticks(default_x_ticks, benchmark_step_size_lst)
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.ylabel('max position error [m]')
plt.xlabel('step size [s]')
plt.show()