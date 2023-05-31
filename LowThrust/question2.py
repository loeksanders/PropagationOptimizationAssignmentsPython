###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# import sys
# sys.path.insert(0, "/home/dominic/Tudat/tudat-bundle/tudat-bundle/cmake-build-default/tudatpy")

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
ref_area = 100.0
radiation_pressure_coefficient = 1.2
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

# Set number of models
number_of_models = 201

# Initialize dictionary to store the results of the simulation
simulation_results = dict()

# Set the interpolation step at which different runs are compared
output_interpolation_step = constants.JULIAN_DAY  # s

np.random.seed(4805232)
grav_lst = []
rad_lst = []

for model_test in range(number_of_models):
    # Define settings for celestial bodies
    bodies_to_create = ['Earth',
                        'Mars',
                        'Sun',
                        'Jupiter',
                        'Moon',
                        'Saturn',
                        'Venus',
                        'Neptune',
                        'Mercury',
                        'Uranus'
                        ]
    # Define coordinate system
    global_frame_origin = 'SSB'
    global_frame_orientation = 'ECLIPJ2000'
    # Create body settings
    body_settings = environment_setup.get_default_body_settings(bodies_to_create,
                                                                global_frame_origin,
                                                                global_frame_orientation)
    grav_para_mars = spice_interface.get_body_gravitational_parameter('Mars')
    if model_test == 0 or model_test > 100:
        body_settings.get('Mars').gravity_field_settings.gravitational_parameter = grav_para_mars
    else:
        body_settings.get('Mars').gravity_field_settings.gravitational_parameter = np.random.normal(grav_para_mars,10**11)
        grav_lst.append(body_settings.get('Mars').gravity_field_settings.gravitational_parameter)

    new_gravitational_parameter = spice_interface.get_body_gravitational_parameter('Jupiter') + \
                                  spice_interface.get_body_gravitational_parameter('Callisto') + \
                                  spice_interface.get_body_gravitational_parameter('Ganymede') + \
                                  spice_interface.get_body_gravitational_parameter('Io')
    body_settings.get('Jupiter').gravity_field_settings.gravitational_parameter = new_gravitational_parameter

    effective_gravitational_parameter_saturn = spice_interface.get_body_gravitational_parameter('Sun') + \
                                        spice_interface.get_body_gravitational_parameter('Saturn')
    body_settings.get('Saturn').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
        'Saturn', initial_propagation_time, effective_gravitational_parameter_saturn, 'Sun', global_frame_orientation)

    effective_gravitational_parameter_mercury = spice_interface.get_body_gravitational_parameter('Sun') + \
                                        spice_interface.get_body_gravitational_parameter('Mercury')
    body_settings.get('Mercury').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
        'Mercury', initial_propagation_time, effective_gravitational_parameter_mercury, 'Sun', global_frame_orientation)

    body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.simple_from_spice(
        global_frame_orientation, 'IAU_Earth', 'IAU_Earth', initial_propagation_time)

    body_settings.get('Mars').rotation_model_settings = environment_setup.rotation_model.simple_from_spice(
        global_frame_orientation, 'IAU_Mars', 'IAU_Mars', initial_propagation_time)

    body_settings.get('Jupiter').rotation_model_settings = environment_setup.rotation_model.simple_from_spice(
        global_frame_orientation, 'IAU_Jupiter', 'IAU_Jupiter', initial_propagation_time)

    body_settings.get('Saturn').rotation_model_settings = environment_setup.rotation_model.simple_from_spice(
        global_frame_orientation, 'IAU_Saturn', 'IAU_Saturn', initial_propagation_time)

    body_settings.get('Neptune').rotation_model_settings = environment_setup.rotation_model.simple_from_spice(
        global_frame_orientation, 'IAU_Neptune', 'IAU_Neptune', initial_propagation_time)

    body_settings.get('Uranus').rotation_model_settings = environment_setup.rotation_model.simple_from_spice(
        global_frame_orientation, 'IAU_Uranus', 'IAU_Uranus', initial_propagation_time)

    body_settings.get('Mercury').rotation_model_settings = environment_setup.rotation_model.simple_from_spice(
        global_frame_orientation, 'IAU_Mercury', 'IAU_Mercury', initial_propagation_time)

    body_settings.get('Venus').rotation_model_settings = environment_setup.rotation_model.simple_from_spice(
        global_frame_orientation, 'IAU_Venus', 'IAU_Venus', initial_propagation_time)

    # Create bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Create vehicle object and add it to the existing system of bodies
    bodies.create_empty_body('Vehicle')
    bodies.get_body('Vehicle').mass = vehicle_mass
    if model_test > 100:
        radiation_pressure_coefficient = np.random.normal(1.2, 0.01)
        rad_lst.append(radiation_pressure_coefficient)
    else:
        radiation_pressure_coefficient = radiation_pressure_coefficient
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        'Sun', ref_area, radiation_pressure_coefficient
    )
    environment_setup.add_radiation_pressure_interface(
        bodies, 'Vehicle',radiation_pressure_settings
    )
    thrust_magnitude_settings = (
        propagation_setup.thrust.custom_thrust_magnitude_fixed_isp(lambda time: 0.0, specific_impulse))
    environment_setup.add_engine_model(
        'Vehicle', 'LowThrustEngine', thrust_magnitude_settings, bodies)
    environment_setup.add_rotation_model(
        bodies, 'Vehicle', environment_setup.rotation_model.custom_inertial_direction_based(
            lambda time: np.array([1, 0, 0]), global_frame_orientation, 'VehicleFixed'))
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

    # Create propagator settings for benchmark (USM7)
    propagator_settings = Util.get_new_propagator_settings(
        trajectory_parameters,
        bodies,
        initial_propagation_time,
        vehicle_mass,
        termination_settings,
        dependent_variables_to_save,
        current_propagator=propagation_setup.propagator.unified_state_model_quaternions,
        model_choice = model_test )

    # integrator index of 3 == rkdp_87
    propagator_settings.integrator_settings = Util.get_integrator_settings(
        0,  3, 0, initial_propagation_time)
    # Propagate dynamics
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings )


    ### OUTPUT OF THE SIMULATION ###
    # Retrieve propagated state and dependent variables
    # NOTE TO STUDENTS, the following retrieve the propagated states, converted to Cartesian states
    state_history = dynamics_simulator.state_history
    dependent_variable_history = dynamics_simulator.dependent_variable_history

    # Save results to a dictionary
    simulation_results[model_test] = [state_history, dependent_variable_history]

    # Get output path
    if model_test == 0:
        subdirectory = '/NominalCase/'
    else:
        subdirectory = '/Model_' + str(model_test) + '/'

    # Decide if output writing is required
    if write_results_to_file:
        output_path = current_dir + subdirectory
    else:
        output_path = None

    # If desired, write output to a file
    if write_results_to_file:
        save2txt(state_history, 'state_history.dat', output_path)
        save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)

"""
NOTE TO STUDENTS
The first index of the dictionary simulation_results refers to the model case, while the second index can be 0 (states)
or 1 (dependent variables).
You can use this dictionary to make all the cross-comparison that you deem necessary. The code below currently compares
every case with respect to the "nominal" one.
"""
# Compare all the model settings with the nominal case

x_diff = []
y_diff = []
z_diff = []
norm_lst = []
x_lst = []
y_lst = []
z_lst = []
for model_test in range(1, number_of_models):
    # Get output path
    output_path = current_dir + '/Model_' + str(model_test) + '/'

    # Set time limits to avoid numerical issues at the boundaries due to the interpolation
    nominal_state_history = simulation_results[0][0]
    nominal_dependent_variable_history = simulation_results[0][1]
    nominal_times = list(nominal_state_history.keys())

    # Retrieve current state and dependent variable history
    current_state_history = simulation_results[model_test][0]
    current_dependent_variable_history = simulation_results[model_test][1]
    current_times = list(current_state_history.keys())

    # Get limit times at which both histories can be validly interpolated
    interpolation_lower_limit = max(nominal_times[3],current_times[3])
    interpolation_upper_limit = min(nominal_times[-10],current_times[-10])

    # Create vector of epochs to be compared (boundaries are referred to the first case)
    unfiltered_interpolation_epochs = np.arange(current_times[0], current_times[-1], output_interpolation_step)
    unfiltered_interpolation_epochs = [n for n in unfiltered_interpolation_epochs if n <= interpolation_upper_limit]
    interpolation_epochs = [n for n in unfiltered_interpolation_epochs if n >= interpolation_lower_limit]

    #interpolation_epochs = unfiltered_interpolation_epochs
    # Compare state history
    state_difference_wrt_nominal = Util.compare_models(current_state_history,
                                                       simulation_results[0][0],
                                                       interpolation_epochs,
                                                       output_path,
                                                       'state_difference_wrt_nominal_case.dat')
    # Compare dependent variable history
    dependent_variable_difference_wrt_nominal = Util.compare_models(current_dependent_variable_history,
                                                                    simulation_results[0][1],
                                                                    interpolation_epochs,
                                                                    output_path,
                                                                    'dependent_variable_difference_wrt_nominal_case.dat')
    x_diff.append(np.array(list(state_difference_wrt_nominal.values()))[-1, 0])
    y_diff.append(np.array(list(state_difference_wrt_nominal.values()))[-1, 1])
    z_diff.append(np.array(list(state_difference_wrt_nominal.values()))[-1, 2])
    norm_lst.append(np.linalg.norm(np.array(list(state_difference_wrt_nominal.values()))[-1,0:3]))
    x_lst.append(np.array(list(current_state_history.values()))[-1, 0])
    y_lst.append(np.array(list(current_state_history.values()))[-1, 1])
    z_lst.append(np.array(list(current_state_history.values()))[-1, 2])
    plt.plot(np.array(list(state_difference_wrt_nominal.keys())),
             np.linalg.norm(np.array(list(state_difference_wrt_nominal.values()))[:,0:3],axis=1))
    if model_test == 100 or model_test == 200:
        plt.ylabel(r'$|\Delta r|$ [m]')
        plt.xlabel('Time [s]')
        plt.grid()
        plt.show()


counts_x, bins_x = np.histogram(x_diff[0:100])
counts_y, bins_y = np.histogram(y_diff[0:100])
counts_z, bins_z = np.histogram(z_diff[0:100])

counts_x_2, bins_x_2 = np.histogram(x_diff[100:200])
counts_y_2, bins_y_2 = np.histogram(y_diff[100:200])
counts_z_2, bins_z_2 = np.histogram(z_diff[100:200])
print(spice_interface.get_body_gravitational_parameter('Mars'))
print(min(grav_lst), max(grav_lst))
print(min(rad_lst), max(rad_lst))

plt.figure(figsize=(3.5,3))

plt.hist(x_diff[0:200], bins_x)
plt.xlabel('Difference [m]')
plt.ylabel('Occurrences [-]')
plt.tight_layout()
plt.show()

plt.figure(figsize=(3.5,3))
plt.hist(y_diff[0:100],bins_y)
plt.xlabel('Difference [m]')
plt.ylabel('Occurrences [-]')
plt.tight_layout()
plt.show()

plt.figure(figsize=(3.5,3))
plt.hist(z_diff[0:100], bins_z)
plt.xlabel('Difference [m]')
plt.ylabel('Occurrences [-]')
plt.tight_layout()
plt.show()

plt.figure(figsize=(3.5,3))
plt.scatter(grav_lst, norm_lst[0:100])
plt.ylabel(r'$|\Delta r(t_1)|$ [m]')
plt.xlabel(r'$\mu$ of Mars [$m^3/s^2$] ')
plt.tight_layout()
plt.show()


# fig = plt.figure(figsize=(6,6), dpi=125)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_lst, y_lst, z_lst)
# #ax.set_aspect('equal', adjustable='box')
# plt.show()

plt.figure(figsize=(3.5,3))
plt.hist(x_diff[100:200], bins_x_2)
plt.xlabel('Difference [m]')
plt.ylabel('Occurrences [-]')
plt.tight_layout()
plt.show()

plt.figure(figsize=(3.5,3))
plt.hist(y_diff[100:200],bins_y_2)
plt.xlabel('Difference [m]')
plt.ylabel('Occurrences [-]')
plt.tight_layout()
plt.show()

plt.figure(figsize=(3.5,3))
plt.hist(z_diff[100:200], bins_z_2)
plt.xlabel('Difference [m]')
plt.ylabel('Occurrences [-]')
plt.tight_layout()
plt.show()

plt.figure(figsize=(3.5,3))
plt.scatter(rad_lst, norm_lst[100:200])
plt.ylabel(r'$|\Delta r(t_1)|$ [m]')
plt.xlabel(r'$C_r$ [-] ')
plt.tight_layout()
plt.show()
