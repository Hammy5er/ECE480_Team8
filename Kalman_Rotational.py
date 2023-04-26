import csv
import math
import numpy as np

def read_sensor_data_from_csv(file_name):
    sensor_data = []

    with open('Fri_Apr_7_12-37-37 2023.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sensor_data.append({
                'roll': float(row['roll']),
                'pitch': float(row['pitch']),
                'yaw': float(row['yaw']),
                'accelx': float(row['raw accelx']),
                'accely': float(row['raw accely']),
                'accelz': float(row['raw accelz']),
                'depth': float(row['depth']) * -1
            })

    return sensor_data

def rotate_coordinates(x, y, z, degrees):
    theta = math.radians(degrees)
    x_rotated = x * math.cos(theta) - y * math.sin(theta)
    y_rotated = x * math.sin(theta) + y * math.cos(theta)
    z_rotated = z
    return x_rotated, y_rotated, z_rotated

def kalman_filter(prev_estimate, prev_covariance, measurement, process_noise, measurement_noise):
    # Predict step
    predicted_estimate = prev_estimate
    predicted_covariance = prev_covariance + process_noise

    # Update step
    kalman_gain = predicted_covariance / (predicted_covariance + measurement_noise)
    new_estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
    new_covariance = (1 - kalman_gain) * predicted_covariance

    return new_estimate, new_covariance

def calculate_robot_position(sensor_data, dt):
    g = 9.81
    x, y, z = -1.34, -0.432, 0

    positions = []

    estimate_accelx, estimate_accely = 0, 0
    cov_accelx, cov_accely = 1, 1
    estimate_roll, estimate_pitch, estimate_yaw = 0, 0, 0
    cov_roll, cov_pitch, cov_yaw = 1, 1, 1

    process_noise_accel = 0.2
    measurement_noise_accel = 0.8
    process_noise_rpy = 0.2
    measurement_noise_rpy = 1

    for data in sensor_data:
        roll, pitch, yaw = data['roll'], data['pitch'], data['yaw']
        accelx, accely, accelz = data['accelx'], data['accely'], data['accelz']

        accelx_global = (accelx * math.cos(roll) + accelz * math.sin(roll)) * math.cos(yaw) - accely * math.sin(yaw)
        accely_global = (accelx * math.cos(pitch) + accelz * math.sin(pitch)) * math.sin(yaw) + accely * math.cos(yaw)
        accelz_global = -accelx * math.sin(pitch) + accelz * math.cos(pitch) - g

        estimate_accelx, cov_accelx = kalman_filter(estimate_accelx, cov_accelx, accelx_global, process_noise_accel, measurement_noise_accel)
        estimate_accely, cov_accely = kalman_filter(estimate_accely, cov_accely, accely_global, process_noise_accel, measurement_noise_accel)
        estimate_roll, cov_roll = kalman_filter(estimate_roll, cov_roll, roll, process_noise_rpy, measurement_noise_rpy)
        estimate_pitch, cov_pitch = kalman_filter(estimate_pitch, cov_pitch, pitch, process_noise_rpy, measurement_noise_rpy)
        estimate_yaw, cov_yaw = kalman_filter(estimate_yaw, cov_yaw, yaw, process_noise_rpy, measurement_noise_rpy)

        accelx_global_filtered, accely_global_filtered = estimate_accelx, estimate_accely
        roll_filtered, pitch_filtered, yaw_filtered = estimate_roll, estimate_pitch, estimate_yaw

        displacement_x = 3.5 * (accelx_global_filtered * dt * dt)
        displacement_y = 1 * (-accely_global_filtered * dt * dt)

        x += displacement_x
        y += displacement_y

        z = data['depth']

        x_rotated, y_rotated, z_rotated = rotate_coordinates(x, y, z, 30)

        positions.append((x_rotated + 0.3, y_rotated - 0.17, z_rotated, accelx_global_filtered, accely_global_filtered, accelz_global, roll_filtered, pitch_filtered, yaw_filtered, z))

    return positions

def write_positions_to_csv(positions, output_file):
    with open(output_file, mode='w', newline='') as csvfile:
        fieldnames = ['x', 'y', 'z', 'accelx_global', 'accely_global', 'accelz_global', 'roll', 'pitch', 'yaw', 'depth']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for position in positions:
            writer.writerow({
                'x': position[0],
                'y': position[1],
                'z': position[2],
                'accelx_global': position[3],
                'accely_global': position[4],
                'accelz_global': position[5],
                'roll': position[6],
                'pitch': position[7],
                'yaw': position[8],
                'depth': position[9]
            })

input_file = 'Fri_Apr_7_12-37-37_2023.csv'
output_file = 'output_data1.csv'
sensor_data = read_sensor_data_from_csv(input_file)
dt = 0.01
positions = calculate_robot_position(sensor_data, dt)
write_positions_to_csv(positions, output_file)
