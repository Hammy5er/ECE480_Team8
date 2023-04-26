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

def kalman_filter(prev_estimate, measurement, process_noise, measurement_noise, prev_covariance):
    predicted_estimate = prev_estimate
    predicted_covariance = prev_covariance + process_noise

    # Update step
    kalman_gain = predicted_covariance / (predicted_covariance + measurement_noise)
    new_estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
    new_covariance = (1 - kalman_gain) * predicted_covariance

    return new_estimate, new_covariance

def calculate_robot_position(sensor_data, dt):
    g = 9.81
    x, y, z = -0.65, -0.874, 0

    positions = []

    estimate_x, estimate_y = x, y
    cov_x, cov_y = 1, 1

    process_noise = 0.3         #increase trust in sensor
    measurement_noise = 0.8     #increase trust in system model (estimation)
    scaling_factorx = 6
    scaling_factory = 4


    for data in sensor_data:
        roll = data['roll']
        pitch = data['pitch']
        yaw = data['yaw']
        accelx = data['accelx']
        accely = data['accely']
        accelz = data['accelz']
        z = data['depth']

        accelx_global = (accelx * math.cos(roll) + accelz * math.sin(roll)) * math.cos(yaw) - accely * math.sin(yaw)
        accely_global = (accelx * math.cos(pitch) + accelz * math.sin(pitch)) * math.sin(yaw) + accely * math.cos(yaw)
        accelz_global = -accelx * math.sin(pitch) + accelz * math.cos(pitch) - g

        displacement_x = scaling_factorx * (accelx_global * dt * dt)
        displacement_y = scaling_factory * (-accely_global * dt * dt)

        # Update position estimates with Kalman filter
        estimate_x, cov_x = kalman_filter(estimate_x, estimate_x + displacement_x, process_noise, measurement_noise, cov_x)
        estimate_y, cov_y = kalman_filter(estimate_y, estimate_y + displacement_y, process_noise, measurement_noise, cov_y)

        shifted_estimate_x = estimate_x
        shifted_estimate_y = estimate_y + 0.4

        positions.append((shifted_estimate_x, shifted_estimate_y, z, accelx_global, accely_global, accelz_global, roll, pitch, yaw, z))

    return positions

def write_positions_to_csv(positions):
    with open('output_data.csv', mode='w', newline='') as csvfile:
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

sensor_data = read_sensor_data_from_csv('Fri_Apr_7_12-37-37 2023.csv')
dt = 0.01
positions = calculate_robot_position(sensor_data, dt)
write_positions_to_csv(positions)