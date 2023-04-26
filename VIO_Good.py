import cv2
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from filterpy.kalman import ExtendedKalmanFilter

def process_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_frame, None)
    frame_with_keypoints = cv2.drawKeypoints(gray_frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return frame_with_keypoints

def match_features(descriptors1, descriptors2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return good_matches
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



def sensor_fusion(imu_positions, frame_positions):

                
    def compute_HJacobian(state, *args):

        # Extract the relevant variables from the state estimate
        x = state[0]
        y = state[1]
        z = state[2]
        
        denom_xy = np.sqrt(x**2 + y**2)
        denom_xy_sq = x**2 + y**2
        
        # Check for potential division by zero
        if denom_xy == 0:
            denom_xy = 1e-8
        if denom_xy_sq == 0:
            denom_xy_sq = 1e-8
            
        HJacobian = np.array([
            [x / denom_xy, y / denom_xy, 0, 0, 0, 0],
            [-y / denom_xy_sq, x / denom_xy_sq, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        return HJacobian


    def compute_Hx(state, *args):

        # Extract the relevant variables from the state estimate
        x = state[0]
        y = state[1]
        z = state[2]

        # Compute the expected measurement
        Hx = np.array([
            np.sqrt(x**2 + y**2),
            np.arctan2(y, x),
            z
        ])

        return Hx

    # Initialize the Extended Kalman Filter
    ekf = ExtendedKalmanFilter(dim_x=6, dim_z=3) # Assuming state vector has 6 elements and measurement has 3

    # Set the initial state and covariance matrices
    ekf.x = np.array([0, 0, 0, 0, 0, 0]) # Example: [x, y, z, roll, pitch, yaw]
    ekf.P = np.eye(6) * 1000

    # Set the process noise and measurement noise covariance matrices
    ekf.Q = np.eye(6) * 0.1
    ekf.R = np.eye(3) * 1.0

    ekf.B = np.zeros((6, 10))  # Example of a zero matrix with shape (6, 10)


    fused_positions = []

    for imu_position, frame_position in zip(imu_positions, frame_positions):
        # Update the EKF with the IMU data (control input) and keypoint-based position (measurement)
        ekf.predict(u=imu_position)
        HJacobian = compute_HJacobian(ekf.x)  # Compute the Jacobian matrix of the measurement model
        Hx = compute_Hx(ekf.x)  # Compute the expected measurement based on the state estimate
        ekf.update(z=frame_position, HJacobian=compute_HJacobian, Hx=compute_Hx)

        # Store the fused position
        fused_positions.append(ekf.x[:3])  # Assuming the first 3 elements of the state are x, y, and z

    return fused_positions           

# Read IMU data from the CSV file
sensor_data = read_sensor_data_from_csv('tester.csv')
dt = 0.01

theta = 2.0

imu_positions = calculate_robot_position(sensor_data, dt)
principal_point = (320, 240)
focal_length = 500
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
K = np.array([[focal_length, 0, principal_point[0]], [0, focal_length, principal_point[1]], [0, 0, 1]])  # Camera intrinsic matrix
motion_model = np.eye(4)  # Motion model: [I  dt*I; 0 I], where I is the 3x3 identity matrix and dt is the time interval between frames

video_capture = cap = cv2.VideoCapture(r"C:\Users\djham\Desktop\ece480\tester1.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video = cv2.VideoWriter('output_video.mp4', fourcc, 22.0, (frame_width, frame_height))

prev_frame = None
prev_keypoints = None
prev_descriptors = None
frame_positions = []

while video_capture.isOpened():
    ret, frame = video_capture.read()

    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_frame, None)

    if prev_frame is not None:
        # Match keypoints between the current and previous frames
        good_matches = match_features(prev_descriptors, descriptors)

        # Extract the matched keypoints' coordinates
        src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute the relative motion (e.g., using findHomography, recoverPose, or other methods)
        # You may need to adjust this part based on the specific requirements of your application
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Assuming you have the camera calibration matrix K

        normalized_H = np.linalg.inv(K) @ H @ K
        U, S, Vt = np.linalg.svd(normalized_H)

        R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        t = S[0] * U[:, 2]

        frame_positions.append((t[0], t[1], t[2]))

        
        # Assuming you have the camera calibration matrix K
        # Compute the essential matrix
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        # Recover the relative pose (rotation and translation)
        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, K, mask)

        # Store the relative motion in the frame_positions list
        # In this example, we store only the translation components
        frame_positions.append((t[0, 0], t[1, 0], t[2, 0]))


    prev_frame = frame
    prev_keypoints = keypoints
    prev_descriptors = descriptors

    frame_with_keypoints = cv2.drawKeypoints(gray_frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the processed frame
    cv2.imshow('Frame with Keypoints', frame_with_keypoints)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    output_video.write(frame_with_keypoints)

# Release the video capture and close the display window.
video_capture.release()
output_video.release()
cv2.destroyAllWindows()

# Combine the IMU data and the keypoints to compute the robot's position
# Here, you need to implement your sensor fusion algorithm, such as an EKF
fused_positions = sensor_fusion(imu_positions, frame_positions)

# Visualize the robot's position in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Use fused_positions instead of imu_positions if you have implemented sensor fusion
x = [pos[0] for pos in imu_positions]
y = [pos[1] for pos in imu_positions]
z = [pos[2] for pos in imu_positions]

# Write the data to a CSV file
with open('positions.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write the header row
    csvwriter.writerow(['X', 'Y', 'Z'])
    
    # Write the data rows
    for i in range(len(x)):
        csvwriter.writerow([x[i], y[i], z[i]])

ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()
