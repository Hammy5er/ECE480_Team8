import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pykalman import KalmanFilter


def interpolate_imu_data(data, src_timestamps, dst_timestamps):
    interpolated_data = np.interp(dst_timestamps, src_timestamps, data)
    return interpolated_data


def complementary_filter(visual_data, imu_data, alpha):
    return alpha * visual_data + (1 - alpha) * imu_data


imu_data_file = r'C:\Users\djham\Desktop\output_data.csv'
imu_data = pd.read_csv(imu_data_file)

roll_data = imu_data['roll']
pitch_data = imu_data['pitch']
yaw_data = imu_data['yaw']

# Set parameters
trajectory = []
cumulative_R = np.eye(3)
cumulative_t = np.zeros((3, 1))

cap = cv2.VideoCapture(r"C:\Users\djham\Desktop\tester.mp4")
ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame. Please check the video file.")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
detector = cv2.SimpleBlobDetector_create(params)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
principal_point = (320, 240)
focal_length = 500
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
K = np.array([[focal_length, 0, principal_point[0]], [0, focal_length, principal_point[1]], [0, 0, 1]])  # Camera intrinsic matrix
motion_model = np.eye(4)  # Motion model: [I  dt*I; 0 I], where I is the 3x3 identity matrix and dt is the time interval between frames

# Open video file
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize state vector
x = np.zeros((4, frame_count))
P = np.eye(4) * 1e-3  # Initial covariance matrix

# Set video writer properties
video_out = 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_out, fourcc, fps, (prev_frame.shape[1], prev_frame.shape[0]))


state_dim = 9  # 3 for position (x, y, z) and 3 for orientation (roll, pitch, yaw)
initial_state_mean = np.zeros(state_dim)
initial_state_covariance = np.eye(state_dim) * 1e-3

kf = KalmanFilter(
    transition_matrices=np.eye(state_dim),
    observation_matrices=np.eye(state_dim),
    initial_state_mean=initial_state_mean,
    initial_state_covariance=initial_state_covariance,
    transition_covariance=0.01 * np.eye(state_dim),
    observation_covariance=1.0 * np.eye(state_dim)
)


# Loop through video frames
prev_frame = None
filtered_state_mean = initial_state_mean
filtered_state_cov = initial_state_covariance

trajectory_filtered = []

for i in range(frame_count):
    # Read video frame
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints = detector.detect(gray)
    if len(keypoints) < 5:
        print(f"Skipping frame {i} due to insufficient keypoints: {len(keypoints)}")
        continue

    # Draw keypoints on the frame
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 255, 0),
                                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Write the frame to the output video
    out.write(frame_with_keypoints)

    pts1 = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints]).astype(np.float32)

    if prev_frame is not None:
        # Feature tracking
        pts2, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, gray, pts1, None, **lk_params)
        src_pts = np.array([pts1[i] for i in range(len(st)) if st[i, 0] == 1], dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array([pts2[i] for i in range(len(st)) if st[i, 0] == 1], dtype=np.float32).reshape(-1, 1, 2)

        # Check if there are enough valid points
        if len(src_pts) < 5 or len(dst_pts) < 5:
            print(f"Skipping frame {i} due to insufficient valid points (src_pts: {len(src_pts)}, dst_pts: {len(dst_pts)})")
            continue

        # Essential matrix calculation
        E, _ = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.5, threshold=0.5)
        if E is None or E.shape != (3, 3):
            print(f"Skipping frame {i} due to incorrect E shape: {E.shape if E is not None else 'None'}")
            continue

        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, K)

        # You can now use the rotation matrix (R) and translation vector (t) for your application
        # Update the cumulative rotation and translation
        cumulative_R = R @ cumulative_R
        cumulative_t = -cumulative_R.T @ (R @ cumulative_t + t)
        alpha = .9  # Adjust the value to balance the contribution of the visual odometry and IMU data (between 0 and 1)
        current_yaw = np.arctan2(R[1, 0], R[0, 0])
        current_pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        current_roll = np.arctan2(R[2, 1], R[2, 2])

        predicted_state_mean, predicted_state_cov = kf.filter_update(
            filtered_state_mean, filtered_state_cov, observation=None
        )

        filtered_yaw = complementary_filter(current_yaw, yaw_data[i], alpha)
        filtered_pitch = complementary_filter(current_pitch, pitch_data[i], alpha)
        filtered_roll = complementary_filter(current_roll, roll_data[i], alpha)

        # Compute the filtered rotation matrix
        Rx = np.array([[1, 0, 0], [0, np.cos(filtered_roll), -np.sin(filtered_roll)],
                       [0, np.sin(filtered_roll), np.cos(filtered_roll)]])
        Ry = np.array([[np.cos(filtered_pitch), 0, np.sin(filtered_pitch)], [0, 1, 0],
                       [-np.sin(filtered_pitch), 0, np.cos(filtered_pitch)]])
        Rz = np.array(
            [[np.cos(filtered_yaw), -np.sin(filtered_yaw), 0], [np.sin(filtered_yaw), np.cos(filtered_yaw), 0],
             [0, 0, 1]])

        filtered_R = Rz @ Ry @ Rx
        cumulative_R = filtered_R @ cumulative_R
        cumulative_t = -cumulative_R.T @ (filtered_R @ cumulative_t + t)

        # Append the current position (the translation vector) to the trajectory
        trajectory.append(cumulative_t.flatten())
        predicted_state_mean, predicted_state_cov = kf.filter_update(
            filtered_state_mean, filtered_state_cov, observation=None
        )

        # Update the state estimate with the measurements from the IMU and VIO.
        # For simplicity, we are assuming that the measurement is [x, y, z, roll, pitch, yaw].
        # You should replace this with your actual measurement.
        dt = 1 / fps  # Assuming constant time intervals between frames
        velocity = (cumulative_t.flatten() - filtered_state_mean[:3]) / dt
        measurement = np.hstack((cumulative_t.flatten(), velocity, [filtered_roll, filtered_pitch, filtered_yaw]))
        filtered_state_mean, filtered_state_cov = kf.filter_update(
            predicted_state_mean, predicted_state_cov, observation=measurement
        )

        # Save the filtered state (position and orientation) for the trajectory
        trajectory_filtered.append(filtered_state_mean)

    # Save current frame as previous frame
    prev_frame = gray

# Release video capture and destroy windows
cap.release()
out.release()
cv2.destroyAllWindows()
trajectory_array = np.array(trajectory)

# Import the necessary module
from mpl_toolkits.mplot3d import Axes3D

# Create a new figure
fig = plt.figure()

# Create a 3D axis
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory as a continuous line
ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2], linestyle='-', marker='o')

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set the title
ax.set_title('Camera Trajectory')

# Show the plot
plt.show()
