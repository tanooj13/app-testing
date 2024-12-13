import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # Last point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to check shoulder press posture
def is_shoulder_press_correct(landmarks, mp_pose):
    # Get coordinates of shoulder, elbow, and wrist (left arm as example)
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    # Calculate angle at the elbow (shoulder, elbow, wrist)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    # Check if the motion is vertical (wrist higher than elbow)
    if wrist[1] < elbow[1] and elbow[1] < shoulder[1]:
        # Ensure proper angle range for a shoulder press
        if 160 <= elbow_angle <= 180:
            return "Shoulder Press: Correct", (0, 255, 0)  # Green for correct
        else:
            return "Shoulder Press: Incorrect - Elbow angle", (0, 255, 255)  # Yellow for improper angle
    else:
        return "Shoulder Press: Incorrect - Alignment", (255, 0, 0)  # Red for alignment issue

# Streamlit App
st.title("Shoulder Press Detection Web App")

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])

if uploaded_file is not None:
    # Save uploaded video to a temporary location
    temp_video_path = "uploaded_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    # Open video with OpenCV
    cap = cv2.VideoCapture(0)

    stframe = st.empty()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process the image for pose detection
            results = pose.process(image)

            # Convert back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Check shoulder press posture
                feedback, color = is_shoulder_press_correct(landmarks, mp_pose)

                # Display feedback
                cv2.putText(image, feedback, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

                # Draw landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            else:
                # Warn if no landmarks are detected
                cv2.putText(image, "No body detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Resize frame for Streamlit
            resized_frame = cv2.resize(image, (640, 480))
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")

    cap.release()