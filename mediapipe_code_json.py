
#
import os
import cv2
import json
import numpy as np
import mediapipe as mp

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

mediapipe_to_openpose = {
    0: 0,   # Nose
    11: 5,  # Left Shoulder
    12: 2,  # Right Shoulder
    13: 6,  # Left Elbow
    14: 3,  # Right Elbow
    15: 7,  # Left Wrist
    16: 4,  # Right Wrist
    23: 11, # Left Hip
    24: 8,  # Right Hip
    25: 12, # Left Knee
    26: 9,  # Right Knee
    27: 13, # Left Ankle
    28: 10, # Right Ankle
}

def detect_pose(image_path, output_json_path):
    # Read the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # Process the image for pose, face, and hand landmarks
    pose_results = pose.process(image_rgb)
    face_results = face_mesh.process(image_rgb)
    hands_results = hands.process(image_rgb)

    # Initialize landmark lists
    pose_keypoints_2d = []
    face_keypoints_2d = []
    hand_left_keypoints_2d = []
    hand_right_keypoints_2d = []

    # Extract pose landmarks
    if pose_results.pose_landmarks:
        for landmark in pose_results.pose_landmarks.landmark:
            pose_keypoints_2d.extend([landmark.x * width, landmark.y * height, landmark.visibility])

    # Extract face landmarks
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                face_keypoints_2d.extend([landmark.x * width, landmark.y * height, landmark.z * width])

    # Extract hand landmarks
    if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
        for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
            if handedness.classification[0].label == 'Left':
                for landmark in hand_landmarks.landmark:
                    hand_left_keypoints_2d.extend([landmark.x * width, landmark.y * height, landmark.z * width])
            elif handedness.classification[0].label == 'Right':
                for landmark in hand_landmarks.landmark:
                    hand_right_keypoints_2d.extend([landmark.x * width, landmark.y * height, landmark.z * width])

    # Format the landmarks into the desired JSON structure
    output_data = {
        "version": 1.3,
        "people": [{
            "person_id": [-1],
            "pose_keypoints_2d": pose_keypoints_2d,
            "face_keypoints_2d": face_keypoints_2d,
            "hand_left_keypoints_2d": hand_left_keypoints_2d,
            "hand_right_keypoints_2d": hand_right_keypoints_2d,
        }]
    }

    # Save the formatted landmarks to a JSON file
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Landmarks saved to {output_json_path}")

def convert_mediapipe_to_openpose(mediapipe_json_path, openpose_json_path):
    # Load the Mediapipe JSON file
    with open(mediapipe_json_path, 'r') as f:
        mediapipe_data = json.load(f)

    # Initialize OpenPose keypoints
    openpose_keypoints = [0] * 25 * 3  # 25 keypoints, each with x, y, confidence

    for mp_idx, op_idx in mediapipe_to_openpose.items():
        if mp_idx < len(mediapipe_data['people'][0]['pose_keypoints_2d']) // 3:
            mp_keypoint = mediapipe_data['people'][0]['pose_keypoints_2d'][mp_idx * 3: mp_idx * 3 + 3]
            openpose_keypoints[op_idx * 3: op_idx * 3 + 3] = mp_keypoint

    openpose_data = {
        "version": 1.3,
        "people": [{
            "pose_keypoints_2d": openpose_keypoints
        }]
    }

    # Save the OpenPose JSON file
    with open(openpose_json_path, 'w') as f:
        json.dump(openpose_data, f, indent=4)

def process_images_in_folder(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            json_output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_keypoints.json")

            # Detect pose and save to JSON
            detect_pose(image_path, json_output_path)

            # Convert to OpenPose format and save
            convert_mediapipe_to_openpose(json_output_path, json_output_path)  # Overwrite the same file

# Example usage
process_images_in_folder(
    'F:\\VITON-HD-main\\VITON-HD-main2\\datasets\\test\\image',
    'F:\\VITON-HD-main\\VITON-HD-main2\\datasets\\test\\openpose-json'
)