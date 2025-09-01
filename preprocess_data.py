import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def process_video_file(filepath, sequence_length):
    """
    Processes a single video file to extract pose keypoints using MediaPipe.
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print(f"Error: Could not open video file {filepath}")
        return None

    frames_data = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark], dtype=np.float32)
            frames_data.append(landmarks)
        else:
            frames_data.append(np.zeros((33, 3), dtype=np.float32))

    cap.release()

    if not frames_data:
        return None

    frames_np = np.array(frames_data)
    
    left_hip = frames_np[:, mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = frames_np[:, mp_pose.PoseLandmark.RIGHT_HIP.value]
    origin = (left_hip + right_hip) / 2
    frames_np[:, :, :2] = frames_np[:, :, :2] - origin[:, np.newaxis, :2]
    
    num_frames = frames_np.shape[0]
    indices = np.linspace(0, num_frames - 1, sequence_length, dtype=int)
    resampled_frames = frames_np[indices]
    
    return resampled_frames.reshape(sequence_length, -1)

def main():
    DATASET_PATH = r'C:\Users\John Thomas\Downloads\UCF101\UCF-101'
    SEQUENCE_LENGTH = 30
    
    if not os.path.exists(DATASET_PATH):
        print("\n" + "="*50)
        print(f"  ERROR: The directory '{DATASET_PATH}' was not found.")
        print("  Please double-check the path. If your video class folders (e.g., 'ApplyEyeMakeup')")
        print("  are directly inside 'C:\\Users\\John Thomas\\Downloads\\UCF101',")
        print("  then remove the final '\\UCF-101' from the DATASET_PATH variable in this script.")
        print("="*50 + "\n")
        return
        
    all_sequences = []
    all_labels = []

    class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    class_map = {name: i for i, name in enumerate(class_names)}

    print(f"Found {len(class_names)} classes in '{DATASET_PATH}'. Starting processing...")

    for class_name, class_idx in tqdm(class_map.items(), desc="Processing Classes"):
        class_path = os.path.join(DATASET_PATH, class_name)
        video_files = [f for f in os.listdir(class_path) if f.endswith('.avi')]
        
        for video_name in tqdm(video_files, desc=f"Videos in {class_name}", leave=False):
            filepath = os.path.join(class_path, video_name)
            
            processed_sequence = process_video_file(filepath, SEQUENCE_LENGTH)
            
            if processed_sequence is not None:
                all_sequences.append(processed_sequence)
                all_labels.append(class_idx)

    X_data = np.array(all_sequences)
    y_labels = np.array(all_labels)

    print(f"\nProcessed data shape (X): {X_data.shape}")
    print(f"Processed labels shape (y): {y_labels.shape}")

    print("Saving processed data to .npy files...")
    np.save('X_data_ucf101.npy', X_data)
    np.save('y_labels_ucf101.npy', y_labels)
    print("Done! You can now run the train.py script.")

if __name__ == '__main__':
    main()