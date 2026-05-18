import cv2
import csv
import time
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import deque
from sklearn.preprocessing import StandardScaler
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils
from calculations import EAR as ear
from calculations import GazeRatio as gaze

# --- Model Architecture ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, num_classes=3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_length, input_size)
        out, _ = self.lstm(x)
        # Take hidden state of the last time step
        out = out[:, -1, :]
        logits = self.fc(out)
        return logits

def main():
    # --- Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load the trained LSTM model
    print("Loading model...")
    model = LSTMClassifier().to(device)
    model.load_state_dict(torch.load("focus_lstm_model.pth", map_location=device))
    model.eval()
    print("Model loaded successfully.")
    
    # 2. Fit the StandardScaler using the training data
    print("Fitting StandardScaler on training data...")
    df = pd.read_csv("lstm_training_data.csv")
    features = df[["avg_ear", "avg_gaze", "nose_y_delta"]].values
    scaler = StandardScaler()
    scaler.fit(features)
    print("StandardScaler ready.")
    
    # 3. Initialize sequence buffer and label mapping
    sequence_buffer = deque(maxlen=30)
    label_names = {0: "Focused", 1: "Distracted", 2: "Drowsy"}
    
    # 4. MediaPipe FaceLandmarker Setup
    mp_drawing = drawing_utils
    mp_drawing_styles = drawing_styles
    base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options, 
        running_mode=vision.RunningMode.IMAGE, 
        num_faces=1
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)
    
    # Landmark indices constants
    left_eye_index = [33, 160, 158, 133, 153, 144]
    right_eye_index = [362, 385, 387, 263, 373, 380]
    left_eye_iris = 473
    right_eye_iris = 468
    
    # --- Live Video Loop ---
    cap = cv2.VideoCapture(0)
    
    # --- Initialize Logger ---
    csv_file = open("inference_debug_log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'raw_ear', 'raw_gaze', 'raw_nose', 'scaled_ear', 'scaled_gaze', 'scaled_nose', 'prob_focused', 'prob_distracted', 'prob_drowsy', 'predicted_label'])
    baseline_y = None
    current_state = "Gathering data..."
    
    print("\nStarting live inference...")
    print("Press 'b' to set the baseline posture (required for accurate predictions).")
    print("Press 'q' to quit.")

    with face_landmarker as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert frame for MediaPipe processing
            frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mirrored_frame_rgb = cv2.flip(frame_rgb, 1)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mirrored_frame_rgb)
            results = face_mesh.detect(mp_image)
            
            # Prepare frame for OpenCV display
            mirrored_frame = cv2.cvtColor(mirrored_frame_rgb, cv2.COLOR_RGB2BGR)
            
            current_nose_y = None
            if results.face_landmarks:
                for face_landmarks in results.face_landmarks:
                    # Draw face mesh on the frame
                    mp_drawing.draw_landmarks(
                        image=mirrored_frame,
                        landmark_list=face_landmarks,
                        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
                    # -- Feature 1: EAR Calculation --
                    left_eye_extract = [face_landmarks[i] for i in left_eye_index]
                    right_eye_extract = [face_landmarks[i] for i in right_eye_index]
                    left_eye_ear = ear.ear(left_eye_extract)
                    right_eye_ear = ear.ear(right_eye_extract)
                    avg_ear = (left_eye_ear + right_eye_ear) / 2
                    
                    # -- Feature 2: Nose Y Delta (Posture) --
                    current_nose_y = face_landmarks[1].y
                    nose_y_delta = current_nose_y - baseline_y if baseline_y is not None else 0.0
                    
                    # -- Feature 3: Gaze Calculation --
                    right_iris = face_landmarks[right_eye_iris]
                    right_eye_corner = [face_landmarks[33], face_landmarks[133]]
                    right_gaze = gaze.gaze_ratio(right_eye_corner[0], right_eye_corner[1], right_iris)
                    
                    left_iris = face_landmarks[left_eye_iris]
                    left_eye_corner = [face_landmarks[362], face_landmarks[263]]
                    left_gaze = gaze.gaze_ratio(left_eye_corner[0], left_eye_corner[1], left_iris)
                    
                    avg_gaze = (right_gaze + left_gaze) / 2
                    
                    # -- Append Features to Buffer --
                    sequence_buffer.append([avg_ear, avg_gaze, nose_y_delta])
                    
                    # -- Real-Time Prediction --
                    if len(sequence_buffer) == 30:
                        # 1. Convert buffer to numpy array
                        seq_array = np.array(sequence_buffer)
                        
                        # 2. Transform the sequence using the fitted StandardScaler
                        seq_scaled = scaler.transform(seq_array)
                        
                        # 3. Convert to PyTorch tensor with batch dimension [1, 30, 3]
                        tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        # 4. Model inference
                        with torch.no_grad():
                            output = model(tensor)
                            pred_idx = torch.argmax(output, dim=1).item()
                            current_state = label_names.get(pred_idx, "Unknown")
                            
                            # Log probabilities and states
                            probs = F.softmax(output, dim=1).squeeze().cpu().numpy()
                            prob_foc, prob_dist, prob_drow = probs[0], probs[1], probs[2]
                            
                            raw_latest = sequence_buffer[-1]
                            scaled_latest = seq_scaled[-1]
                            
                            csv_writer.writerow([
                                time.time(),
                                raw_latest[0], raw_latest[1], raw_latest[2],
                                scaled_latest[0], scaled_latest[1], scaled_latest[2],
                                prob_foc, prob_dist, prob_drow,
                                pred_idx
                            ])
                            csv_file.flush()
                            
            else:
                # No Face Detected (Pause buffer appending, display warning)
                cv2.putText(mirrored_frame, "No Face Detected", (100, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # --- Visual Output ---
            color = (0, 255, 0) # Green for Focused
            if current_state == "Distracted":
                color = (0, 165, 255) # Orange
            elif current_state == "Drowsy":
                color = (0, 0, 255) # Red
                
            cv2.putText(
                img=mirrored_frame,
                text=f"State: {current_state}",
                org=(10, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.2,
                color=color,
                thickness=3
            )
            
            # Baseline prompt
            if baseline_y is None:
                cv2.putText(
                    img=mirrored_frame,
                    text="Press 'b' to set baseline posture",
                    org=(10, 80),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=(255, 255, 255),
                    thickness=2
                )

            cv2.imshow("Live Cognitive State Inference", mirrored_frame)
            
            # Key Bindings
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                if current_nose_y is not None:
                    baseline_y = current_nose_y
                    print(f"Baseline Y set to: {baseline_y:.4f}")

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

if __name__ == "__main__":
    main()
