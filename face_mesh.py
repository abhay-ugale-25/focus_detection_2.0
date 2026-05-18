import cv2
import csv
import time
import mediapipe as mp
from mediapipe import tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils
from calculations import EAR as ear
from calculations import GazeRatio as gaze

# Mediapipe Face Mesh
mp_face_mesh = vision.FaceLandmarker
mp_drawing = drawing_utils
mp_drawing_styles = drawing_styles


class focus_cv:
    def video_capture():
        # Constants
        # Eye landmark indices
        left_eye_index = [33, 160, 158, 133, 153, 144]
        right_eye_index = [362, 385, 387, 263, 373, 380]
        # Iris landmark indices for gaze
        left_eye_iris = 473
        right_eye_iris = 468
        # Camera
        cap = cv2.VideoCapture(0)
        model = python.BaseOptions(model_asset_path="face_landmarker.task")
        landmark = vision.FaceLandmarkerOptions(base_options=model, running_mode=vision.RunningMode.IMAGE, num_faces=1)
        face_mesh_obj = vision.FaceLandmarker.create_from_options(landmark)

        # Data logging setup
        current_label = 0
        csv_file = open("lstm_training_data.csv", "a", newline="")
        csv_writer = csv.writer(csv_file)
        # Write header only if the file is empty
        if csv_file.tell() == 0:
            csv_writer.writerow(["timestamp", "avg_ear", "avg_gaze", "nose_y_delta", "label"])

        label_names = {0: "Focused", 1: "Distracted", 2: "Drowsy"}

        with face_mesh_obj as face_mesh:
            # Posture tracking
            baseline_y = None
            current_nose_y = None
            while True:
                ret, frame = cap.read()
                if ret:
                    frame.flags.writeable = False
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mirrored_frame = cv2.flip(frame, 1)
                    results = face_mesh.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=mirrored_frame))
                    mirrored_frame.flags.writeable = True
                    mirrored_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_RGB2BGR)
                    if results.face_landmarks:
                        for face_landmarks in results.face_landmarks:
                            mp_drawing.draw_landmarks(
                                image=mirrored_frame,
                                landmark_list=face_landmarks,
                                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                            )
                            # EAR calculation
                            left_eye_extract = []
                            right_eye_extract = []
                            for index in left_eye_index:
                                left_eye_extract.append(face_landmarks[index])
                            for index in right_eye_index:
                                right_eye_extract.append(face_landmarks[index])
                            left_eye_ear = ear.ear(left_eye_extract)
                            right_eye_ear = ear.ear(right_eye_extract)
                            avg_ear = (left_eye_ear + right_eye_ear) / 2
                            # Nose Y (posture) calculation
                            current_nose_y = face_landmarks[1].y
                            nose_y_delta = current_nose_y - baseline_y if baseline_y is not None else 0.0
                            # Gaze calculation
                            right_iris = face_landmarks[right_eye_iris]
                            right_eye_corner = [face_landmarks[33], face_landmarks[133]]
                            right_gaze = gaze.gaze_ratio(right_eye_corner[0], right_eye_corner[1], right_iris)
                            left_iris = face_landmarks[left_eye_iris]
                            left_eye_corner = [face_landmarks[362], face_landmarks[263]]
                            left_gaze = gaze.gaze_ratio(left_eye_corner[0], left_eye_corner[1], left_iris)
                            avg_gaze = (right_gaze + left_gaze) / 2
                            # Write data row to CSV
                            csv_writer.writerow([time.time(), avg_ear, avg_gaze, nose_y_delta, current_label])
                            csv_file.flush()
                    # No Face Detected
                    else:
                        baseline_y = None
                        cv2.putText(img=mirrored_frame, text="No Face Detected", org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=3)

                    # Debug overlay: show current recording label
                    cv2.putText(
                        img=mirrored_frame,
                        text=f"Label: {current_label} ({label_names.get(current_label, '?')})",
                        org=(10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        color=(0, 255, 0),
                        thickness=2,
                    )
                    # Display
                    cv2.imshow("focus_cv", mirrored_frame)
                    # Key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    # Set baseline Y
                    elif key == ord('b'):
                        if current_nose_y is not None:
                            baseline_y = current_nose_y
                            print(f"Baseline Y: {baseline_y} is set.")
                    # Label controls
                    elif key == ord('0'):
                        current_label = 0
                        print("Label set to 0 (Focused)")
                    elif key == ord('1'):
                        current_label = 1
                        print("Label set to 1 (Distracted)")
                    elif key == ord('2'):
                        current_label = 2
                        print("Label set to 2 (Drowsy)")
                else:
                    break
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
