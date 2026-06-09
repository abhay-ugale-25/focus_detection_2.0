import math

class EAR:
    def euclidean_distance(point1, point2):
        return math.hypot(point2.x - point1.x, point2.y - point1.y)

    def ear(eye):
        vericle_1 = EAR.euclidean_distance(eye[1], eye[5])
        vericle_2 = EAR.euclidean_distance(eye[2], eye[4])
        horizontal = EAR.euclidean_distance(eye[0], eye[3])
        ear = (vericle_1 + vericle_2) / (2 * horizontal)
        return ear

class GazeRatio:
    def gaze_ratio(left_corner, right_corner, iris_center):
        return float(EAR.euclidean_distance(left_corner, iris_center) / EAR.euclidean_distance(left_corner, right_corner))

def get_head_pitch_ratio(face_landmarks):
    """Calculate a scale-invariant head pitch ratio.

    Compares the vertical distance from forehead to nose against
    nose to chin. This ratio is independent of camera distance.

    Args:
        face_landmarks: MediaPipe face landmarks list.

    Returns:
        float: upper_face_distance / lower_face_distance ratio.
    """
    forehead_y = face_landmarks[10].y
    nose_y = face_landmarks[1].y
    chin_y = face_landmarks[152].y

    upper_face_distance = abs(nose_y - forehead_y)
    total_face_height = abs(chin_y - forehead_y)

    return upper_face_distance / (total_face_height + 1e-6)
