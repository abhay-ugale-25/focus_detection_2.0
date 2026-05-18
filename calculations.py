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
