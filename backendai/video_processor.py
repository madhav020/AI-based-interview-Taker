import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

# Initialize MediaPipe Face Mesh with higher confidence threshold
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define facial landmark indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LEFT_EYEBROW = [276, 283, 282, 295, 285]
RIGHT_EYEBROW = [46, 53, 52, 65, 55]


def get_aspect_ratio(landmarks, points):
    """Calculate the aspect ratio of facial features"""
    points_np = np.array([(landmarks.landmark[point].x, landmarks.landmark[point].y) for point in points])

    if len(points_np) > 4:  # For eyes and mouth
        hull = cv2.convexHull(np.float32(points_np))
        rect = cv2.minAreaRect(hull)
        width = rect[1][0]
        height = rect[1][1]
        return height / width if width > 0 else 0
    else:  # For eyebrows
        return distance.euclidean(points_np[0], points_np[-1])


def detect_blinking(eye_ratio):
    """Detect if eyes are blinking based on aspect ratio"""
    return eye_ratio < 0.2


def calculate_eye_direction(landmarks, eye_points):
    """Calculate eye gaze direction more accurately"""
    eye_region = np.array([(landmarks.landmark[point].x, landmarks.landmark[point].y, landmarks.landmark[point].z)
                           for point in eye_points])

    # Calculate eye center
    eye_center = np.mean(eye_region, axis=0)

    # Calculate gaze vector (from eye center to pupil)
    pupil_index = eye_points[len(eye_points) // 2]  # Approximate pupil position
    pupil = np.array([landmarks.landmark[pupil_index].x,
                      landmarks.landmark[pupil_index].y,
                      landmarks.landmark[pupil_index].z])

    gaze_vector = pupil - eye_center

    # Calculate angle with forward direction
    forward_vector = np.array([0, 0, 1])
    angle = np.arccos(np.dot(gaze_vector, forward_vector) /
                      (np.linalg.norm(gaze_vector) * np.linalg.norm(forward_vector)))

    return angle


def detect_micro_expressions(landmarks):
    """Detect micro expressions for better confidence/nervousness assessment"""
    # Calculate asymmetry in facial features
    left_eye_ratio = get_aspect_ratio(landmarks, LEFT_EYE)
    right_eye_ratio = get_aspect_ratio(landmarks, RIGHT_EYE)
    eye_asymmetry = abs(left_eye_ratio - right_eye_ratio)

    # Analyze mouth tension
    lips_ratio = get_aspect_ratio(landmarks, LIPS)

    # Analyze eyebrow position
    left_brow = get_aspect_ratio(landmarks, LEFT_EYEBROW)
    right_brow = get_aspect_ratio(landmarks, RIGHT_EYEBROW)
    brow_asymmetry = abs(left_brow - right_brow)

    return {
        "eye_asymmetry": eye_asymmetry,
        "lips_tension": lips_ratio,
        "brow_asymmetry": brow_asymmetry
    }


def analyze_frame(frame):
    """Enhanced frame analysis with more accurate metrics"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0]

    # Enhanced eye contact detection
    left_eye_direction = calculate_eye_direction(landmarks, LEFT_EYE)
    right_eye_direction = calculate_eye_direction(landmarks, RIGHT_EYE)

    # Average gaze direction
    avg_eye_direction = (left_eye_direction + right_eye_direction) / 2
    eye_contact = 1 - (avg_eye_direction / np.pi)  # Normalize to 0-1

    # Detect micro expressions
    micro_expressions = detect_micro_expressions(landmarks)

    # Calculate confidence based on multiple factors
    confidence = 1.0 - (
            micro_expressions["eye_asymmetry"] * 0.3 +
            micro_expressions["brow_asymmetry"] * 0.3 +
            (1 - micro_expressions["lips_tension"]) * 0.4
    )

    # Calculate nervousness based on micro movements
    nervousness = (
            micro_expressions["eye_asymmetry"] * 0.4 +
            micro_expressions["brow_asymmetry"] * 0.3 +
            (1 - micro_expressions["lips_tension"]) * 0.3
    )

    return {
        "confidence": confidence,
        "nervousness": nervousness,
        "eye_contact": eye_contact,
        "micro_expressions": micro_expressions
    }


def calculate_score(metrics):
    """Enhanced scoring system"""
    confidence_weight = 0.4
    eye_contact_weight = 0.4
    nervousness_weight = 0.2

    score = (
            (metrics["confidence"] * confidence_weight * 10) +
            (metrics["eye_contact"] * eye_contact_weight * 10) -
            (metrics["nervousness"] * nervousness_weight * 10)
    )

    return min(max(score, 0), 10)


def generate_detailed_feedback(metrics):
    """Generate more detailed and accurate feedback"""
    feedback = []

    # Eye Contact Analysis
    if metrics["eye_contact"] > 0.7:
        feedback.append("You maintained excellent eye contact throughout the interview.")
    elif metrics["eye_contact"] > 0.4:
        feedback.append("Your eye contact was good but could be more consistent.")
    elif metrics["eye_contact"] > 0.2:
        feedback.append("Try to maintain more regular eye contact with the interviewer.")
    else:
        feedback.append("Work on maintaining better eye contact. Looking at the camera helps establish connection.")

    # Confidence Analysis
    if metrics["confidence"] > 0.7:
        feedback.append("Your body language shows strong confidence and composure.")
    elif metrics["confidence"] > 0.4:
        feedback.append("You appear fairly confident, though there's room for improvement in your body language.")
    elif metrics["confidence"] > 0.2:
        feedback.append("Try to project more confidence through your facial expressions and posture.")
    else:
        feedback.append("Work on appearing more confident by maintaining a more relaxed facial expression.")

    # Nervousness Analysis
    if metrics["nervousness"] < 0.2:
        feedback.append("You appeared very calm and composed.")
    elif metrics["nervousness"] < 0.3:
        feedback.append("You showed good composure with minimal signs of nervousness.")
    elif metrics["nervousness"] < 0.5:
        feedback.append("Some signs of nervousness were visible. Try deep breathing exercises before interviews.")
    else:
        feedback.append("You appeared notably nervous. Practice more mock interviews to build confidence.")

    return " ".join(feedback)


def process_video(video_path):
    """Process video with enhanced analysis"""
    cap = cv2.VideoCapture(video_path)
    frame_results = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 3rd frame to improve performance while maintaining accuracy
        if frame_count % 3 == 0:
            analysis = analyze_frame(frame)
            if analysis:
                frame_results.append(analysis)

        frame_count += 1

    cap.release()

    if not frame_results:
        return {
            "score": 0,
            "feedback": "No face detected in the video. Please ensure proper lighting and camera positioning."
        }

    # Calculate overall metrics
    overall_metrics = {
        "confidence": np.mean([r["confidence"] for r in frame_results]),
        "nervousness": np.mean([r["nervousness"] for r in frame_results]),
        "eye_contact": np.mean([r["eye_contact"] for r in frame_results])
    }

    score = calculate_score(overall_metrics)
    feedback = generate_detailed_feedback(overall_metrics)

    return {
        "score": round(score, 1),
        "feedback": feedback,
        "metrics": overall_metrics
    }


if __name__ == "__main__":
    video_path = "your_video.mp4"
    results = process_video(video_path)
    print("\nDetailed Analysis:")
    print(f"Score: {results['score']}/10")
    print(f"Feedback: {results['feedback']}")
    print("\nDetailed Metrics:")
    print(f"Confidence: {results['metrics']['confidence']:.2f}")
    print(f"Eye Contact: {results['metrics']['eye_contact']:.2f}")
    print(f"Nervousness: {results['metrics']['nervousness']:.2f}")