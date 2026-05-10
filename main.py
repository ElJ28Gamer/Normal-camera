import cv2 as cv
import random as r
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe import Image as MediapipeImage, ImageFormat
import time
import numpy as np
import os

# Try to download the hand landmarker model if it doesn't exist
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    try:
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Could not download model: {e}")
        print("Using alternative approach without model file...")
        model_path = None

# Initialize MediaPipe Hand detection using new Tasks API
detector = None
try:
    if model_path and os.path.exists(model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        detector = vision.HandLandmarker.create_from_options(options)
        print("Hand detector initialized successfully!")
    else:
        raise Exception("Model not available")
except Exception as e:
    print(f"Error initializing detector: {e}")
    print("Hand detection will be disabled")

# FPS counter variables
pTime = 0
cTime = 0
frame_count = 0

delay = 1
random = 0
random_stop = True
custom_color = cv.COLOR_BGR2BGRA
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("NO")
    exit()

print("Camera initialized. Press 'q' to quit.")
print("Move your hand in front of the camera to test hand detection.")

def get_finger_state(hand_landmarks, debug=False):
    """
    Detects which fingers are raised/down and returns a state
    Returns a value representing the finger configuration:
    0 = Fist (all fingers down)
    1 = Thumb up
    2 = Peace sign (index + middle up)
    3 = Three fingers (index + middle + ring up)
    4 = Four fingers (all but thumb up)
    5 = Open hand (all fingers up)
    """
    if hand_landmarks is None:
        return -1
    
    # Handle different API formats
    landmarks = None
    if isinstance(hand_landmarks, list):
        # New API returns landmarks as a list
        landmarks = hand_landmarks
    elif hasattr(hand_landmarks, 'landmarks'):
        landmarks = hand_landmarks.landmarks
    elif hasattr(hand_landmarks, 'landmark'):
        landmarks = hand_landmarks.landmark
    else:
        return -1
    
    if not landmarks or len(landmarks) < 21:
        return -1
    
    # Check each finger (comparing tip to PIP joint)
    # Thumb (4 > 3)
    thumb_up = landmarks[4].y < landmarks[3].y
    # Index (8 > 6)
    index_up = landmarks[8].y < landmarks[6].y
    # Middle (12 > 10)
    middle_up = landmarks[12].y < landmarks[10].y
    # Ring (16 > 14)
    ring_up = landmarks[16].y < landmarks[14].y
    # Pinky (20 > 18)
    pinky_up = landmarks[20].y < landmarks[18].y
    
    if debug:
        print(f"Fingers - Thumb: {thumb_up}, Index: {index_up}, Middle: {middle_up}, Ring: {ring_up}, Pinky: {pinky_up}")
    
    fingers_up = sum([index_up, middle_up, ring_up, pinky_up])
    
    # Determine hand gesture - check Open hand first
    if all([index_up, middle_up, ring_up, pinky_up, thumb_up]):
        return 5  # Open hand (all 5 fingers)
    elif index_up and middle_up and ring_up and pinky_up:
        return 4  # Four fingers (without thumb)
    elif index_up and middle_up and ring_up:
        return 3  # Three fingers (index + middle + ring)
    elif index_up and middle_up:
        return 2  # Peace sign (index + middle)
    elif thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return 1  # Thumb up (only thumb)
    else:
        return 0  # Fist

def draw_hand_landmarks(frame, hand_landmarks):
    """Draw hand landmarks on frame"""
    h, w, c = frame.shape
    
    if hand_landmarks is None:
        return frame
    
    # Handle different API formats
    landmarks = None
    if isinstance(hand_landmarks, list):
        # New API returns landmarks as a list
        landmarks = hand_landmarks
    elif hasattr(hand_landmarks, 'landmarks'):
        landmarks = hand_landmarks.landmarks
    elif hasattr(hand_landmarks, 'landmark'):
        landmarks = hand_landmarks.landmark
    else:
        return frame
    
    if not landmarks:
        return frame
    
    # Draw circles for each landmark
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv.circle(frame, (x, y), 3, (0, 255, 0), -1)
    
    # Draw connections between landmarks
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Palm lines
    ]
    
    for start, end in connections:
        x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
        x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
        cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return frame

while True:
    ret, frame = cap.read()
    frame_count += 1
    
    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    key = cv.waitKey(delay) & 0xFF

    if key == ord("n") or key == ord("N"):
        random_stop = True
        random = 0
        custom_color = cv.COLOR_BGR2BGRA
    
    elif key == ord("r") or key == ord("R"):
        random_stop = True
        random = r.randint(1, 5)

    elif key == ord("c") or key == ord("C"):
        random_stop = False

    elif key == ord("q") or key == ord("Q"):
        break

    if random_stop == False:
        random = r.randint(1, 5)
    
    if random == 1:
        custom_color = cv.COLOR_RGB2YUV
    
    elif random == 2:
        custom_color = cv.COLOR_RGB2BGR

    elif random == 3:
        custom_color = cv.COLOR_RGB2GRAY

    elif random == 4:
        custom_color = cv.COLOR_RGB2HLS

    elif random == 5:
        custom_color = cv.COLOR_RGB2HSV

    if not ret:
        print("exit")
        break
    
    # Hand detection
    hand_gesture = -1
    hand_detected = False
    if detector:
        try:
            # Convert frame to RGB and create MediaPipe image
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # Ensure the array is C-contiguous and correct type
            rgb_frame = np.ascontiguousarray(rgb_frame)
            mp_image = MediapipeImage(image_format=ImageFormat.SRGB, data=rgb_frame)
            results = detector.detect(mp_image)
            
            # Debug: Print detection status every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: Detected {len(results.hand_landmarks) if results.hand_landmarks else 0} hands")
            
            # Change color based on hand gesture
            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                hand_detected = True
                hand_landmarks = results.hand_landmarks[0]
                debug_mode = (frame_count % 10 == 0)  # Debug every 10 frames
                hand_gesture = get_finger_state(hand_landmarks, debug=debug_mode)
                
                # Change color based on gesture
                if hand_gesture == 1:  # Thumb up
                    custom_color = cv.COLOR_RGB2YUV
                elif hand_gesture == 2:  # Peace sign
                    custom_color = cv.COLOR_RGB2GRAY
                elif hand_gesture == 3:  # Three fingers
                    custom_color = cv.COLOR_RGB2HLS
                elif hand_gesture == 4:  # Four fingers
                    custom_color = cv.COLOR_RGB2HSV
                elif hand_gesture == 5:  # Open hand
                    custom_color = cv.COLOR_RGB2BGR
                elif hand_gesture == 0:  # Fist
                    custom_color = cv.COLOR_BGR2BGRA
        except Exception as e:
            if frame_count % 30 == 0:
                print(f"Detection error: {e}")
                import traceback
                traceback.print_exc()

    default_color = custom_color
    
    color = cv.cvtColor(frame, default_color)
    
    # Draw hand landmarks on frame using already detected results
    if detector:
        try:
            # Re-detect for drawing
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame)
            mp_image = MediapipeImage(image_format=ImageFormat.SRGB, data=rgb_frame)
            results = detector.detect(mp_image)
            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                for hand_landmarks in results.hand_landmarks:
                    color = draw_hand_landmarks(color, hand_landmarks)
        except Exception as e:
            pass
    
    # Display FPS on frame
    fps_text = f"FPS: {int(fps)}"
    cv.putText(color, fps_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display current gesture
    gesture_names = {-1: "No Hand", 0: "Fist", 1: "Thumb Up", 2: "Peace", 3: "Three", 4: "Four", 5: "Open Hand"}
    gesture_text = f"Gesture: {gesture_names.get(hand_gesture, 'Unknown')}"
    cv.putText(color, gesture_text, (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv.imshow("CUSTOM CAMERA", color)

cap.release()
cv.destroyAllWindows()
if detector:
    detector.close()