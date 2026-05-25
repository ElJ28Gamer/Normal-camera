import cv2 as cv
import random as r
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe import Image as MediapipeImage, ImageFormat
import time
import numpy as np
import os
import json
import pickle
import threading
import urllib.request
from pathlib import Path
from collections import deque

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

# Initialize face detection using OpenCV cascade classifiers
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

# Create image cache directory
image_cache_dir = Path("image_cache")
image_cache_dir.mkdir(exist_ok=True)

# FPS counter variables
pTime = 0
cTime = 0
frame_count = 0
target_fps = 60
frame_times = []  # Track last 10 frame times for smooth FPS calculation

delay = 16  # ~60 FPS (1000ms / 60 frames)
random = 0
random_stop = True
distortion_type = 0  # 0 = none, 1-5 = different distortions

# Gesture stability variables
gesture_history = []
stable_gesture = -1
gesture_stability_frames = 5  # Require 5 consistent frames for stable gesture

# Detection mode variables
detection_mode = "hand"  # hand, face, or both
detected_faces = {}  # Store detected face info per frame

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("NO")
    exit()

print("Camera initialized. Press 'q' to quit.")
print("Detection modes: 'H'=hand, 'F'=face, 'B'=both, 'M'=toggle")
print("Move your hand in front of the camera to test detection.")

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
    
    # Wrist position (landmark 0) as reference
    wrist = landmarks[0]
    
    # Check each finger (comparing tip to PIP joint)
    # Thumb detection: requires thumb to be significantly extended AND to the side
    thumb_tip = landmarks[4]
    thumb_pip = landmarks[3]
    thumb_mcp = landmarks[2]
    
    # Require thumb to extend outward (laterally away from hand)
    # Check: thumb tip is far from wrist AND extends to the side
    thumb_distance_from_wrist = np.sqrt((thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2)
    thumb_lateral_extension = abs(thumb_tip.x - wrist.x)  # Horizontal distance (side to side)
    thumb_vertical_extension = abs(thumb_tip.y - wrist.y)  # Vertical distance
    
    # Thumb is up only if: extended significantly (0.25), lateral extension is substantial, 
    # and thumb tip is above MCP joint
    thumb_up = (thumb_distance_from_wrist > 0.25) and (thumb_lateral_extension > 0.15) and (landmarks[4].y < landmarks[2].y)
    
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
    elif index_up and middle_up and ring_up and pinky_up and not thumb_up:
        return 4  # Four fingers (index, middle, ring, pinky - thumb down)
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

def apply_pixelation(frame, block_size=20):
    """Apply pixelation effect to frame"""
    h, w = frame.shape[:2]
    # Downsample
    small = cv.resize(frame, (w // block_size, h // block_size), interpolation=cv.INTER_LINEAR)
    # Upsample
    pixelated = cv.resize(small, (w, h), interpolation=cv.INTER_NEAREST)
    return pixelated

def apply_wave_distortion(frame, strength=10, frequency=0.02):
    """Apply wave distortion effect"""
    h, w = frame.shape[:2]
    output = np.zeros_like(frame)
    
    for y in range(h):
        wave = int(strength * np.sin(y * frequency * 2 * np.pi))
        if 0 <= y < h:
            output[y] = np.roll(frame[y], wave, axis=0)
    
    return output

def apply_spiral_distortion(frame, strength=0.003):
    """Apply spiral distortion effect using vectorized operations"""
    h, w = frame.shape[:2]
    center_x, center_y = w / 2, h / 2
    
    # Create coordinate grids
    yy, xx = np.mgrid[0:h, 0:w]
    dx = xx - center_x
    dy = yy - center_y
    
    # Calculate distance and angle
    distance = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx) + distance * strength
    
    # Calculate source coordinates
    src_x = np.clip((center_x + distance * np.cos(angle)).astype(int), 0, w - 1)
    src_y = np.clip((center_y + distance * np.sin(angle)).astype(int), 0, h - 1)
    
    # Map pixels
    return frame[src_y, src_x]

def apply_lens_distortion(frame, strength=0.0008):
    """Apply barrel/lens distortion effect using vectorized operations"""
    h, w = frame.shape[:2]
    center_x, center_y = w / 2, h / 2
    
    # Create coordinate grids
    yy, xx = np.mgrid[0:h, 0:w]
    dx = (xx - center_x) / center_x
    dy = (yy - center_y) / center_y
    r = np.sqrt(dx**2 + dy**2)
    
    # Barrel distortion formula
    r_distorted = r * (1 + strength * r**2)
    
    # Avoid division by zero
    r_safe = np.where(r > 0, r, 1)
    
    src_x = np.clip((center_x + (dx / r_safe) * r_distorted * center_x).astype(int), 0, w - 1)
    src_y = np.clip((center_y + (dy / r_safe) * r_distorted * center_y).astype(int), 0, h - 1)
    
    # Map pixels
    return frame[src_y, src_x]

def apply_motion_blur(frame, kernel_size=9, angle=45):
    """Apply motion blur effect"""
    # Use simple blur instead of complex kernel for performance
    return cv.blur(frame, (kernel_size, kernel_size))

def apply_distortion(frame, distortion_type):
    """Apply distortion based on type"""
    if distortion_type == 0:
        return frame
    elif distortion_type == 1:
        return apply_pixelation(frame, block_size=15)
    elif distortion_type == 2:
        return apply_wave_distortion(frame, strength=8)
    elif distortion_type == 3:
        return apply_spiral_distortion(frame, strength=0.003)
    elif distortion_type == 4:
        return apply_lens_distortion(frame, strength=0.0008)
    elif distortion_type == 5:
        return apply_motion_blur(frame, kernel_size=21, angle=45)
    else:
        return frame

class ExpressionDetector:
    """Detects facial expressions using simple image-based heuristics"""
    
    def __init__(self, stability_frames=5):
        self.stability_frames = stability_frames
        self.expression_history = {
            'smile': deque(maxlen=stability_frames),
            'blink': deque(maxlen=stability_frames),
            'mouth_open': deque(maxlen=stability_frames),
            'eyebrows_raised': deque(maxlen=stability_frames)
        }
    
    def detect_expressions(self, frame, face_roi, eyes):
        """Detect expressions from face region"""
        if frame is None or face_roi is None:
            return {}
        
        x, y, w, h = face_roi
        face_img = frame[y:y+h, x:x+w]
        
        if face_img.size == 0:
            return {}
        
        # Convert to grayscale for analysis
        gray_face = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
        
        # Simple heuristics (can be improved)
        smile = self._detect_smile(gray_face)
        blink = self._detect_blink(eyes)
        mouth_open = self._detect_mouth_open(gray_face)
        eyebrows_raised = self._detect_eyebrows(gray_face)
        
        return {
            'smile': smile,
            'blink': blink,
            'mouth_open': mouth_open,
            'eyebrows_raised': eyebrows_raised
        }
    
    def _detect_smile(self, gray_face):
        """Detect smile by analyzing mouth area"""
        # Lower half of face where mouth is
        h = gray_face.shape[0]
        mouth_region = gray_face[int(h * 0.6):, :]
        
        # Check for bright mouth area (smile indicator)
        mean_brightness = np.mean(mouth_region)
        return mean_brightness > 120
    
    def _detect_blink(self, eyes):
        """Detect blink when eyes are closed"""
        if not eyes:
            return True
        
        # If no eyes detected in region, likely blinking/eyes closed
        return len(eyes) < 2
    
    def _detect_mouth_open(self, gray_face):
        """Detect open mouth"""
        # Lower region of face
        h = gray_face.shape[0]
        mouth_region = gray_face[int(h * 0.7):, :]
        
        if mouth_region.size == 0:
            return False
        
        # Detect dark pixels (inside mouth)
        dark_pixels = np.sum(mouth_region < 100)
        total_pixels = mouth_region.size
        
        return (dark_pixels / total_pixels) > 0.15
    
    def _detect_eyebrows(self, gray_face):
        """Detect raised eyebrows"""
        # Upper region of face
        eyebrow_region = gray_face[:int(gray_face.shape[0] * 0.3), :]
        
        if eyebrow_region.size == 0:
            return False
        
        # Raised eyebrows show more forehead area
        dark_pixels = np.sum(eyebrow_region < 100)
        return dark_pixels > eyebrow_region.size * 0.2
    
    def is_expression_stable(self, expression_dict):
        """Check if expression is stable (consistent over N frames)"""
        stable_expressions = {}
        for expr_name, detected in expression_dict.items():
            self.expression_history[expr_name].append(detected)
            if len(self.expression_history[expr_name]) == self.stability_frames:
                if all(self.expression_history[expr_name]):
                    stable_expressions[expr_name] = True
        return stable_expressions

class FaceDatabase:
    """Manages face recognition and person identification"""
    
    def __init__(self, db_path="face_database.pkl"):
        self.db_path = db_path
        self.faces = {}
        self.load_database()
    
    def load_database(self):
        """Load face database from file"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                self.faces = pickle.load(f)
    
    def save_database(self):
        """Save face database to file"""
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.faces, f)
    
    def add_face(self, person_id, person_name, face_roi):
        """Add a face to the database (face_roi: [x, y, w, h])"""
        if face_roi is None:
            return False
        
        self.faces[person_id] = {
            'name': person_name,
            'roi': tuple(face_roi),
            'created_at': time.time()
        }
        self.save_database()
        return True
    
    def find_face(self, face_roi, threshold=50):
        """Find matching face in database (simple distance comparison)"""
        if face_roi is None or not self.faces:
            return None
        
        x, y, w, h = face_roi
        current_center = (x + w//2, y + h//2)
        
        min_distance = float('inf')
        best_match = None
        
        for person_id, face_data in self.faces.items():
            stored_roi = face_data['roi']
            stored_x, stored_y, stored_w, stored_h = stored_roi
            stored_center = (stored_x + stored_w//2, stored_y + stored_h//2)
            
            distance = np.sqrt((current_center[0] - stored_center[0])**2 + 
                            (current_center[1] - stored_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                best_match = person_id if distance < threshold else None
        
        return best_match

class ImageWindowManager:
    """Manages downloading and displaying images from URLs"""
    
    def __init__(self, cache_dir="image_cache", timeout=5):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.timeout = timeout
        self.open_windows = {}
        self.window_timers = {}
    
    def get_image_for_url(self, url):
        """Download and cache image from URL"""
        import hashlib
        
        if url is None:
            return None
        
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_path = self.cache_dir / f"{url_hash}.jpg"
        
        # Return cached image if exists
        if cache_path.exists():
            return cv.imread(str(cache_path))
        
        # Download and cache
        try:
            urllib.request.urlretrieve(url, str(cache_path))
            return cv.imread(str(cache_path))
        except Exception as e:
            print(f"Failed to download image from {url}: {e}")
            return None
    
    def display_image_for_expression(self, person_id, expression_name, image_url):
        """Display image in a window for a detected expression"""
        if image_url is None:
            return
        
        window_name = f"{person_id}_{expression_name}"
        
        # Get or download image
        image = self.get_image_for_url(image_url)
        if image is None:
            return
        
        # Resize to reasonable size
        h, w = image.shape[:2]
        max_width, max_height = 400, 300
        if w > max_width or h > max_height:
            scale = min(max_width / w, max_height / h)
            image = cv.resize(image, (int(w * scale), int(h * scale)))
        
        cv.imshow(window_name, image)
        self.open_windows[window_name] = time.time()
        self.window_timers[window_name] = time.time()
    
    def update_windows(self):
        """Close windows that have exceeded timeout"""
        current_time = time.time()
        windows_to_close = []
        
        for window_name, start_time in self.window_timers.items():
            if current_time - start_time > self.timeout:
                windows_to_close.append(window_name)
        
        for window_name in windows_to_close:
            cv.destroyWindow(window_name)
            del self.window_timers[window_name]
            if window_name in self.open_windows:
                del self.open_windows[window_name]

# Initialize face detection components (must be after class definitions)
expression_detector = ExpressionDetector(stability_frames=5)
face_database = FaceDatabase("face_database.pkl")
image_manager = ImageWindowManager("image_cache", timeout=5)

# Load expression-to-URL config
face_config = {}
if os.path.exists("face_config.json"):
    with open("face_config.json", "r") as f:
        face_config = json.load(f)

def update_stable_gesture(new_gesture, gesture_history, stable_gesture, stability_frames=5):
    """Update gesture only when stable across multiple frames"""
    # Add new gesture to history
    gesture_history.append(new_gesture)
    
    # Keep only the last N frames
    if len(gesture_history) > stability_frames:
        gesture_history.pop(0)
    
    # If we have enough history and all recent gestures are the same, update stable gesture
    if len(gesture_history) == stability_frames and len(set(gesture_history)) == 1:
        stable_gesture = new_gesture
    
    return stable_gesture, gesture_history

while True:
    ret, frame = cap.read()
    frame_count += 1
    
    # Calculate FPS with smoothing
    cTime = time.time()
    frame_delta = cTime - pTime
    pTime = cTime
    
    # Track frame times for smoothing
    if frame_delta > 0:
        frame_times.append(frame_delta)
    if len(frame_times) > 10:
        frame_times.pop(0)
    
    # Calculate smoothed FPS
    if frame_times:
        avg_frame_time = sum(frame_times) / len(frame_times)
        fps = 1 / avg_frame_time if avg_frame_time > 0 else 0
    else:
        fps = 0
    
    # Dynamically adjust delay to maintain 60 FPS
    if fps > 0:
        # If fps is too high, increase delay; if too low, decrease
        if fps > target_fps:
            delay = max(10, delay + 1)
        elif fps < target_fps * 0.9:
            delay = max(5, delay - 1)

    key = cv.waitKey(delay) & 0xFF

    if key == ord("n") or key == ord("N"):
        random_stop = True
        random = 0
        distortion_type = 0
    
    elif key == ord("r") or key == ord("R"):
        random_stop = True
        random = r.randint(1, 5)

    elif key == ord("c") or key == ord("C"):
        random_stop = False
    
    elif key == ord("h") or key == ord("H"):
        detection_mode = "hand"
        print("Mode: HAND DETECTION")
    
    elif key == ord("f") or key == ord("F"):
        detection_mode = "face"
        print("Mode: FACE DETECTION")
    
    elif key == ord("b") or key == ord("B"):
        detection_mode = "both"
        print("Mode: HAND + FACE DETECTION")
    
    elif key == ord("m") or key == ord("M"):
        modes = ["hand", "face", "both"]
        current_idx = modes.index(detection_mode)
        detection_mode = modes[(current_idx + 1) % len(modes)]
        print(f"Mode: {detection_mode.upper()}")

    elif key == ord("q") or key == ord("Q"):
        break

    if random_stop == False:
        random = r.randint(1, 5)
    
    if random == 1:
        distortion_type = 1
    
    elif random == 2:
        distortion_type = 2

    elif random == 3:
        distortion_type = 3

    elif random == 4:
        distortion_type = 4

    elif random == 5:
        distortion_type = 5

    if not ret:
        print("exit")
        break
    
    # Hand detection (every frame)
    hand_gesture = -1
    hand_detected = False
    hand_landmarks_list = []
    
    if detector and (detection_mode == "hand" or detection_mode == "both"):
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
            
            # Change distortion based on hand gesture
            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                hand_detected = True
                hand_landmarks = results.hand_landmarks[0]
                hand_landmarks_list = results.hand_landmarks  # Store for drawing
                debug_mode = (frame_count % 10 == 0)  # Debug every 10 frames
                hand_gesture = get_finger_state(hand_landmarks, debug=debug_mode)
                
                # Update stable gesture with debouncing
                stable_gesture, gesture_history = update_stable_gesture(hand_gesture, gesture_history, stable_gesture, gesture_stability_frames)
                
                # Change distortion based on stable gesture
                if stable_gesture == 1:  # Thumb up
                    distortion_type = 1  # Pixelation
                elif stable_gesture == 2:  # Peace sign
                    distortion_type = 2  # Wave
                elif stable_gesture == 3:  # Three fingers
                    distortion_type = 3  # Spiral
                elif stable_gesture == 4:  # Four fingers
                    distortion_type = 4  # Lens
                elif stable_gesture == 5:  # Open hand
                    distortion_type = 5  # Motion blur
                elif stable_gesture == 0:  # Fist
                    distortion_type = 0  # No distortion
            else:
                # Reset gesture history if no hand detected
                gesture_history = []
                stable_gesture = -1
        except Exception as e:
            if frame_count % 30 == 0:
                print(f"Detection error: {e}")
    
    # Face detection (every frame)
    detected_faces = {}
    if face_cascade and (detection_mode == "face" or detection_mode == "both"):
        try:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for idx, (x, y, w, h) in enumerate(faces):
                face_roi = (x, y, w, h)
                
                # Detect eyes in face region
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                # Detect expressions
                expressions = expression_detector.detect_expressions(frame, face_roi, eyes)
                stable_expressions = expression_detector.is_expression_stable(expressions)
                
                # Try to identify person
                person_id = face_database.find_face(face_roi)
                if person_id is None:
                    person_id = f"Unknown_{idx}"
                
                detected_faces[person_id] = {
                    'roi': face_roi,
                    'eyes': eyes,
                    'expressions': expressions,
                    'stable_expressions': stable_expressions
                }
                
                # Check for triggered expressions and display images
                config_key = str(person_id)
                if config_key in face_config:
                    person_config = face_config[config_key]
                    for expr_name, detected in stable_expressions.items():
                        if detected and expr_name in person_config:
                            image_url = person_config[expr_name]
                            image_manager.display_image_for_expression(person_id, expr_name, image_url)
        except Exception as e:
            if frame_count % 30 == 0:
                print(f"Face detection error: {e}")
    
    # Update image window timeouts
    image_manager.update_windows()

    default_color = distortion_type
    
    color = apply_distortion(frame, default_color)
    
    # Draw hand landmarks on distorted frame (reuse detection results)
    if hand_landmarks_list:
        for hand_landmarks in hand_landmarks_list:
            color = draw_hand_landmarks(color, hand_landmarks)
    
    # Draw face bounding boxes and detected expressions
    if detected_faces:
        for person_id, face_data in detected_faces.items():
            x, y, w, h = face_data['roi']
            
            # Draw bounding box
            cv.rectangle(color, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # Draw person ID/name
            cv.putText(color, str(person_id), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw detected expressions
            expr_text = ", ".join([name for name, detected in face_data['stable_expressions'].items() if detected])
            if expr_text:
                cv.putText(color, f"Expr: {expr_text}", (x, y + h + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display FPS on frame
    fps_text = f"FPS: {int(fps)}"
    cv.putText(color, fps_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display detection mode
    mode_text = f"Mode: {detection_mode.upper()}"
    cv.putText(color, mode_text, (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display current gesture (only in hand mode)
    if detection_mode == "hand" or detection_mode == "both":
        gesture_names = {-1: "No Hand", 0: "Fist", 1: "Thumb Up", 2: "Peace", 3: "Three", 4: "Four", 5: "Open Hand"}
        gesture_text = f"Gesture: {gesture_names.get(stable_gesture, 'Unknown')}"
        cv.putText(color, gesture_text, (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv.imshow("CUSTOM CAMERA", color)

cap.release()
cv.destroyAllWindows()
if detector:
    detector.close()