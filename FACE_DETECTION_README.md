# Face Detection & Expression-Based Image Windows

## New Features

### Face Detection Mode
Detect faces in real-time using MediaPipe Face Mesh and display detected expressions.

**Facial Expressions Detected:**
- **Smile** - Wide mouth opening
- **Blink** - Both eyes closed (via eye aspect ratio)
- **Mouth Open** - Lips separated beyond threshold
- **Eyebrows Raised** - Raised eyebrow position

### Expression-to-Image Mapping
When a registered person makes a detected expression, automatically open an image window with a configured URL.

**Configuration:** `face_config.json`
```json
{
  "person_id": {
    "smile": "https://example.com/smile.jpg",
    "blink": "https://example.com/blink.jpg",
    "mouth_open": "https://example.com/mouth.jpg",
    "eyebrows_raised": "https://example.com/brows.jpg"
  }
}
```

### Face Registration
Faces are stored with 468-point landmark encodings for identification.

**Database:** `face_database.pkl` (auto-created)

---

## Keyboard Controls

### Detection Mode Toggle
- **`H`** - Hand Gesture Mode (original distortion effects)
- **`F`** - Face Expression Mode (new face detection)
- **`B`** - Both Modes (simultaneous hand + face detection)
- **`M`** - Cycle through modes

### Hand Gesture Controls (Mode: Hand)
- **`N`** - No distortion
- **`R`** - Random fixed distortion
- **`C`** - Continuous random distortion
- **`Q`** - Quit

---

## Display Indicators

### Face Detection Mode
- **Yellow bounding box** around detected faces
- **Person ID/Name** above bounding box
- **Detected expressions** below face (green text)
- **Mode indicator** (top-left: "MODE: FACE")

### Hand Gesture Mode
- **Green circles** at hand landmarks
- **Green lines** connecting hand joints
- **Gesture type** displayed (e.g., "THUMB UP")
- **Applied distortion effect** name

### Both Modes
- Shows both hand landmarks and face detections simultaneously
- Mode indicator displays "BOTH"

---

## Features

### 1. Face Recognition
- Stores face encodings in `face_database.pkl`
- Compares detected faces against database
- Handles multiple unknown faces

### 2. Expression Detection
- **5-frame stability debouncing** per expression per person (prevents false positives)
- Threshold-based detection:
  - Smile: mouth_width > 0.15
  - Blink: eye aspect ratio < 0.15
  - Mouth Open: lip distance > 0.05
  - Eyebrows Raised: eyebrow elevation

### 3. Image Window Manager
- **Automatic image download** and caching (MD5 hash filenames)
- **5-second auto-close** timeout per image window
- **Thread-safe** URL fetching
- **Fallback** handling for failed downloads

### 4. Dual Detection
- Run hand gestures AND face expressions simultaneously
- Independent triggering (hand distortion ≠ facial image windows)
- Configurable via mode toggle

---

## File Structure

```
Normal-camera/
├── main.py                  # Main application (~900 lines)
├── face_config.json         # Expression→URL mapping config
├── face_database.pkl        # Stored face encodings (auto-created)
├── image_cache/             # Downloaded images cache (auto-created)
├── hand_landmarker.task     # MediaPipe hand model
└── README.md                # This file
```

---

## Classes

### `ExpressionDetector`
Detects 4 facial expressions from 468-point face landmarks.

```python
detector = ExpressionDetector(stability_frames=5)
expressions = detector.detect_expressions(landmarks)
stable = detector.is_expression_stable(expressions)
```

### `FaceDatabase`
Manages face registration and recognition.

```python
db = FaceDatabase("face_database.pkl")
db.add_face("user_1", "John", landmarks)
person_id = db.find_face(landmarks, threshold=0.5)
```

### `ImageWindowManager`
Downloads and displays images from URLs with timeout.

```python
manager = ImageWindowManager("image_cache", timeout=5)
image = manager.get_image_for_url(url)
manager.display_image_for_expression("user_1", "smile", url)
manager.update_windows()  # Call each frame to close expired windows
```

---

## Usage Example

### 1. Start Camera
```bash
python main.py
```

### 2. Switch to Face Mode
Press **`F`** to enable face detection.

### 3. Show Your Face
Position your face in the camera frame. Yellow bounding box should appear.

### 4. Configure Expressions
Edit `face_config.json` and add your person ID with expression→URL mappings:

```json
{
  "Unknown_0": {
    "smile": "https://via.placeholder.com/400x300?text=You+Smiled",
    "blink": "https://via.placeholder.com/400x300?text=You+Blinked",
    "mouth_open": "https://via.placeholder.com/400x300?text=Mouth+Open",
    "eyebrows_raised": "https://via.placeholder.com/400x300?text=Brows+Up"
  }
}
```

### 5. Make Expressions
- **Smile widely** → Image window opens (5 sec)
- **Blink both eyes** → Image window opens (5 sec)
- **Open mouth** → Image window opens (5 sec)
- **Raise eyebrows** → Image window opens (5 sec)

### 6. Switch Back to Hand Mode
Press **`H`** to return to hand gesture + distortion effects.

---

## Performance

- **Face Detection:** ~20-30 FPS per face (5 faces max)
- **Hand Detection:** ~60 FPS (with optimization)
- **Both Modes:** ~40-50 FPS (combined load)
- **Image Downloads:** Cached locally (no re-download)
- **FPS Stabilization:** Dynamic delay adjustment ±1ms

---

## Troubleshooting

### Face Not Detected
- Ensure good lighting
- Face must be at least 50x50 pixels
- Check `min_detection_confidence=0.5` in Face Mesh init

### Expressions Not Triggering
- Verify `face_config.json` contains your person ID
- Check console for detected expressions: `print(expressions)`
- Adjust thresholds in `ExpressionDetector.detect_expressions()`

### Images Not Downloading
- Check internet connection
- Verify URL is valid and accessible
- Images cached in `image_cache/` directory
- Check console for download errors

### FPS Drops with Both Modes
- Disable hand detection: Press **`F`**
- Or lower detection frequency (modify `frame_count % N`)

---

## Future Enhancements

- [ ] Face registration UI (gesture to register)
- [ ] Custom expression thresholds per person
- [ ] Sound feedback on expression detection
- [ ] Expression logging/statistics
- [ ] Video recording mode
- [ ] Network streaming for remote display

---

## Dependencies

```
opencv-python
mediapipe
numpy
```

Install via:
```bash
pip install opencv-python mediapipe numpy
```
