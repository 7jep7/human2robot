# MediaPipe Object Detection Script

This script uses Google's [MediaPipe](https://developers.google.com/mediapipe) Object Detection API to detect and localize objects in each frame of a video. It prints the (x, y) coordinates of detected objects and displays the annotated video with bounding boxes and labels.

## How It Works

- Loads a TensorFlow Lite object detection model (EfficientDet Lite0 by default).
- Processes each frame of the specified video file.
- Detects objects, prints their center coordinates and confidence scores.
- Draws bounding boxes and labels on the video frames for visualization.

## Setup

1. **Install Dependencies:**
   ```bash
   pip install mediapipe opencv-python numpy
   ```

2. **Download the Model:**
   - Download the `efficientdet_lite0.tflite` model from the [MediaPipe Model Zoo](https://developers.google.com/mediapipe/solutions/vision/object_detector#models).
   - Save the model file to:
     ```
     Models/efficientdet_lite0.tflite
     ```
   - If you use a different model, update the `model_path` argument in the script accordingly.

3. **Set the Video File Path:**
   - By default, the script uses:
     ```python
     video_path = 'Videos/AngleCheck.MOV'
     ```
   - Change this path in the `main()` function to point to your own video file.

## Usage

Run the script:
```bash
python ObjectTrack.py
```

- Press `q` to quit the video window.
- Press `p` to pause and any key to resume.

## Notes

- The script processes up to 200 frames by default for testing. Adjust `max_frames` as needed.
- Make sure your video file and model file paths are correct.
