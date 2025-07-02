import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class MediaPipeObjectDetector:
    def __init__(self, model_path='Models/efficientdet_lite0.tflite', score_threshold=0.5):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(base_options=base_options,
                                              score_threshold=score_threshold)
        self.detector = vision.ObjectDetector.create_from_options(options)
        self.frame_count = 0

    def detect_objects(self, frame):
        # Convert the BGR image to RGB and to MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)
        detections = []
        if results.detections:
            ih, iw, _ = frame.shape
            for detection in results.detections:
                bboxC = detection.bounding_box
                x = int(bboxC.origin_x)
                y = int(bboxC.origin_y)
                w = int(bboxC.width)
                h = int(bboxC.height)
                cx = x + w // 2
                cy = y + h // 2
                score = detection.categories[0].score if detection.categories else 0.0
                label = detection.categories[0].category_name if detection.categories else "object"
                detections.append({
                    "bbox": (x, y, x + w, y + h),
                    "center": (cx, cy),
                    "score": score,
                    "label": label
                })
        return detections

    def draw_detections(self, frame, detections):
        annotated_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["center"]
            label = det.get("label", "object")
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
            label_text = f"{label} ({cx}, {cy}) {det['score']:.2f}"
            cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return annotated_frame

    def process_video(self, video_path, skip_frames=1, max_frames=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        print(f"Processing video: {video_path}")
        print("Press 'q' to quit, 'p' to pause")

        frame_skip_counter = 0
        processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            frame_skip_counter += 1

            if frame_skip_counter < skip_frames:
                continue
            frame_skip_counter = 0
            processed_frames += 1

            if max_frames and processed_frames > max_frames:
                break

            detections = self.detect_objects(frame)
            print(f"Frame {self.frame_count}:")
            for i, det in enumerate(detections):
                cx, cy = det["center"]
                label = det.get("label", "object")
                print(f"  Object {i+1}: {label} Center at ({cx}, {cy}), Score: {det['score']:.2f}")

            annotated_frame = self.draw_detections(frame, detections)
            cv2.imshow("MediaPipe Object Detection", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                print("Paused. Press any key to continue...")
                cv2.waitKey(0)

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessing complete! Processed {processed_frames} frames.")

def main():
    video_path = 'Videos/AngleCheck.MOV'  # Change as needed
    detector = MediaPipeObjectDetector()
    detector.process_video(
        video_path=video_path,
        skip_frames=1,   # Process every frame
        max_frames=200    # Limit for testing
    )

if __name__ == "__main__":
    main()
