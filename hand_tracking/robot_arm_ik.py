#!/usr/bin/env python3.10
"""
Robot Arm Inverse Kinematics with Hand Tracking Integration
==========================================================

This script integrates with the hand tracking system to:
1. Extract thumb and index finger positions from video
2. Perform inverse kinematics for a 5DOF planar arm + 1DOF gripper
3. Generate plots of joint angles over time
4. Create an overlaid video showing the robot end-effector

Author: Generated for human2robot project
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple, Optional
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class RobotArmIK:
    """5DOF planar robot arm with inverse kinematics"""
    
    def __init__(self, link_lengths: List[float] = None):
        """
        Initialize robot arm with link lengths
        
        Args:
            link_lengths: List of 5 link lengths in meters [default: [0.2, 0.2, 0.2, 0.2, 0.2]]
        """
        self.link_lengths = link_lengths or [0.2, 0.2, 0.2, 0.2, 0.2]
        self.num_joints = len(self.link_lengths)
        
    def forward_kinematics(self, joint_angles: List[float]) -> Tuple[float, float]:
        """
        Calculate end-effector position from joint angles
        
        Args:
            joint_angles: List of joint angles in radians
            
        Returns:
            (x, y) position of end-effector
        """
        x = 0.0
        y = 0.0
        cumulative_angle = 0.0
        
        for i, (angle, length) in enumerate(zip(joint_angles, self.link_lengths)):
            cumulative_angle += angle
            x += length * np.cos(cumulative_angle)
            y += length * np.sin(cumulative_angle)
            
        return x, y
    
    def inverse_kinematics_numerical(self, target_x: float, target_y: float, 
                                   initial_guess: List[float] = None) -> List[float]:
        """
        Improved numerical inverse kinematics with better continuity
        
        Args:
            target_x: Target x position
            target_y: Target y position
            initial_guess: Initial joint angles [default: all zeros]
            
        Returns:
            List of joint angles in radians
        """
        if initial_guess is None:
            initial_guess = [0.0] * self.num_joints
            
        angles = np.array(initial_guess, dtype=float)
        learning_rate = 0.05  # Reduced for smoother convergence
        max_iterations = 500  # Reduced iterations for real-time performance
        tolerance = 1e-3  # Slightly relaxed tolerance
        
        # Add joint limits for realistic motion
        joint_limits = [(-np.pi, np.pi)] * self.num_joints  # ±180 degrees
        
        for iteration in range(max_iterations):
            # Current position
            current_x, current_y = self.forward_kinematics(angles)
            
            # Error
            error_x = target_x - current_x
            error_y = target_y - current_y
            error_magnitude = np.sqrt(error_x**2 + error_y**2)
            
            if error_magnitude < tolerance:
                break
                
            # Compute Jacobian numerically
            jacobian = np.zeros((2, self.num_joints))
            delta = 1e-5
            
            for i in range(self.num_joints):
                angles_plus = angles.copy()
                angles_plus[i] += delta
                x_plus, y_plus = self.forward_kinematics(angles_plus)
                
                angles_minus = angles.copy()
                angles_minus[i] -= delta
                x_minus, y_minus = self.forward_kinematics(angles_minus)
                
                jacobian[0, i] = (x_plus - x_minus) / (2 * delta)
                jacobian[1, i] = (y_plus - y_minus) / (2 * delta)
            
            # Damped least squares (more stable than pseudo-inverse)
            damping = 0.01
            jacobian_T = jacobian.T
            try:
                A = jacobian @ jacobian_T + damping * np.eye(2)
                error_vector = np.array([error_x, error_y])
                delta_angles = jacobian_T @ np.linalg.solve(A, error_vector)
                
                # Adaptive learning rate based on error magnitude
                adaptive_lr = learning_rate * min(1.0, error_magnitude / 0.1)
                angles += adaptive_lr * delta_angles
                
                # Apply joint limits
                for i, (min_angle, max_angle) in enumerate(joint_limits):
                    angles[i] = np.clip(angles[i], min_angle, max_angle)
                    
            except np.linalg.LinAlgError:
                # If singular, use small random perturbation
                angles += np.random.normal(0, 0.001, self.num_joints)
        
        return angles.tolist()


class HandTrackingRobotIntegration:
    """Integration class for hand tracking and robot arm control"""
    
    def __init__(self, model_path: str, video_path: str):
        """
        Initialize hand tracking and robot arm
        
        Args:
            model_path: Path to MediaPipe hand landmark model
            video_path: Path to input video
        """
        self.model_path = model_path
        self.video_path = video_path
        self.robot_arm = RobotArmIK()
        
        # Setup MediaPipe
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        self.options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.1,
            min_hand_presence_confidence=0.1,
            min_tracking_confidence=0.3
        )
        
        # Data storage
        self.timestamps = []
        self.thumb_positions = []
        self.index_positions = []
        self.joint_angles = []
        self.gripper_states = []
        self.gripper_orientations = []  # Track gripper orientation
        
    def smooth_positions(self, positions: List[Tuple[float, float]], window_size: int = 5) -> List[Tuple[float, float]]:
        """
        Apply moving average smoothing to position data
        
        Args:
            positions: List of (x, y) positions
            window_size: Size of smoothing window
            
        Returns:
            Smoothed positions
        """
        if len(positions) < window_size:
            return positions
            
        smoothed = []
        for i in range(len(positions)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(positions), i + window_size // 2 + 1)
            
            window_positions = positions[start_idx:end_idx]
            avg_x = sum(pos[0] for pos in window_positions) / len(window_positions)
            avg_y = sum(pos[1] for pos in window_positions) / len(window_positions)
            
            smoothed.append((avg_x, avg_y))
            
        return smoothed
    
    def smooth_values(self, values: List[float], window_size: int = 5) -> List[float]:
        """
        Apply moving average smoothing to scalar values
        
        Args:
            values: List of scalar values
            window_size: Size of smoothing window
            
        Returns:
            Smoothed values
        """
        if len(values) < window_size:
            return values
            
        smoothed = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(values), i + window_size // 2 + 1)
            
            window_values = values[start_idx:end_idx]
            avg_value = sum(window_values) / len(window_values)
            smoothed.append(avg_value)
            
        return smoothed
        
    def extract_hand_data(self) -> bool:
        """
        Extract hand tracking data from video
        
        Returns:
            True if successful, False otherwise
        """
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return False
            
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_count = 0
        
        HandLandmarker = mp.tasks.vision.HandLandmarker
        
        with HandLandmarker.create_from_options(self.options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Calculate timestamp
                timestamp_ms = int(frame_count * 1000 / fps) if fps > 0 else frame_count * 33
                
                # Detect hand landmarks
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                
                # Store timestamp
                self.timestamps.append(frame_count / fps if fps > 0 else frame_count * 0.033)
                
                if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
                    # Get first hand
                    hand = detection_result.hand_landmarks[0]
                    
                    # Extract more landmarks for better gripper control
                    wrist = hand[0]         # Wrist
                    thumb_tip = hand[4]     # Thumb tip
                    thumb_ip = hand[3]      # Thumb IP joint
                    index_tip = hand[8]     # Index finger tip
                    index_pip = hand[6]     # Index PIP joint
                    middle_tip = hand[12]   # Middle finger tip
                    
                    # Convert normalized coordinates to workspace coordinates
                    # Assuming workspace is 1m x 1m, centered at origin
                    thumb_x = (thumb_tip.x - 0.5) * 1.0  # Scale to workspace
                    thumb_y = (0.5 - thumb_tip.y) * 1.0  # Flip Y and scale
                    
                    index_x = (index_tip.x - 0.5) * 1.0
                    index_y = (0.5 - index_tip.y) * 1.0
                    
                    self.thumb_positions.append((thumb_x, thumb_y))
                    self.index_positions.append((index_x, index_y))
                    
                    # Improved gripper state calculation - much more sensitive
                    # Calculate distance between thumb and index tips
                    thumb_index_dist = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
                    
                    # More sensitive gripper detection - scale the distance more aggressively
                    # In video coordinates, even small movements should open the gripper
                    base_distance = 0.05  # Minimum distance for closed gripper
                    max_distance = 0.25   # Distance for fully open gripper
                    
                    # Normalize distance to 0-1 range
                    normalized_dist = max(0, min(1, (thumb_index_dist - base_distance) / (max_distance - base_distance)))
                    
                    # Apply exponential scaling to make it more sensitive to small movements
                    gripper_opening = normalized_dist ** 0.5  # Square root for more sensitivity
                    gripper_angle = gripper_opening * np.pi/3  # Max 60 degrees opening
                    
                    self.gripper_states.append(gripper_angle)
                    
                    # Fixed gripper orientation - point horizontally right for demo
                    # In a real robot, this would be calculated from wrist/forearm angles
                    orientation = 0.0  # Always point right (positive X direction)
                    
                    self.gripper_orientations.append(orientation)
                    
                    # Debug gripper detection every 10 frames
                    if frame_count % 10 == 0:
                        print(f"Frame {frame_count}: Distance={thumb_index_dist:.3f}, "
                              f"Normalized={normalized_dist:.3f}, Gripper={gripper_angle:.3f}rad")
                    
                    # Perform inverse kinematics for thumb position
                    prev_angles = self.joint_angles[-1] if self.joint_angles else None
                    joint_angles = self.robot_arm.inverse_kinematics_numerical(
                        thumb_x, thumb_y, prev_angles
                    )
                    self.joint_angles.append(joint_angles)
                    
                else:
                    # No hand detected - use previous values or defaults
                    if self.thumb_positions:
                        self.thumb_positions.append(self.thumb_positions[-1])
                        self.index_positions.append(self.index_positions[-1])
                        self.joint_angles.append(self.joint_angles[-1])
                        self.gripper_states.append(self.gripper_states[-1])
                        self.gripper_orientations.append(self.gripper_orientations[-1])
                    else:
                        self.thumb_positions.append((0.0, 0.0))
                        self.index_positions.append((0.0, 0.0))
                        self.joint_angles.append([0.0] * 5)
                        self.gripper_states.append(0.0)
                        self.gripper_orientations.append(0.0)
                
                frame_count += 1
                
                # Debug output every 30 frames
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames...")
        
        cap.release()
        print(f"Extracted data from {len(self.timestamps)} frames")
        
        # Apply smoothing to positions
        if self.thumb_positions:
            print("Applying position smoothing...")
            self.thumb_positions = self.smooth_positions(self.thumb_positions)
            self.index_positions = self.smooth_positions(self.index_positions)
            
            # Smooth gripper states and orientations
            self.gripper_states = self.smooth_values(self.gripper_states)
            self.gripper_orientations = self.smooth_values(self.gripper_orientations)
        
        return True
    
    def plot_joint_angles(self, save_path: str = "joint_angles.png"):
        """
        Plot joint angles over time - single plot with all 6 joints (presentation-ready)
        
        Args:
            save_path: Path to save the plot
        """
        if not self.joint_angles:
            print("No joint angle data to plot")
            return
            
        # Set presentation-ready style
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))  # Larger figure
        
        # Convert data to numpy arrays for easier plotting
        times = np.array(self.timestamps)
        angles = np.array(self.joint_angles)
        grippers = np.array(self.gripper_states)
        
        # SO-101 joint names (from base to end-effector)
        joint_names = ['Base', 'Shoulder', 'Elbow', 'Forearm', 'Wrist', 'Gripper']
        # High contrast colors for presentation
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#E67E22']
        
        # Plot all joint angles with thick lines
        for i in range(5):
            ax.plot(times, angles[:, i], color=colors[i], linewidth=4.5, 
                   label=joint_names[i], alpha=0.9, solid_capstyle='round')
        
        # Plot gripper with thick line
        ax.plot(times, grippers, color=colors[5], linewidth=4.5, 
               label=joint_names[5], alpha=0.9, solid_capstyle='round')
        
        # Large, bold title
        ax.set_title('SO-101 Robot Arm Joint Angles Over Time', 
                    fontsize=28, fontweight='bold', pad=30)
        
        # Large axis labels
        ax.set_xlabel('Time (s)', fontsize=22, fontweight='bold', labelpad=15)
        ax.set_ylabel('Joint Angle (radians)', fontsize=22, fontweight='bold', labelpad=15)
        
        # Thick grid lines
        ax.grid(True, alpha=0.4, linewidth=1.5)
        
        # Large legend with thicker lines
        legend = ax.legend(loc='upper right', fontsize=18, frameon=True, 
                          fancybox=True, shadow=True, framealpha=0.9)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(2)
        
        # Make legend lines thicker
        for line in legend.get_lines():
            line.set_linewidth(6)
        
        # Improve plot aesthetics
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        # Larger tick labels
        ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=8)
        ax.tick_params(axis='both', which='minor', width=1, length=4)
        
        # Add minor ticks for better readability
        ax.minorticks_on()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"Presentation-ready joint angles plot saved to {save_path}")
    
    def draw_so101_gripper(self, frame, center_x: int, center_y: int, gripper_angle: float, 
                          orientation: float = 0.0, scale: float = 1.0):
        """
        Draw SO-101 style crocodile gripper end-effector with proper orientation
        
        Args:
            frame: OpenCV frame to draw on
            center_x: Center X position in pixels
            center_y: Center Y position in pixels
            gripper_angle: Gripper opening angle in radians
            orientation: Gripper pointing direction in radians (0 = right, π/2 = up, etc.)
            scale: Scale factor for gripper size
        """
        # Gripper parameters
        base_length = int(30 * scale)
        max_jaw_length = int(50 * scale)
        jaw_width = int(8 * scale)
        
        # Calculate jaw opening based on gripper angle
        jaw_opening = min(gripper_angle, np.pi/2)  # Max 90 degrees opening
        
        # Rotation matrix for orientation
        cos_orient = np.cos(orientation)
        sin_orient = np.sin(orientation)
        
        def rotate_point(x, y):
            """Rotate point around origin by orientation angle"""
            rot_x = x * cos_orient - y * sin_orient
            rot_y = x * sin_orient + y * cos_orient
            return int(center_x + rot_x), int(center_y + rot_y)
        
        # Base triangle (gripper body) - points forward in direction of orientation
        base_pts = np.array([
            rotate_point(-base_length//2, -jaw_width//2),
            rotate_point(-base_length//2, jaw_width//2),
            rotate_point(0, 0)  # Point at center
        ], np.int32)
        
        # Upper jaw - starts horizontal and rotates up when opening
        upper_angle = -jaw_opening / 2
        upper_tip_x_local = max_jaw_length * np.cos(upper_angle)
        upper_tip_y_local = max_jaw_length * np.sin(upper_angle)
        
        upper_jaw_pts = np.array([
            rotate_point(0, -jaw_width//3),
            rotate_point(0, 0),
            rotate_point(upper_tip_x_local, upper_tip_y_local)
        ], np.int32)
        
        # Lower jaw - starts horizontal and rotates down when opening
        lower_angle = jaw_opening / 2
        lower_tip_x_local = max_jaw_length * np.cos(lower_angle)
        lower_tip_y_local = max_jaw_length * np.sin(lower_angle)
        
        lower_jaw_pts = np.array([
            rotate_point(0, 0),
            rotate_point(0, jaw_width//3),
            rotate_point(lower_tip_x_local, lower_tip_y_local)
        ], np.int32)
        
        # Draw gripper components with better colors
        # Base (dark green)
        cv.fillPoly(frame, [base_pts], (0, 150, 0))
        cv.polylines(frame, [base_pts], True, (0, 100, 0), 3)
        
        # Upper jaw (bright green)
        cv.fillPoly(frame, [upper_jaw_pts], (50, 255, 50))
        cv.polylines(frame, [upper_jaw_pts], True, (0, 200, 0), 3)
        
        # Lower jaw (bright green)
        cv.fillPoly(frame, [lower_jaw_pts], (50, 255, 50))
        cv.polylines(frame, [lower_jaw_pts], True, (0, 200, 0), 3)
        
        # Add gripper joint/pivot point
        cv.circle(frame, (center_x, center_y), int(6 * scale), (0, 100, 0), -1)
        cv.circle(frame, (center_x, center_y), int(6 * scale), (255, 255, 255), 2)
        
        # Add teeth/serrations when mostly closed
        if jaw_opening < np.pi/4:  # Show teeth when less than 45 degrees open
            # Upper jaw teeth
            for i in range(2):
                tooth_base_x = 15 + i * 15
                tooth_x1, tooth_y1 = rotate_point(tooth_base_x, -2)
                tooth_x2, tooth_y2 = rotate_point(tooth_base_x, 2)
                cv.line(frame, (tooth_x1, tooth_y1), (tooth_x2, tooth_y2), (255, 255, 255), 2)
            
            # Lower jaw teeth
            for i in range(2):
                tooth_base_x = 15 + i * 15
                tooth_x1, tooth_y1 = rotate_point(tooth_base_x, -2)
                tooth_x2, tooth_y2 = rotate_point(tooth_base_x, 2)
                cv.line(frame, (tooth_x1, tooth_y1), (tooth_x2, tooth_y2), (255, 255, 255), 2)
        
        # Add direction indicator (small arrow showing gripper direction)
        arrow_start = rotate_point(-15, 0)
        arrow_end = rotate_point(max_jaw_length + 10, 0)
        cv.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 3, tipLength=0.2)

    def create_overlay_video(self, output_path: str = "robot_arm.mp4"):
        """
        Create video with SO-101 style robot end-effector overlay
        
        Args:
            output_path: Path to save the output video
        """
        if not self.thumb_positions:
            print("No tracking data to create overlay")
            return
            
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return
            
        # Get video properties
        fps = cap.get(cv.CAP_PROP_FPS)
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= len(self.thumb_positions):
                break
                
            # Get current positions and gripper state
            thumb_x, thumb_y = self.thumb_positions[frame_idx]
            index_x, index_y = self.index_positions[frame_idx]
            gripper_angle = self.gripper_states[frame_idx]
            gripper_orientation = self.gripper_orientations[frame_idx]
            
            # Convert workspace coordinates back to pixel coordinates
            pixel_thumb_x = int((thumb_x / 1.0 + 0.5) * width)
            pixel_thumb_y = int((0.5 - thumb_y / 1.0) * height)
            pixel_index_x = int((index_x / 1.0 + 0.5) * width)
            pixel_index_y = int((0.5 - index_y / 1.0) * height)
            
            # Draw SO-101 style crocodile gripper with proper orientation
            self.draw_so101_gripper(frame, pixel_thumb_x, pixel_thumb_y, 
                                  gripper_angle, gripper_orientation, scale=1.2)
            
            # Draw tracking points (smaller and more subtle)
            cv.circle(frame, (pixel_thumb_x, pixel_thumb_y), 6, (0, 0, 255), 2)  # Red for thumb (hollow)
            cv.circle(frame, (pixel_index_x, pixel_index_y), 6, (255, 0, 0), 2)  # Blue for index (hollow)
            
            # Add improved text overlay with background
            overlay = frame.copy()
            cv.rectangle(overlay, (5, 5), (400, 100), (0, 0, 0), -1)
            cv.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv.putText(frame, f"Frame: {frame_idx:03d} | Time: {frame_idx/fps:.2f}s", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(frame, f"Gripper Opening: {gripper_angle:.3f} rad", (10, 55), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(frame, f"Position: ({thumb_x:.3f}, {thumb_y:.3f}) m", (10, 80), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx} frames for video overlay...")
        
        cap.release()
        out.release()
        print(f"SO-101 style overlay video saved to {output_path}")


def main():
    """Main function to run the complete pipeline"""
    print("Starting Robot Arm IK with Hand Tracking Integration...")
    
    # Configuration
    model_path = 'hand_landmarker.task'
    video_path = 'training_vids/push_2D_c3_to_c5_1.mp4'
    
    # Create integration object
    integration = HandTrackingRobotIntegration(model_path, video_path)
    
    # Extract hand tracking data
    print("Extracting hand tracking data...")
    if not integration.extract_hand_data():
        print("Failed to extract hand data")
        return
    
    # Generate plots
    print("Generating joint angle plots...")
    integration.plot_joint_angles("joint_angles_plot.png")
    
    # Create overlay video
    print("Creating overlay video...")
    integration.create_overlay_video("robot_arm.mp4")
    
    print("Complete! Check the generated files:")
    print("- joint_angles_plot.png: Joint angles over time")
    print("- robot_arm.mp4: Video with robot end-effector overlay")


if __name__ == "__main__":
    main()
