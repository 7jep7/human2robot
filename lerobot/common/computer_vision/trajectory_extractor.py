# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Video Trajectory Extractor

Interface for extracting end-effector trajectories from human demonstration videos.
This module provides the bridge between Dami + Omar's computer vision work 
and Jonas's inverse kinematics pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


class VideoTrajectoryExtractor:
    """
    Extracts end-effector trajectories from human demonstration videos
    
    This class provides a standardized interface for the CV pipeline to output
    trajectories that can be consumed by the inverse kinematics solver.
    """
    
    def __init__(self, fps: int = 30):
        self.fps = fps
        
    def extract_from_video(self, video_path: str) -> Dict:
        """
        Extract trajectory from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing trajectory data:
                - positions: Array of 3D positions [N, 3] or 2D positions [N, 2]
                - timestamps: Array of timestamps [N]
                - object_positions: Optional array of object positions [N, 3]
                - confidence_scores: Optional confidence scores [N]
                - metadata: Dictionary with extraction metadata
                
        NOTE: This is a placeholder implementation. Dami + Omar will replace
        this with their actual computer vision pipeline.
        """
        
        logging.info(f"Extracting trajectory from video: {video_path}")
        
        # PLACEHOLDER: Mock trajectory extraction
        # Dami + Omar will replace this with real CV implementation
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Mock: Generate a simple trajectory for testing
        trajectory = self._generate_mock_trajectory()
        
        logging.info(f"Extracted {len(trajectory['timestamps'])} trajectory points")
        
        return trajectory
    
    def _generate_mock_trajectory(self) -> Dict:
        """
        Generate mock trajectory for testing (PLACEHOLDER)
        
        This simulates a simple pick-and-place operation:
        1. Start at position A (high)
        2. Move down to pick object
        3. Move to position B while carrying object
        4. Move down to place object
        5. Move up to final position
        
        Dami + Omar will replace this with real CV extraction.
        """
        
        # Mock parameters
        duration = 5.0  # 5 second demonstration
        num_frames = int(duration * self.fps)
        timestamps = np.linspace(0, duration, num_frames)
        
        # Create simple trajectory: pick from (0.1, 0.1) and place at (0.15, 0.15)
        pick_pos = np.array([0.1, 0.1])
        place_pos = np.array([0.15, 0.15])
        
        # Generate 2D trajectory
        positions_2d = np.zeros((num_frames, 2))
        
        # Phase 1: Move to pick position (20% of trajectory)
        phase1_end = int(0.2 * num_frames)
        start_pos = np.array([0.12, 0.08])  # Start slightly offset
        for i in range(phase1_end):
            t = i / phase1_end
            positions_2d[i] = start_pos + t * (pick_pos - start_pos)
        
        # Phase 2: Stay at pick position (10% of trajectory)
        phase2_end = int(0.3 * num_frames)
        positions_2d[phase1_end:phase2_end] = pick_pos
        
        # Phase 3: Move to place position (50% of trajectory)
        phase3_end = int(0.8 * num_frames)
        for i in range(phase2_end, phase3_end):
            t = (i - phase2_end) / (phase3_end - phase2_end)
            positions_2d[i] = pick_pos + t * (place_pos - pick_pos)
        
        # Phase 4: Stay at place position (10% of trajectory)
        phase4_end = int(0.9 * num_frames)
        positions_2d[phase3_end:phase4_end] = place_pos
        
        # Phase 5: Move away (10% of trajectory)
        final_pos = place_pos + np.array([0.02, -0.02])
        for i in range(phase4_end, num_frames):
            t = (i - phase4_end) / (num_frames - phase4_end)
            positions_2d[i] = place_pos + t * (final_pos - place_pos)
        
        # Mock object detection - object follows hand during pick-place
        object_positions = positions_2d.copy()
        
        # Object doesn't move during approach and departure
        object_positions[:phase2_end] = pick_pos
        object_positions[phase4_end:] = place_pos
        
        # Add some noise to make it more realistic
        noise_std = 0.002
        positions_2d += np.random.normal(0, noise_std, positions_2d.shape)
        object_positions += np.random.normal(0, noise_std, object_positions.shape)
        
        # Generate mock confidence scores
        confidence_scores = np.random.uniform(0.8, 0.98, num_frames)
        
        return {
            "positions": positions_2d,
            "timestamps": timestamps,
            "object_positions": object_positions,
            "confidence_scores": confidence_scores,
            "metadata": {
                "video_fps": self.fps,
                "extraction_method": "mock_placeholder",
                "coordinate_system": "image_2d",
                "units": "meters",
                "task_type": "pick_place"
            }
        }
    
    def extract_from_image_sequence(self, image_paths: list) -> Dict:
        """
        Extract trajectory from sequence of images
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Same format as extract_from_video
            
        NOTE: Placeholder for Dami + Omar's implementation
        """
        
        logging.info(f"Extracting trajectory from {len(image_paths)} images")
        
        # Mock implementation
        duration = len(image_paths) / self.fps
        timestamps = np.linspace(0, duration, len(image_paths))
        
        # For now, generate mock trajectory
        trajectory = self._generate_mock_trajectory()
        
        # Adjust for actual number of images
        if len(image_paths) != len(trajectory["timestamps"]):
            # Resample trajectory to match image count
            trajectory = self._resample_trajectory(trajectory, len(image_paths))
        
        return trajectory
    
    def _resample_trajectory(self, trajectory: Dict, new_length: int) -> Dict:
        """Resample trajectory to new length"""
        
        old_length = len(trajectory["timestamps"])
        
        # Create new timestamps
        new_timestamps = np.linspace(0, trajectory["timestamps"][-1], new_length)
        
        # Resample positions
        new_positions = np.zeros((new_length, trajectory["positions"].shape[1]))
        for dim in range(trajectory["positions"].shape[1]):
            new_positions[:, dim] = np.interp(
                new_timestamps, 
                trajectory["timestamps"], 
                trajectory["positions"][:, dim]
            )
        
        # Resample other arrays similarly
        resampled = {
            "positions": new_positions,
            "timestamps": new_timestamps,
            "metadata": trajectory["metadata"].copy()
        }
        
        # Resample optional fields if present
        if "object_positions" in trajectory:
            new_object_pos = np.zeros((new_length, trajectory["object_positions"].shape[1]))
            for dim in range(trajectory["object_positions"].shape[1]):
                new_object_pos[:, dim] = np.interp(
                    new_timestamps, 
                    trajectory["timestamps"], 
                    trajectory["object_positions"][:, dim]
                )
            resampled["object_positions"] = new_object_pos
        
        if "confidence_scores" in trajectory:
            resampled["confidence_scores"] = np.interp(
                new_timestamps, 
                trajectory["timestamps"], 
                trajectory["confidence_scores"]
            )
        
        return resampled
    
    def validate_trajectory(self, trajectory: Dict) -> Tuple[bool, list]:
        """
        Validate extracted trajectory for completeness and quality
        
        Args:
            trajectory: Trajectory dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        
        issues = []
        
        # Check required fields
        required_fields = ["positions", "timestamps"]
        for field in required_fields:
            if field not in trajectory:
                issues.append(f"Missing required field: {field}")
        
        if issues:  # Early return if missing critical fields
            return False, issues
        
        # Check array lengths match
        positions = trajectory["positions"]
        timestamps = trajectory["timestamps"]
        
        if len(positions) != len(timestamps):
            issues.append(f"Position and timestamp arrays have different lengths: {len(positions)} vs {len(timestamps)}")
        
        # Check minimum trajectory length
        if len(positions) < 10:
            issues.append(f"Trajectory too short: {len(positions)} frames (minimum 10)")
        
        # Check for valid positions (no NaN/inf)
        if np.any(~np.isfinite(positions)):
            issues.append("Trajectory contains invalid values (NaN/inf)")
        
        # Check temporal ordering
        if not np.all(np.diff(timestamps) > 0):
            issues.append("Timestamps are not strictly increasing")
        
        # Check confidence scores if present
        if "confidence_scores" in trajectory:
            scores = trajectory["confidence_scores"]
            if len(scores) != len(positions):
                issues.append("Confidence scores length doesn't match positions")
            
            mean_confidence = np.mean(scores)
            if mean_confidence < 0.5:
                issues.append(f"Low average confidence: {mean_confidence:.2f}")
        
        # Check trajectory smoothness
        if len(positions) > 2:
            velocities = np.diff(positions, axis=0)
            velocity_magnitudes = np.linalg.norm(velocities, axis=1)
            
            if np.max(velocity_magnitudes) > 1.0:  # > 1 m/frame is suspicious
                issues.append("Trajectory has very high velocities (possible noise)")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logging.warning(f"Trajectory validation failed with {len(issues)} issues")
        
        return is_valid, issues
