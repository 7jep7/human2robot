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
Trajectory Processor

Converts computer vision trajectories to robot workspace trajectories,
handling coordinate transformations and trajectory smoothing.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np


class TrajectoryProcessor:
    """Processes trajectories between different coordinate systems and formats"""
    
    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.robot_type = robot_config.get("type", "unknown")
        
        # Load robot-specific workspace parameters
        self._load_workspace_parameters()
        
    def _load_workspace_parameters(self):
        """Load robot-specific workspace bounds and transformations"""
        
        # Default workspace bounds (can be overridden per robot type)
        self.workspace_bounds = {
            "x_min": -0.3, "x_max": 0.3,
            "y_min": -0.3, "y_max": 0.3,
            "z_min": 0.0, "z_max": 0.4
        }
        
        # Robot-specific configurations
        if self.robot_type == "so101":
            self.workspace_bounds.update({
                "x_min": -0.25, "x_max": 0.25,
                "y_min": -0.25, "y_max": 0.25,
                "z_min": 0.05, "z_max": 0.35
            })
            # Base to end-effector transform for SO-101
            self.base_transform = np.array([0.0, 0.0, 0.15])  # Base height offset
            
        elif self.robot_type == "koch":
            self.workspace_bounds.update({
                "x_min": -0.2, "x_max": 0.2,
                "y_min": -0.2, "y_max": 0.2,
                "z_min": 0.02, "z_max": 0.25
            })
            self.base_transform = np.array([0.0, 0.0, 0.1])
            
        elif self.robot_type == "aloha":
            self.workspace_bounds.update({
                "x_min": -0.4, "x_max": 0.4,
                "y_min": -0.3, "y_max": 0.3,
                "z_min": 0.05, "z_max": 0.5
            })
            self.base_transform = np.array([0.0, 0.0, 0.2])
            
        else:
            logging.warning(f"Unknown robot type {self.robot_type}, using default workspace")
            self.base_transform = np.array([0.0, 0.0, 0.0])
    
    def cv_to_robot_trajectory(self, cv_trajectory: Dict) -> Dict:
        """
        Convert computer vision trajectory to robot workspace trajectory
        
        Args:
            cv_trajectory: Dictionary containing CV trajectory data:
                - positions: Array of 2D or 3D positions from CV
                - timestamps: Array of timestamps
                - image_coords: Optional image coordinates
                - world_scale: Optional world scale factor
                
        Returns:
            Dictionary containing robot trajectory:
                - positions: Array of 3D positions in robot workspace
                - timestamps: Array of timestamps (same as input)
                - orientations: Array of end-effector orientations
        """
        cv_positions = cv_trajectory["positions"]
        timestamps = cv_trajectory["timestamps"]
        
        # Convert CV positions to robot workspace
        robot_positions = self._transform_cv_to_robot(cv_positions, cv_trajectory)
        
        # Generate orientations (for now, use default downward-facing)
        orientations = self._generate_default_orientations(len(robot_positions))
        
        # Smooth trajectory if needed
        if len(robot_positions) > 3:
            robot_positions = self._smooth_trajectory(robot_positions)
            orientations = self._smooth_trajectory(orientations)
        
        # Validate trajectory is within workspace
        valid_indices = self._validate_workspace_positions(robot_positions)
        
        if np.sum(valid_indices) < len(robot_positions) * 0.8:
            logging.warning(f"Only {np.sum(valid_indices)}/{len(robot_positions)} trajectory points are within robot workspace")
        
        return {
            "positions": robot_positions,
            "orientations": orientations,
            "timestamps": timestamps,
            "valid_indices": valid_indices
        }
    
    def _transform_cv_to_robot(self, cv_positions: np.ndarray, cv_trajectory: Dict) -> np.ndarray:
        """Transform CV positions to robot coordinate system"""
        
        # Handle 2D vs 3D CV positions
        if cv_positions.shape[1] == 2:
            # 2D CV positions - need to estimate Z
            robot_positions = self._estimate_3d_from_2d(cv_positions, cv_trajectory)
        else:
            # 3D CV positions
            robot_positions = cv_positions.copy()
        
        # Apply coordinate transformation
        # Assume CV uses image coordinates, need to transform to robot base frame
        
        # Example transformation (this would be calibrated for each setup):
        # 1. Scale from pixels/meters to robot workspace
        world_scale = cv_trajectory.get("world_scale", 0.001)  # Default: 1mm per pixel
        robot_positions *= world_scale
        
        # 2. Rotate and translate to robot base frame
        # For simplicity, assume CV Z-up, robot Z-up, but might need rotation
        
        # 3. Add base transform offset
        robot_positions += self.base_transform
        
        # 4. Center the trajectory in the workspace
        robot_positions = self._center_trajectory_in_workspace(robot_positions)
        
        return robot_positions
    
    def _estimate_3d_from_2d(self, positions_2d: np.ndarray, cv_trajectory: Dict) -> np.ndarray:
        """Estimate 3D positions from 2D CV trajectory"""
        
        num_points = len(positions_2d)
        positions_3d = np.zeros((num_points, 3))
        
        # Copy X, Y from 2D
        positions_3d[:, :2] = positions_2d
        
        # Estimate Z based on trajectory type
        task_type = cv_trajectory.get("task_type", "pick_place")
        
        if task_type == "pick_place":
            # Simple pick-place: start high, go down, back up
            z_high = 0.2
            z_low = 0.05
            
            # Find lowest point in trajectory (pick/place location)
            mid_point = num_points // 2
            
            # Create Z trajectory: high -> low -> high
            z_trajectory = np.ones(num_points) * z_high
            
            # Go down to pick
            pick_start = max(0, mid_point - num_points // 4)
            pick_end = mid_point
            z_trajectory[pick_start:pick_end] = np.linspace(z_high, z_low, pick_end - pick_start)
            
            # Stay low during transport
            transport_end = min(num_points, mid_point + num_points // 4)
            z_trajectory[pick_end:transport_end] = z_low
            
            # Go back up
            if transport_end < num_points:
                z_trajectory[transport_end:] = np.linspace(z_low, z_high, num_points - transport_end)
            
            positions_3d[:, 2] = z_trajectory
            
        else:
            # Default: constant height
            positions_3d[:, 2] = 0.15  # 15cm above base
        
        return positions_3d
    
    def _center_trajectory_in_workspace(self, positions: np.ndarray) -> np.ndarray:
        """Center trajectory within robot workspace bounds"""
        
        centered_positions = positions.copy()
        
        for axis, (min_bound, max_bound) in enumerate([
            (self.workspace_bounds["x_min"], self.workspace_bounds["x_max"]),
            (self.workspace_bounds["y_min"], self.workspace_bounds["y_max"]),
            (self.workspace_bounds["z_min"], self.workspace_bounds["z_max"])
        ]):
            pos_min = np.min(positions[:, axis])
            pos_max = np.max(positions[:, axis])
            pos_range = pos_max - pos_min
            
            # Center of workspace
            workspace_center = (min_bound + max_bound) / 2
            workspace_range = max_bound - min_bound
            
            # If trajectory fits, center it
            if pos_range <= workspace_range:
                trajectory_center = (pos_min + pos_max) / 2
                offset = workspace_center - trajectory_center
                centered_positions[:, axis] += offset
            else:
                # Scale down if trajectory is too large
                scale = workspace_range / pos_range * 0.9  # 90% of workspace
                trajectory_center = (pos_min + pos_max) / 2
                centered_positions[:, axis] = (
                    workspace_center + (positions[:, axis] - trajectory_center) * scale
                )
                
                logging.warning(f"Trajectory scaled down by {scale:.2f} for axis {axis}")
        
        return centered_positions
    
    def _generate_default_orientations(self, num_points: int) -> np.ndarray:
        """Generate default end-effector orientations"""
        
        # Default: gripper pointing down (for pick-place tasks)
        # Euler angles: [roll, pitch, yaw] = [0, π/2, 0] (pointing down)
        default_orientation = np.array([0.0, np.pi/2, 0.0])
        
        orientations = np.tile(default_orientation, (num_points, 1))
        
        return orientations
    
    def _smooth_trajectory(self, trajectory: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Apply smoothing filter to trajectory"""
        
        if len(trajectory) < window_size:
            return trajectory
        
        # Simple moving average filter
        smoothed = trajectory.copy()
        half_window = window_size // 2
        
        for i in range(half_window, len(trajectory) - half_window):
            smoothed[i] = np.mean(trajectory[i - half_window:i + half_window + 1], axis=0)
        
        return smoothed
    
    def _validate_workspace_positions(self, positions: np.ndarray) -> np.ndarray:
        """Check which positions are within robot workspace"""
        
        valid_flags = np.ones(len(positions), dtype=bool)
        
        # Check X bounds
        valid_flags &= (positions[:, 0] >= self.workspace_bounds["x_min"])
        valid_flags &= (positions[:, 0] <= self.workspace_bounds["x_max"])
        
        # Check Y bounds
        valid_flags &= (positions[:, 1] >= self.workspace_bounds["y_min"])
        valid_flags &= (positions[:, 1] <= self.workspace_bounds["y_max"])
        
        # Check Z bounds
        valid_flags &= (positions[:, 2] >= self.workspace_bounds["z_min"])
        valid_flags &= (positions[:, 2] <= self.workspace_bounds["z_max"])
        
        return valid_flags
    
    def robot_to_dataset_format(
        self, 
        joint_trajectory: Dict,
        task_description: str,
        fps: int
    ) -> Dict:
        """
        Convert robot joint trajectory to LeRobot dataset format
        
        Args:
            joint_trajectory: Output from IK solver
            task_description: Human-readable task description
            fps: Frames per second for the dataset
            
        Returns:
            Dictionary in LeRobot dataset format
        """
        timestamps = joint_trajectory["timestamps"]
        joint_positions = joint_trajectory["joint_positions"]
        joint_velocities = joint_trajectory["joint_velocities"]
        
        # Prepare dataset frames
        frames = []
        
        for i, (timestamp, position, velocity) in enumerate(zip(timestamps, joint_positions, joint_velocities)):
            frame = {
                "timestamp": float(timestamp),
                "observation.state": position.astype(np.float32),
                "action": position.astype(np.float32),  # For imitation learning
                "observation.velocity": velocity.astype(np.float32),
                "task": task_description,
                "frame_index": i,
                "episode_index": 0,  # Single episode for now
                "next.reward": 0.0,  # No reward signal from human demo
                "next.success": i == len(timestamps) - 1,  # Success at end
            }
            frames.append(frame)
        
        return {
            "frames": frames,
            "fps": fps,
            "num_frames": len(frames),
            "num_episodes": 1,
            "task": task_description
        }
