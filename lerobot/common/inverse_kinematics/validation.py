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
Trajectory Validation

Provides validation functions for robot trajectories to ensure
safety and feasibility before execution.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np


class TrajectoryValidator:
    """Validates robot trajectories for safety and feasibility"""
    
    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.robot_type = robot_config.get("type", "unknown")
        
        # Load robot-specific validation parameters
        self._load_validation_parameters()
    
    def _load_validation_parameters(self):
        """Load robot-specific validation parameters"""
        
        # Default parameters
        self.max_joint_velocity = 2.0  # rad/s
        self.max_joint_acceleration = 5.0  # rad/s^2
        self.max_end_effector_velocity = 0.5  # m/s
        self.min_safety_distance = 0.02  # m (minimum distance to obstacles)
        
        # Robot-specific overrides
        if self.robot_type in ["so101", "so100"]:
            self.max_joint_velocity = 1.5
            self.max_joint_acceleration = 3.0
            self.max_end_effector_velocity = 0.3
            
        elif self.robot_type == "koch":
            self.max_joint_velocity = 2.0
            self.max_joint_acceleration = 4.0
            self.max_end_effector_velocity = 0.4
            
        elif self.robot_type == "aloha":
            self.max_joint_velocity = 3.0
            self.max_joint_acceleration = 6.0
            self.max_end_effector_velocity = 0.6
    
    def validate_trajectory(self, trajectory: Dict) -> Dict:
        """
        Comprehensive trajectory validation
        
        Args:
            trajectory: Dictionary containing trajectory data:
                - joint_positions: Array of joint configurations
                - joint_velocities: Array of joint velocities
                - timestamps: Array of timestamps
                
        Returns:
            Dictionary with validation results:
                - is_valid: Overall validity flag
                - violations: List of validation violations
                - safe_indices: Array of safe trajectory indices
                - recommendations: List of recommendations for fixes
        """
        joint_positions = trajectory["joint_positions"]
        joint_velocities = trajectory["joint_velocities"]
        timestamps = trajectory["timestamps"]
        
        violations = []
        safe_indices = np.ones(len(joint_positions), dtype=bool)
        recommendations = []
        
        # 1. Validate joint limits
        joint_limit_violations, joint_safe_indices = self._validate_joint_limits(joint_positions)
        violations.extend(joint_limit_violations)
        safe_indices &= joint_safe_indices
        
        # 2. Validate velocity limits
        velocity_violations, velocity_safe_indices = self._validate_velocity_limits(
            joint_velocities, timestamps
        )
        violations.extend(velocity_violations)
        safe_indices &= velocity_safe_indices
        
        # 3. Validate acceleration limits
        acceleration_violations, accel_safe_indices = self._validate_acceleration_limits(
            joint_velocities, timestamps
        )
        violations.extend(acceleration_violations)
        safe_indices &= accel_safe_indices
        
        # 4. Validate trajectory smoothness
        smoothness_violations, smoothness_recommendations = self._validate_smoothness(
            joint_positions, timestamps
        )
        violations.extend(smoothness_violations)
        recommendations.extend(smoothness_recommendations)
        
        # 5. Validate workspace bounds
        workspace_violations, workspace_safe_indices = self._validate_workspace_bounds(trajectory)
        violations.extend(workspace_violations)
        safe_indices &= workspace_safe_indices
        
        # Generate overall recommendations
        if len(violations) > 0:
            recommendations.extend(self._generate_fix_recommendations(violations))
        
        is_valid = len(violations) == 0
        
        return {
            "is_valid": is_valid,
            "violations": violations,
            "safe_indices": safe_indices,
            "recommendations": recommendations,
            "safety_score": np.mean(safe_indices),
            "num_violations": len(violations)
        }
    
    def _validate_joint_limits(self, joint_positions: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """Validate joint position limits"""
        
        violations = []
        safe_indices = np.ones(len(joint_positions), dtype=bool)
        
        # Import robot model to get joint limits
        try:
            from .robot_models import create_robot_model
            robot_model = create_robot_model(self.robot_config)
            joint_limits = robot_model.get_joint_limits()
            
            for joint_idx in range(joint_positions.shape[1]):
                min_limit, max_limit = joint_limits[joint_idx]
                
                # Check violations
                under_limit = joint_positions[:, joint_idx] < min_limit
                over_limit = joint_positions[:, joint_idx] > max_limit
                
                if np.any(under_limit):
                    violations.append(f"Joint {joint_idx} under minimum limit in {np.sum(under_limit)} frames")
                    safe_indices[under_limit] = False
                
                if np.any(over_limit):
                    violations.append(f"Joint {joint_idx} over maximum limit in {np.sum(over_limit)} frames")
                    safe_indices[over_limit] = False
                    
        except ImportError:
            logging.warning("Could not import robot model for joint limit validation")
        
        return violations, safe_indices
    
    def _validate_velocity_limits(
        self, 
        joint_velocities: np.ndarray, 
        timestamps: np.ndarray
    ) -> Tuple[List[str], np.ndarray]:
        """Validate joint velocity limits"""
        
        violations = []
        safe_indices = np.ones(len(joint_velocities), dtype=bool)
        
        # Check velocity magnitudes
        velocity_magnitudes = np.abs(joint_velocities)
        
        for joint_idx in range(joint_velocities.shape[1]):
            over_limit = velocity_magnitudes[:, joint_idx] > self.max_joint_velocity
            
            if np.any(over_limit):
                max_velocity = np.max(velocity_magnitudes[:, joint_idx])
                violations.append(
                    f"Joint {joint_idx} velocity exceeds limit "
                    f"({max_velocity:.2f} > {self.max_joint_velocity:.2f} rad/s) "
                    f"in {np.sum(over_limit)} frames"
                )
                safe_indices[over_limit] = False
        
        return violations, safe_indices
    
    def _validate_acceleration_limits(
        self, 
        joint_velocities: np.ndarray, 
        timestamps: np.ndarray
    ) -> Tuple[List[str], np.ndarray]:
        """Validate joint acceleration limits"""
        
        violations = []
        safe_indices = np.ones(len(joint_velocities), dtype=bool)
        
        if len(timestamps) < 2:
            return violations, safe_indices
        
        # Compute accelerations
        dt = np.diff(timestamps)
        accelerations = np.diff(joint_velocities, axis=0) / dt[:, None]
        
        # Pad with zeros for first frame
        accelerations = np.vstack([np.zeros((1, joint_velocities.shape[1])), accelerations])
        
        # Check acceleration magnitudes
        acceleration_magnitudes = np.abs(accelerations)
        
        for joint_idx in range(accelerations.shape[1]):
            over_limit = acceleration_magnitudes[:, joint_idx] > self.max_joint_acceleration
            
            if np.any(over_limit):
                max_acceleration = np.max(acceleration_magnitudes[:, joint_idx])
                violations.append(
                    f"Joint {joint_idx} acceleration exceeds limit "
                    f"({max_acceleration:.2f} > {self.max_joint_acceleration:.2f} rad/s²) "
                    f"in {np.sum(over_limit)} frames"
                )
                safe_indices[over_limit] = False
        
        return violations, safe_indices
    
    def _validate_smoothness(
        self, 
        joint_positions: np.ndarray, 
        timestamps: np.ndarray
    ) -> Tuple[List[str], List[str]]:
        """Validate trajectory smoothness"""
        
        violations = []
        recommendations = []
        
        if len(timestamps) < 3:
            return violations, recommendations
        
        # Compute second derivatives (acceleration from positions)
        dt = np.diff(timestamps)
        velocities = np.diff(joint_positions, axis=0) / dt[:, None]
        
        dt2 = dt[1:]
        accelerations = np.diff(velocities, axis=0) / dt2[:, None]
        
        # Check for high accelerations (indication of jerky motion)
        acceleration_threshold = self.max_joint_acceleration * 0.5
        
        for joint_idx in range(accelerations.shape[1]):
            high_accel_frames = np.abs(accelerations[:, joint_idx]) > acceleration_threshold
            
            if np.sum(high_accel_frames) > len(accelerations) * 0.1:  # More than 10% of frames
                violations.append(f"Joint {joint_idx} trajectory is jerky (high accelerations)")
                recommendations.append(f"Consider smoothing joint {joint_idx} trajectory")
        
        # Check for sudden direction changes
        for joint_idx in range(velocities.shape[1]):
            velocity_signs = np.sign(velocities[:, joint_idx])
            sign_changes = np.diff(velocity_signs) != 0
            
            if np.sum(sign_changes) > len(velocities) * 0.2:  # More than 20% sign changes
                recommendations.append(f"Joint {joint_idx} has many direction changes - consider simplifying")
        
        return violations, recommendations
    
    def _validate_workspace_bounds(self, trajectory: Dict) -> Tuple[List[str], np.ndarray]:
        """Validate end-effector workspace bounds"""
        
        violations = []
        safe_indices = np.ones(len(trajectory["joint_positions"]), dtype=bool)
        
        try:
            from .robot_models import create_robot_model
            robot_model = create_robot_model(self.robot_config)
            workspace_bounds = robot_model.get_workspace_bounds()
            
            # Compute forward kinematics for each configuration
            for i, joint_config in enumerate(trajectory["joint_positions"]):
                try:
                    pose = robot_model.forward_kinematics(joint_config)
                    position = pose[:3]  # Extract position
                    
                    # Check bounds
                    if not (workspace_bounds["x_min"] <= position[0] <= workspace_bounds["x_max"]):
                        safe_indices[i] = False
                    if not (workspace_bounds["y_min"] <= position[1] <= workspace_bounds["y_max"]):
                        safe_indices[i] = False
                    if not (workspace_bounds["z_min"] <= position[2] <= workspace_bounds["z_max"]):
                        safe_indices[i] = False
                        
                except Exception:
                    # If FK fails, mark as unsafe
                    safe_indices[i] = False
            
            unsafe_count = np.sum(~safe_indices)
            if unsafe_count > 0:
                violations.append(f"End-effector outside workspace in {unsafe_count} frames")
                
        except ImportError:
            logging.warning("Could not import robot model for workspace validation")
        
        return violations, safe_indices
    
    def _generate_fix_recommendations(self, violations: List[str]) -> List[str]:
        """Generate recommendations to fix trajectory violations"""
        
        recommendations = []
        
        # Joint limit violations
        if any("limit" in v for v in violations):
            recommendations.append("Scale trajectory to fit within joint limits")
            recommendations.append("Check robot calibration and workspace bounds")
        
        # Velocity violations
        if any("velocity" in v for v in violations):
            recommendations.append("Reduce trajectory speed or increase time duration")
            recommendations.append("Apply velocity smoothing filter")
        
        # Acceleration violations
        if any("acceleration" in v for v in violations):
            recommendations.append("Apply acceleration smoothing")
            recommendations.append("Increase trajectory time to reduce accelerations")
        
        # Workspace violations
        if any("workspace" in v for v in violations):
            recommendations.append("Scale trajectory to fit within robot workspace")
            recommendations.append("Adjust trajectory origin/center point")
        
        return recommendations
    
    def apply_safety_filtering(self, trajectory: Dict, validation_results: Dict) -> Dict:
        """
        Apply safety filtering to remove unsafe trajectory segments
        
        Args:
            trajectory: Original trajectory
            validation_results: Results from validate_trajectory
            
        Returns:
            Filtered safe trajectory
        """
        safe_indices = validation_results["safe_indices"]
        
        # Keep only safe trajectory points
        filtered_trajectory = {
            "joint_positions": trajectory["joint_positions"][safe_indices],
            "joint_velocities": trajectory["joint_velocities"][safe_indices],
            "timestamps": trajectory["timestamps"][safe_indices],
            "success_rate": trajectory.get("success_rate", 1.0)
        }
        
        # Recalculate velocities for filtered trajectory
        if len(filtered_trajectory["timestamps"]) > 1:
            dt = np.diff(filtered_trajectory["timestamps"])
            filtered_velocities = np.diff(filtered_trajectory["joint_positions"], axis=0) / dt[:, None]
            
            # Pad with zero velocity for first frame
            filtered_velocities = np.vstack([
                np.zeros((1, filtered_trajectory["joint_positions"].shape[1])),
                filtered_velocities
            ])
            
            filtered_trajectory["joint_velocities"] = filtered_velocities
        
        logging.info(f"Filtered trajectory: {len(trajectory['timestamps'])} -> {len(filtered_trajectory['timestamps'])} frames")
        
        return filtered_trajectory
