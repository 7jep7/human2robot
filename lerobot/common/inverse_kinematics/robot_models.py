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
Robot Kinematic Models

Provides forward/inverse kinematics models for different robot types
supported by the human2robot pipeline.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np


class RobotKinematicModel(ABC):
    """Abstract base class for robot kinematic models"""
    
    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.robot_type = robot_config.get("type", "unknown")
        self.num_joints = self._get_num_joints()
        
    @abstractmethod
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics
        
        Args:
            joint_angles: Array of joint angles
            
        Returns:
            End-effector pose [x, y, z, rx, ry, rz] or [x, y, z]
        """
        pass
    
    @abstractmethod
    def compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix at given joint configuration
        
        Args:
            joint_angles: Array of joint angles
            
        Returns:
            Jacobian matrix (6 x num_joints) or (3 x num_joints)
        """
        pass
    
    @abstractmethod
    def get_joint_limits(self) -> np.ndarray:
        """
        Get joint limits for the robot
        
        Returns:
            Array of shape (num_joints, 2) with [min, max] for each joint
        """
        pass
    
    @abstractmethod
    def get_workspace_bounds(self) -> Dict:
        """
        Get workspace bounds for the robot
        
        Returns:
            Dictionary with x_min, x_max, y_min, y_max, z_min, z_max
        """
        pass
    
    @abstractmethod
    def _get_num_joints(self) -> int:
        """Get number of joints for this robot"""
        pass


class SimpleArmModel(RobotKinematicModel):
    """Simple 6-DOF arm model for SO-101, Koch, etc."""
    
    def __init__(self, robot_config: Dict):
        super().__init__(robot_config)
        self._setup_dh_parameters()
        
    def _setup_dh_parameters(self):
        """Setup Denavit-Hartenberg parameters for the robot"""
        
        if self.robot_type in ["so101", "so100"]:
            # SO-101 DH parameters (example values - need actual measurements)
            self.dh_params = np.array([
                # [a, alpha, d, theta_offset]
                [0.0,    np.pi/2,  0.15,  0.0],      # Base to shoulder
                [0.12,   0.0,      0.0,   -np.pi/2], # Shoulder to upper arm
                [0.10,   0.0,      0.0,   0.0],      # Upper arm to forearm
                [0.0,    np.pi/2,  0.08,  0.0],      # Forearm to wrist
                [0.0,    -np.pi/2, 0.0,   0.0],      # Wrist pitch
                [0.0,    0.0,      0.06,  0.0]       # Wrist to end-effector
            ])
            
        elif self.robot_type == "koch":
            # Koch DH parameters (example values)
            self.dh_params = np.array([
                [0.0,    np.pi/2,  0.10,  0.0],
                [0.08,   0.0,      0.0,   -np.pi/2],
                [0.07,   0.0,      0.0,   0.0],
                [0.0,    np.pi/2,  0.05,  0.0],
                [0.0,    -np.pi/2, 0.0,   0.0],
                [0.0,    0.0,      0.04,  0.0]
            ])
            
        elif self.robot_type == "aloha":
            # Aloha DH parameters (example values)
            self.dh_params = np.array([
                [0.0,    np.pi/2,  0.16,  0.0],
                [0.15,   0.0,      0.0,   -np.pi/2],
                [0.12,   0.0,      0.0,   0.0],
                [0.0,    np.pi/2,  0.10,  0.0],
                [0.0,    -np.pi/2, 0.0,   0.0],
                [0.0,    0.0,      0.08,  0.0]
            ])
            
        else:
            # Default generic 6-DOF arm
            logging.warning(f"Using default DH parameters for unknown robot type: {self.robot_type}")
            self.dh_params = np.array([
                [0.0,    np.pi/2,  0.12,  0.0],
                [0.10,   0.0,      0.0,   -np.pi/2],
                [0.08,   0.0,      0.0,   0.0],
                [0.0,    np.pi/2,  0.06,  0.0],
                [0.0,    -np.pi/2, 0.0,   0.0],
                [0.0,    0.0,      0.05,  0.0]
            ])
    
    def _get_num_joints(self) -> int:
        """Simple arms typically have 6 joints"""
        return 6
    
    def _dh_transform(self, a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """Compute DH transformation matrix"""
        
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct,    -st * ca,  st * sa,   a * ct],
            [st,    ct * ca,   -ct * sa,  a * st],
            [0,     sa,        ca,        d],
            [0,     0,         0,         1]
        ])
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute forward kinematics using DH parameters"""
        
        if len(joint_angles) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joint angles, got {len(joint_angles)}")
        
        # Start with identity transform
        T = np.eye(4)
        
        # Apply each joint transformation
        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + theta_offset
            T_i = self._dh_transform(a, alpha, d, theta)
            T = T @ T_i
        
        # Extract position and orientation
        position = T[:3, 3]
        
        # Extract Euler angles from rotation matrix (ZYX convention)
        R = T[:3, :3]
        
        # Check for gimbal lock
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        
        if not singular:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0
        
        return np.concatenate([position, [rx, ry, rz]])
    
    def compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute Jacobian using numerical differentiation"""
        
        epsilon = 1e-6
        jacobian = np.zeros((6, self.num_joints))
        
        # Current pose
        pose_0 = self.forward_kinematics(joint_angles)
        
        # Compute partial derivatives
        for i in range(self.num_joints):
            joint_angles_plus = joint_angles.copy()
            joint_angles_plus[i] += epsilon
            
            pose_plus = self.forward_kinematics(joint_angles_plus)
            
            # Numerical derivative
            jacobian[:, i] = (pose_plus - pose_0) / epsilon
        
        return jacobian
    
    def get_joint_limits(self) -> np.ndarray:
        """Get joint limits based on robot type"""
        
        if self.robot_type in ["so101", "so100"]:
            # SO-101 joint limits (in radians)
            return np.array([
                [-np.pi, np.pi],        # Base rotation
                [-np.pi/2, np.pi/2],    # Shoulder pitch
                [-np.pi, 0],            # Elbow pitch
                [-np.pi, np.pi],        # Wrist roll
                [-np.pi/2, np.pi/2],    # Wrist pitch
                [-np.pi, np.pi]         # End-effector roll
            ])
            
        elif self.robot_type == "koch":
            # Koch joint limits
            return np.array([
                [-np.pi, np.pi],
                [-np.pi/3, np.pi/3],
                [-np.pi/2, np.pi/2],
                [-np.pi, np.pi],
                [-np.pi/2, np.pi/2],
                [-np.pi, np.pi]
            ])
            
        elif self.robot_type == "aloha":
            # Aloha joint limits
            return np.array([
                [-np.pi, np.pi],
                [-np.pi/2, np.pi/2],
                [-np.pi, 0],
                [-np.pi, np.pi],
                [-np.pi/2, np.pi/2],
                [-np.pi, np.pi]
            ])
            
        else:
            # Default conservative limits
            return np.array([
                [-np.pi/2, np.pi/2],
                [-np.pi/3, np.pi/3],
                [-np.pi/2, np.pi/2],
                [-np.pi/2, np.pi/2],
                [-np.pi/3, np.pi/3],
                [-np.pi/2, np.pi/2]
            ])
    
    def get_workspace_bounds(self) -> Dict:
        """Get workspace bounds based on robot type and DH parameters"""
        
        # Estimate reach from DH parameters
        total_reach = np.sum(np.abs(self.dh_params[:, 0])) + np.sum(np.abs(self.dh_params[:, 2]))
        
        if self.robot_type in ["so101", "so100"]:
            return {
                "x_min": -0.25, "x_max": 0.25,
                "y_min": -0.25, "y_max": 0.25,
                "z_min": 0.05, "z_max": 0.35
            }
        elif self.robot_type == "koch":
            return {
                "x_min": -0.2, "x_max": 0.2,
                "y_min": -0.2, "y_max": 0.2,
                "z_min": 0.02, "z_max": 0.25
            }
        elif self.robot_type == "aloha":
            return {
                "x_min": -0.4, "x_max": 0.4,
                "y_min": -0.3, "y_max": 0.3,
                "z_min": 0.05, "z_max": 0.5
            }
        else:
            # Default based on estimated reach
            margin = 0.8  # Use 80% of theoretical reach
            reach = total_reach * margin
            return {
                "x_min": -reach, "x_max": reach,
                "y_min": -reach, "y_max": reach,
                "z_min": 0.05, "z_max": reach
            }


def create_robot_model(robot_config: Dict) -> RobotKinematicModel:
    """
    Factory function to create appropriate robot model
    
    Args:
        robot_config: Robot configuration dictionary
        
    Returns:
        Appropriate robot kinematic model
    """
    robot_type = robot_config.get("type", "unknown")
    
    if robot_type in ["so101", "so100", "koch", "aloha"]:
        return SimpleArmModel(robot_config)
    else:
        logging.warning(f"Unknown robot type {robot_type}, using SimpleArmModel")
        return SimpleArmModel(robot_config)
