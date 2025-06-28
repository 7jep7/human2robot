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
Inverse Kinematics Solver

Provides various IK solving algorithms to convert end-effector poses
to joint configurations for different robot types.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Some IK solvers may not work.")


class IKSolver(ABC):
    """Abstract base class for inverse kinematics solvers"""
    
    @abstractmethod
    def solve(self, target_pose: np.ndarray, initial_guess: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        """
        Solve inverse kinematics for a single target pose
        
        Args:
            target_pose: Target end-effector pose [x, y, z, rx, ry, rz] or [x, y, z]
            initial_guess: Initial joint configuration guess
            
        Returns:
            Tuple of (joint_angles, success_flag)
        """
        pass


class JacobianPseudoInverseSolver(IKSolver):
    """Jacobian pseudo-inverse based IK solver"""
    
    def __init__(
        self, 
        robot_model, 
        max_iterations: int = 100,
        tolerance: float = 1e-3,
        step_size: float = 0.1,
        joint_limits: Optional[np.ndarray] = None
    ):
        self.robot_model = robot_model
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.step_size = step_size
        self.joint_limits = joint_limits
        
    def solve(self, target_pose: np.ndarray, initial_guess: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        """Solve IK using Jacobian pseudo-inverse method"""
        
        if initial_guess is None:
            # Use middle of joint range as initial guess
            if self.joint_limits is not None:
                initial_guess = (self.joint_limits[:, 0] + self.joint_limits[:, 1]) / 2
            else:
                initial_guess = np.zeros(self.robot_model.num_joints)
        
        q = initial_guess.copy()
        
        for iteration in range(self.max_iterations):
            # Get current end-effector pose
            current_pose = self.robot_model.forward_kinematics(q)
            
            # Compute pose error
            if len(target_pose) == 3:  # Position only
                pose_error = target_pose - current_pose[:3]
            else:  # Position + orientation
                pose_error = target_pose - current_pose
            
            # Check convergence
            if np.linalg.norm(pose_error) < self.tolerance:
                return q, True
            
            # Compute Jacobian
            jacobian = self.robot_model.compute_jacobian(q)
            
            # If position-only target, use only position part of Jacobian
            if len(target_pose) == 3:
                jacobian = jacobian[:3, :]
            
            # Compute pseudo-inverse
            jacobian_pinv = np.linalg.pinv(jacobian)
            
            # Update joint angles
            delta_q = self.step_size * jacobian_pinv @ pose_error
            q += delta_q
            
            # Apply joint limits
            if self.joint_limits is not None:
                q = np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])
        
        # Max iterations reached without convergence
        return q, False


class NumericalIKSolver(IKSolver):
    """Numerical optimization-based IK solver using scipy"""
    
    def __init__(
        self, 
        robot_model,
        method: str = "SLSQP",
        tolerance: float = 1e-6,
        joint_limits: Optional[np.ndarray] = None
    ):
        self.robot_model = robot_model
        self.method = method
        self.tolerance = tolerance
        self.joint_limits = joint_limits
        
        try:
            from scipy.optimize import minimize
            self.minimize = minimize
        except ImportError:
            raise ImportError("scipy is required for NumericalIKSolver")
    
    def solve(self, target_pose: np.ndarray, initial_guess: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        """Solve IK using numerical optimization"""
        
        if initial_guess is None:
            if self.joint_limits is not None:
                initial_guess = (self.joint_limits[:, 0] + self.joint_limits[:, 1]) / 2
            else:
                initial_guess = np.zeros(self.robot_model.num_joints)
        
        def objective(q):
            current_pose = self.robot_model.forward_kinematics(q)
            if len(target_pose) == 3:
                error = target_pose - current_pose[:3]
            else:
                error = target_pose - current_pose
            return np.sum(error**2)
        
        # Set up bounds
        bounds = None
        if self.joint_limits is not None:
            bounds = [(low, high) for low, high in self.joint_limits]
        
        # Solve optimization
        result = self.minimize(
            objective,
            initial_guess,
            method=self.method,
            bounds=bounds,
            options={'ftol': self.tolerance}
        )
        
        return result.x, result.success


class InverseKinematicsSolver:
    """Main IK solver class that handles different robot types and solver algorithms"""
    
    def __init__(
        self,
        robot_config: Dict,
        solver_type: str = "jacobian_pseudo_inverse",
        max_iterations: int = 100,
        tolerance: float = 1e-3,
        **solver_kwargs
    ):
        self.robot_config = robot_config
        self.solver_type = solver_type
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Import robot model
        from .robot_models import create_robot_model
        self.robot_model = create_robot_model(robot_config)
        
        # Create solver
        self._create_solver(**solver_kwargs)
        
    def _create_solver(self, **kwargs):
        """Create the appropriate IK solver"""
        
        joint_limits = self.robot_model.get_joint_limits()
        
        if self.solver_type == "jacobian_pseudo_inverse":
            self.solver = JacobianPseudoInverseSolver(
                self.robot_model,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
                joint_limits=joint_limits,
                **kwargs
            )
        elif self.solver_type == "numerical":
            self.solver = NumericalIKSolver(
                self.robot_model,
                tolerance=self.tolerance,
                joint_limits=joint_limits,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")
    
    def solve_single_pose(
        self, 
        target_pose: Union[np.ndarray, List[float]], 
        initial_guess: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, bool]:
        """
        Solve IK for a single target pose
        
        Args:
            target_pose: Target end-effector pose [x, y, z] or [x, y, z, rx, ry, rz]
            initial_guess: Initial joint configuration
            
        Returns:
            Tuple of (joint_angles, success_flag)
        """
        target_pose = np.array(target_pose)
        return self.solver.solve(target_pose, initial_guess)
    
    def solve_trajectory(self, trajectory: Dict) -> Dict:
        """
        Solve IK for an entire trajectory
        
        Args:
            trajectory: Dictionary containing:
                - positions: Array of 3D positions
                - timestamps: Array of timestamps
                - orientations: Optional array of orientations
                
        Returns:
            Dictionary containing:
                - joint_positions: Array of joint configurations
                - joint_velocities: Array of joint velocities
                - timestamps: Same as input timestamps
                - success_rate: Percentage of successful solutions
        """
        positions = trajectory["positions"]
        timestamps = trajectory["timestamps"]
        orientations = trajectory.get("orientations", None)
        
        num_waypoints = len(positions)
        joint_positions = []
        success_flags = []
        
        # Use previous solution as initial guess for next waypoint
        initial_guess = None
        
        for i, position in enumerate(positions):
            # Construct target pose
            if orientations is not None:
                target_pose = np.concatenate([position, orientations[i]])
            else:
                target_pose = position
            
            # Solve IK
            joint_config, success = self.solve_single_pose(target_pose, initial_guess)
            
            joint_positions.append(joint_config)
            success_flags.append(success)
            
            # Use current solution as initial guess for next waypoint
            initial_guess = joint_config
            
            if i % 10 == 0:
                logging.info(f"IK Progress: {i+1}/{num_waypoints} waypoints solved")
        
        joint_positions = np.array(joint_positions)
        success_rate = np.mean(success_flags)
        
        # Compute joint velocities using finite differences
        dt = np.diff(timestamps)
        joint_velocities = np.zeros_like(joint_positions)
        joint_velocities[1:] = np.diff(joint_positions, axis=0) / dt[:, None]
        
        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "timestamps": timestamps,
            "success_rate": success_rate,
            "success_flags": success_flags
        }
    
    def validate_workspace(self, positions: np.ndarray) -> Dict:
        """
        Check if positions are reachable by the robot
        
        Args:
            positions: Array of 3D positions to check
            
        Returns:
            Dictionary with validation results
        """
        workspace_bounds = self.robot_model.get_workspace_bounds()
        reachable_flags = []
        
        for position in positions:
            # Check if position is within workspace bounds
            within_bounds = (
                workspace_bounds["x_min"] <= position[0] <= workspace_bounds["x_max"] and
                workspace_bounds["y_min"] <= position[1] <= workspace_bounds["y_max"] and
                workspace_bounds["z_min"] <= position[2] <= workspace_bounds["z_max"]
            )
            reachable_flags.append(within_bounds)
        
        return {
            "reachable_positions": np.array(reachable_flags),
            "reachability_rate": np.mean(reachable_flags),
            "workspace_bounds": workspace_bounds
        }
