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
Inverse Kinematics Module for Human2Robot Pipeline

This module provides inverse kinematics solving capabilities to convert
end-effector trajectories from computer vision into robot joint trajectories.
"""

from .solver import InverseKinematicsSolver
from .trajectory_processor import TrajectoryProcessor
from .robot_models import RobotKinematicModel
from .validation import TrajectoryValidator

__all__ = [
    "InverseKinematicsSolver",
    "TrajectoryProcessor", 
    "RobotKinematicModel",
    "TrajectoryValidator"
]
