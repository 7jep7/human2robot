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
Computer Vision Data Format Specifications

Defines standardized data formats for communication between CV pipeline
and inverse kinematics pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CVTrajectoryFormat:
    """
    Standardized format for CV trajectory output
    
    This defines the contract between Dami + Omar's CV work and Jonas's IK work.
    """
    
    # Required fields
    positions: np.ndarray  # Shape: [N, 2] or [N, 3] - end-effector positions
    timestamps: np.ndarray  # Shape: [N] - time stamps for each position
    
    # Optional fields
    object_positions: Optional[np.ndarray] = None  # Shape: [N, 2] or [N, 3]
    confidence_scores: Optional[np.ndarray] = None  # Shape: [N] - detection confidence
    orientations: Optional[np.ndarray] = None  # Shape: [N, 3] or [N, 4] - end-effector orientations
    
    # Metadata
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the trajectory format after initialization"""
        self._validate()
    
    def _validate(self):
        """Validate the trajectory data"""
        
        # Check required fields
        if self.positions is None or self.timestamps is None:
            raise ValueError("positions and timestamps are required")
        
        # Check shapes
        if len(self.positions.shape) != 2:
            raise ValueError(f"positions must be 2D array, got shape {self.positions.shape}")
        
        if len(self.timestamps.shape) != 1:
            raise ValueError(f"timestamps must be 1D array, got shape {self.timestamps.shape}")
        
        # Check length consistency
        if len(self.positions) != len(self.timestamps):
            raise ValueError(f"positions and timestamps must have same length: "
                           f"{len(self.positions)} vs {len(self.timestamps)}")
        
        # Check position dimensions
        pos_dim = self.positions.shape[1]
        if pos_dim not in [2, 3]:
            raise ValueError(f"positions must be 2D or 3D, got {pos_dim}D")
        
        # Validate optional fields
        if self.object_positions is not None:
            if len(self.object_positions) != len(self.positions):
                raise ValueError("object_positions length must match positions")
        
        if self.confidence_scores is not None:
            if len(self.confidence_scores) != len(self.positions):
                raise ValueError("confidence_scores length must match positions")
            
            if not np.all((self.confidence_scores >= 0) & (self.confidence_scores <= 1)):
                raise ValueError("confidence_scores must be in range [0, 1]")
        
        if self.orientations is not None:
            if len(self.orientations) != len(self.positions):
                raise ValueError("orientations length must match positions")
    
    @property
    def num_frames(self) -> int:
        """Number of frames in the trajectory"""
        return len(self.positions)
    
    @property
    def duration(self) -> float:
        """Duration of the trajectory in seconds"""
        return self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0.0
    
    @property
    def fps(self) -> float:
        """Estimated frames per second"""
        return (self.num_frames - 1) / self.duration if self.duration > 0 else 0.0
    
    @property
    def is_2d(self) -> bool:
        """Whether positions are 2D"""
        return self.positions.shape[1] == 2
    
    @property
    def is_3d(self) -> bool:
        """Whether positions are 3D"""
        return self.positions.shape[1] == 3
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format for serialization"""
        
        result = {
            "positions": self.positions.tolist(),
            "timestamps": self.timestamps.tolist(),
            "metadata": self.metadata
        }
        
        if self.object_positions is not None:
            result["object_positions"] = self.object_positions.tolist()
        
        if self.confidence_scores is not None:
            result["confidence_scores"] = self.confidence_scores.tolist()
        
        if self.orientations is not None:
            result["orientations"] = self.orientations.tolist()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CVTrajectoryFormat":
        """Create from dictionary format"""
        
        trajectory = cls(
            positions=np.array(data["positions"]),
            timestamps=np.array(data["timestamps"]),
            metadata=data.get("metadata", {})
        )
        
        if "object_positions" in data:
            trajectory.object_positions = np.array(data["object_positions"])
        
        if "confidence_scores" in data:
            trajectory.confidence_scores = np.array(data["confidence_scores"])
        
        if "orientations" in data:
            trajectory.orientations = np.array(data["orientations"])
        
        return trajectory
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for the trajectory"""
        
        stats = {
            "num_frames": self.num_frames,
            "duration": self.duration,
            "fps": self.fps,
            "position_dimensions": self.positions.shape[1],
            "position_range": {
                "min": self.positions.min(axis=0).tolist(),
                "max": self.positions.max(axis=0).tolist(),
                "mean": self.positions.mean(axis=0).tolist(),
                "std": self.positions.std(axis=0).tolist()
            }
        }
        
        if self.confidence_scores is not None:
            stats["confidence"] = {
                "mean": float(self.confidence_scores.mean()),
                "min": float(self.confidence_scores.min()),
                "max": float(self.confidence_scores.max()),
                "std": float(self.confidence_scores.std())
            }
        
        if self.object_positions is not None:
            stats["object_tracking"] = {
                "available": True,
                "range": {
                    "min": self.object_positions.min(axis=0).tolist(),
                    "max": self.object_positions.max(axis=0).tolist()
                }
            }
        
        return stats


def create_cv_trajectory(
    positions: List[List[float]],
    timestamps: List[float],
    object_positions: Optional[List[List[float]]] = None,
    confidence_scores: Optional[List[float]] = None,
    metadata: Optional[Dict] = None
) -> CVTrajectoryFormat:
    """
    Convenience function to create CV trajectory from lists
    
    Args:
        positions: List of position coordinates [[x, y], ...] or [[x, y, z], ...]
        timestamps: List of timestamps
        object_positions: Optional list of object positions
        confidence_scores: Optional list of confidence scores
        metadata: Optional metadata dictionary
    
    Returns:
        CVTrajectoryFormat object
    """
    
    trajectory = CVTrajectoryFormat(
        positions=np.array(positions),
        timestamps=np.array(timestamps),
        metadata=metadata or {}
    )
    
    if object_positions is not None:
        trajectory.object_positions = np.array(object_positions)
    
    if confidence_scores is not None:
        trajectory.confidence_scores = np.array(confidence_scores)
    
    return trajectory


# Example usage and format documentation
def example_cv_output() -> CVTrajectoryFormat:
    """
    Example of how Dami + Omar should format their CV output
    
    This shows the expected data structure that the IK pipeline will consume.
    """
    
    # Example: 3-second pick and place trajectory at 30 FPS
    fps = 30
    duration = 3.0
    num_frames = int(fps * duration)
    
    # Generate example 2D trajectory (image coordinates converted to world coordinates)
    timestamps = np.linspace(0, duration, num_frames)
    
    # Simple pick-place: start -> pick location -> place location -> end
    start_pos = [0.1, 0.1]
    pick_pos = [0.15, 0.12]
    place_pos = [0.2, 0.15]
    end_pos = [0.18, 0.13]
    
    # Interpolate between waypoints
    positions = []
    for i, t in enumerate(timestamps):
        progress = t / duration
        
        if progress < 0.3:  # Move to pick
            alpha = progress / 0.3
            pos = [(1-alpha) * start_pos[j] + alpha * pick_pos[j] for j in range(2)]
        elif progress < 0.7:  # Move to place
            alpha = (progress - 0.3) / 0.4
            pos = [(1-alpha) * pick_pos[j] + alpha * place_pos[j] for j in range(2)]
        else:  # Move to end
            alpha = (progress - 0.7) / 0.3
            pos = [(1-alpha) * place_pos[j] + alpha * end_pos[j] for j in range(2)]
        
        positions.append(pos)
    
    # Mock object tracking - object follows during transport
    object_positions = []
    for i, t in enumerate(timestamps):
        progress = t / duration
        
        if progress < 0.3:  # Object stays at pick location
            obj_pos = pick_pos
        elif progress < 0.7:  # Object moves with hand
            obj_pos = positions[i]
        else:  # Object stays at place location
            obj_pos = place_pos
        
        object_positions.append(obj_pos)
    
    # Mock confidence scores (higher during stable phases)
    confidence_scores = []
    for i, t in enumerate(timestamps):
        progress = t / duration
        
        # Lower confidence during rapid movements
        if 0.25 < progress < 0.35 or 0.65 < progress < 0.75:
            confidence = np.random.uniform(0.7, 0.85)
        else:
            confidence = np.random.uniform(0.85, 0.95)
        
        confidence_scores.append(confidence)
    
    # Create trajectory
    trajectory = create_cv_trajectory(
        positions=positions,
        timestamps=timestamps.tolist(),
        object_positions=object_positions,
        confidence_scores=confidence_scores,
        metadata={
            "video_source": "demo_video.mp4",
            "extraction_method": "hand_tracking",
            "coordinate_system": "world_2d",
            "units": "meters",
            "task_type": "pick_place",
            "object_type": "chess_piece"
        }
    )
    
    return trajectory


if __name__ == "__main__":
    # Example usage
    example_trajectory = example_cv_output()
    print("Example CV Trajectory:")
    print(f"  Frames: {example_trajectory.num_frames}")
    print(f"  Duration: {example_trajectory.duration:.2f}s")
    print(f"  FPS: {example_trajectory.fps:.1f}")
    print(f"  Dimensions: {example_trajectory.positions.shape[1]}D")
    
    # Show summary stats
    stats = example_trajectory.get_summary_stats()
    print(f"  Position range: {stats['position_range']}")
    print(f"  Mean confidence: {stats['confidence']['mean']:.3f}")
