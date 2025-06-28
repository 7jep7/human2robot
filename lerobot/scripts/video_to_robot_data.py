#!/usr/bin/env python3
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
Video-to-Robot Data Generation Pipeline

This script converts human hand videos into robot training datasets using computer vision
and inverse kinematics. It replaces teleoperation-based data collection with video-based
trajectory extraction and transformation to robot space.

Examples of usage:

- Process single video file:
```bash
python lerobot/scripts/video_to_robot_data.py \
    --video_path=/path/to/hand_video.mp4 \
    --robot.type=so101 \
    --task="Pick up the red block" \
    --output_dataset=my_dataset
```

- Process video directory:
```bash
python lerobot/scripts/video_to_robot_data.py \
    --video_dir=/path/to/videos/ \
    --robot.type=so101 \
    --task="Assembly task" \
    --output_dataset=assembly_dataset \
    --fps=30
```

- Generate dataset with validation:
```bash
python lerobot/scripts/video_to_robot_data.py \
    --video_path=/path/to/video.mp4 \
    --robot.type=so101 \
    --task="Manipulation task" \
    --output_dataset=validated_dataset \
    --validate_trajectory=true \
    --smooth_trajectory=true \
    --workspace_bounds="[-0.5,0.5,-0.3,0.3,0.0,0.4]"
```
"""

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from rich.console import Console
from rich.progress import track

from lerobot.common.computer_vision.trajectory_extractor import VideoTrajectoryExtractor
from lerobot.common.computer_vision.data_format import CVTrajectoryFormat
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.inverse_kinematics.solver import InverseKinematicsSolver
from lerobot.common.inverse_kinematics.trajectory_processor import TrajectoryProcessor
from lerobot.common.inverse_kinematics.robot_models import get_robot_model
from lerobot.common.inverse_kinematics.validation import TrajectoryValidator
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.utils.utils import init_logging
from lerobot.configs import parser

console = Console()


@dataclass
class VideoToDataConfig:
    """Configuration for video-to-robot-data pipeline."""
    # Input video configuration
    video_path: Optional[str] = None
    video_dir: Optional[str] = None
    
    # Output configuration
    output_dataset: str = "video_generated_dataset"
    output_root: Optional[str] = None
    
    # Task configuration
    task: str = "Manipulation task from video"
    
    # Processing configuration
    fps: int = 30
    smooth_trajectory: bool = True
    validate_trajectory: bool = True
    
    # Workspace configuration
    workspace_bounds: Optional[List[float]] = None  # [x_min, x_max, y_min, y_max, z_min, z_max]
    
    # CV configuration
    cv_model_type: str = "mediapipe"  # placeholder for CV team integration
    cv_confidence_threshold: float = 0.8
    
    # IK configuration
    ik_solver: str = "jacobian"  # or "numerical"
    ik_max_iterations: int = 1000
    ik_tolerance: float = 1e-4
    
    # Dataset configuration
    push_to_hub: bool = False
    private: bool = False
    tags: Optional[List[str]] = None


class VideoToRobotDataPipeline:
    """Main pipeline for converting videos to robot training data."""
    
    def __init__(self, config: VideoToDataConfig, robot_config):
        """Initialize the pipeline with configuration."""
        self.config = config
        self.robot_config = robot_config
        
        # Initialize components
        self.cv_extractor = VideoTrajectoryExtractor(
            model_type=config.cv_model_type,
            confidence_threshold=config.cv_confidence_threshold
        )
        
        # Create robot model for IK
        self.robot_model = get_robot_model(robot_config.type)
        
        # Initialize IK solver
        self.ik_solver = InverseKinematicsSolver(
            robot_model=self.robot_model,
            solver_type=config.ik_solver,
            max_iterations=config.ik_max_iterations,
            tolerance=config.ik_tolerance
        )
        
        # Initialize trajectory processor
        self.trajectory_processor = TrajectoryProcessor(
            robot_model=self.robot_model,
            workspace_bounds=config.workspace_bounds
        )
        
        # Initialize validator
        self.trajectory_validator = TrajectoryValidator(
            robot_model=self.robot_model
        )
        
        console.print("[green]✓[/green] Video-to-Robot pipeline initialized")
    
    def process_single_video(self, video_path: Path) -> Dict:
        """Process a single video file and return trajectory data."""
        console.print(f"[blue]Processing video:[/blue] {video_path}")
        
        # Step 1: Extract CV trajectory from video
        console.print("  [yellow]1.[/yellow] Extracting hand trajectory from video...")
        cv_trajectory = self.cv_extractor.extract_trajectory(video_path)
        
        if cv_trajectory is None:
            console.print(f"  [red]✗[/red] Failed to extract trajectory from {video_path}")
            return None
        
        console.print(f"  [green]✓[/green] Extracted {len(cv_trajectory.positions)} trajectory points")
        
        # Step 2: Process and transform trajectory to robot workspace
        console.print("  [yellow]2.[/yellow] Processing trajectory for robot workspace...")
        processed_trajectory = self.trajectory_processor.process_cv_trajectory(
            cv_trajectory,
            smooth=self.config.smooth_trajectory
        )
        
        # Step 3: Apply inverse kinematics
        console.print("  [yellow]3.[/yellow] Solving inverse kinematics...")
        joint_trajectories = []
        
        for i, target_pose in enumerate(track(processed_trajectory.positions, 
                                             description="  Computing joint angles...")):
            joint_angles = self.ik_solver.solve(target_pose)
            if joint_angles is not None:
                joint_trajectories.append(joint_angles)
            else:
                console.print(f"  [orange]⚠[/orange] IK failed for pose {i}, using previous solution")
                if joint_trajectories:
                    joint_trajectories.append(joint_trajectories[-1])
        
        if not joint_trajectories:
            console.print("  [red]✗[/red] No valid joint solutions found")
            return None
        
        # Step 4: Validate trajectory
        if self.config.validate_trajectory:
            console.print("  [yellow]4.[/yellow] Validating trajectory...")
            is_valid, validation_report = self.trajectory_validator.validate_joint_trajectory(
                joint_trajectories
            )
            
            if not is_valid:
                console.print(f"  [orange]⚠[/orange] Trajectory validation warnings: {validation_report}")
            else:
                console.print("  [green]✓[/green] Trajectory validation passed")
        
        # Create episode data
        episode_data = {
            'video_path': str(video_path),
            'cv_trajectory': cv_trajectory,
            'processed_trajectory': processed_trajectory,
            'joint_trajectories': joint_trajectories,
            'task': self.config.task,
            'fps': self.config.fps
        }
        
        console.print("  [green]✓[/green] Video processing complete")
        return episode_data
    
    def create_robot_dataset(self, episodes_data: List[Dict]) -> LeRobotDataset:
        """Create a LeRobot dataset from processed episode data."""
        console.print(f"[blue]Creating dataset with {len(episodes_data)} episodes...[/blue]")
        
        # Create mock robot for dataset creation
        robot = make_robot_from_config(self.robot_config)
        
        # Create dataset
        dataset = LeRobotDataset.create(
            repo_id=self.config.output_dataset,
            fps=self.config.fps,
            root=self.config.output_root,
            robot=robot,
            use_videos=False,  # We don't need video storage for joint trajectories
        )
        
        # Add episodes to dataset
        for episode_idx, episode_data in enumerate(episodes_data):
            console.print(f"  [yellow]Adding episode {episode_idx + 1}/{len(episodes_data)}...[/yellow]")
            
            joint_trajectories = episode_data['joint_trajectories']
            
            # Add frames to dataset
            for frame_idx, joint_angles in enumerate(joint_trajectories):
                # Create observation (current joint state)
                observation = {
                    'observation.state': torch.tensor(joint_angles, dtype=torch.float32)
                }
                
                # Create action (target joint state - for now same as observation)
                action = {
                    'action': torch.tensor(joint_angles, dtype=torch.float32)
                }
                
                # Create frame
                frame = {
                    **observation,
                    **action,
                    'task': episode_data['task']
                }
                
                dataset.add_frame(frame)
            
            # Save episode
            dataset.save_episode()
        
        console.print("[green]✓[/green] Dataset creation complete")
        return dataset
    
    def run(self) -> LeRobotDataset:
        """Run the complete video-to-robot-data pipeline."""
        console.print("[bold blue]Starting Video-to-Robot Data Pipeline[/bold blue]")
        
        # Collect video files
        video_files = []
        
        if self.config.video_path:
            video_files.append(Path(self.config.video_path))
        elif self.config.video_dir:
            video_dir = Path(self.config.video_dir)
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
            for ext in video_extensions:
                video_files.extend(video_dir.glob(f'*{ext}'))
                video_files.extend(video_dir.glob(f'*{ext.upper()}'))
        else:
            raise ValueError("Either video_path or video_dir must be specified")
        
        if not video_files:
            raise ValueError("No video files found")
        
        console.print(f"[blue]Found {len(video_files)} video file(s) to process[/blue]")
        
        # Process all videos
        episodes_data = []
        for video_file in video_files:
            episode_data = self.process_single_video(video_file)
            if episode_data:
                episodes_data.append(episode_data)
        
        if not episodes_data:
            raise RuntimeError("No valid episodes generated from videos")
        
        console.print(f"[green]✓[/green] Successfully processed {len(episodes_data)} videos")
        
        # Create dataset
        dataset = self.create_robot_dataset(episodes_data)
        
        # Push to hub if requested
        if self.config.push_to_hub:
            console.print("[blue]Pushing dataset to Hugging Face Hub...[/blue]")
            dataset.push_to_hub(tags=self.config.tags, private=self.config.private)
            console.print("[green]✓[/green] Dataset pushed to Hub")
        
        console.print("[bold green]✓ Pipeline completed successfully![/bold green]")
        return dataset


def main():
    """Main entry point for the video-to-robot-data script."""
    init_logging()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Convert videos to robot training data")
    
    # Input arguments
    parser.add_argument("--video_path", type=str, help="Path to single video file")
    parser.add_argument("--video_dir", type=str, help="Directory containing video files")
    
    # Output arguments
    parser.add_argument("--output_dataset", type=str, default="video_generated_dataset",
                       help="Name of output dataset")
    parser.add_argument("--output_root", type=str, help="Root directory for dataset")
    
    # Task arguments
    parser.add_argument("--task", type=str, default="Manipulation task from video",
                       help="Description of the task being performed")
    
    # Processing arguments
    parser.add_argument("--fps", type=int, default=30, help="Target FPS for dataset")
    parser.add_argument("--smooth_trajectory", action="store_true", default=True,
                       help="Apply trajectory smoothing")
    parser.add_argument("--validate_trajectory", action="store_true", default=True,
                       help="Validate generated trajectories")
    
    # Workspace arguments
    parser.add_argument("--workspace_bounds", type=str,
                       help="Workspace bounds as JSON list [x_min,x_max,y_min,y_max,z_min,z_max]")
    
    # CV arguments
    parser.add_argument("--cv_model_type", type=str, default="mediapipe",
                       help="Computer vision model type")
    parser.add_argument("--cv_confidence_threshold", type=float, default=0.8,
                       help="CV confidence threshold")
    
    # IK arguments
    parser.add_argument("--ik_solver", type=str, default="jacobian",
                       choices=["jacobian", "numerical"], help="IK solver type")
    parser.add_argument("--ik_max_iterations", type=int, default=1000,
                       help="Maximum IK iterations")
    parser.add_argument("--ik_tolerance", type=float, default=1e-4,
                       help="IK convergence tolerance")
    
    # Dataset arguments
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push dataset to Hugging Face Hub")
    parser.add_argument("--private", action="store_true",
                       help="Make dataset private on Hub")
    parser.add_argument("--tags", type=str, nargs="*",
                       help="Tags for dataset")
    
    # Robot configuration (using existing parser pattern)
    parser.add_argument("--robot.type", type=str, required=True,
                       help="Robot type (e.g., so101, koch, aloha)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.video_path and not args.video_dir:
        parser.error("Either --video_path or --video_dir must be specified")
    
    # Parse workspace bounds if provided
    workspace_bounds = None
    if args.workspace_bounds:
        import json
        workspace_bounds = json.loads(args.workspace_bounds)
        if len(workspace_bounds) != 6:
            parser.error("workspace_bounds must have exactly 6 values")
    
    # Create configuration
    config = VideoToDataConfig(
        video_path=args.video_path,
        video_dir=args.video_dir,
        output_dataset=args.output_dataset,
        output_root=args.output_root,
        task=args.task,
        fps=args.fps,
        smooth_trajectory=args.smooth_trajectory,
        validate_trajectory=args.validate_trajectory,
        workspace_bounds=workspace_bounds,
        cv_model_type=args.cv_model_type,
        cv_confidence_threshold=args.cv_confidence_threshold,
        ik_solver=args.ik_solver,
        ik_max_iterations=args.ik_max_iterations,
        ik_tolerance=args.ik_tolerance,
        push_to_hub=args.push_to_hub,
        private=args.private,
        tags=args.tags
    )
    
    # Create mock robot config (this would need to be integrated with actual robot configs)
    from types import SimpleNamespace
    robot_config = SimpleNamespace(type=getattr(args, 'robot.type'))
    
    # Create and run pipeline
    pipeline = VideoToRobotDataPipeline(config, robot_config)
    dataset = pipeline.run()
    
    console.print(f"[bold green]Dataset created successfully: {config.output_dataset}[/bold green]")
    console.print(f"[blue]Total episodes: {dataset.num_episodes}[/blue]")
    console.print(f"[blue]Total frames: {len(dataset)}[/blue]")


if __name__ == "__main__":
    main()
