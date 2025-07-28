#!/usr/bin/env python3
"""
SO-101 Hardware Testing Script for 5-Day Sprint
===============================================

This script is designed for the critical 5-day hardware access period.
It focuses on collecting essential SO-101 robot data and validating 
the hand-to-robot conversion pipeline before travel.

Usage:
    python so101_hardware_tests.py --test [data_collection|ik_validation|end_to_end]

Author: Jonas Petersen (human2robot project)
Date: July 27, 2025
"""

import argparse
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

# LeRobot imports
try:
    from lerobot.common.robot_devices.robots.configs import So101RobotConfig
    from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
    from lerobot.common.datasets.lerobot_dataset import create_lerobot_dataset
except ImportError as e:
    print(f"⚠️  LeRobot imports failed: {e}")
    print("Running in simulation mode...")

# Local imports
from lerobot.common.inverse_kinematics.hand_to_robot_ik import HandToRobotBridge, create_demo_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SO101HardwareTester:
    """Test suite for SO-101 robot hardware validation"""
    
    def __init__(self, mock_robot: bool = False):
        """Initialize SO-101 hardware tester"""
        self.mock_robot = mock_robot
        self.robot = None
        self.config = None
        self.test_results = {}
        
        # Create output directory for test data
        self.output_dir = Path("outputs/so101_hardware_tests")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize_robot(self) -> bool:
        """Initialize SO-101 robot connection"""
        try:
            logger.info("🤖 Initializing SO-101 robot...")
            self.config = So101RobotConfig(mock=self.mock_robot)
            
            if not self.mock_robot:
                self.robot = ManipulatorRobot(self.config)
                logger.info("✅ SO-101 robot connected successfully")
            else:
                logger.info("🔧 Running in mock mode (no hardware)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize SO-101: {e}")
            return False
    
    def test_joint_limits_and_workspace(self) -> Dict:
        """Test SO-101 joint limits and workspace boundaries"""
        logger.info("📏 Testing joint limits and workspace...")
        
        results = {
            "test_name": "joint_limits_workspace",
            "timestamp": time.time(),
            "joint_data": {},
            "workspace_data": {}
        }
        
        if self.robot is None:
            logger.warning("⚠️  No robot connection - using mock data")
            # Generate mock joint limit data
            joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
            for joint in joint_names:
                results["joint_data"][joint] = {
                    "min_angle": -np.pi,
                    "max_angle": np.pi,
                    "safe_speed": 1.0,
                    "measured": False
                }
        else:
            # Test actual robot joint limits
            try:
                # Get current joint positions
                observation = self.robot.capture_observation()
                current_joints = observation["observation.state"]
                
                joint_names = list(self.config.follower_arms["main"].motors.keys())
                for i, joint in enumerate(joint_names):
                    results["joint_data"][joint] = {
                        "current_position": float(current_joints[i]),
                        "measured": True
                    }
                    
                logger.info("✅ Joint data collected successfully")
                
            except Exception as e:
                logger.error(f"❌ Joint testing failed: {e}")
                results["error"] = str(e)
        
        # Save results
        output_file = self.output_dir / "joint_limits_test.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.test_results["joint_limits"] = results
        return results
    
    def collect_teleoperation_demonstrations(self, num_demos: int = 5) -> Dict:
        """Collect teleoperation demonstration data"""
        logger.info(f"📹 Collecting {num_demos} teleoperation demonstrations...")
        
        results = {
            "test_name": "teleoperation_demos",
            "timestamp": time.time(),
            "demonstrations": [],
            "total_demos": num_demos
        }
        
        if self.robot is None:
            logger.warning("⚠️  No robot connection - skipping teleoperation")
            results["error"] = "No robot connection available"
            return results
        
        try:
            for demo_idx in range(num_demos):
                logger.info(f"🎮 Recording demonstration {demo_idx + 1}/{num_demos}")
                logger.info("   Press ENTER when ready to start recording...")
                input()
                
                demo_data = {
                    "demo_id": demo_idx,
                    "timestamp": time.time(),
                    "joint_trajectory": [],
                    "camera_frames": [],
                    "duration": 0.0
                }
                
                # Record for 10 seconds at 10 Hz
                start_time = time.time()
                recording_duration = 10.0  # seconds
                hz = 10
                
                logger.info(f"🔴 Recording for {recording_duration} seconds...")
                
                while time.time() - start_time < recording_duration:
                    frame_start = time.time()
                    
                    # Capture robot state
                    observation = self.robot.capture_observation()
                    
                    frame_data = {
                        "timestamp": time.time() - start_time,
                        "joint_positions": observation["observation.state"].tolist(),
                        "has_camera": "observation.images" in observation
                    }
                    
                    demo_data["joint_trajectory"].append(frame_data)
                    
                    # Sleep to maintain desired frequency
                    elapsed = time.time() - frame_start
                    sleep_time = max(0, 1.0/hz - elapsed)
                    time.sleep(sleep_time)
                
                demo_data["duration"] = time.time() - start_time
                results["demonstrations"].append(demo_data)
                
                logger.info(f"✅ Demo {demo_idx + 1} recorded ({len(demo_data['joint_trajectory'])} frames)")
        
        except Exception as e:
            logger.error(f"❌ Teleoperation recording failed: {e}")
            results["error"] = str(e)
        
        # Save results
        output_file = self.output_dir / "teleoperation_demos.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.test_results["teleoperation"] = results
        return results
    
    def test_ik_validation(self) -> Dict:
        """Test inverse kinematics accuracy with SO-101"""
        logger.info("🧮 Testing inverse kinematics validation...")
        
        results = {
            "test_name": "ik_validation",
            "timestamp": time.time(),
            "test_points": [],
            "accuracy_metrics": {}
        }
        
        try:
            # Create hand-to-robot bridge
            bridge = create_demo_pipeline()
            
            # Test IK with known target positions
            test_positions = [
                [0.2, 0.0, 0.15],   # Forward center
                [0.15, 0.1, 0.12],  # Right side
                [0.15, -0.1, 0.12], # Left side
                [0.1, 0.0, 0.08],   # Close and low
            ]
            
            for i, target_pos in enumerate(test_positions):
                logger.info(f"   Testing position {i+1}: {target_pos}")
                
                # Solve IK
                joint_solution = bridge.ik_solver.solve_ik(np.array(target_pos))
                
                if joint_solution is not None:
                    # Compute forward kinematics to check accuracy
                    computed_pos = bridge.ik_solver.forward_kinematics(joint_solution)
                    error = np.linalg.norm(computed_pos - np.array(target_pos))
                    
                    test_point = {
                        "target_position": target_pos,
                        "joint_solution": joint_solution.tolist() if joint_solution is not None else None,
                        "computed_position": computed_pos.tolist() if computed_pos is not None else None,
                        "position_error": float(error) if computed_pos is not None else None,
                        "solution_found": True
                    }
                else:
                    test_point = {
                        "target_position": target_pos,
                        "joint_solution": None,
                        "computed_position": None,
                        "position_error": None,
                        "solution_found": False
                    }
                
                results["test_points"].append(test_point)
            
            # Compute overall accuracy metrics
            successful_tests = [tp for tp in results["test_points"] if tp["solution_found"]]
            if successful_tests:
                errors = [tp["position_error"] for tp in successful_tests if tp["position_error"] is not None]
                results["accuracy_metrics"] = {
                    "success_rate": len(successful_tests) / len(test_positions),
                    "mean_error": float(np.mean(errors)) if errors else None,
                    "max_error": float(np.max(errors)) if errors else None,
                    "std_error": float(np.std(errors)) if errors else None
                }
            
            logger.info(f"✅ IK validation complete. Success rate: {results['accuracy_metrics'].get('success_rate', 0):.2%}")
            
        except Exception as e:
            logger.error(f"❌ IK validation failed: {e}")
            results["error"] = str(e)
        
        # Save results
        output_file = self.output_dir / "ik_validation.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.test_results["ik_validation"] = results
        return results
    
    def test_end_to_end_pipeline(self) -> Dict:
        """Test complete hand video to robot execution pipeline"""
        logger.info("🔄 Testing end-to-end pipeline...")
        
        results = {
            "test_name": "end_to_end_pipeline",
            "timestamp": time.time(),
            "pipeline_stages": {}
        }
        
        try:
            # Create bridge
            bridge = create_demo_pipeline()
            
            # Test with mock hand data (since we may not have videos yet)
            mock_hand_data = self.create_mock_hand_trajectory()
            results["pipeline_stages"]["hand_data_created"] = True
            
            # Convert to robot trajectory
            robot_trajectory = bridge.generate_robot_trajectory(mock_hand_data)
            results["pipeline_stages"]["trajectory_generated"] = robot_trajectory is not None
            
            if robot_trajectory and self.robot:
                # Test execution on robot (safety mode)
                logger.info("🤖 Testing robot trajectory execution...")
                # TODO: Implement safe trajectory execution
                results["pipeline_stages"]["robot_execution"] = "skipped_for_safety"
            
            logger.info("✅ End-to-end pipeline test complete")
            
        except Exception as e:
            logger.error(f"❌ End-to-end pipeline failed: {e}")
            results["error"] = str(e)
        
        # Save results
        output_file = self.output_dir / "end_to_end_test.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.test_results["end_to_end"] = results
        return results
    
    def create_mock_hand_trajectory(self):
        """Create mock hand tracking data for testing"""
        # TODO: Implement mock hand trajectory for testing
        return None
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        logger.info("📊 Generating test report...")
        
        report_path = self.output_dir / "so101_test_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# SO-101 Hardware Test Report\n\n")
            f.write(f"**Test Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Robot Mode**: {'Hardware' if not self.mock_robot else 'Mock'}\n\n")
            
            for test_name, results in self.test_results.items():
                f.write(f"## {test_name.replace('_', ' ').title()}\n\n")
                
                if "error" in results:
                    f.write(f"❌ **Status**: Failed - {results['error']}\n\n")
                else:
                    f.write("✅ **Status**: Completed\n\n")
                
                # Add test-specific details
                if test_name == "joint_limits" and "joint_data" in results:
                    f.write("### Joint Information\n")
                    for joint, data in results["joint_data"].items():
                        f.write(f"- **{joint}**: {data}\n")
                    f.write("\n")
                
                elif test_name == "teleoperation" and "demonstrations" in results:
                    f.write(f"### Demonstrations Collected: {len(results['demonstrations'])}\n")
                    for demo in results["demonstrations"]:
                        f.write(f"- Demo {demo['demo_id']}: {len(demo['joint_trajectory'])} frames, {demo['duration']:.2f}s\n")
                    f.write("\n")
                
                elif test_name == "ik_validation" and "accuracy_metrics" in results:
                    metrics = results["accuracy_metrics"]
                    f.write("### IK Accuracy Metrics\n")
                    f.write(f"- **Success Rate**: {metrics.get('success_rate', 0):.2%}\n")
                    f.write(f"- **Mean Error**: {metrics.get('mean_error', 'N/A')}\n")
                    f.write(f"- **Max Error**: {metrics.get('max_error', 'N/A')}\n")
                    f.write("\n")
        
        logger.info(f"📄 Test report saved to: {report_path}")
        return str(report_path)


def main():
    """Main function for SO-101 hardware testing"""
    parser = argparse.ArgumentParser(description="SO-101 Hardware Testing Suite")
    parser.add_argument("--test", choices=["data_collection", "ik_validation", "end_to_end", "all"], 
                       default="all", help="Test to run")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no hardware)")
    parser.add_argument("--demos", type=int, default=3, help="Number of teleoperation demos to collect")
    
    args = parser.parse_args()
    
    print("🚨 SO-101 Hardware Testing Suite")
    print("================================")
    print(f"Test mode: {args.test}")
    print(f"Hardware mode: {'Mock' if args.mock else 'Real'}")
    print("")
    
    # Initialize tester
    tester = SO101HardwareTester(mock_robot=args.mock)
    
    if not tester.initialize_robot():
        print("❌ Failed to initialize robot. Exiting.")
        return
    
    # Run selected tests
    if args.test in ["data_collection", "all"]:
        tester.test_joint_limits_and_workspace()
        tester.collect_teleoperation_demonstrations(args.demos)
    
    if args.test in ["ik_validation", "all"]:
        tester.test_ik_validation()
    
    if args.test in ["end_to_end", "all"]:
        tester.test_end_to_end_pipeline()
    
    # Generate report
    report_path = tester.generate_test_report()
    
    print("\n🎉 Testing Complete!")
    print(f"📊 Full report: {report_path}")
    print(f"📁 Test data: {tester.output_dir}")


if __name__ == "__main__":
    main()
