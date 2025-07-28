# Implementation Roadmap for human2robot MVP

## 📋 Week 1: MVP Demo Tasks

### Task 1: Complete Inverse Kinematics Bridge (Priority: CRITICAL)
**File**: `lerobot/common/inverse_kinematics/hand_to_robot_ik.py`

**Sub-tasks**:
1. **WorkspaceMapper.map_coordinates()** [2-3 hours]
   - Implement linear scaling transformation
   - Add bounds checking and clipping
   - Test with sample coordinates

2. **RobotArmIK.solve_ik()** [4-6 hours]
   - Implement 2D planar arm IK (simplified for chess task)
   - Use geometric/analytical solution for 5DOF arm
   - Add joint limit constraints
   - Test forward/inverse kinematics consistency

3. **HandToRobotBridge.extract_hand_data_from_video()** [2-3 hours]
   - Integrate existing MediaPipe pipeline from `hand_tracking/hand_coordinates.py`
   - Extract thumb/index finger tip positions
   - Convert to consistent coordinate system

4. **HandToRobotBridge.generate_robot_trajectory()** [3-4 hours]
   - Map hand positions to robot workspace
   - Solve IK for each timestamp
   - Handle IK solution failures gracefully
   - Generate gripper commands from hand closure detection

### Task 2: Data Pipeline Integration (Priority: HIGH)
**Files**: `hand_to_robot_ik.py`, new dataset creation scripts

**Sub-tasks**:
1. **Trajectory Smoothing** [2-3 hours]
   - Implement moving average or spline smoothing
   - Add velocity/acceleration constraints
   - Test on real hand tracking data

2. **LeRobot Dataset Conversion** [4-5 hours]
   - Study LeRobot dataset format (`lerobot/common/datasets/`)
   - Create observation.state (joint positions) and action (target joints) pairs
   - Add proper episode structure and metadata
   - Test dataset loading with LeRobot utilities

### Task 3: Demo Implementation (Priority: MEDIUM)
**Goal**: End-to-end chess piece movement demo

**Sub-tasks**:
1. **Simple Test Case** [2-3 hours]
   - Use existing chess movement video (`push_2D_c3_to_c5_1.mp4`)
   - Run full pipeline: video → hand data → robot trajectory → dataset
   - Visualize results and debug issues

2. **Policy Training** [1-2 hours]
   - Train ACT policy on generated dataset
   - Use existing LeRobot training scripts as reference
   - Evaluate policy performance (even if robot execution isn't available)

## 🧠 Learning & Understanding Focus

### Foundational Concepts to Master:
1. **Inverse Kinematics Mathematics**
   - Understand DH parameters and transformation matrices
   - Learn analytical vs numerical IK solving methods
   - Practice with different robot configurations

2. **Imitation Learning Theory**
   - Study behavioral cloning loss functions
   - Understand observation-action space design
   - Learn about covariate shift and distribution mismatch

3. **Temporal Sequence Modeling**
   - Understand why action chunking works
   - Learn about recurrent vs transformer approaches
   - Study trajectory optimization principles

### Hands-on Implementation Skills:
1. **Build Core Algorithms from Scratch**
   - Implement IK solver without libraries (educational value)
   - Code basic transformer blocks for understanding ACT
   - Write custom loss functions for trajectory learning

2. **Experiment Design**
   - Create systematic evaluation metrics
   - Design ablation studies (e.g., smoothing effects, workspace scaling)
   - Learn to visualize and debug robot trajectories

## 🚀 Research Extensions (Weeks 2-4)

### High-Impact Research Questions:
1. **How does workspace scaling affect learning performance?**
2. **Can we learn optimal coordinate mappings end-to-end?**
3. **What temporal smoothing strategies work best for different tasks?**
4. **How do different hand gesture representations affect robot performance?**

### Implementation-Driven Research:
1. **Comparative Analysis Framework**
   - Implement multiple IK solving methods
   - Compare ACT vs Diffusion Policy on your data
   - Create benchmarks for video-to-robot learning

2. **Novel Architecture Components**
   - Design task-specific observation encoders
   - Implement uncertainty-aware trajectory generation
   - Create multi-modal fusion architectures

### 🔬 Advanced Research Directions (Post-Hardware Access)

#### Sensor Fusion & Multi-modal Learning:
1. **iPhone LiDAR Integration** [Week 3-4]
   - Extract 256x192 depth maps from iPhone recordings
   - Fuse RGB + depth for improved 3D understanding
   - Compare depth-assisted vs RGB-only trajectory generation

2. **Multi-sensor Data Pipeline** [Simulation Phase]
   - Design unified sensor fusion architecture
   - Implement modality-specific encoders (RGB, depth, IMU)
   - Study sensor dropout robustness

#### Comprehensive Imitation Learning Survey:
1. **Classical IL Methods** [Weeks 2-3]
   - **Behavioral Cloning variants**: BC, BC-O (Offline), BC with data augmentation
   - **Inverse RL**: Maximum Entropy IRL, Deep Maximum Entropy IRL
   - **Distribution Matching**: Moments Matching, Wasserstein BC

2. **Modern IL Approaches** [Weeks 3-4]
   - **Adversarial IL**: GAIL, ValueDice, IQ-Learn
   - **Offline IL**: SQIL, f-BRAC, AWR-BC
   - **Few-shot IL**: One-shot imitation, Model-Agnostic Meta-Learning

3. **Implementation Priority**: Start with BC variants, then GAIL, then ValueDice

#### Hybrid IL+RL Pipeline:
1. **Scene Reset Mastery via IL** [Week 4]
   - Train reliable reset policies from human demonstrations
   - Implement automatic scene state verification
   - Design robust failure detection and recovery

2. **Task Optimization via RL** [Simulation Phase]
   - Use reset policy to enable overnight RL training
   - Implement PPO/SAC for fine-tuning task execution
   - Design reward functions that complement IL demonstrations

#### Isaac Lab Simulation Integration:
1. **Simulation Environment Setup** [Week 3]
   - Install and configure Isaac Lab
   - Create chess manipulation environment
   - Implement human2robot data replay in simulation

2. **Sim2Real Transfer Pipeline** [Simulation Phase]
   - Domain adaptation techniques for visual transfer
   - Physics parameter randomization
   - Policy robustness evaluation

### 💻 Hardware-Constrained Development Strategy

#### Data Collection Sprint (Next 5 Days):
1. **Comprehensive Demonstration Recording**
   - Multiple camera angles for same task
   - Varied lighting conditions and backgrounds  
   - Different hand sizes and manipulation styles
   - LiDAR + RGB simultaneous recording

2. **Sensor Data Harvesting**
   - iPhone LiDAR depth maps
   - IMU data during demonstrations
   - Multiple object types and sizes

#### Simulation-First Development (2-4 Month Period):
1. **Pure Simulation Development**
   - Isaac Lab environment development
   - Algorithm implementation and testing
   - Comparative studies between IL methods

2. **Data Efficiency Research**
   - How much real data is needed?
   - Synthetic data augmentation strategies
   - Transfer learning from simulation

3. **Preparation for Hardware Return**
   - Simulation-validated algorithms ready for deployment
   - Automated evaluation pipelines
   - Hardware integration scripts prepared

## 📚 Study Resources for Depth:

### Papers to Implement:
1. **"Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"** (ACT)
2. **"Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"**
3. **"Where2Act: From Pixels to Actions for Articulated 3D Objects"**

### Imitation Learning Deep Dive:
**Classical Foundations:**
1. **"Apprenticeship Learning via Inverse Reinforcement Learning"** (Abbeel & Ng, 2004)
2. **"Maximum Entropy Inverse Reinforcement Learning"** (Ziebart et al., 2008)
3. **"Generative Adversarial Imitation Learning"** (Ho & Ermon, 2016)

**Modern Approaches:**
1. **"ValueDice: Learning State Value Functions and State Distributions"** (Kostrikov et al., 2019)
2. **"IQ-Learn: Inverse soft-Q Learning for Imitation"** (Garg et al., 2021)
3. **"Strictly Batch Imitation Learning by Energy-based Distribution Matching"** (f-BRAC)

**Offline & Few-shot IL:**
1. **"SQIL: Imitation Learning via Regularized Behavioral Cloning"** (Reddy et al., 2019)
2. **"One-Shot Imitation Learning"** (Duan et al., 2017)
3. **"Model-Agnostic Meta-Learning"** (Finn et al., 2017)

### Reinforcement Learning Integration:
1. **"Proximal Policy Optimization"** (Schulman et al., 2017)
2. **"Soft Actor-Critic"** (Haarnoja et al., 2018)
3. **"Reset-Free Reinforcement Learning via Multi-Task Learning"** (Eysenbach et al., 2022)

### Mathematical Foundations:
1. **Robot Kinematics**: "Introduction to Robotics" by Craig
2. **Probabilistic Robotics**: Thrun, Burgard, Fox
3. **Deep Learning**: Goodfellow, Bengio, Courville (especially sequence modeling chapters)

---

This roadmap balances practical MVP delivery with deep foundational learning. Each task builds understanding while creating tangible progress toward the demo goal.
