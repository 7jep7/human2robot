# human2robot: Computer Vision to Robot Training Data Pipeline

> **Note**: This project is cloned/forked from [Hugging Face's lerobot repository](https://github.com/huggingface/lerobot) and adapted for our hackathon goals.

<p align="center">
  <strong>A 6-hour hackathon project transforming human demonstrations into robot training data</strong>
  <br/>
  <em>Converting video demonstrations to teleoperated imitation learning datasets</em>
</p>

## 📋 Pitch Deck

**[View our project pitch deck →](https://www.canva.com/design/DAGrp1t2CbU/rpmUw4W6mIyQFrgV70_NXA/edit?utm_content=DAGrp1t2CbU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)**

---

## 🎯 Project Overview

**human2robot** is a hackathon project that bridges the gap between human video demonstrations and robot training data. Instead of requiring expensive teleoperation hardware setups, we enable robot training from simple video recordings of human task demonstrations.

### 🚀 The Vision

Traditional robot imitation learning requires:
1. Expensive leader-follower robot pairs for teleoperation
2. Expert operators to demonstrate tasks
3. Complex hardware setups and calibration

**Our solution**: Record a human performing a task → Generate robot training data → Train robots via imitation learning

### 👥 Team & Timeline
- **Duration**: 6-hour hackathon
- **Team**: 3 people
  - **Dami + Omar**: UCL Robotics students (Computer Vision)
  - **Jonas**: Inverse Kinematics & Robot Training Pipeline

---

## 🔧 Technical Architecture

### Task 1: Computer Vision Pipeline (Dami + Omar)
**Goal**: Extract end-effector motion and object interaction from video

- **Input**: Video of human hand performing a simple task (e.g., moving a chess rook)
- **Output**: 2D trajectory of end-effector position and object movement
- **Scope**: Focus on simple push operations (rook from one square to adjacent square)
- **Tech Stack**: Computer vision, object tracking, motion analysis

### Task 2: Inverse Kinematics Engine (Jonas)
**Goal**: Convert CV outputs to robot joint trajectories

- **Input**: End-effector trajectories from Task 1
- **Output**: Time series of joint positions (motor encoder data)
- **Challenge**: Generate realistic robot motion that achieves the same task
- **Result**: Training data equivalent to teleoperated demonstrations

### Task 3: Marketplace Demo (Future)
**Goal**: End-to-end demonstration platform

- **Component A**: Task specification interface
  - Users define desired robot behaviors
  - Specify hardware requirements and constraints
- **Component B**: Data pipeline demonstration
  - Human demonstrates task via video
  - human2robot converts to training data
  - Imitation learning trains the model

---

## 🤔 Open Research Questions

### The Camera Problem
**Challenge**: Real robots need visual input during operation, not just joint trajectories.

- **Current state**: We generate joint motion data from video
- **Missing piece**: How does the robot "see" during execution?
- **Questions**:
  - Can we generate synthetic camera views for the robot's perspective?
  - How do we bridge human hand demonstrations to robot end-effector views?
  - Can we train vision models to translate between human and robot perspectives?

### Potential Solutions
1. **Domain Transfer**: Train vision models to map human→robot viewpoints
2. **Synthetic Data**: Generate robot-perspective videos from human demonstrations
3. **Multi-modal Training**: Combine trajectory data with vision adaptation
4. **View Synthesis**: Use computer graphics to render robot's perspective

---

## 🛠 Getting Started

```bash
# Clone the repository
git clone https://github.com/your-username/human2robot.git
cd human2robot

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python scripts/video_to_robot_data.py --input demo_video.mp4 --robot_config so101
```

## 📊 Pipeline Overview

```
Human Video → CV Analysis → Inverse Kinematics → Robot Training Data → Imitation Learning
     ↓              ↓              ↓                    ↓                    ↓
  demo.mp4    trajectories.json  joint_data.csv   lerobot_dataset/    trained_policy.pt
```

## 🎯 Current Status

- [x] Project setup and architecture design
- [ ] Computer vision pipeline for motion extraction
- [ ] Inverse kinematics solver implementation
- [ ] Integration testing with sample data
- [ ] LeRobot dataset format compatibility
- [ ] Demo marketplace interface

## 🤝 Contributing

This is a hackathon exploration project. We welcome:
- Ideas for solving the camera perspective problem
- Improvements to the CV→IK pipeline
- Real-world testing and validation
- Extensions to new robot platforms

---

## � Development Roadmap

### 📱 MVP Demo (Week 1)
**Goal**: End-to-end pipeline from video → robot policy

**Core Tasks:**
1. **Complete Inverse Kinematics Bridge** (`hand_to_robot_ik.py`)
   - Hand coordinate → robot workspace mapping
   - 5DOF arm + gripper IK solver integration
   - Temporal trajectory generation

2. **Data Pipeline Integration**
   - Convert CV+IK outputs to LeRobot dataset format
   - Implement observation-action pair generation
   - Add temporal synchronization and smoothing

3. **Demo Implementation**
   - Simple chess piece movement task
   - Single camera, fixed workspace setup
   - ACT policy training on generated data

### 🚀 High-Impact Research Extensions (Weeks 2-4)

**Vision & Perception Research:**
- **Multi-view Geometry**: Camera calibration and 3D reconstruction from human demonstrations
- **Domain Adaptation**: Learning visual mappings between human and robot perspectives
- **Temporal Action Segmentation**: Automatic detection of action primitives in demonstrations

**Robotics & Control:**
- **Workspace Scaling**: Adaptive mapping between human and robot workspaces
- **Trajectory Optimization**: Physics-informed smoothing and feasibility constraints
- **Multi-robot Coordination**: Extending to bimanual or multi-agent scenarios

**Machine Learning Foundations:**
- **Comparative Policy Analysis**: Systematic evaluation of ACT vs Diffusion Policy performance
- **Data Efficiency**: Few-shot learning from minimal human demonstrations
- **Uncertainty Quantification**: Confidence estimation in generated robot trajectories

**Novel Research Directions:**
- **Synthetic Data Augmentation**: Physics simulation for expanding training datasets
- **Cross-embodiment Transfer**: Learning mappings between different robot morphologies
- **Interactive Learning**: Real-time feedback and correction mechanisms

**Sensor Fusion & Modalities:**
- **LiDAR Integration**: iPhone LiDAR (256x192) depth sensing for 3D workspace understanding
- **Multi-sensor Fusion**: Combine RGB, depth, IMU, and tactile feedback
- **Sensor Modality Transfer**: Learn mappings between different sensor types

**Beyond Imitation Learning:**
- **Hybrid IL+RL**: Use IL for reliable scene reset, then overnight RL for task optimization
- **Comprehensive IL Survey**: Implement and compare SQIL, ValueDice, IQ-Learn, f-BRAC
- **Advanced IL Methods**: Adversarial IL (GAIL), Distribution Matching (PM, BC-O)
- **Simulation-to-Reality**: Isaac Lab integration for physics-based training

### 🎯 Learning & Research Skills Development

**Foundational Understanding:**
- Implement core algorithms from scratch (IK solvers, transformers, diffusion models)
- Mathematical foundations: robotics kinematics, probabilistic models, optimization
- Experimental design: hypothesis formation, systematic evaluation, statistical analysis

**Innovation Opportunities:**
- **Novel Loss Functions**: Task-specific objectives for trajectory generation
- **Architecture Design**: Custom neural network components for robotics
- **Benchmark Creation**: Standardized evaluation protocols for video-to-robot learning

### ⚠️ Hardware Access Timeline
**Critical Constraint**: Hardware access ends in 5 days, then 2-4 month gap
- **Priority**: Focus on simulation-based development (Isaac Lab, MuJoCo)
- **Data Collection Sprint**: Gather comprehensive demonstration videos while hardware available
- **Simulation-First Approach**: Develop and validate algorithms in simulation for future hardware deployment

---

## �🔗 Built on LeRobot Foundation

This project builds upon the excellent [LeRobot](https://github.com/huggingface/lerobot) framework for the imitation learning components.

### Original LeRobot Description

🤗 LeRobot aims to provide models, datasets, and tools for real-world robotics in PyTorch. The goal is to lower the barrier to entry to robotics so that everyone can contribute and benefit from sharing datasets and pretrained models.

🤗 LeRobot contains state-of-the-art approaches that have been shown to transfer to the real-world with a focus on imitation learning and reinforcement learning.

🤗 LeRobot hosts pretrained models and datasets on this Hugging Face community page: [huggingface.co/lerobot](https://huggingface.co/lerobot)

## 📝 License

This project maintains the same Apache 2.0 license as the original LeRobot framework.

---

**Note**: This project explores novel approaches to robot training data generation. The techniques are experimental and intended for research and demonstration purposes.
