# 🚨 U### Day 1-2: SO-101 Robot Data Collection & Testing
**Priority 1: Robot Teleoperation Data**
- [x] Record SO-101 teleoperation demonstrations (chess piece movements)
- [x] Collect 6DOF joint position data (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
- [x] Record robot's wrist camera feed during demonstrations
- [x] Test robot workspace bounds and joint limits (sts3215 Feetech servos)

**Priority 2: Robot-Specific Calibration**
- [x] SO-101 wrist camera calibration and workspace mapping
- [x] Validate Feetech sts3215 joint angle accuracy and safety limits
- [ ] End-effector pose accuracy testing with built-in kinematics
- [ ] Robot response time and smoothness evaluation (leader-follower latency)

**Priority 3: SO-101 Hardware Documentation**
- [ ] Document actual DH parameters and link lengths
- [ ] Record joint limits and safe operating speeds
- [ ] Test leader arm torque mode and gripper behavior
- [ ] Measure workspace reachability and singularities LeRobot SO-101 Sprint Plan

## Critical Mission: Maximize SO-101 Robot Testing Before Travel

> **Hardware Constraint**: SO-101 robot cannot travel, but hand video collection can continue later

### Day 1-2: SO-101 Robot Data Collection & Testing
**Priority 1: Robot Teleoperation Data**
- [x] Record SO-101 teleoperation demonstrations (chess piece movements)
- [x] Collect robot joint position data during teleoperation
- [x] Record robot's camera feed during demonstrations
- [x] Test robot workspace bounds and joint limits

**Priority 2: Robot-Specific Calibration**
- [ ] SO-101 camera calibration and workspace mapping
- [ ] Joint angle validation and safety limits
- [ ] End-effector pose accuracy testing
- [ ] Robot response time and smoothness evaluation

### Day 3-4: MVP Implementation & Robot Integration
- [ ] Complete `WorkspaceMapper.map_coordinates()` with SO-101 workspace
- [ ] Implement IK solver for SO-101's specific kinematics
- [ ] Test generated trajectories on actual SO-101 robot
- [ ] Validate safety and joint limit enforcement

### Day 5: Robot Documentation & Remote Development Setup
- [ ] Document SO-101 specifications and limitations
- [ ] Create robot simulation model for Isaac Lab
- [ ] Test remote development workflow (simulation-to-robot pipeline)
- [ ] Establish baseline performance metrics for future comparison

## Post-Hardware Strategy (2-4 Months)

### Week 1-2: Simulation Foundation + Hand Video Collection
- Isaac Lab SO-101 environment creation
- **Continue hand demonstration recording** (iPhone LiDAR + RGB)
- Replay SO-101 teleoperation data in simulation
- Implement IL algorithm comparisons

### Week 3-8: Research Deep Dive
- Comprehensive IL survey implementation  
- Hand-to-robot trajectory mapping development
- Hybrid IL+RL pipeline design using SO-101 simulation
- Multi-modal learning (hand videos + robot data fusion)

### Week 9-16: Advanced Research
- Novel architecture design for cross-embodiment transfer
- Sim2Real transfer techniques validated on SO-101 model
- Preparation for robot access return with tested algorithms

## Key Research Questions to Answer During Robot-Free Period:
1. **How accurately can we map hand demonstrations to SO-101 trajectories?**
2. **Which IL algorithm works best for SO-101 manipulation tasks?**
3. **How can we effectively combine human demos + robot teleoperation data?**
4. **What's the optimal hand-to-robot transfer strategy?**

## Ongoing Data Collection (While Traveling):
- [ ] Hand demonstration videos (multiple camera angles)
- [ ] iPhone LiDAR + RGB recordings
- [ ] Different manipulation styles and objects
- [ ] Varied environments and lighting conditions

---

**Strategy**: Use the 5-day sprint to master SO-101 robotics fundamentals, then leverage the travel period to become an expert in simulation-based development and hand-video analysis. Return with validated algorithms ready for immediate robot deployment!
