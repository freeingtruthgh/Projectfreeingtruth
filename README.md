# Projectfreeingtruth
Projectfreeingtruth is an activate project that for now contains a README for what I plan to complete as a class project.

# Dataset: D4RL Walker2D (walker2d-medium-v2)
## Description
This project uses the Walker2D locomotion dataset from the D4RL benchmark suite.

The dataset conatins state-action trajectories collected from a simulated two-dimensional bipedal robot using the MuJoCo physics engine. The objective of the agent is to learn stable walking behaviour while maximizing forward velocity.

Due to limited availability of large-scale real-world biped locomotion datasets, physics-based simulation datasets such as D4RL are commonly used as a proxy in robotics research.

# Dataset Features
Each sample inn the dataset consists of a transition tuple:\
$$(s_t, a_t, r_t, s_{t+1})$$

Where:
- State (observations):
    - Joint positions (angles)
    - Joint velocities
    - Torso position and velocity
- Actions:
    - Continuous joint torques applied at each joint
- Rewards:
    - Scalar reward encouraging forward motion and penalizing instability and excessive energy use
- Next State:
    - Resulting state after applying the action
- Terminals:
    - Indicates episode termination (e.g., falling)

# Dataset Size
- Approximately 1,000 trajectories (episodes)
- Over 1,000,000 state-action samples (timesteps)

## Problem and Methodology
# Problem statement
The goal of this project is to develop a learning-based control policy for bipedal locomotion. Specifically, we aim to:
- Learn stable walking gait patterns
- Maintain balance using feedback from position and velocity states
- Maximize forward velocity while minimizing instability

This represents a classical robotics control control problem involving stability, gait generation, and optimal control.

# Methodology
We use an offline reinforcement learning approach based on the Soft Actor-Critic (SAC) framework.
- The **actor network** learns a policy mapping states to actions
- The **critic network** learns a value function that evaluates the quality of actions

The dataset provides supervision in the form of state transitions and rewards, allowing the model to learn without interacting with a live environment.

The learned policy effectively serves as a feedback controller for the bipedal system.

# How to Obtain the Dataset
**Option 1: Official Repository**
- GitHub: <https://github.com/Farama-Foundation/D4RL>

**Option 2: Direct Acccess via Python**
This code was ran in colab and worked. The conversion is a cleaner dataset and after this python 3.12 can be used for Deep Learning
```python
!pip install "numpy<2.0"
!pip install "gym==0.23.1"
!pip install d4rl
!pip install robomimic
!python convert_d4rl.py --env walker2d-medium-expert-v2
```

Through the terminal:
```bash
# by default, download to robomimic/datasets
$ python convert_d4rl.py --env walker2d-medium-expert-v2
# download to specific folder
$ python convert_d4rl.py --env walker2d-medium-expert-v2 --folder /path/to/output/folder/

```
The dataset was introduced by researchers at UC Berkeley as part of the D4RL benchmark for offline reinforcement learning.

What is contained in the dataset can better be viewed at <https://www.tensorflow.org/datasets/catalog/d4rl_mujoco_walker2d>. The conversion is given by robotmimic which can be viewed at <https://robomimic.github.io/docs/datasets/d4rl.html>.