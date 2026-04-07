# Projectfreeingtruth
Projectfreeingtruth is an activate project that implements an offline reinforcement learning approach for bipedal locomotion using the D$RL Walker 2D dataset. The goal is to develop a stable control policy from fixed trajectory data and explore improvements to modern offline RL methods.

# Dataset: D4RL Walker2D (walker2d-medium-expert-v2)
## Description
This project uses the Walker2D locomotion dataset from the D4RL benchmark suite.

The dataset conatins state-action trajectories collected from a simulated two-dimensional bipedal robot using the MuJoCo physics engine. The objective of the agent is to learn stable walking behaviour while maximizing forward velocity.

Due to limited availability of large-scale real-world biped locomotion datasets, physics-based simulation datasets such as D4RL are commonly used as a proxy in robotics research.

# Dataset Features
Each sample in the dataset consists of a transition tuple:\
$$(s_t, a_t, r_t, s_{t+1})$$

Where:
- **State (observations)**:
    - Joint positions (angles)
    - Joint velocities
    - Torso position and velocity
    -17-dimensional continuous state vector
- **Actions**:
    - Continuous joint torques applied at each joint
    - 6-dimensional action space
- **Rewards**:
    - Scalar reward encouraging forward motion and penalizing instability and excessive energy use
- **Next State**:
    - Resulting state after applying the action
- **Terminals**:
    - Indicates episode termination (e.g., falling)

# Dataset Size
- Approximately 1,000+ trajectories (episodes)
- Over 1,000,000 state-action samples (timesteps)
- Each episode contains ~1000 steps

# How to Obtain the Dataset
The dataset is accessed using TensorFlow Datasets:

```python
import tensorflow_datasets as tfds

ds = tfds.load("d4rl_mujoco_walker2d/v2-medium-expert", split="train")
```

Through the terminal:
```bash
# by default, download to robomimic/datasets
$ python convert_d4rl.py --env walker2d-medium-expert-v2
# download to specific folder
$ python convert_d4rl.py --env walker2d-medium-expert-v2 --folder /path/to/output/folder/

```
The dataset is provided in RLDS (Reinforcement Learning Dataset Standard) format and is converted into transition tuples for training.

The dataset was introduced by researchers at UC Berkeley as part of the D4RL benchmark for offline reinforcement learning. What is contained in the dataset can better be viewed at <https://www.tensorflow.org/datasets/catalog/d4rl_mujoco_walker2d>.

## Problem and Methodology

# Problem statement
The goal of this project is to develop a learning-based control policy for bipedal locomotion. Specifically, we aim to:
- Learn stable walking gait patterns
- Maintain balance using feedback from position and velocity states
- Maximize forward velocity while minimizing instability

This represents a classical robotics control control problem involving stability, gait generation, and optimal control.

# Methodology

We use an offline reinforcement learning approach based on **Implicit Q-Learning (IQL)**.
- The **value function** learns expected returns from in-dataset actions
- The **critic (Q-function)** evaluates state-action pairs
- The **actor network** learns a policy using advantage-weighted regression

Unlike standard off-policy algorithms such as SAC, IQL avoids evaluating out-of-distribution actions, making it well-suited for learning from fixed datasets.

The learned policy effectively serves as a feedback controller for the Walker2d system.

