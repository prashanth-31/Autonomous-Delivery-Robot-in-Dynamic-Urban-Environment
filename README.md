
# Autonomous Delivery Robot Simulator using Reinforcement Learning


An interactive simulator that trains a **Q-learning** agent to drive an autonomous delivery robot across a **dynamic 20 × 20 grid-world** full of static and moving obstacles.  
The project blends classic reinforcement learning with a Q-value-influenced **A\*** fallback planner, visualised in real time through a **Tkinter** GUI.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents
1. [Features](#features)  
2. [Quick Start](#quick-start)  
3. [Usage](#usage)  
4. [Project Structure](#project-structure)  
5. [Algorithm Details](#algorithm-details)  
6. [Configuration](#configuration)  
7. [License](#license)  

---

## Features
- **Model-free Q-Learning** with:
  - Epsilon–greedy exploration and exponential decay  
  - Distance-shaped rewards (+100 goal, -10 collision, -1 step, +Δdistance)  
- **Hybrid pathfinding** ➜ Q-table first, Q-weighted A\* if no valid policy  
- **Dynamic obstacles** that move stochastically, forcing online replanning  
- **Tkinter GUI**:
  - Start / pause / reset missions  
  - Generate new maps or add extra obstacles on the fly  
  - Adjustable animation speed and training episode count  
  - Live statistics panel and Matplotlib reward curves  
- **Model persistence** – save and reload learned Q-tables in JSON  
- **Threaded training** so the UI never freezes

---

## Quick Start

### 1. Clone and Install
```bash
git clone https://github.com/<your-username>/autonomous-delivery-robot.git
cd autonomous-delivery-robot

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Run the Simulator
```bash
python delivery_app.py
```

> **Tip:** On Linux you may need `sudo apt-get install python3-tk` for Tkinter support.

---

## Usage

| Button / Control | Description |
|------------------|-------------|
| **Train Agent**  | Runs the selected number of episodes (default = 100) in a background thread, updating the reward plot live. |
| **Start Mission**| Lets the robot follow the learned policy (or A\*) from its current position to the goal. |
| **Pause / Resume** | Toggle animation at any time. |
| **Reset** | Re-initialise the environment and agent. |
| **Generate New Map** | Create fresh static/dynamic obstacles. |
| **Add Random Obstacles** | Increase difficulty mid-session. |
| **Speed Slider** | 1 = slow-motion, 10 = fast-forward. |

---

## Project Structure
```text
├── delivery_app.py     # GUI, environment, animation loop
├── ql_agent.py         # Q-Learning agent class
├── requirements.txt    # pip dependencies
└── README.md
```

---

## Algorithm Details

### Q-Learning Update  
\[
Q(s, a) ← Q(s, a) + α [r + γ * max(Q(s', a')) - Q(s, a)]
\]

* **State (s):** Robot’s grid coordinate  
* **Action (a):** {up, down, left, right} if within bounds & not an obstacle  
* **Reward (r):**  
  * +100 for reaching the goal  
  * –10 for collisions / invalid moves  
  * –1 per step  
  * +2 × (Δ manhattan distance towards goal)  
* **α (alpha):** 0.1 **γ (gamma):** 0.95  
* **ε (epsilon) decay:** 0.995 → min 0.01

### A\* Fallback  
If the Q-table cannot produce a viable path (unexplored area or dynamic blockage), an A\* search is executed where each neighbour’s cost is **discounted by its learned Q-value**—biasing the heuristic toward states the agent rates highly.

---

## Configuration

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `GRID_SIZE` | 20 | Width/height of the square grid |
| `NUM_OBSTACLES` | 50 | Total obstacles (70 % static / 30 % dynamic) |
| `OBSTACLE_MOVE_PROB` | 0.3 | Per-step probability that a dynamic obstacle moves |
| `alpha`, `gamma` | 0.1 / 0.95 | Q-learning rate / discount |
| `epsilon_decay` | 0.995 | Exploration decay per episode |
| Etc. |

---

## License

This project is released under the **MIT License** – see the [LICENSE](LICENSE) file for details.


> **Enjoy teaching your robot to deliver! 🚚🤖**  
> Star ⭐ the repo if you find it useful.
