# Adaptive CNC Machining Using Reinforcement Learning
## Standalone Class-Project Demonstration

This is a small, self-contained educational simulation showing how a reinforcement-learning agent can select CNC milling parameters and learn from the consequences.

### What it demonstrates
- A simplified CNC milling environment.
- A tabular Q-learning agent.
- Three adjustable parameters: spindle speed, feed rate, and depth of cut.
- A reward function balancing productivity, tool wear, vibration, and temperature.
- A comparison against a fixed-parameter baseline.
- Automatically generated graphs and CSV evaluation results.

### Requirements
- Python 3.8 or later
- NumPy
- Matplotlib

Install dependencies:

```bash
python -m pip install numpy matplotlib
```

### Run

```bash
python cnc_rl_demo.py
```

The simulation trains for 500 episodes and evaluates the learned policy over 100 matched episodes against a fixed baseline. It should complete quickly on an ordinary laptop.

### Output
The `results/` folder contains:
- `training_curve.png`: episode reward during learning.
- `controller_comparison.png`: normalized comparison of the two controllers.
- `evaluation_summary.csv`: average evaluation metrics.
- `episode_results.csv`: per-episode evaluation data.

### How the RL agent works
1. The environment reports a state based on tool wear, vibration, and temperature.
2. The agent selects one combination of spindle speed, feed rate, and depth of cut.
3. The simulated machining process calculates productivity, wear, vibration, and temperature.
4. A reward is returned. Productive settings earn reward, while excessive wear and unsafe operating conditions are penalized.
5. Q-learning updates the value of the chosen action for that state.
6. Repeating this process allows the agent to learn a policy.

### Important limitation
The equations and thresholds are deliberately simplified and illustrative. They are not calibrated against a real CNC machine, cutting tool, workpiece, or experimental dataset. This project demonstrates the RL workflow only and must not be used to control physical machinery.
