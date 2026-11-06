"""
Standalone CNC Reinforcement Learning Demonstration
---------------------------------------------------
A small tabular Q-learning project for a simulated CNC milling process.

This is an educational simulation, not a physically calibrated CNC controller.
Dependencies: numpy, matplotlib
Run: python cnc_rl_demo.py
"""

from __future__ import annotations
import csv
import itertools
import os
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


# --------------------------- CNC SIMULATOR ---------------------------

SPINDLE_LEVELS = [3000, 4500, 6000]       # rpm
FEED_LEVELS = [100, 160, 220]             # mm/min (illustrative)
DEPTH_LEVELS = [0.5, 0.8, 1.1]            # mm
ACTIONS = list(itertools.product(range(3), range(3), range(3)))
MAX_STEPS = 60


@dataclass
class CNCState:
    wear: float = 0.05
    vibration: float = 0.45
    temperature: float = 65.0


class CNCMillingEnv:
    """Toy CNC environment with simplified, illustrative process equations."""

    def __init__(self, seed: int = 0, max_steps: int = MAX_STEPS):
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.state = CNCState(
            wear=self.rng.uniform(0.02, 0.08),
            vibration=self.rng.uniform(0.30, 0.55),
            temperature=self.rng.uniform(58, 72),
        )
        self.steps = 0
        self.total_reward = 0.0
        self.total_volume = 0.0
        self.violations = 0
        return self._discretize()

    def _discretize(self):
        # 5 wear bins x 3 vibration bins x 3 temperature bins = 45 states
        w = min(4, int(self.state.wear / 0.20))
        v = 0 if self.state.vibration < 0.70 else (1 if self.state.vibration < 1.0 else 2)
        t = 0 if self.state.temperature < 80 else (1 if self.state.temperature < 110 else 2)
        return (w * 3 + v) * 3 + t

    def step(self, action_index: int):
        si, fi, di = ACTIONS[action_index]
        spindle = SPINDLE_LEVELS[si]
        feed = FEED_LEVELS[fi]
        depth = DEPTH_LEVELS[di]

        sr, fr, dr = spindle / 4500.0, feed / 160.0, depth / 0.8

        # Simplified process indicators with small process variation.
        noise_v = self.rng.uniform(-0.035, 0.035)
        noise_t = self.rng.uniform(-1.5, 1.5)
        vibration = max(0.05, 0.30 + 0.20 * sr + 0.22 * fr + 0.28 * dr
                        + 0.18 * max(0, dr - 1) * max(0, fr - 1) + noise_v)
        temperature = max(20, 42 + 17 * sr + 13 * fr + 15 * dr + noise_t)

        # Wear increases faster under high speed, feed, depth and vibration.
        wear_increment = (0.0015 * (sr ** 1.35) * (fr ** 1.15) * (dr ** 1.20)
                          * (1 + 0.45 * max(0, vibration - 0.75)))
        self.state.wear = min(1.2, self.state.wear + wear_increment)
        self.state.vibration = vibration
        self.state.temperature = temperature

        # Material removal proxy: larger feed/depth improves productivity.
        productivity = (feed * depth) / (160.0 * 0.8)
        self.total_volume += productivity

        vibration_bad = vibration > 1.0
        temperature_bad = temperature > 105
        wear_bad = self.state.wear >= 0.85
        if vibration_bad or temperature_bad or wear_bad:
            self.violations += 1

        # Reward encourages productivity but penalizes wear and unsafe conditions.
        reward = (
            2.0 * productivity
            - 1.8 * wear_increment * 100
            - 1.4 * max(0, vibration - 0.85) ** 2 * 5
            - 0.025 * max(0, temperature - 90)
            - (2.0 if vibration_bad else 0.0)
            - (1.5 if temperature_bad else 0.0)
            - (3.0 if wear_bad else 0.0)
        )

        self.steps += 1
        done = self.steps >= self.max_steps or wear_bad
        self.total_reward += reward
        info = {
            "spindle_rpm": spindle,
            "feed_mm_min": feed,
            "depth_mm": depth,
            "productivity": productivity,
            "wear": self.state.wear,
            "vibration": vibration,
            "temperature": temperature,
            "violations": self.violations,
            "volume_proxy": self.total_volume,
        }
        return self._discretize(), reward, done, info


# --------------------------- Q-LEARNING AGENT ---------------------------

class QLearningAgent:
    def __init__(self, n_states=45, n_actions=len(ACTIONS), alpha=0.15,
                 gamma=0.92, epsilon=1.0, epsilon_min=0.04,
                 epsilon_decay=0.992, seed=0):
        self.q = np.zeros((n_states, n_actions), dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(ACTIONS)))
        return int(np.argmax(self.q[state]))

    def learn(self, state, action, reward, next_state, done):
        target = reward if done else reward + self.gamma * np.max(self.q[next_state])
        self.q[state, action] += self.alpha * (target - self.q[state, action])

    def finish_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# --------------------------- TRAINING / EVALUATION ---------------------------

def run_episode(env, agent=None, training=False, fixed_action=None):
    state = env.reset()
    total_reward = 0.0
    last_info = {}
    for _ in range(env.max_steps):
        if fixed_action is not None:
            action = fixed_action
        else:
            action = agent.choose_action(state, explore=training)
        next_state, reward, done, info = env.step(action)
        if training:
            agent.learn(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
        last_info = info
        if done:
            break
    return {
        "reward": total_reward,
        "steps": env.steps,
        "violations": env.violations,
        "wear": env.state.wear,
        "volume_proxy": env.total_volume,
        "final_temperature": env.state.temperature,
        "final_vibration": env.state.vibration,
    }


def train_agent(episodes=500, seed=7):
    env = CNCMillingEnv(seed=seed)
    agent = QLearningAgent(seed=seed)
    history = []
    for ep in range(episodes):
        result = run_episode(env, agent, training=True)
        agent.finish_episode()
        history.append(result["reward"])
    return agent, history


def evaluate(agent, episodes=100, seed=1000):
    rl_results, baseline_results = [], []
    # Fixed medium setting: 4500 rpm, 160 mm/min, 0.8 mm.
    baseline_action = ACTIONS.index((1, 1, 1))
    for i in range(episodes):
        rl_env = CNCMillingEnv(seed=seed + i)
        base_env = CNCMillingEnv(seed=seed + i)
        rl_results.append(run_episode(rl_env, agent, training=False))
        baseline_results.append(run_episode(base_env, fixed_action=baseline_action))
    return rl_results, baseline_results


def mean_metric(rows, key):
    return float(np.mean([r[key] for r in rows]))


def save_outputs(history, rl, baseline, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    metrics = ["reward", "steps", "violations", "wear", "volume_proxy"]
    summary = {}
    for label, rows in [("Q-learning agent", rl), ("Fixed baseline", baseline)]:
        summary[label] = {m: mean_metric(rows, m) for m in metrics}

    with open(os.path.join(out_dir, "evaluation_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Controller"] + metrics)
        for label, vals in summary.items():
            writer.writerow([label] + [vals[m] for m in metrics])

    with open(os.path.join(out_dir, "episode_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["controller", "episode"] + metrics + ["final_temperature", "final_vibration"])
        for label, rows in [("Q-learning agent", rl), ("Fixed baseline", baseline)]:
            for i, r in enumerate(rows, 1):
                writer.writerow([label, i] + [r[m] for m in metrics]
                                + [r["final_temperature"], r["final_vibration"]])

    # Training curve with moving average.
    plt.figure(figsize=(9, 4.8))
    plt.plot(history, alpha=0.28, label="Episode reward")
    window = min(25, max(5, len(history) // 10))
    if len(history) >= window:
        moving = np.convolve(history, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(history)), moving, linewidth=2, label=f"{window}-episode moving average")
    plt.title("Q-learning training progress")
    plt.xlabel("Training episode")
    plt.ylabel("Total episode reward")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curve.png"), dpi=160)
    plt.close()

    # Controller comparison.
    labels = ["Reward", "Steps", "Violations", "Tool wear", "Material removal\n(proxy)"]
    keys = ["reward", "steps", "violations", "wear", "volume_proxy"]
    x = np.arange(len(keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    # Normalize each metric by the larger absolute mean so unlike units can be compared visually.
    rl_vals = [summary["Q-learning agent"][k] for k in keys]
    base_vals = [summary["Fixed baseline"][k] for k in keys]
    scales = [max(abs(a), abs(b), 1e-9) for a, b in zip(rl_vals, base_vals)]
    ax.bar(x - width/2, [a/s for a, s in zip(rl_vals, scales)], width, label="Q-learning")
    ax.bar(x + width/2, [b/s for b, s in zip(base_vals, scales)], width, label="Fixed baseline")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Normalized mean (per metric)")
    ax.set_title("Controller comparison (each metric normalized separately)")
    ax.axhline(0, linewidth=0.8)
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "controller_comparison.png"), dpi=160)
    plt.close(fig)

    return summary


def main():
    episodes = 500
    print("\n=== Standalone Adaptive CNC RL Demonstration ===")
    print("Training a tabular Q-learning agent...")
    agent, history = train_agent(episodes=episodes, seed=7)
    print(f"Training complete. Final exploration rate: {agent.epsilon:.3f}")

    print("\nEvaluating learned policy against a fixed baseline...")
    rl, baseline = evaluate(agent, episodes=100, seed=1000)
    summary = save_outputs(history, rl, baseline)

    print("\nAverage results over 100 matched evaluation episodes:")
    print(f"{'Metric':<22}{'Q-learning':>15}{'Fixed baseline':>18}")
    for metric in ["reward", "steps", "violations", "wear", "volume_proxy"]:
        a = summary["Q-learning agent"][metric]
        b = summary["Fixed baseline"][metric]
        print(f"{metric:<22}{a:>15.3f}{b:>18.3f}")

    print("\nFiles saved in ./results/")
    print("- training_curve.png")
    print("- controller_comparison.png")
    print("- evaluation_summary.csv")
    print("- episode_results.csv")
    print("\nNote: This is a simplified educational model, not a calibrated CNC process model.")


if __name__ == "__main__":
    main()
