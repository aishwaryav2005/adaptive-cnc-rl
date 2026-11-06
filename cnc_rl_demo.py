"""
Standalone CNC Reinforcement Learning Demonstration
---------------------------------------------------
A small tabular Q-learning project for a simulated CNC milling process.

This is an educational simulation, not a physically calibrated CNC controller.
Dependencies: numpy, matplotlib
Run: python cnc_rl_demo.py

The unmodified first version of this file is kept in original_backup/.
"""

from __future__ import annotations

import csv
import itertools
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# --------------------------- SIMULATION SETTINGS ---------------------------

SPINDLE_LEVELS = [3000, 4500, 6000]       # rpm
FEED_LEVELS = [100, 160, 220]             # mm/min (illustrative)
DEPTH_LEVELS = [0.5, 0.8, 1.1]            # mm
ACTIONS = list(itertools.product(range(3), range(3), range(3)))
MAX_STEPS = 60

# Limits. Exceeding one counts as a violation; reaching the wear limit ends the episode.
VIBRATION_LIMIT = 1.0                     # normalised vibration index
TEMPERATURE_LIMIT = 105.0                 # illustrative cutting temperature, deg C
WEAR_LIMIT = 0.85                         # normalised tool wear (1.0 = fully worn)
# Above these "soft" levels the reward starts to apply a graded penalty.
VIBRATION_SOFT = 0.85
TEMPERATURE_SOFT = 90.0

WEAR_RATE = 0.012                         # wear per step at the medium setting (before vibration effect)
WEAR_PENALTY = 22.5                       # reward cost per unit of wear
WEAR_VIBRATION_GAIN = 0.25                # a worn tool vibrates more
WEAR_TEMPERATURE_GAIN = 20.0              # a worn tool runs hotter

MEDIUM_SETTING = (1, 1, 1)                # 4500 rpm, 160 mm/min, 0.8 mm

# Seeds are kept apart so that nothing is tuned on the evaluation episodes.
TRAINING_SEED = 7
TRAINING_EPISODES = 5000                  # chosen on tuning seeds: 500 varied too much between training seeds
EVALUATION_SEED = 1000                    # episodes 1000..1099
EVALUATION_EPISODES = 100
TUNING_SEED = 5000                        # used only to choose the best fixed setting
TUNING_EPISODES = 50
MOVING_AVERAGE_WINDOW = 100
SEED_STUDY = [7, 11, 23, 42, 101]         # chosen before any result was seen

# 5 wear bins x 3 vibration bins x 3 temperature bins = 45 states
N_STATES = 45
WEAR_BIN_NAMES = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8+"]
VIBRATION_BIN_NAMES = ["normal", "elevated", "over limit"]
TEMPERATURE_BIN_NAMES = ["normal", "elevated", "over limit"]


def action_settings(action_index: int):
    """(spindle rpm, feed mm/min, depth mm) of an action index."""
    si, fi, di = ACTIONS[action_index]
    return SPINDLE_LEVELS[si], FEED_LEVELS[fi], DEPTH_LEVELS[di]


def state_bins(state_index: int):
    """(wear bin, vibration bin, temperature bin) of a state index."""
    return state_index // 9, (state_index // 3) % 3, state_index % 3


# --------------------------- CNC SIMULATOR ---------------------------

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
        # Bin edges coincide with the reward's soft levels and the violation limits.
        w = min(4, int(self.state.wear / 0.20))
        v = 0 if self.state.vibration < VIBRATION_SOFT else (1 if self.state.vibration <= VIBRATION_LIMIT else 2)
        t = 0 if self.state.temperature < TEMPERATURE_SOFT else (1 if self.state.temperature <= TEMPERATURE_LIMIT else 2)
        return (w * 3 + v) * 3 + t

    def step(self, action_index: int):
        spindle, feed, depth = action_settings(action_index)
        sr, fr, dr = spindle / 4500.0, feed / 160.0, depth / 0.8
        wear_before = self.state.wear

        # Simplified process indicators with small process variation. Wear feeds back into
        # vibration and temperature, so a setting that is safe for a new tool may not stay safe.
        noise_v = self.rng.uniform(-0.035, 0.035)
        noise_t = self.rng.uniform(-1.5, 1.5)
        vibration = max(0.05, 0.20 + 0.20 * sr + 0.22 * fr + 0.28 * dr
                        + 0.18 * max(0, dr - 1) * max(0, fr - 1)
                        + WEAR_VIBRATION_GAIN * wear_before + noise_v)
        temperature = max(20, 42 + 17 * sr + 13 * fr + 15 * dr
                          + WEAR_TEMPERATURE_GAIN * wear_before + noise_t)

        # Wear increases faster under high speed, feed, depth and vibration.
        wear_increment = (WEAR_RATE * (sr ** 1.35) * (fr ** 1.15) * (dr ** 1.20)
                          * (1 + 0.45 * max(0, vibration - 0.75)))
        self.state.wear = min(1.2, wear_before + wear_increment)
        self.state.vibration = vibration
        self.state.temperature = temperature

        # Material removal proxy: larger feed/depth improves productivity.
        productivity = (feed * depth) / (160.0 * 0.8)
        self.total_volume += productivity

        vibration_bad = vibration > VIBRATION_LIMIT
        temperature_bad = temperature > TEMPERATURE_LIMIT
        wear_bad = self.state.wear >= WEAR_LIMIT
        if vibration_bad or temperature_bad or wear_bad:
            self.violations += 1

        # Reward encourages productivity but penalizes wear and unsafe conditions.
        reward = (
            2.0 * productivity
            - WEAR_PENALTY * wear_increment
            - 1.4 * max(0, vibration - VIBRATION_SOFT) ** 2 * 5
            - 0.025 * max(0, temperature - TEMPERATURE_SOFT)
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
            "tool_failed": wear_bad,       # a true terminal state
            "vibration_bad": vibration_bad,
            "temperature_bad": temperature_bad,
        }
        return self._discretize(), reward, done, info


# --------------------------- Q-LEARNING AGENT ---------------------------

class QLearningAgent:
    def __init__(self, n_states=N_STATES, n_actions=len(ACTIONS), alpha=0.15,
                 gamma=0.92, epsilon=1.0, epsilon_min=0.04,
                 epsilon_decay=0.992, seed=0):
        self.q = np.zeros((n_states, n_actions), dtype=float)
        self.visits = np.zeros((n_states, n_actions), dtype=int)
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

    def learn(self, state, action, reward, next_state, terminal):
        # Only a real terminal state (tool failure) cuts off the future. Reaching the step limit
        # does not: the state has no clock in it, so the process would otherwise look as if it
        # randomly stopped being worth anything.
        target = reward if terminal else reward + self.gamma * np.max(self.q[next_state])
        self.q[state, action] += self.alpha * (target - self.q[state, action])
        self.visits[state, action] += 1

    def finish_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# --------------------------- TRAINING / EVALUATION ---------------------------

def run_episode(env, agent=None, training=False, fixed_action=None):
    state = env.reset()
    total_reward = 0.0
    tool_failed = False
    for _ in range(env.max_steps):
        if fixed_action is not None:
            action = fixed_action
        else:
            action = agent.choose_action(state, explore=training)
        next_state, reward, done, info = env.step(action)
        if training:
            agent.learn(state, action, reward, next_state, info["tool_failed"])
        total_reward += reward
        state = next_state
        tool_failed = info["tool_failed"]
        if done:
            break
    return {
        "reward": total_reward,
        "steps": env.steps,
        "violations": env.violations,
        "wear": env.state.wear,
        "volume_proxy": env.total_volume,
        "tool_failed": float(tool_failed),
        "final_temperature": env.state.temperature,
        "final_vibration": env.state.vibration,
    }


def moving_average(values, window):
    """Trailing mean; the first entries average over as many values as exist so far."""
    values = np.asarray(values, dtype=float)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    out = np.empty(len(values))
    for i in range(len(values)):
        start = max(0, i + 1 - window)
        out[i] = (cumulative[i + 1] - cumulative[start]) / (i + 1 - start)
    return out


def train_agent(episodes=TRAINING_EPISODES, seed=TRAINING_SEED, window=MOVING_AVERAGE_WINDOW):
    env = CNCMillingEnv(seed=seed)
    agent = QLearningAgent(seed=seed)
    history = []
    for ep in range(1, episodes + 1):
        result = run_episode(env, agent, training=True)
        history.append({"episode": ep, "epsilon": agent.epsilon, "reward": result["reward"],
                        "steps": result["steps"], "violations": result["violations"],
                        "wear": result["wear"], "volume_proxy": result["volume_proxy"]})
        agent.finish_episode()
    averaged = moving_average([h["reward"] for h in history], window)
    for h, m in zip(history, averaged):
        h["moving_average_reward"] = float(m)
    return agent, history


def find_best_fixed_action(seed=TUNING_SEED, episodes=TUNING_EPISODES):
    """Best constant setting, chosen on tuning seeds that are never used for evaluation."""
    means = []
    for action in range(len(ACTIONS)):
        rows = [run_episode(CNCMillingEnv(seed=seed + i), fixed_action=action) for i in range(episodes)]
        means.append(float(np.mean([r["reward"] for r in rows])))
    return int(np.argmax(means))


def make_controllers(agent, best_fixed_action) -> Dict[str, Callable]:
    medium = ACTIONS.index(MEDIUM_SETTING)
    return {
        "Q-learning agent": lambda env: run_episode(env, agent, training=False),
        "Fixed baseline (medium)": lambda env: run_episode(env, fixed_action=medium),
        "Best fixed setting": lambda env: run_episode(env, fixed_action=best_fixed_action),
    }


def evaluate(controllers, episodes=EVALUATION_EPISODES, seed=EVALUATION_SEED):
    """Every controller runs on the same seeds: same start state and same noise sequence."""
    return {label: [fn(CNCMillingEnv(seed=seed + i)) for i in range(episodes)]
            for label, fn in controllers.items()}


METRICS = ["reward", "steps", "violations", "wear", "volume_proxy", "tool_failed"]
METRIC_LABELS = {"reward": "Total reward", "steps": "Machining steps", "violations": "Violations",
                 "wear": "Final tool wear", "volume_proxy": "Material removal (proxy)",
                 "tool_failed": "Tool-failure rate"}
Q_LABEL = "Q-learning agent"


def mean_metric(rows, key):
    return float(np.mean([r[key] for r in rows]))


def summarize(results):
    return {label: {m: mean_metric(rows, m) for m in METRICS} for label, rows in results.items()}


# Two-sided 95 % Student-t value for df = 99 (100 paired episodes).
T_975_DF99 = 1.984


def paired_differences(results, reference=Q_LABEL):
    """Mean of (reference - other) over matched episodes, with a 95 % confidence half-width."""
    rows = []
    for label, other in results.items():
        if label == reference:
            continue
        for m in METRICS:
            d = np.array([a[m] - b[m] for a, b in zip(results[reference], other)], dtype=float)
            half = T_975_DF99 * d.std(ddof=1) / np.sqrt(len(d))
            rows.append({"comparison": f"{reference} minus {label}", "metric": m, "mean_difference": float(d.mean()),
                         "ci95_half_width": float(half), "excludes_zero": bool(abs(d.mean()) > half)})
    return rows


def training_seed_study(best_fixed_action, seeds=SEED_STUDY, episodes=TRAINING_EPISODES):
    """Retrain with several seeds and evaluate each agent on the same evaluation episodes."""
    fixed = [run_episode(CNCMillingEnv(seed=EVALUATION_SEED + i), fixed_action=best_fixed_action)
             for i in range(EVALUATION_EPISODES)]
    rows = []
    for s in seeds:
        agent, _ = train_agent(episodes=episodes, seed=s)
        rl = [run_episode(CNCMillingEnv(seed=EVALUATION_SEED + i), agent) for i in range(EVALUATION_EPISODES)]
        rows.append({"training_seed": s, **{m: mean_metric(rl, m) for m in METRICS},
                     "reward_minus_best_fixed": mean_metric(rl, "reward") - mean_metric(fixed, "reward")})
    return rows


def simulate_episode(controller: Callable[[int], int], seed: int) -> List[dict]:
    """Per-step record of one episode. `controller(state)` returns an action index."""
    env = CNCMillingEnv(seed=seed)
    state = env.reset()
    rows, cumulative = [], 0.0
    for t in range(1, env.max_steps + 1):
        action = controller(state)
        next_state, reward, done, info = env.step(action)
        cumulative += reward
        rows.append({"step": t, "state": state, "action": action, "reward": reward,
                     "cumulative_reward": cumulative, "next_state": next_state, **info})
        state = next_state
        if done:
            break
    return rows


def greedy_policy_rows(agent):
    """Learned action per visited state (states never visited in training are left out)."""
    rows = []
    for s in range(N_STATES):
        visits = int(agent.visits[s].sum())
        if visits == 0:
            continue
        w, v, t = state_bins(s)
        spindle, feed, depth = action_settings(int(np.argmax(agent.q[s])))
        rows.append({"state": s, "wear": WEAR_BIN_NAMES[w], "vibration": VIBRATION_BIN_NAMES[v],
                     "temperature": TEMPERATURE_BIN_NAMES[t], "visits": visits, "spindle_rpm": spindle,
                     "feed_mm_min": feed, "depth_mm": depth, "best_q_value": float(np.max(agent.q[s]))})
    return rows


# --------------------------- OUTPUT FILES ---------------------------

CONTROLLER_COLORS = {"Q-learning agent": "#2a78d6", "Fixed baseline (medium)": "#eb6834", "Best fixed setting": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SIM_NOTE = "Simplified educational simulation; not calibrated against a real machine."


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _write_csv(path, rows, fields=None):
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curve(history, path):
    episodes = [h["episode"] for h in history]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    top.plot(episodes, [h["reward"] for h in history], color="#a9c8ee", linewidth=0.6, label="Episode reward")
    top.plot(episodes, [h["moving_average_reward"] for h in history], color=CONTROLLER_COLORS[Q_LABEL], linewidth=2,
             label=f"Moving average ({MOVING_AVERAGE_WINDOW} episodes)")
    top.set_ylabel("Total episode reward", color=MUTED)
    top.set_title("Q-learning training progress", color=INK, loc="left", fontsize=12)
    top.legend(frameon=False, fontsize=9, loc="lower right")
    bottom.plot(episodes, [h["epsilon"] for h in history], color=CONTROLLER_COLORS[Q_LABEL], linewidth=2)
    bottom.set_ylabel("Exploration rate", color=MUTED)
    bottom.set_xlabel("Training episode", color=MUTED)
    bottom.set_ylim(0, 1.05)
    for ax in (top, bottom):
        _style(ax)
    fig.text(0.99, 0.005, SIM_NOTE, ha="right", fontsize=7, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_comparison(results, summary, path):
    """One small panel per metric, each on its own scale, so units are never mixed."""
    keys = ["reward", "violations", "wear", "steps", "volume_proxy"]
    labels = list(summary)
    fig, axes = plt.subplots(1, len(keys), figsize=(12.5, 3.9))
    for ax, key in zip(axes, keys):
        values = [summary[c][key] for c in labels]
        half = [T_975_DF99 * np.std([r[key] for r in results[c]], ddof=1) / np.sqrt(len(results[c])) for c in labels]
        ax.bar(range(len(labels)), values, width=0.62, yerr=half, ecolor=MUTED, capsize=3, error_kw={"linewidth": 0.8},
               color=[CONTROLLER_COLORS[c] for c in labels])
        for i, v in enumerate(values):
            ax.annotate(f"{v:.3f}" if key == "wear" else f"{v:.1f}", (i, max(v, 0) + half[i]), ha="center", va="bottom",
                        xytext=(0, 3), textcoords="offset points", fontsize=8.5, color=INK)
        ax.set_title(METRIC_LABELS[key], fontsize=10, color=INK)
        ax.set_xticks([])
        ax.margins(y=0.14)
        _style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CONTROLLER_COLORS[c]) for c in labels]
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9.5)
    fig.text(0.99, 0.01, f"Mean of {len(results[labels[0]])} matched episodes; error bars = 95 % CI. {SIM_NOTE}",
             ha="right", fontsize=7, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_experiment_summary(path, agent, history, best_action, summary, paired, seed_rows, episodes):
    best = action_settings(best_action)
    medium = action_settings(ACTIONS.index(MEDIUM_SETTING))
    lines = [
        "# Experiment summary", "",
        "*Generated by `cnc_rl_demo.py`. Simulation results only: the process model is simplified and is not "
        "calibrated against a real machine.*", "",
        "## Settings", "",
        f"- Algorithm: tabular Q-learning, {N_STATES} states, {len(ACTIONS)} actions",
        f"- Learning rate {agent.alpha}, discount factor {agent.gamma}, epsilon 1.0 decaying by "
        f"{agent.epsilon_decay} per episode to {agent.epsilon_min}",
        f"- Training: {len(history)} episodes of up to {MAX_STEPS} steps, seed {TRAINING_SEED}",
        f"- Limits: vibration {VIBRATION_LIMIT}, temperature {TEMPERATURE_LIMIT} C, tool wear {WEAR_LIMIT}", "",
        "## Evaluation conditions", "",
        f"- {episodes} episodes on seeds {EVALUATION_SEED}-{EVALUATION_SEED + episodes - 1}, never used in training",
        "- Every controller runs on the same seeds, so start state and noise sequence match",
        "- Q-learning runs greedily (no exploration, no further learning)",
        f"- Fixed baseline (medium): {medium[0]} rpm, {medium[1]} mm/min, {medium[2]} mm",
        f"- Best fixed setting: {best[0]} rpm, {best[1]} mm/min, {best[2]} mm, the best of all {len(ACTIONS)} constant "
        f"settings on separate tuning seeds {TUNING_SEED}-{TUNING_SEED + TUNING_EPISODES - 1}", "",
        "## Results (mean per episode)", "",
        "| Metric | " + " | ".join(summary) + " |",
        "|---|" + "---|" * len(summary),
    ]
    for m in METRICS:
        lines.append(f"| {METRIC_LABELS[m]} | " + " | ".join(f"{summary[c][m]:.3f}" for c in summary) + " |")
    lines += ["", "## Paired differences (Q-learning minus the other controller, 95 % CI)", "",
              "| Compared with | Metric | Mean difference | Excludes zero |", "|---|---|---|---|"]
    for r in paired:
        lines.append(f"| {r['comparison'].split(' minus ')[1]} | {METRIC_LABELS[r['metric']]} | "
                     f"{r['mean_difference']:+.3f} ± {r['ci95_half_width']:.3f} | {'yes' if r['excludes_zero'] else 'no'} |")
    lines += ["", "## Training-seed study (same evaluation episodes)", "",
              "| Training seed | Reward | Violations | Tool-failure rate | Reward minus best fixed |", "|---|---|---|---|---|"]
    for r in seed_rows:
        lines.append(f"| {r['training_seed']} | {r['reward']:.2f} | {r['violations']:.2f} | {r['tool_failed']:.2f} | "
                     f"{r['reward_minus_best_fixed']:+.2f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def save_outputs(agent, history, results, summary, paired, seed_rows, best_action, out_dir=RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    _write_csv(os.path.join(out_dir, "training_history.csv"), history,
               ["episode", "epsilon", "reward", "moving_average_reward", "steps", "violations", "wear", "volume_proxy"])
    _write_csv(os.path.join(out_dir, "evaluation_summary.csv"),
               [{"controller": c, **summary[c]} for c in summary], ["controller"] + METRICS)
    _write_csv(os.path.join(out_dir, "episode_results.csv"),
               [{"controller": c, "episode": i, "seed": EVALUATION_SEED + i - 1, **r}
                for c, rows in results.items() for i, r in enumerate(rows, 1)],
               ["controller", "episode", "seed"] + METRICS + ["final_temperature", "final_vibration"])
    _write_csv(os.path.join(out_dir, "paired_comparison.csv"), paired)
    _write_csv(os.path.join(out_dir, "training_seed_study.csv"), seed_rows)
    _write_csv(os.path.join(out_dir, "learned_policy.csv"), greedy_policy_rows(agent))
    plot_training_curve(history, os.path.join(out_dir, "training_curve.png"))
    plot_comparison(results, summary, os.path.join(out_dir, "controller_comparison.png"))
    write_experiment_summary(os.path.join(out_dir, "experiment_summary.md"), agent, history, best_action, summary,
                             paired, seed_rows, len(next(iter(results.values()))))


def run_experiment(episodes=TRAINING_EPISODES, seed=TRAINING_SEED, seed_study=True):
    """Train, choose the best fixed setting, evaluate. Deterministic for fixed arguments."""
    agent, history = train_agent(episodes=episodes, seed=seed)
    best_action = find_best_fixed_action()
    results = evaluate(make_controllers(agent, best_action))
    return {"agent": agent, "history": history, "best_action": best_action, "results": results,
            "summary": summarize(results), "paired": paired_differences(results),
            "seed_rows": training_seed_study(best_action, episodes=episodes) if seed_study else []}


def main():
    print("\n=== Standalone Adaptive CNC RL Demonstration ===")
    print(f"Training a tabular Q-learning agent for {TRAINING_EPISODES} episodes...")
    exp = run_experiment()
    print(f"Training complete. Final exploration rate: {exp['agent'].epsilon:.3f}")
    print("Best fixed setting (chosen on tuning seeds): %d rpm, %d mm/min, %.1f mm" % action_settings(exp["best_action"]))
    save_outputs(exp["agent"], exp["history"], exp["results"], exp["summary"], exp["paired"], exp["seed_rows"],
                 exp["best_action"])

    labels = list(exp["summary"])
    print(f"\nAverage results over {EVALUATION_EPISODES} matched evaluation episodes:")
    print(f"{'Metric':<26}" + "".join(f"{c:>26}" for c in labels))
    for m in METRICS:
        print(f"{METRIC_LABELS[m]:<26}" + "".join(f"{exp['summary'][c][m]:>26.3f}" for c in labels))
    print(f"\nFiles saved in {RESULTS_DIR}")
    print("\nNote: This is a simplified educational model, not a calibrated CNC process model.")


if __name__ == "__main__":
    main()
