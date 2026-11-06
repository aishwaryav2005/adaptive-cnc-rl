"""Checks of the simulator, the Q-learning update and the evaluation protocol."""
import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cnc_rl_demo as C  # noqa: E402

MEDIUM = C.ACTIONS.index(C.MEDIUM_SETTING)
SLOW = C.ACTIONS.index((0, 0, 0))
AGGRESSIVE = C.ACTIONS.index((2, 2, 2))


def test_action_and_state_spaces():
    assert len(C.ACTIONS) == 27 and C.action_settings(MEDIUM) == (4500, 160, 0.8)
    assert C.action_settings(SLOW) == (3000, 100, 0.5) and C.action_settings(AGGRESSIVE) == (6000, 220, 1.1)
    seen = set()
    for w in range(5):
        for v in range(3):
            for t in range(3):
                s = (w * 3 + v) * 3 + t
                assert C.state_bins(s) == (w, v, t)
                seen.add(s)
    assert seen == set(range(C.N_STATES))


def test_state_stays_in_range_and_wear_never_falls():
    rng = np.random.default_rng(0)
    env = C.CNCMillingEnv(seed=1)
    for _ in range(100):
        state, done = env.reset(), False
        wear = env.state.wear
        while not done:
            assert 0 <= state < C.N_STATES
            state, _, done, info = env.step(int(rng.integers(len(C.ACTIONS))))
            assert info["wear"] >= wear
            wear = info["wear"]
        assert 0 <= state < C.N_STATES


def test_reward_matches_hand_calculation():
    env = C.CNCMillingEnv(seed=3)
    env.reset()
    wear0 = env.state.wear
    _, reward, _, info = env.step(MEDIUM)
    vib, temp = info["vibration"], info["temperature"]
    increment = C.WEAR_RATE * (1 + 0.45 * max(0, vib - 0.75))
    expected = (2.0 * 1.0 - C.WEAR_PENALTY * increment - 7.0 * max(0, vib - C.VIBRATION_SOFT) ** 2
                - 0.025 * max(0, temp - C.TEMPERATURE_SOFT) - (2.0 if vib > C.VIBRATION_LIMIT else 0.0))
    assert reward == pytest.approx(expected)
    assert info["wear"] == pytest.approx(wear0 + increment)
    assert info["volume_proxy"] == pytest.approx(1.0)


def test_every_limit_can_actually_be_reached():
    """The first version of the simulator could never reach its temperature or wear limit."""
    counts = {"vibration_bad": 0, "temperature_bad": 0, "tool_failed": 0}
    env = C.CNCMillingEnv(seed=2)
    env.reset()
    done = False
    while not done:
        _, _, done, info = env.step(AGGRESSIVE)
        for k in counts:
            counts[k] += int(info[k])
    assert all(v > 0 for v in counts.values()), counts
    assert env.steps < C.MAX_STEPS  # the tool wears out before the step limit


def test_medium_setting_is_safe_for_a_new_tool_and_not_later():
    """The baseline must not sit on a limit from the first step, and wear must matter."""
    rows = C.simulate_episode(lambda s: MEDIUM, seed=5)
    assert not any(r["vibration_bad"] for r in rows[:5])
    assert any(r["vibration_bad"] for r in rows[-5:])


def test_episode_ends_at_step_limit_or_tool_failure():
    env = C.CNCMillingEnv(seed=4)
    assert C.run_episode(env, fixed_action=SLOW)["steps"] == C.MAX_STEPS
    r = C.run_episode(C.CNCMillingEnv(seed=4), fixed_action=AGGRESSIVE)
    assert r["steps"] < C.MAX_STEPS and r["tool_failed"] == 1.0 and r["wear"] >= C.WEAR_LIMIT


def test_q_update_formula():
    agent = C.QLearningAgent(n_states=3, n_actions=2, alpha=0.5, gamma=0.9)
    agent.q[1] = [2.0, 4.0]
    agent.learn(0, 1, reward=1.0, next_state=1, terminal=False)
    assert agent.q[0, 1] == pytest.approx(0.5 * (1.0 + 0.9 * 4.0))
    agent.learn(2, 0, reward=1.0, next_state=1, terminal=True)
    assert agent.q[2, 0] == pytest.approx(0.5)


def test_step_limit_bootstraps_but_tool_failure_does_not():
    """Reaching the step limit is not a terminal state of the process; a failed tool is."""
    agent = C.QLearningAgent(seed=0, alpha=1.0, epsilon=0.0)
    agent.q[:] = 50.0
    env = C.CNCMillingEnv(seed=0, max_steps=1)
    state = env.reset()
    action = agent.choose_action(state)
    next_state, reward, done, info = env.step(action)
    assert done and not info["tool_failed"]
    agent.learn(state, action, reward, next_state, info["tool_failed"])
    assert agent.q[state, action] == pytest.approx(reward + agent.gamma * 50.0)


def test_epsilon_decays_to_its_floor_and_greedy_ignores_it():
    agent = C.QLearningAgent(seed=0)
    for _ in range(2000):
        agent.finish_episode()
    assert agent.epsilon == agent.epsilon_min
    agent.q[0, 5] = 1.0
    assert {agent.choose_action(0, explore=False) for _ in range(50)} == {5}


def test_training_and_evaluation_are_reproducible():
    a1, h1 = C.train_agent(episodes=200, seed=3)
    a2, h2 = C.train_agent(episodes=200, seed=3)
    a3, _ = C.train_agent(episodes=200, seed=4)
    assert (a1.q == a2.q).all() and h1 == h2
    assert not (a1.q == a3.q).all()
    r1 = C.evaluate(C.make_controllers(a1, SLOW), episodes=5)
    r2 = C.evaluate(C.make_controllers(a2, SLOW), episodes=5)
    assert r1 == r2


def test_evaluation_conditions_are_matched():
    """Same seed, same start state and same noise for every action, step by step."""
    a, b = C.CNCMillingEnv(seed=1000), C.CNCMillingEnv(seed=1000)
    a.reset(), b.reset()
    assert a.state == b.state
    for _ in range(10):
        a.step(SLOW), b.step(MEDIUM)
        assert a.rng.getstate() == b.rng.getstate()


def test_evaluation_does_not_change_the_agent():
    agent, _ = C.train_agent(episodes=50, seed=1)
    q, epsilon = agent.q.copy(), agent.epsilon
    C.evaluate(C.make_controllers(agent, SLOW), episodes=3)
    assert (agent.q == q).all() and agent.epsilon == epsilon


def test_moving_average():
    assert C.moving_average([1, 3, 5, 7], 2).tolist() == [1, 2, 4, 6]


def test_saved_files_are_valid(tmp_path):
    exp = C.run_experiment(episodes=100, seed_study=False)
    exp["seed_rows"] = C.training_seed_study(exp["best_action"], seeds=[1], episodes=50)
    C.save_outputs(exp["agent"], exp["history"], exp["results"], exp["summary"], exp["paired"], exp["seed_rows"],
                   exp["best_action"], out_dir=str(tmp_path))
    for name in ("training_curve.png", "controller_comparison.png", "experiment_summary.md"):
        assert (tmp_path / name).stat().st_size > 0
    with open(tmp_path / "episode_results.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * C.EVALUATION_EPISODES
    assert all(np.isfinite(float(r["reward"])) for r in rows)
    with open(tmp_path / "training_history.csv") as f:
        assert len(list(csv.DictReader(f))) == 100
    with open(tmp_path / "evaluation_summary.csv") as f:
        summary = {r["controller"]: float(r["reward"]) for r in csv.DictReader(f)}
    mine = [float(r["reward"]) for r in rows if r["controller"] == C.Q_LABEL]
    assert summary[C.Q_LABEL] == pytest.approx(np.mean(mine))
