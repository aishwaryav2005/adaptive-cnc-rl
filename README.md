# Adaptive CNC Machining Parameter Optimization Using Reinforcement Learning

A small, self-contained class project. A tabular Q-learning agent learns to choose spindle speed, feed rate and depth
of cut in a simulated milling process, and is compared with fixed-parameter machining on identical test episodes.

> **Simulation only.** The equations and limits are simplified and illustrative. They are not calibrated against a real
> CNC machine, tool or workpiece. This project demonstrates the reinforcement learning workflow and must not be used
> to control physical machinery.

## Install

Python 3.8 or later. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python cnc_rl_demo.py       # train, evaluate, write everything in results/ (about 15 s)
streamlit run app.py        # dashboard; run it from this folder so the dark theme loads
python -m pytest tests      # automated checks
```

The dashboard opens at http://localhost:8501. It trains its agent on start-up with the same seed as the report (about
3 s), so it does not need `results/` except for the training-seed table.

## Results (100 matched test episodes)

| Metric | Q-learning agent | Fixed baseline (medium) | Best fixed setting |
|---|---|---|---|
| Total reward | 113.914 | 13.908 | 108.163 |
| Machining steps | 60.000 | 59.480 | 60.000 |
| Violating steps | 0.070 | 32.190 | 0.000 |
| Final tool wear | 0.539 | 0.848 | 0.496 |
| Material removal proxy | 64.186 | 59.480 | 60.000 |
| Tool-failure rate | 0.000 | 0.560 | 0.000 |

Against the best fixed setting (3000 rpm, 160 mm/min, 0.8 mm, chosen on separate tuning seeds), the agent gains
+5.75 ± 0.27 in reward and +4.19 ± 0.14 in material removed, with +0.043 tool wear and +0.07 violating steps per
episode. The gain held for all five training seeds (+4.13 to +7.17). The agent uses a higher feed while the tool is new,
then returns to the best fixed setting.

## Files

| Path | Content |
|---|---|
| `cnc_rl_demo.py` | simulator, Q-learning agent, training, evaluation, saved outputs |
| `app.py`, `.streamlit/config.toml` | Streamlit dashboard and its dark theme |
| `results/training_history.csv` | per-episode reward, moving average, ε, steps, violations, wear, material |
| `results/evaluation_summary.csv` | mean metrics per controller |
| `results/episode_results.csv` | per-episode test data for every controller |
| `results/paired_comparison.csv` | paired differences with 95 % confidence intervals |
| `results/training_seed_study.csv` | test results for five training seeds |
| `results/learned_policy.csv` | best action per visited state |
| `results/experiment_summary.md` | settings, evaluation conditions and results |
| `results/training_curve.png`, `results/controller_comparison.png` | charts |
| `documentation/` | `PROJECT_REPORT.md`, `PRESENTATION_CONTENT.md`, `VIVA_QUESTIONS.md` |
| `tests/` | simulator, Q-learning, evaluation and dashboard checks |
| `original_backup/` | the unmodified first version of the project |

## How the agent works

1. The simulator reports a state built from tool wear, vibration and temperature (45 states).
2. The agent picks one combination of spindle speed, feed rate and depth of cut (27 actions).
3. The simulator returns a new state and a reward. Material removal earns reward; wear and unsafe conditions cost reward.
4. Q-learning updates the value of that state and action.
5. Exploration falls over repeated episodes, and the agent relies more on what it has learned.

## Main limitations

- The model is not calibrated, so absolute values have no physical meaning.
- Spindle speed adds wear, vibration and heat but no material removal, so the agent's preference for 3000 rpm is a model
  artefact. Feed per tooth is not modelled.
- The only comparison controllers are fixed settings; there is no rule-based adaptive controller.

See `documentation/PROJECT_REPORT.md` for the full list, and for the changes made to the first version.
