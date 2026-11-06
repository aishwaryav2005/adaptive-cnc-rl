# Presentation Content (8 slides)

All results are from a simplified simulation. Say this early and keep it on the results slide.

---

## Slide 1: Title

**Adaptive CNC Machining Parameter Optimization Using Reinforcement Learning**

- Tabular Q-learning on a simulated milling process
- Name, roll number, department, guide, date

*Say:* "This project asks whether a software agent can learn to adjust machining parameters as a tool wears. It is a simulation study."

---

## Slide 2: Introduction and motivation

- Spindle speed, feed rate and depth of cut decide productivity and tool life
- Higher feed and depth remove material faster, but raise vibration, heat and wear
- Fixed programs do not respond as the tool wears
- Reinforcement learning can learn a trade-off from reward signals

*Say:* "A setting that is safe for a new tool may not stay safe for a worn one."

---

## Slide 3: Problem statement and objectives

**Problem:** choose parameters step by step, to maximise material removal while staying within limits.

**Objectives:**
- Simulate tool wear, vibration and temperature
- Train a Q-learning agent to choose the parameters
- Design a balanced reward
- Compare fairly with fixed machining on identical episodes
- Build an interactive dashboard

---

## Slide 4: System architecture

```
Q-learning agent  -- spindle, feed, depth -->  CNC simulation
   (Q-table)      <-- state and reward ------  (wear, vibration, temperature)
```

- `cnc_rl_demo.py`: simulator, agent, training, evaluation
- `app.py`: Streamlit dashboard
- `results/`: CSV files, charts, experiment summary

*Visual:* this diagram, or a screenshot of the dashboard's CNC Simulation tab.

---

## Slide 5: Methodology and Q-learning

- **State (45):** 5 wear × 3 vibration × 3 temperature bins
- **Action (27):** 3 spindle × 3 feed × 3 depth levels
- **Reward:** pays for material removed; penalises wear, vibration, heat and limit violations
- **Update:** Q(s,a) ← Q(s,a) + α [r + γ max Q(s′,a′) − Q(s,a)]
- **Settings:** α = 0.15, γ = 0.92, ε from 1.0 to 0.04

*Say:* "The agent explores at first, then gradually relies on what it has learned."

---

## Slide 6: Simulation and implementation

- One episode: up to 60 steps, ending early if the tool reaches its wear limit
- Limits: vibration 1.0, temperature 105 °C, tool wear 0.85
- A worn tool vibrates more and runs hotter
- Training: 5000 episodes, about 2 seconds
- 21 automated tests; reproducible seeds

*Visual:* `results/training_curve.png`

*Say:* "Testing the first version showed that its limits were mostly unreachable and that its baseline sat on a limit. I fixed that before collecting the results shown next."

---

## Slide 7: Results and comparison

**100 identical test episodes (simulation)**

| | Q-learning | Medium fixed | Best fixed |
|---|---|---|---|
| Reward | **113.9** | 13.9 | 108.2 |
| Violating steps | 0.07 | 32.19 | 0.00 |
| Final tool wear | 0.539 | 0.848 | 0.496 |
| Material removal | **64.2** | 59.5 | 60.0 |

- Against the best fixed setting: reward +5.75 ± 0.27, material +4.19 ± 0.14
- Costs: wear +0.043, violations +0.07 per episode
- Learned behaviour: higher feed while the tool is new, then the best fixed setting
- Gain held for all five training seeds (+4.13 to +7.17)

*Visual:* `results/controller_comparison.png`

*Say:* "The big gain over the medium setting mostly shows that it was a poor setting. The fair comparison is the best fixed setting, and there the gain is modest but consistent."

---

## Slide 8: Conclusion and future scope

**Conclusion**
- Q-learning learned to adapt to tool wear, within the simulation
- Modest, consistent gain over the best fixed setting
- A fair baseline changed the conclusion: it removed a misleading large gain

**Limitations**
- Not calibrated; not for real machines
- Spindle speed has no productivity benefit in this model

**Future scope**
- Calibrate with cutting experiments
- Add feed per tooth and a better chatter model
- Compare with a rule-based adaptive controller

*Say:* "Thank you. I can show the live dashboard now."
