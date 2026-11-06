# Viva Questions and Answers

## CNC machining fundamentals

**1. What do spindle speed, feed rate and depth of cut mean?**
Spindle speed is how fast the tool rotates (rpm). Feed rate is how fast the tool moves through the workpiece
(mm/min). Depth of cut is how deep the tool engages the material (mm). Together they determine the material removal
rate, the cutting forces, the heat and the tool life.

**2. What is material removal rate, and what is your "proxy"?**
In milling, MRR = axial depth × radial width × feed rate. My model assumes a constant radial width, so I use
feed × depth, divided by its value at the medium setting. It is a relative number, not a real volume.

**3. Why does tool wear matter?**
A worn cutting edge raises forces and temperature, worsens the surface finish, and can lead to tool failure and a
scrapped part. Tool replacement also costs money and machine time. In my model, wear raises vibration and temperature,
and the episode ends when wear reaches 0.85.

**4. What causes vibration in milling, and why limit it?**
Causes include cutting forces and self-excited vibration called chatter. Vibration harms surface finish and can damage
the tool and spindle. My model uses a simple vibration index with a limit of 1.0. It does not simulate real chatter.

**5. Why does cutting temperature matter?**
Cutting energy turns mostly into heat. High temperature softens the tool and speeds up wear. In my model, temperature
rises with speed, feed, depth and wear, with a limit of 105 °C. These values are illustrative.

## Reinforcement learning and Q-learning

**6. What is reinforcement learning?**
An agent interacts with an environment: it observes a state, takes an action and receives a reward. It learns a policy
that maximises long-term reward. Unlike supervised learning, it has no labelled correct answers.

**7. What is Q-learning?**
A model-free RL method that keeps a table of values Q(s, a), the expected long-term reward of action a in state s. The
update is Q(s,a) ← Q(s,a) + α [r + γ max Q(s′,a′) − Q(s,a)]. It is off-policy: it learns the value of acting greedily
while it explores.

**8. Why did you choose tabular Q-learning and not a neural network?**
The discretised problem is small (45 states, 27 actions), so a table is enough. It trains in seconds, and I can read
the learned values directly, which suits a class project. Deep RL would be needed for much larger or continuous state spaces.

**9. Explain exploration versus exploitation.**
Exploitation uses the best action known so far. Exploration tries other actions, which may turn out better. Too little
exploration can trap the agent in a poor habit; too much wastes reward. I use ε-greedy selection to balance the two.

**10. What are your state, action and reward?**
State: 5 wear bins × 3 vibration bins × 3 temperature bins = 45 states. Action: one of 3 spindle × 3 feed × 3 depth
levels = 27 actions. Reward: 2 × productivity, minus penalties for wear, high vibration and temperature, with extra
penalties for breaking a limit.

**11. What does the learning rate do?**
α = 0.15 sets how far each update moves the old estimate toward the new one. A large value learns fast but reacts
strongly to noise; a small value is stable but slow.

**12. What does the discount factor do?**
γ = 0.92 sets the weight of future reward. A rough planning horizon is 1 / (1 − γ) = 12.5 steps. A low γ would
ignore later consequences such as wear.

**13. How does epsilon decay work?**
ε starts at 1.0 (all random) and is multiplied by 0.992 after each episode, down to a minimum of 0.04. It reaches the
minimum after about 400 episodes, because ln(0.04) / ln(0.992) ≈ 401. The minimum keeps a little exploration going. In
testing, ε is not used at all.

**14. How did you design the reward?**
It pays for material removed, so doing nothing earns nothing. Graded penalties start before the limits (vibration above
0.85, temperature above 90), which gives an early signal. Fixed penalties apply when a limit is broken. Tool failure
also loses all future reward. The weights are my design choices, and different weights would produce a different policy.

**15. How do you know the agent learned something?**
Training reward rose and levelled off. On unseen test seeds the agent beat the best of all 27 fixed settings: reward
+5.75 ± 0.27. I also read its behaviour: at wear below 0.2 it mostly uses 220 mm/min, and after that mostly 160 mm/min.
This does not prove the policy is optimal.

## Evaluation and baselines

**16. Is your comparison fair?**
Each controller uses the same 100 test seeds, so the starting state and noise match. The agent does not explore or learn
during testing, and those seeds were not used in training. I also added the best fixed setting, chosen on separate
tuning seeds, because comparing only with an arbitrary setting would flatter the agent.

**17. Why is the gain over the medium setting so large (113.9 vs 13.9) but small over the best fixed setting (113.9 vs 108.2)?**
The medium setting becomes unsafe when wear reaches about 0.4: it averages 32 violating steps and reaches the wear
limit in 56 % of episodes. So that gap mostly shows that the setting is a poor choice. The best fixed setting is the
fair test, and there the gain is about 5 %.

**18. Does the agent have any disadvantage compared with the best fixed setting?**
Yes. Final tool wear is higher (0.539 vs 0.496), and it has 0.07 violating steps per episode against 0. In exchange it
removes about 7 % more material. Whether that is worthwhile depends on the cost of tools.

**19. What problems did you find in the first version?**
Its baseline sat exactly on the vibration limit, the temperature and wear limits could never be reached, and only 7 of
45 states were reachable. Its comparison (102.4 vs 33.1) looked strong, but a constant setting scored 106.8 and beat the
agent. I corrected the simulator and added a fair baseline, then re-ran everything.

**20. Why does the agent almost always use 3000 rpm? Is that realistic?**
In my model, spindle speed adds wear, vibration and heat but adds nothing to material removal, so the lowest speed is
always preferred. That is a limitation of the model. In real milling, feed per tooth = feed / (spindle speed × number of
teeth), so a low speed with a high feed overloads each tooth. I do not present this as machining advice.

## Limitations and future work

**21. Can this controller be used on a real CNC machine?**
No. The equations are illustrative and not calibrated. A real system would need cutting experiments, model validation,
testing in shadow mode, and independent safety limits in the machine.

**22. What are the main limitations?**
No calibration; no feed-per-tooth effect; no real chatter or stability model; discrete settings; coarse states; no
elapsed time in the state; 16 of 45 states unvisited; comparison limited to fixed controllers.

**23. How would you improve it?**
Calibrate it with measurements, add feed per tooth and a stability-lobe chatter model, compare it with a rule-based
adaptive controller, use finer states or a deep Q-network, and add disturbances such as material hardness changes.

**24. Why not test on real data?**
I had no machine or measured dataset for this project, and exploring random actions on a real machine could damage the
tool or machine. Simulation allows safe trial and error, but its results apply only to the simulation.
