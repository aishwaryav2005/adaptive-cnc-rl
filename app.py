"""Interactive dashboard for the Q-learning CNC class project.

    streamlit run app.py

Every number is computed live by cnc_rl_demo.py with fixed seeds, so the default view matches results/.
The dashboard writes no files. Simulation only: the process model is not calibrated.
"""
import os
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import cnc_rl_demo as C

st.set_page_config(page_title="Adaptive CNC Q-learning", layout="wide")

SURFACE, PANEL, BORDER = "#14161a", "#1e2127", "#2e323a"
INK, MUTED, GRID = "#f0efec", "#a9aba8", "#2a2d33"
CONTROLLER_COLORS = {C.Q_LABEL: "#3987e5", "Fixed baseline (medium)": "#d95926", "Best fixed setting": "#199e70"}
RAW_LINE = "#6b6f76"
WARNING = "#e0a030"

st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1280px; }}
  [data-testid="stDecoration"] {{ display: none; }}
  [data-testid="stHeader"] {{ background: transparent; }}
  .legend {{ display: flex; gap: 22px; flex-wrap: wrap; margin: 0.6rem 0 0.2rem 0; font-size: 0.9rem; }}
  .legend i {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 7px; vertical-align: -1px; }}
  h1 {{ font-size: 1.75rem !important; font-weight: 650 !important; letter-spacing: -0.01em; padding-bottom: 0 !important; }}
  h3 {{ font-size: 1.1rem !important; font-weight: 600 !important; }}
  [data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; padding: 10px 14px; }}
  [data-testid="stMetricLabel"] p {{ font-size: 0.78rem !important; color: {MUTED}; }}
  [data-testid="stMetricValue"] {{ font-size: 1.35rem !important; font-variant-numeric: tabular-nums; }}
  .sim-note {{ color: {MUTED}; font-size: 0.82rem; margin: -0.3rem 0 0.6rem 0; }}
  .decision {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; padding: 10px 14px; margin: 0.5rem 0;
               font-size: 0.92rem; line-height: 1.6; }}
  .decision b {{ color: {INK}; }}
  .flag {{ color: {WARNING}; font-weight: 600; }}
  .ok {{ color: {MUTED}; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_resource(show_spinner="Training the Q-learning agent...")
def get_experiment(episodes: int, seed: int):
    return C.run_experiment(episodes=episodes, seed=seed, seed_study=False)


@st.cache_data
def load_seed_study():
    path = os.path.join(C.RESULTS_DIR, "training_seed_study.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


st.session_state.setdefault("train_episodes", C.TRAINING_EPISODES)
st.session_state.setdefault("train_seed", C.TRAINING_SEED)
exp = get_experiment(st.session_state.train_episodes, st.session_state.train_seed)
agent, history, summary = exp["agent"], exp["history"], exp["summary"]
is_report_config = (st.session_state.train_episodes, st.session_state.train_seed) == (C.TRAINING_EPISODES, C.TRAINING_SEED)
best_setting = C.action_settings(exp["best_action"])
medium_setting = C.action_settings(C.ACTIONS.index(C.MEDIUM_SETTING))


def setting_text(setting):
    return f"{setting[0]} rpm, {setting[1]} mm/min, {setting[2]} mm"


def controller_fn(label):
    if label == C.Q_LABEL:
        return lambda state: agent.choose_action(state, explore=False)
    action = exp["best_action"] if label == "Best fixed setting" else C.ACTIONS.index(C.MEDIUM_SETTING)
    return lambda state: action


def state_text(state):
    w, v, t = C.state_bins(state)
    return (f"wear {C.WEAR_BIN_NAMES[w]}, vibration {C.VIBRATION_BIN_NAMES[v]}, "
            f"temperature {C.TEMPERATURE_BIN_NAMES[t]}")


# ------------------------------------------------------------------ chart helpers
def themed(chart, height=135):
    return (chart.properties(height=height, background="transparent")
            .configure(font="Source Sans Pro, sans-serif")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor=BORDER, tickColor=BORDER,
                            labelFontSize=11, titleFontSize=11, titleFontWeight="normal")
            .configure_axisX(grid=False)
            .configure_title(color=INK, fontSize=14, fontWeight=600, anchor="start")
            .configure_legend(labelColor=INK, titleColor=MUTED, labelFontSize=12, orient="top", title=None))


def trace_chart(df, column, title, domain, limit=None, step=False):
    """One process variable against step. The x axis stays fixed so the episode's length is visible."""
    x = alt.X("step:Q", scale=alt.Scale(domain=[0, C.MAX_STEPS]), title="Step", axis=alt.Axis(tickCount=6))
    y = alt.Y(f"{column}:Q", scale=alt.Scale(domain=domain, nice=False), title=None, axis=alt.Axis(tickCount=4))
    line = (alt.Chart(df).mark_line(color=CONTROLLER_COLORS[C.Q_LABEL], strokeWidth=2,
                                    interpolate="step-after" if step else "linear",
                                    point=alt.OverlayMarkDef(opacity=0, size=90))
            .encode(x=x, y=y, tooltip=[alt.Tooltip("step:Q", title="Step"), alt.Tooltip(f"{column}:Q", title=title, format=".3~f")]))
    layers = [line]
    if limit is not None:
        limit_df = pd.DataFrame({"limit": [limit], "label": [f"limit {limit:g}"], "x": [C.MAX_STEPS]})
        layers.append(alt.Chart(limit_df).mark_rule(color=WARNING, strokeDash=[5, 4], strokeWidth=1.2).encode(y="limit:Q"))
        layers.append(alt.Chart(limit_df).mark_text(color=WARNING, align="right", dy=-7, fontSize=11)
                      .encode(x="x:Q", y="limit:Q", text="label:N"))
    return themed(alt.layer(*layers).properties(title=title))


def comparison_panel(frame, metric, title):
    labels = list(CONTROLLER_COLORS)
    data = frame[frame["metric"] == metric]
    fmt = ".3f" if metric == "wear" else ".1f"
    base = alt.Chart(data).encode(
        x=alt.X("controller:N", sort=labels, axis=None, title=None),
        tooltip=[alt.Tooltip("controller:N", title="Controller"), alt.Tooltip("mean:Q", title=title, format=fmt),
                 alt.Tooltip("ci:Q", title="95 % CI ±", format=fmt)])
    bars = base.mark_bar(cornerRadiusEnd=4, size=30).encode(
        y=alt.Y("mean:Q", title=None, scale=alt.Scale(zero=True), axis=alt.Axis(tickCount=4)),
        color=alt.Color("controller:N", scale=alt.Scale(domain=labels, range=[CONTROLLER_COLORS[c] for c in labels]), legend=None))
    errors = base.mark_rule(color=INK, strokeWidth=1).encode(y="low:Q", y2="high:Q")
    text = base.mark_text(color=INK, dy=-9, fontSize=12).encode(y="high:Q", text=alt.Text("mean:Q", format=fmt))
    return alt.layer(bars, errors, text).properties(title=title)


# ------------------------------------------------------------------ page
st.title("Adaptive CNC Machining Parameter Optimization Using Reinforcement Learning")
st.markdown('<div class="sim-note">Educational simulation with simplified, illustrative equations. '
            'Not calibrated against a real machine and not for controlling real machinery.</div>', unsafe_allow_html=True)

tab_overview, tab_sim, tab_train, tab_compare, tab_explain = st.tabs(
    ["Project Overview", "CNC Simulation", "Training Results", "Performance Comparison", "Explanation"])

# ------------------------------------------------------------------ A. overview
with tab_overview:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Project description")
        st.write("A milling machine must balance productivity against tool wear, vibration and heat. A fixed program "
                 "cannot react as the tool wears. This project trains a software agent to choose machining parameters "
                 "in a simulated milling process, learning from reward and penalty signals instead of from fixed rules.")
        st.subheader("Objective")
        st.write("Learn a policy that removes as much material as possible while staying within the vibration, "
                 "temperature and tool-wear limits, and compare it fairly with fixed-parameter machining on "
                 "identical test episodes.")
        st.subheader("Reinforcement learning algorithm")
        st.write(f"Tabular **Q-learning** with ε-greedy exploration: {C.N_STATES} states, {len(C.ACTIONS)} actions, "
                 f"learning rate {agent.alpha}, discount factor {agent.gamma}.")
    with right:
        st.subheader("Parameters controlled by the agent")
        st.table(pd.DataFrame({
            "Parameter": ["Spindle speed", "Feed rate", "Depth of cut"],
            "Unit": ["rpm", "mm/min", "mm"],
            "Levels": [", ".join(map(str, C.SPINDLE_LEVELS)), ", ".join(map(str, C.FEED_LEVELS)),
                       ", ".join(map(str, C.DEPTH_LEVELS))]}).set_index("Parameter"))
        st.subheader("Limits in the simulation")
        st.table(pd.DataFrame({
            "Quantity": ["Vibration index", "Cutting temperature", "Tool wear"],
            "Limit": [f"{C.VIBRATION_LIMIT:g}", f"{C.TEMPERATURE_LIMIT:g} °C", f"{C.WEAR_LIMIT:g} (episode ends)"]
        }).set_index("Quantity"))
    st.subheader("Main result on the 100 test episodes")
    c1, c2, c3 = st.columns(3)
    for col, label in zip((c1, c2, c3), CONTROLLER_COLORS):
        col.metric(f"{label}: mean reward", f"{summary[label]['reward']:.1f}")

# ------------------------------------------------------------------ B. simulation
with tab_sim:
    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    sim_controller = c1.selectbox("Controller", list(CONTROLLER_COLORS), key="sim_controller")
    sim_episode = c2.number_input("Test episode (1 to 100)", min_value=1, max_value=C.EVALUATION_EPISODES, value=1,
                                  key="sim_episode")
    sim_speed = c3.selectbox("Playback", ["Normal", "Fast", "Instant"], key="sim_speed")
    c4.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    start = c4.button("Start simulation", type="primary", use_container_width=True, key="sim_start")

    progress_slot = st.empty()
    metric_cols = st.columns(7)
    metric_slots = [c.empty() for c in metric_cols]
    decision_slot = st.empty()
    process_cols = st.columns(3)
    setting_cols = st.columns(4)
    chart_slots = [c.empty() for c in process_cols] + [c.empty() for c in setting_cols]
    outcome_slot = st.empty()

    def render_frame(rows, label):
        now = rows[-1]
        progress_slot.progress(len(rows) / C.MAX_STEPS, text=f"Episode progress: step {len(rows)} of {C.MAX_STEPS}")
        values = [("Spindle speed (rpm)", f"{now['spindle_rpm']}"), ("Feed rate (mm/min)", f"{now['feed_mm_min']}"),
                  ("Depth of cut (mm)", f"{now['depth_mm']}"), ("Tool wear", f"{now['wear']:.3f}"),
                  ("Vibration", f"{now['vibration']:.3f}"), ("Temperature (°C)", f"{now['temperature']:.1f}"),
                  ("Step reward", f"{now['reward']:.2f}")]
        for slot, (name, value) in zip(metric_slots, values):
            slot.metric(name, value)
        flags = [text for key, text in (("vibration_bad", "Vibration over limit"), ("temperature_bad", "Temperature over limit"),
                                        ("tool_failed", "Tool wear limit reached")) if now[key]]
        status = (f'<span class="flag">⚠ {", ".join(flags)}</span>' if flags else '<span class="ok">✓ Within limits</span>')
        decision_slot.markdown(
            f'<div class="decision"><b>{label}</b> saw: {state_text(now["state"])}<br>'
            f'Chose: <b>{setting_text(C.action_settings(now["action"]))}</b> &nbsp;·&nbsp; {status} &nbsp;·&nbsp; '
            f'violations so far: {now["violations"]} &nbsp;·&nbsp; total reward: {now["cumulative_reward"]:.1f}</div>',
            unsafe_allow_html=True)

    def render_charts(rows):
        df = pd.DataFrame(rows)
        specs = [("wear", "Tool wear", [0, 1.0], C.WEAR_LIMIT, False),
                 ("vibration", "Vibration index", [0.4, 1.6], C.VIBRATION_LIMIT, False),
                 ("temperature", "Temperature (°C)", [50, 130], C.TEMPERATURE_LIMIT, False),
                 ("spindle_rpm", "Spindle speed (rpm)", [2500, 6500], None, True),
                 ("feed_mm_min", "Feed rate (mm/min)", [80, 240], None, True),
                 ("depth_mm", "Depth of cut (mm)", [0.4, 1.2], None, True),
                 ("cumulative_reward", "Total reward so far", [min(0, float(df.cumulative_reward.min())) - 5,
                                                               max(130, float(df.cumulative_reward.max()) + 5)], None, False)]
        for slot, (col, title, domain, limit, step) in zip(chart_slots, specs):
            slot.altair_chart(trace_chart(df, col, title, domain, limit, step), use_container_width=True, theme=None)

    if start:
        rows = C.simulate_episode(controller_fn(sim_controller), seed=C.EVALUATION_SEED + int(sim_episode) - 1)
        delay = {"Normal": 0.12, "Fast": 0.03, "Instant": 0.0}[sim_speed]
        if delay:
            for t in range(1, len(rows) + 1):
                render_frame(rows[:t], sim_controller)
                if t % 2 == 0 or t == len(rows):
                    render_charts(rows[:t])
                time.sleep(delay)
        st.session_state.sim_result = {"rows": rows, "label": sim_controller, "episode": int(sim_episode)}

    result = st.session_state.get("sim_result")
    if result:
        render_frame(result["rows"], result["label"])
        render_charts(result["rows"])
        last = result["rows"][-1]
        ending = "the tool reached its wear limit" if last["tool_failed"] else f"all {C.MAX_STEPS} steps were completed"
        outcome_slot.markdown(
            f'<div class="decision"><b>Episode finished</b> ({result["label"]}, test episode {result["episode"]}): '
            f'{ending} after {len(result["rows"])} steps. Total reward {last["cumulative_reward"]:.1f}, '
            f'material removal proxy {last["volume_proxy"]:.1f}, {last["violations"]} violating steps.</div>',
            unsafe_allow_html=True)
    else:
        progress_slot.progress(0.0, text="Episode progress: press Start simulation")

# ------------------------------------------------------------------ C. training
with tab_train:
    hist = pd.DataFrame(history)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training episodes", f"{len(hist)}")
    c2.metric("Final exploration rate (ε)", f"{hist.epsilon.iloc[-1]:.3f}")
    c3.metric(f"Moving-average reward, last {C.MOVING_AVERAGE_WINDOW}", f"{hist.moving_average_reward.iloc[-1]:.1f}")
    c4.metric("States visited", f"{int((agent.visits.sum(axis=1) > 0).sum())} of {C.N_STATES}")

    series = pd.concat([
        hist.assign(series="Episode reward", value=hist.reward)[["episode", "series", "value"]],
        hist.assign(series=f"Moving average ({C.MOVING_AVERAGE_WINDOW} episodes)", value=hist.moving_average_reward)[["episode", "series", "value"]],
    ])
    names = list(series.series.unique())
    reward_chart = (alt.Chart(series).mark_line().encode(
        x=alt.X("episode:Q", title="Training episode", axis=alt.Axis(tickCount=10)), y=alt.Y("value:Q", title="Total episode reward"),
        color=alt.Color("series:N", scale=alt.Scale(domain=names, range=[RAW_LINE, CONTROLLER_COLORS[C.Q_LABEL]]),
                        legend=alt.Legend(labelLimit=260)),
        strokeWidth=alt.StrokeWidth("series:N", scale=alt.Scale(domain=names, range=[0.5, 2.2]), legend=None),
        order=alt.Order("series:N", sort="ascending"),
        tooltip=[alt.Tooltip("episode:Q", title="Episode"), alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("value:Q", title="Reward", format=".1f")])
        .properties(title="Reward during training"))
    st.altair_chart(themed(reward_chart, 270), use_container_width=True, theme=None)
    epsilon_chart = (alt.Chart(hist[hist.episode % 5 == 0]).mark_line(color=CONTROLLER_COLORS[C.Q_LABEL], strokeWidth=2).encode(
        x=alt.X("episode:Q", title="Training episode", axis=alt.Axis(tickCount=10)),
        y=alt.Y("epsilon:Q", title="ε", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(tickCount=3)),
        tooltip=[alt.Tooltip("episode:Q", title="Episode"), alt.Tooltip("epsilon:Q", title="ε", format=".3f")])
        .properties(title="Exploration rate"))
    st.altair_chart(themed(epsilon_chart, 110), use_container_width=True, theme=None)

    st.subheader("Train again")
    st.caption("Same simulator, different training run. The agent is replaced in all tabs until you restore the report settings.")
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    new_episodes = c1.selectbox("Episodes", [500, 1000, 2000, 5000], index=3, key="new_episodes")
    new_seed = c2.number_input("Training seed", min_value=0, max_value=9999, value=C.TRAINING_SEED, key="new_seed")
    c3.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    c4.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    if c3.button("Retrain agent", use_container_width=True, key="retrain"):
        st.session_state.train_episodes, st.session_state.train_seed = int(new_episodes), int(new_seed)
        st.session_state.pop("sim_result", None)
        st.rerun()
    if c4.button("Restore report settings", use_container_width=True, key="restore", disabled=is_report_config):
        st.session_state.train_episodes, st.session_state.train_seed = C.TRAINING_EPISODES, C.TRAINING_SEED
        st.session_state.pop("sim_result", None)
        st.rerun()
    with st.expander("Learned policy: the best action in each visited state"):
        st.dataframe(pd.DataFrame(C.greedy_policy_rows(agent)), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ D. comparison
with tab_compare:
    if not is_report_config:
        st.warning(f"Showing a custom training run ({st.session_state.train_episodes} episodes, seed "
                   f"{st.session_state.train_seed}). The report uses {C.TRAINING_EPISODES} episodes, seed {C.TRAINING_SEED}.")
    st.markdown(f"All controllers ran the same **{C.EVALUATION_EPISODES} test episodes** (seeds {C.EVALUATION_SEED}"
                f"-{C.EVALUATION_SEED + C.EVALUATION_EPISODES - 1}): same starting state, same noise. "
                "Q-learning runs without exploration or further learning. Error bars and ± values are 95 % confidence intervals.")
    frames = []
    for label, rows in exp["results"].items():
        for m in C.METRICS:
            v = np.array([r[m] for r in rows], dtype=float)
            half = C.T_975_DF99 * v.std(ddof=1) / np.sqrt(len(v))
            frames.append({"controller": label, "metric": m, "mean": v.mean(), "ci": half, "low": v.mean() - half, "high": v.mean() + half})
    frame = pd.DataFrame(frames)

    table = {C.METRIC_LABELS[m]: {c: summary[c][m] for c in summary} for m in
             ["reward", "wear", "violations", "steps", "volume_proxy", "tool_failed"]}
    st.dataframe(pd.DataFrame(table).T.style.format("{:.3f}"), use_container_width=True)

    panels = [comparison_panel(frame, m, t) for m, t in
              (("reward", "Average reward"), ("wear", "Average tool wear"), ("violations", "Average violations"),
               ("steps", "Average machining steps"), ("volume_proxy", "Material removal proxy"))]
    st.markdown('<div class="legend">' + "".join(f'<span><i style="background:{CONTROLLER_COLORS[c]}"></i>{c}</span>'
                                                 for c in CONTROLLER_COLORS) + "</div>", unsafe_allow_html=True)
    for col, panel in zip(st.columns(5), panels):
        col.altair_chart(themed(panel, 200), use_container_width=True, theme=None)

    def compared(other):
        return {r["metric"]: r for r in exp["paired"] if r["comparison"].endswith(other)}

    d_best, d_med = compared("Best fixed setting"), compared("Fixed baseline (medium)")
    st.subheader("What the comparison shows")
    st.markdown(
        f"- **Against the medium fixed setting** ({setting_text(medium_setting)}): reward difference "
        f"{d_med['reward']['mean_difference']:+.1f} ± {d_med['reward']['ci95_half_width']:.1f}. This setting is safe for a new "
        f"tool but breaks the vibration limit as the tool wears, and reached the wear limit in "
        f"{summary['Fixed baseline (medium)']['tool_failed'] * 100:.0f} % of episodes.\n"
        f"- **Against the best fixed setting** ({setting_text(best_setting)}), the fairer test, chosen on separate tuning seeds: "
        f"reward difference {d_best['reward']['mean_difference']:+.2f} ± {d_best['reward']['ci95_half_width']:.2f}, material removal "
        f"{d_best['volume_proxy']['mean_difference']:+.2f} ± {d_best['volume_proxy']['ci95_half_width']:.2f}, tool wear "
        f"{d_best['wear']['mean_difference']:+.3f} ± {d_best['wear']['ci95_half_width']:.3f}, violations "
        f"{d_best['violations']['mean_difference']:+.2f} ± {d_best['violations']['ci95_half_width']:.2f} per episode.\n"
        "- These numbers describe this simplified simulator only.")
    with st.expander("Paired differences on matched episodes"):
        pd_table = pd.DataFrame(exp["paired"])
        pd_table["metric"] = pd_table["metric"].map(C.METRIC_LABELS)
        st.dataframe(pd_table.rename(columns={"comparison": "Comparison", "metric": "Metric", "mean_difference": "Mean difference",
                                              "ci95_half_width": "95 % CI ±", "excludes_zero": "CI excludes zero"}),
                     use_container_width=True, hide_index=True)
    seed_study = load_seed_study()
    if seed_study is not None:
        with st.expander(f"Does it depend on the training seed? ({C.TRAINING_EPISODES} episodes each, from results/)"):
            st.dataframe(seed_study[["training_seed", "reward", "violations", "wear", "volume_proxy", "reward_minus_best_fixed"]],
                         use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ E. explanation
with tab_explain:
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("What is Q-learning?")
        st.write("Q-learning is a reinforcement learning method. The agent keeps a table of scores, Q(s, a): how good it "
                 "expects action a to be in state s, counting future reward as well as immediate reward. It needs no "
                 "equations of the process; it learns only from trying actions and seeing the rewards.")
        st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,]")
        st.write(f"α = {agent.alpha} is the learning rate (how far each update moves the score). γ = {agent.gamma} is the "
                 "discount factor (how much future reward counts). If the tool fails, the episode really ends, so the "
                 "future term is dropped.")
        st.subheader("How states are represented")
        st.write(f"The measured values are grouped into bins, giving {C.N_STATES} states: 5 tool-wear bins × 3 vibration "
                 "bins (normal, elevated, over limit) × 3 temperature bins (normal, elevated, over limit).")
        st.subheader("How actions are selected")
        st.write(f"An action is one combination of spindle speed, feed rate and depth of cut (3 × 3 × 3 = {len(C.ACTIONS)}). "
                 f"During training the agent uses **ε-greedy** selection: with probability ε it tries a random action "
                 f"(exploration), otherwise it takes the highest-scoring one (exploitation). ε starts at 1.0 and decays by "
                 f"{agent.epsilon_decay} per episode to {agent.epsilon_min}. During testing it always takes the best action.")
    with right:
        st.subheader("How rewards are calculated")
        st.latex(r"r = 2P - %g\,\Delta w - 7\,[v - %g]_+^2 - 0.025\,[T - %g]_+ - \text{penalties}"
                 % (C.WEAR_PENALTY, C.VIBRATION_SOFT, C.TEMPERATURE_SOFT))
        st.write("P is the material removal proxy for the step (feed × depth, equal to 1 at the medium setting), Δw is the wear "
                 "added in the step, v is vibration and T is temperature. [x]₊ means x when positive and 0 otherwise. Limit penalties: "
                 "2 for vibration over the limit, 1.5 for temperature over the limit, 3 when the wear limit is reached.")
        st.subheader("How the agent learns through repeated episodes")
        st.markdown("1. Start with all scores at zero and a fresh tool.\n2. Observe the state and choose an action.\n"
                    "3. The simulator returns the new state and a reward.\n4. Update that score with the formula.\n"
                    f"5. Repeat for up to {C.MAX_STEPS} steps or until the tool fails.\n"
                    f"6. Reduce ε and start again. The report run uses {C.TRAINING_EPISODES} episodes.")
        st.subheader("Look inside the Q-table")
        c1, c2, c3 = st.columns(3)
        w = c1.selectbox("Wear", C.WEAR_BIN_NAMES, index=1, key="q_wear")
        v = c2.selectbox("Vibration", C.VIBRATION_BIN_NAMES, index=1, key="q_vib")
        t = c3.selectbox("Temperature", C.TEMPERATURE_BIN_NAMES, index=0, key="q_temp")
        s = (C.WEAR_BIN_NAMES.index(w) * 3 + C.VIBRATION_BIN_NAMES.index(v)) * 3 + C.TEMPERATURE_BIN_NAMES.index(t)
        if agent.visits[s].sum() == 0:
            st.info("This state was never visited in training, so all its scores are still zero.")
        else:
            top = np.argsort(agent.q[s])[::-1][:3]
            st.table(pd.DataFrame([{"Rank": i + 1, "Setting": setting_text(C.action_settings(int(a))),
                                    "Q-value": f"{agent.q[s, a]:.2f}", "Tried": int(agent.visits[s, a])}
                                   for i, a in enumerate(top)]).set_index("Rank"))
