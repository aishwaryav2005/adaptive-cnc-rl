"""Headless dashboard checks using Streamlit's AppTest: every button, on every controller."""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


@pytest.fixture()
def app():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    return at


def test_all_sections_render(app):
    assert [t.label for t in app.tabs] == ["Project Overview", "CNC Simulation", "Training Results",
                                           "Performance Comparison", "Explanation"]
    assert len(app.metric) >= 7
    assert any("Q-learning" in s.value for s in app.subheader) or len(app.subheader) > 0


@pytest.mark.parametrize("controller", ["Q-learning agent", "Fixed baseline (medium)", "Best fixed setting"])
def test_start_simulation_shows_every_required_value(app, controller):
    app.selectbox(key="sim_controller").select(controller)
    app.selectbox(key="sim_speed").select("Instant")
    app.button(key="sim_start").click().run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    for name in ("Spindle speed (rpm)", "Feed rate (mm/min)", "Depth of cut (mm)", "Tool wear", "Vibration",
                 "Temperature (°C)", "Step reward"):
        assert name in labels
    assert any("Episode finished" in m.value for m in app.markdown)


def test_animated_playback_runs(app):
    app.selectbox(key="sim_speed").select("Fast")
    app.number_input(key="sim_episode").set_value(7)
    app.button(key="sim_start").click().run()
    assert not app.exception
    assert any("test episode 7" in m.value for m in app.markdown)


def test_retrain_and_restore(app):
    assert app.button(key="restore").disabled
    app.selectbox(key="new_episodes").select(500)
    app.number_input(key="new_seed").set_value(3)
    app.button(key="retrain").click().run()
    assert not app.exception
    assert any("custom training run" in w.value for w in app.warning)
    assert any(m.label == "Training episodes" and m.value == "500" for m in app.metric)
    app.button(key="restore").click().run()
    assert not app.exception
    assert not app.warning
    assert any(m.label == "Training episodes" and m.value == "5000" for m in app.metric)


def test_q_table_inspector_handles_unvisited_state(app):
    app.selectbox(key="q_wear").select("0.8+")
    app.selectbox(key="q_vib").select("normal")
    app.selectbox(key="q_temp").select("over limit")
    app.run()
    assert not app.exception
    assert any("never visited" in i.value for i in app.info)
