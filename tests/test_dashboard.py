from streamlit.testing.v1 import AppTest

from riskwatch.config import PROJECT_ROOT

APP = str(PROJECT_ROOT / "app" / "dashboard.py")


def test_dashboard_renders_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Portfolio risk", "Model & threshold", "Monitoring"]


def test_recommended_threshold_matches_report():
    at = AppTest.from_file(APP, default_timeout=60).run()
    caught = next(m for m in at.metric if m.label == "Frauds caught")
    alarms = next(m for m in at.metric if m.label == "False alarms")
    assert caught.value == "60 / 74"
    assert alarms.value == "65"


def test_switching_to_logistic_regression():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio[0].set_value("logistic_regression").run()
    assert not at.exception
    assert next(m for m in at.metric if m.label == "Frauds caught").value == "58 / 74"
