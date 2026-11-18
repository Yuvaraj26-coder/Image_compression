from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_loads():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    assert not app.exception


def test_all_stages_execute():
    stages = [
        "2. The Decomposition",
        "3. Where the Information Lives",
        "4. Building the Image Up",
        "5. The Compression Maths",
        "6. Quality Analysis",
        "Bonus: Image Denoising",
    ]
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    for stage in stages:
        guided = next(radio for radio in app.radio if radio.label == "Guided stages")
        guided.set_value(stage).run(timeout=30)
        assert not app.exception
