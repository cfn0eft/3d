import io

from poselab.progress import ProgressReporter, format_duration


def test_format_duration():
    assert format_duration(0) == "0:00"
    assert format_duration(75) == "1:15"
    assert format_duration(3661) == "1:01:01"


def test_render_with_total():
    reporter = ProgressReporter(total=100, stream=io.StringIO(), min_interval=0)
    reporter.update(50)
    text = reporter.render()
    assert "50.0%" in text
    assert "50/100" in text
    assert "残り" in text


def test_render_without_total():
    reporter = ProgressReporter(total=None, stream=io.StringIO(), min_interval=0)
    reporter.update(30)
    text = reporter.render()
    assert "30 フレーム" in text
    assert "経過" in text


def test_disabled_writes_nothing():
    stream = io.StringIO()
    reporter = ProgressReporter(total=10, stream=stream, enabled=False)
    reporter.update(5)
    reporter.finish()
    assert stream.getvalue() == ""


def test_finish_writes_newline():
    stream = io.StringIO()
    reporter = ProgressReporter(total=10, stream=stream, min_interval=0)
    reporter.update(10)
    reporter.finish()
    assert stream.getvalue().endswith("\n")
    assert "100.0%" in stream.getvalue()
