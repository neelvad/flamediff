import glob
import json
import os

import pytest
from typer.testing import CliRunner

from flamediff import load_checkpoint
from flamediff.cli import app
from flamediff.detect import Event
from flamediff.report import Watcher, build_report, severity_of

RUN = "fixtures/run_1782312586"


def _ekey(ee):
    e = ee.event
    return (e.index, e.table, e.metric, e.method)


def _link(paths, dst):
    for p in paths:
        os.symlink(os.path.abspath(p), os.path.join(dst, os.path.basename(p)))


def _run():
    if len(glob.glob(f"{RUN}/ckpt_*")) < 2:
        pytest.skip("no trajectory fixture under fixtures/run_*")
    return [load_checkpoint(p) for p in sorted(glob.glob(f"{RUN}/ckpt_*"))]


def test_severity_of_prefers_calibrated():
    base = dict(index=0, step=10, table="t", metric="m", value=1.0, baseline=0.0,
                direction="down", method="robust_z")
    assert severity_of(Event(score=-3.0, calibrated_severity=2.5, **base)) == 2.5
    assert severity_of(Event(score=-3.0, calibrated_severity=None, **base)) == 3.0


@pytest.mark.integration
def test_build_report_fuses_detection_and_attribution():
    rep = build_report(_run(), run=RUN, min_severity=1.0)
    assert rep.events  # the saturation regime change is anomalous
    assert {ee.why.kind for ee in rep.events} <= {"churn", "drift", "geometry", "dense"}
    assert "churn" in {ee.why.kind for ee in rep.events}  # insertion/eviction spikes

    sevs = [ee.severity for ee in rep.events]
    assert sevs == sorted(sevs, reverse=True)            # sorted by severity desc
    assert rep.worst_severity() == sevs[0]

    churn = next(ee for ee in rep.events if ee.why.kind == "churn")
    assert churn.why.detail["churn"]["inserted"] >= 0    # structured churn breakdown
    drift = [ee for ee in rep.events if ee.why.kind == "drift"]
    if drift:
        assert "top_movers" in drift[0].why.detail        # attribution attached


@pytest.mark.integration
def test_report_json_and_markdown():
    rep = build_report(_run(), run=RUN)
    d = json.loads(rep.to_json())
    assert d["n_events"] == len(rep.events)
    assert d["worst_severity"] == round(rep.worst_severity(), 3)
    assert d["series"] and d["series"][0]["values"]  # the charts' trajectory data
    assert rep.to_markdown().startswith("# flamediff report")


@pytest.mark.integration
def test_report_html_is_self_contained():
    import re

    html = build_report(_run(), run=RUN).to_html()
    assert html.startswith("<!doctype html>")
    assert "http://" not in html and "https://" not in html  # no external deps -> works offline
    data = json.loads(re.search(r'application/json">(.*?)</script>', html, re.S)
                       .group(1).replace("<\\/", "</"))
    assert data["series"] and data["events"]


@pytest.mark.integration
def test_cli_html_output(tmp_path):
    _run()
    out = tmp_path / "report.html"
    res = CliRunner().invoke(app, ["report", RUN, "--html", str(out)])
    assert res.exit_code == 0
    assert out.exists() and out.read_text().startswith("<!doctype html>")


@pytest.mark.integration
def test_cli_report_and_gate():
    _run()  # skip-guard
    runner = CliRunner()
    assert "ANOMALIES" in runner.invoke(app, ["report", RUN]).stdout
    assert runner.invoke(app, ["report", RUN, "--fail-on", "5"]).exit_code == 1
    assert runner.invoke(app, ["report", RUN, "--fail-on", "9999"]).exit_code == 0
    assert runner.invoke(app, ["report", "fixtures"]).exit_code == 2  # <2 checkpoints


@pytest.mark.integration
def test_cli_json_output():
    _run()
    res = CliRunner().invoke(app, ["report", RUN, "--json", "--min-severity", "3"])
    assert res.exit_code == 0
    payload = json.loads(res.stdout)
    assert payload["run"] == RUN and payload["events"]


@pytest.mark.integration
def test_watcher_surfaces_new_events_once(tmp_path):
    src = sorted(glob.glob(f"{RUN}/ckpt_*"))
    if len(src) < 10:
        pytest.skip("no trajectory fixture")
    run = tmp_path / "run"
    run.mkdir()
    w = Watcher(str(run), min_severity=1.0)

    _link(src[:6], run)
    first = w.poll()
    _link(src[6:], run)             # the later batch carries the saturation regime change
    second = w.poll()

    assert second                   # new checkpoints surface new events
    assert w.poll() == []           # no new checkpoints -> nothing surfaced
    assert {_ekey(e) for e in first}.isdisjoint({_ekey(e) for e in second})  # never twice
    assert not isinstance(w._last, list)   # bounded: keeps only the last checkpoint
    assert len(w._diffs) == len(src) - 1


@pytest.mark.integration
def test_cli_watch_max_polls_and_gate(tmp_path):
    src = sorted(glob.glob(f"{RUN}/ckpt_*"))
    if len(src) < 5:
        pytest.skip("no trajectory fixture")
    run = tmp_path / "run"
    run.mkdir()
    _link(src, run)
    runner = CliRunner()
    res = runner.invoke(app, ["watch", str(run), "--max-polls", "1", "--min-severity", "5"])
    assert res.exit_code == 0 and "step" in res.stdout         # one poll, surfaced events, exited
    gated = runner.invoke(app, ["watch", str(run), "--max-polls", "1", "--fail-on", "5"])
    assert gated.exit_code == 1                                 # a >=5x event trips the guard
