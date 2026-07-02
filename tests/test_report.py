import glob
import json
import os

import pytest
from typer.testing import CliRunner

from flamediff import load_checkpoint
from flamediff.cli import app
from flamediff.detect import Event
from flamediff.report import EnrichedEvent, Watcher, Why, build_report, group_incidents, severity_of

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


def _ee(index, sev, *, table="t", metric="m", method="robust_z", step=None):
    e = Event(index=index, step=step if step is not None else index * 50, table=table,
              metric=metric, value=1.0, baseline=0.0, score=sev, direction="up",
              method=method, calibrated_severity=sev)
    return EnrichedEvent(e, Why("churn", "x"))


def test_group_incidents_chains_adjacent_indices_across_series():
    # one cause fires many series at indices 4/5 (robust_z at the spike, PH one step later);
    # an unrelated event fires at index 9 -> two incidents, not five anomalies
    events = [_ee(4, 17.6, metric="inserted_rate"), _ee(4, 4.5, metric="evicted_rate", table="u"),
              _ee(5, 3.2, method="page_hinkley"), _ee(5, 2.0, method="pelt"),
              _ee(9, 6.0, metric="frozen_max")]
    incs = group_incidents(events)
    assert [len(i.events) for i in incs] == [4, 1]        # severity-desc incident order
    assert incs[0].headline.severity == 17.6              # headlined by the strongest signal
    assert incs[0].steps == [200, 250] and incs[0].tables == ["t", "u"]
    assert incs[0].span() == "steps 200–250" and incs[1].span() == "step 450"
    sevs = [ee.severity for ee in incs[0].events]
    assert sevs == sorted(sevs, reverse=True)             # events severity-desc within incident
    assert group_incidents([]) == []


def test_group_incidents_gap_splits_distant_events():
    a, b = _ee(1, 5.0), _ee(4, 3.0)
    assert len(group_incidents([a, b], gap=1)) == 2
    assert len(group_incidents([a, b], gap=3)) == 1


@pytest.mark.integration
def test_report_incidents_group_the_flat_events():
    rep = build_report(_run(), run=RUN, min_severity=1.0)
    incs = rep.incidents()
    assert 0 < len(incs) < len(rep.events)                # the whole point: fewer incidents
    assert sum(len(i.events) for i in incs) == len(rep.events)  # a partition, nothing dropped
    d = rep.to_dict()
    assert d["n_incidents"] == len(incs)
    flat = [i for inc in d["incidents"] for i in inc["events"]]
    assert sorted(flat) == list(range(len(rep.events)))   # JSON refs cover every event once
    assert "INCIDENTS" in rep.to_text() and "SUMMARY" in rep.to_text()
    assert "## incident 1" in rep.to_markdown()


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
def test_serve_endpoints(tmp_path):
    import threading
    import urllib.request

    if len(glob.glob(f"{RUN}/ckpt_*")) < 5:
        pytest.skip("no trajectory fixture")
    run = tmp_path / "run"
    run.mkdir()
    _link(sorted(glob.glob(f"{RUN}/ckpt_*")), run)

    from flamediff.serve import make_server
    httpd = make_server(str(run), host="127.0.0.1", port=0, interval=1.0, min_severity=1.0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        html = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10).read().decode()
        assert html.startswith("<!doctype html>")
        assert "const POLL=1000" in html  # live mode wired: interval 1s -> re-fetch every 1000ms
        payload = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{port}/data.json", timeout=10).read())
        assert payload["events"] and payload["series"]
    finally:
        httpd.shutdown()


@pytest.mark.integration
def test_cli_report_and_gate():
    _run()  # skip-guard
    runner = CliRunner()
    assert "INCIDENTS" in runner.invoke(app, ["report", RUN]).stdout
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
    # bounded in run length: only the last checkpoint + reduced per-step forms are retained
    assert w._last is not None and not isinstance(w._last, list)  # holds one checkpoint's weights
    assert len(w._rows) == len(src) - 1                           # one scalar-features row per step
    assert not hasattr(w, "_diffs")                               # full per-id diffs not retained
    assert all(len(e["attr"][1]) <= 10 for e in w._why.values())  # mover ids capped (no arrays)


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
