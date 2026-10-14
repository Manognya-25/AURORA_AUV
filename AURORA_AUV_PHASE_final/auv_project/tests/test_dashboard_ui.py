"""Browser test of the real dashboard page (headless Chromium via Playwright): click every control and read
back what the UI displays. Skipped (and reported as skipped) if Playwright/Chromium is unavailable.

Backend: real Flask app + real child process (stand-in vehicle when PyBullet is missing; see test_dashboard_api)."""
import re
import threading
import time

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

from werkzeug.serving import make_server                                   # noqa: E402

from dashboard.app import create_app                                       # noqa: E402
from dashboard.simulation_manager import SimulationManager                 # noqa: E402
from src.utils.procutil import pid_alive                                   # noqa: E402,F401
from tests.helpers.proc_signals import resume_pid, suspend_pid, terminate_pid  # noqa: E402
from tests.helpers.standin_viewer import STANDIN_VIEWER_MODULE             # noqa: E402
from tests.test_dashboard_api import CHILD                                 # noqa: E402


@pytest.fixture()
def page(tmp_path):
    mgr = SimulationManager(runtime_dir=tmp_path / "rt", child_module=CHILD, telemetry_timeout_s=3.0,
                            viewer_module=STANDIN_VIEWER_MODULE)           # live 3-D window without a display
    mgr.configure({"realtime_factor": 1.0, "scenario": "mapping_validation"})
    srv = make_server("127.0.0.1", 0, create_app(mgr), threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:                                             # browser binary missing
            srv.shutdown()
            pytest.skip(f"Chromium unavailable: {exc}")
        pg = browser.new_page(viewport={"width": 1500, "height": 1300})
        pg.errors = []
        pg.on("pageerror", lambda e: pg.errors.append(f"pageerror: {e}"))
        pg.on("console", lambda m: pg.errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
        pg.goto(f"http://127.0.0.1:{srv.server_port}/")
        pg.base = f"http://127.0.0.1:{srv.server_port}"
        pg.mgr = mgr
        yield pg
        browser.close()
    mgr.shutdown()
    srv.shutdown()


def state(pg):
    return pg.inner_text("#state").strip()


def wait_state(pg, want, timeout=40):
    pg.wait_for_function("w => document.getElementById('state').textContent.trim() === w", arg=want, timeout=timeout * 1000)


def row(pg, table, label):
    """Value cell of the row whose first cell contains `label`."""
    return pg.evaluate("""([t, l]) => { for (const tr of document.querySelectorAll('#' + t + ' tr')) {
        const c = tr.querySelectorAll('td'); if (c.length >= 2 && c[0].textContent.includes(l)) return c[1].textContent.trim(); } return null; }""", [table, label])


def cell(pg, table, label, column):
    """Text of the cell in `column` (matched against the header) of the row whose first cell equals `label`."""
    return pg.evaluate("""([t, l, col]) => {
        const rows = [...document.querySelectorAll('#' + t + ' tr')];
        const head = rows.find(r => r.querySelector('th')) ;
        const heads = head ? [...head.querySelectorAll('th')].map(h => h.textContent.trim()) : [];
        const i = heads.findIndex(h => h === col);
        if (i < 0) return null;
        for (const r of rows) {
            const c = r.querySelectorAll('td');
            if (c.length > i && c[0].textContent.trim() === l) return c[i].textContent.trim();
        }
        return null; }""", [table, label, column])


def canvas_colors(pg, cid):
    return pg.evaluate("""(id) => { const c = document.getElementById(id), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        const s = new Set(); for (let i = 0; i < d.length; i += 4 * 97) s.add(d[i] + ',' + d[i+1] + ',' + d[i+2] + ',' + d[i+3]); return [...s]; }""", cid)


def test_initial_page_is_idle_with_correct_enabled_states_and_no_fake_numbers(page):
    wait_state(page, "IDLE")
    assert page.is_enabled("#btnStart") and page.is_disabled("#btnStop") and page.is_enabled("#btnReset")
    for b in ("#btnMapOn", "#btnMapOff", "#btnMapReset", "#selSpeed"):
        assert page.is_disabled(b), b                                         # acts on a running sim: disabled when idle
    for sel in page.query_selector_all("#faultTable select"):
        assert sel.is_disabled()
    assert page.inner_text("#simTime") == "–" and page.inner_text("#tAge") == "–"
    assert "Start the simulation" in page.inner_text("#thrusters")
    assert page.errors == [], page.errors
    scenarios = page.eval_on_selector_all("#selScenario option", "o => o.map(x => x.value)")
    assert "mapping_validation" in scenarios and "narrow_corridor" in scenarios and "legacy_week4" in scenarios


def test_start_button_starts_a_real_process_and_the_page_shows_live_values(page):
    page.click("#btnStart")
    wait_state(page, "RUNNING")
    assert page.mgr.alive and page.is_disabled("#btnStart") and page.is_enabled("#btnStop")
    page.wait_for_function("() => parseFloat(document.getElementById('simTime').textContent) > 2")
    t1 = float(page.inner_text("#simTime").split()[0]); x1 = row(page, "navTable", "x / y")
    time.sleep(2.0)
    t2 = float(page.inner_text("#simTime").split()[0]); x2 = row(page, "navTable", "x / y")
    assert t2 > t1 and x1 != x2                                               # time and estimated position advance
    assert row(page, "ekfTable", "filter status") in ("NOMINAL", "DEAD_RECKONING", "COASTING")
    assert int(row(page, "ekfTable", "predictions")) > 100
    assert "stand-in" in page.inner_text("#dynMode") or page.inner_text("#dynMode") in ("legacy", "6dof")
    assert page.inner_text("#scenario").startswith("mapping_validation")
    assert "healthy" in page.inner_text("#sensorTable")
    assert int(row(page, "mapTable", "map updates")) > 3
    assert int(row(page, "costTable", "traversable cells")) > 0
    page.wait_for_function("() => document.getElementById('sonarTable').textContent.includes('valid / invalid rays')")
    assert page.errors == [], page.errors


def test_occupancy_and_cost_canvases_draw_real_map_content(page):
    page.click("#btnStart")
    wait_state(page, "RUNNING")
    page.wait_for_function("() => parseInt((document.getElementById('occMeta').textContent.match(/\\d+/) || [0])[0]) > 6", timeout=40000)
    time.sleep(1.5)
    occ = canvas_colors(page, "occCanvas")
    assert "68,82,92,255" in occ and "220,236,241,255" in occ                  # unknown (grey) AND free (light) really drawn
    cost = canvas_colors(page, "costCanvas")
    assert "68,82,92,255" in cost and len(cost) >= 3
    assert page.inner_text("#occLegend").count("unknown") == 1 and "free" in page.inner_text("#costLegend")
    import os, tempfile
    page.screenshot(path=os.path.join(tempfile.gettempdir(), "aurora_dashboard_running.png"), full_page=True)
    assert page.errors == [], page.errors


def test_map_reset_button_resets_the_map_the_page_displays(page):
    page.click("#btnStart"); wait_state(page, "RUNNING")
    page.wait_for_function("() => parseInt(document.querySelectorAll('#mapTable tr')[1]?.querySelectorAll('td')[1]?.textContent) > 8", timeout=40000)
    page.click("#btnMapOff")
    page.wait_for_function("() => document.querySelector('#mapTable tr td:last-child')?.textContent.trim() === 'false'")
    known_before = int(row(page, "mapTable", "known / unknown cells").split("/")[0])
    assert known_before > 500
    page.click("#btnMapReset")
    page.wait_for_function("() => document.getElementById('lastAction').textContent.includes('Reset map')")
    page.wait_for_function("() => document.querySelectorAll('#mapTable tr')[2]?.querySelectorAll('td')[1]?.textContent.trim().startsWith('0 /')", timeout=15000)
    assert row(page, "mapTable", "map updates") == "0"
    assert "ok" in page.get_attribute("#lastAction", "class").split()               # shown as success, not error
    assert "reset map" in page.inner_text("#lastAction").lower()
    page.wait_for_function("() => { const c = document.getElementById('occCanvas'); const d = c.getContext('2d').getImageData(0,0,c.width,c.height).data;"
                           " let free = 0; for (let i = 0; i < d.length; i += 4) if (d[i] === 220 && d[i+1] === 236) free++; return free === 0; }", timeout=15000)
    page.click("#btnMapOn")
    page.wait_for_function("() => document.querySelector('#mapTable tr td:last-child')?.textContent.trim() === 'true'")
    page.wait_for_function("() => parseInt(document.querySelectorAll('#mapTable tr')[1]?.querySelectorAll('td')[1]?.textContent) > 0", timeout=15000)
    assert page.errors == [], page.errors


def test_sensor_fault_selector_changes_the_displayed_health_and_failed_attempts_are_not_shown(page):
    page.click("#btnStart"); wait_state(page, "RUNNING")
    page.wait_for_function("() => document.querySelectorAll('#faultTable select')[3] && !document.querySelectorAll('#faultTable select')[3].disabled")
    page.select_option("#faultTable select[data-sensor=sonar]", "FAILED")
    # Wait for the HEALTH pill of the sonar row to read failed, not merely for the word "failed" in the row: the
    # snapshot published right after the injection still carries the previous (healthy) scan while its source cell
    # already says "forced (failed)", so a looser wait races the sonar panel below.
    page.wait_for_function("""() => [...document.querySelectorAll('#sensorTable tr')].some(r => {
        const c = r.querySelectorAll('td');
        return c.length >= 2 && c[0].textContent.trim() === 'sonar' && c[1].textContent.trim() === 'failed'; })""", timeout=15000)
    # the state is applied instantly, the next SCAN arrives at the sonar's own rate (5 Hz): wait for the failed scan
    page.wait_for_function("""() => { const rows = document.querySelectorAll('#sonarTable tr');
        return rows.length > 1 && rows[1].querySelectorAll('td')[1].textContent.trim().startsWith('0 /'); }""", timeout=15000)
    assert "failed" in page.inner_text("#sonarTable").lower()
    assert row(page, "sonarTable", "valid / invalid rays").startswith("0 /")
    page.select_option("#faultTable select[data-sensor=sonar]", "HEALTHY")
    page.wait_for_function("""() => [...document.querySelectorAll('#sensorTable tr')].some(r => {
        const c = r.querySelectorAll('td');
        return c.length >= 2 && c[0].textContent.trim() === 'sonar' && c[1].textContent.trim() === 'healthy'; })""", timeout=15000)
    assert page.errors == [], page.errors


def test_a_healthy_sensor_shows_a_changing_reading_and_a_failed_one_shows_zero_that_stays_zero(page):
    """The reading columns are the sensor's own current sample: they change while it is healthy. Failing the sensor
    leaves it sampling, but with a stuck-at-zero output, so every reading for it is 0.00 and does not change (Sensor
    health panel, fault-injection panel and the IMU chart read-out). Clearing the fault brings the numbers back."""
    page.click("#btnStart"); wait_state(page, "RUNNING")
    # 1. the healthy IMU publishes a real acceleration that keeps changing (this is live data, not a snapshot)
    page.wait_for_function("""() => { const r = [...document.querySelectorAll('#sensorTable tr')]
        .find(t => t.querySelector('td') && t.querySelector('td').textContent.trim() === 'imu');
        const c = r && r.querySelectorAll('td')[3]; return !!c && c.textContent.includes('m/s') && /[0-9]/.test(c.textContent); }""", timeout=30000)
    first = cell(page, "sensorTable", "imu", "live reading")
    assert first and "m/s" in first
    # ...and keeps changing, so the panel is showing live samples rather than one frozen number
    page.evaluate("(v) => { window.__first = v; }", first)
    page.wait_for_function("""() => { const r = [...document.querySelectorAll('#sensorTable tr')]
        .find(t => t.querySelector('td') && t.querySelector('td').textContent.trim() === 'imu');
        const c = r && r.querySelectorAll('td')[3]; return !!c && c.textContent.trim() !== window.__first; }""", timeout=30000)

    # 2. fail the IMU from the fault panel: the output is a real sample stuck at 0.0, and it stays 0.0
    page.wait_for_function("() => !document.querySelector('#faultTable select[data-sensor=imu]').disabled")
    page.select_option("#faultTable select[data-sensor=imu]", "FAILED")
    page.wait_for_function("""() => { const r = [...document.querySelectorAll('#sensorTable tr')]
        .find(t => t.querySelector('td') && t.querySelector('td').textContent.trim() === 'imu');
        const c = r && r.querySelectorAll('td')[1]; return !!c && c.textContent.trim() === 'failed'; }""", timeout=20000)
    page.wait_for_function("""() => { const r = [...document.querySelectorAll('#sensorTable tr')]
        .find(t => t.querySelector('td') && t.querySelector('td').textContent.trim() === 'imu');
        const c = r && r.querySelectorAll('td')[3];
        return !!c && c.textContent.trim().startsWith('a 0.00/0.00/0.00') && /failed/i.test(c.textContent); }""", timeout=20000)
    zeroed = cell(page, "sensorTable", "imu", "live reading")
    assert zeroed.startswith("a 0.00/0.00/0.00") and "failed" in zeroed.lower()
    assert cell(page, "faultTable", "imu", "reading now").startswith("a 0.00/0.00/0.00")   # same rule in the fault panel
    page.evaluate("(v) => { window.__zero = v; }", zeroed)
    page.wait_for_timeout(2500)
    assert page.evaluate("() => { const r = [...document.querySelectorAll('#sensorTable tr')]"
                         ".find(t => t.querySelector('td') && t.querySelector('td').textContent.trim() === 'imu');"
                         "return r.querySelectorAll('td')[3].textContent.trim(); }") == zeroed, "a failed output must not change"
    # the IMU chart is a flat 0.00 line (a real stuck-at-zero sample) and keeps being fed, not an empty read-out
    page.wait_for_function("() => document.getElementById('imuAccNow').textContent.trim().startsWith('0.00')", timeout=20000)
    page.wait_for_timeout(1500)
    assert page.inner_text("#imuAccNow").strip().startswith("0.00"), "the chart must keep plotting the zero sample"

    # 3. clearing the fault brings the real readings back everywhere
    page.select_option("#faultTable select[data-sensor=imu]", "HEALTHY")
    page.wait_for_function("""() => { const r = [...document.querySelectorAll('#sensorTable tr')]
        .find(t => t.querySelector('td') && t.querySelector('td').textContent.trim() === 'imu');
        const c = r && r.querySelectorAll('td')[3]; return !!c && c.textContent.includes('m/s') && !c.textContent.startsWith('a 0.00/0.00/0.00'); }""", timeout=20000)
    page.wait_for_function("() => { const t = document.getElementById('imuAccNow').textContent.trim(); return t !== '–' && !t.startsWith('0.00'); }", timeout=20000)
    assert page.errors == [], page.errors


def test_discarded_and_link_cut_freeze_the_readings_and_discarding_everything_shows_no_values(page):
    """DISCARDED / LINK CUT: the sensor produces no sample at all, so its readings STOP CHANGING (the same frozen
    reading is shown, the chart's read-out drops to no value) while the run keeps going. With every sensor discarded
    the sensor system delivers nothing and no sensor value is shown anywhere."""
    page.click("#btnStart"); wait_state(page, "RUNNING")
    page.wait_for_function("""() => { const r = [...document.querySelectorAll('#sensorTable tr')]
        .find(t => t.querySelector('td') && t.querySelector('td').textContent.trim() === 'imu');
        const c = r && r.querySelectorAll('td')[3]; return !!c && c.textContent.includes('m/s'); }""", timeout=40000)

    for state, label in (("DISCARDED", "discarded"), ("LINK_CUT", "link cut")):
        page.wait_for_function("() => !document.querySelector('#faultTable select[data-sensor=imu]').disabled")
        page.select_option("#faultTable select[data-sensor=imu]", state)
        page.wait_for_function("""(label) => { const r = [...document.querySelectorAll('#sensorTable tr')]
            .find(t => t.querySelector('td') && t.querySelector('td').textContent.trim() === 'imu');
            const c = r && r.querySelectorAll('td')[3];
            return !!c && /frozen/.test(c.textContent) && c.textContent.toLowerCase().includes(label); }""",
            arg=label, timeout=20000)
        frozen = cell(page, "sensorTable", "imu", "live reading")
        assert label in frozen.lower() and "frozen" in frozen.lower(), frozen
        page.wait_for_timeout(2500)
        assert page.evaluate("() => { const r = [...document.querySelectorAll('#sensorTable tr')]"
                             ".find(t => t.querySelector('td') && t.querySelector('td').textContent.trim() === 'imu');"
                             "return r.querySelectorAll('td')[3].textContent.trim(); }") == frozen, \
            "the reading must stop changing while the sensor is discarded / link cut"
        # the chart fed by this sensor shows no live value any more (its real history stays on screen)
        page.wait_for_function("() => document.getElementById('imuAccNow').textContent.trim() === '–'", timeout=20000)
        assert "chart stops" in page.inner_text("#imuAccStats"), page.inner_text("#imuAccStats")

    page.select_option("#faultTable select[data-sensor=imu]", "HEALTHY")
    page.wait_for_function("""() => { const r = [...document.querySelectorAll('#sensorTable tr')]
        .find(t => t.querySelector('td') && t.querySelector('td').textContent.trim() === 'imu');
        const c = r && r.querySelectorAll('td')[3]; return !!c && c.textContent.includes('m/s') && !/frozen/.test(c.textContent); }""", timeout=20000)
    page.wait_for_function("() => document.getElementById('imuAccNow').textContent.trim() !== '–'", timeout=20000)

    # discard everything: no sensor value may be shown anywhere
    for name in ("imu", "dvl", "depth", "compass", "sonar"):
        page.wait_for_function("n => { const s = document.querySelector(`#faultTable select[data-sensor=${n}]`); return s && !s.disabled; }", arg=name, timeout=15000)
        page.select_option(f"#faultTable select[data-sensor={name}]", "DISCARDED")
    page.wait_for_function("() => document.getElementById('sensorTable').innerText.includes('every sensor is discarded')", timeout=25000)
    for name in ("imu", "dvl", "depth", "compass", "sonar"):
        txt = cell(page, "sensorTable", name, "live reading")
        assert not re.search(r"\d", txt), (name, txt)
    assert not re.search(r"\d", cell(page, "faultTable", "imu", "reading now"))
    page.wait_for_function("() => document.getElementById('imuAccNow').textContent.trim() === '–'", timeout=20000)
    assert page.errors == [], page.errors


def test_a_failed_sonar_shows_no_scan_values_and_clearing_it_restores_them(page):
    page.click("#btnStart"); wait_state(page, "RUNNING")
    page.wait_for_function("""() => { const r = [...document.querySelectorAll('#sensorTable tr')]
        .find(t => t.querySelector('td') && t.querySelector('td').textContent.trim() === 'sonar');
        const c = r && r.querySelectorAll('td')[3]; return !!c && /hits/.test(c.textContent); }""", timeout=40000)
    page.wait_for_function("() => !document.querySelector('#faultTable select[data-sensor=sonar]').disabled")
    page.select_option("#faultTable select[data-sensor=sonar]", "FAILED")
    page.wait_for_function("""() => { const r = [...document.querySelectorAll('#sensorTable tr')]
        .find(t => t.querySelector('td') && t.querySelector('td').textContent.trim() === 'sonar');
        const c = r && r.querySelectorAll('td')[1]; return !!c && c.textContent.trim() === 'failed'; }""", timeout=20000)
    n0 = int(cell(page, "sensorTable", "sonar", "valid"))
    page.wait_for_function("() => document.getElementById('clrNow').textContent.trim() === '–'", timeout=20000)
    # 0 hits / 0 valid rays is the sonar's stuck-at-zero output (a 0.0 m RANGE would mean an obstacle on the sensor)
    sonar_txt = cell(page, "faultTable", "sonar", "reading now")
    assert n0 > 0 and "0 valid rays" in sonar_txt and "0 hits" in sonar_txt, sonar_txt
    assert "valid / invalid rays" in page.inner_text("#sonarTable")
    page.select_option("#faultTable select[data-sensor=sonar]", "HEALTHY")
    page.wait_for_function("""() => { const r = [...document.querySelectorAll('#sensorTable tr')]
        .find(t => t.querySelector('td') && t.querySelector('td').textContent.trim() === 'sonar');
        const c = r && r.querySelectorAll('td')[3]; return !!c && /hits/.test(c.textContent); }""", timeout=30000)
    page.wait_for_function("() => document.getElementById('clrNow').textContent.trim() !== '–'", timeout=20000)
    assert page.errors == [], page.errors


def test_speed_selector_is_sent_to_the_running_simulation(page):
    page.click("#btnStart"); wait_state(page, "RUNNING")
    page.wait_for_function("() => !document.getElementById('selSpeed').disabled")
    page.select_option("#selSpeed", "10")
    page.wait_for_function("() => document.getElementById('events').textContent.includes('realtime factor 10')", timeout=15000)
    page.wait_for_function("() => document.getElementById('pace').textContent.includes('real time')")
    assert page.errors == [], page.errors


def test_next_start_form_saves_real_config_and_the_next_run_uses_it(page):
    page.select_option("#selScenario", "dense_obstacles")
    page.fill("#inSeed", "5")
    page.fill("#inCount", "6")
    page.select_option("#selDyn", "6dof")
    page.select_option("#selShow3d", "true")                                   # request the 3D PyBullet window...
    page.click("#btnApply")
    page.wait_for_function("() => document.getElementById('pendingNote').textContent.includes('dense_obstacles')", timeout=15000)
    note = page.inner_text("#pendingNote")
    assert '"seed":5' in note.replace(" ", "") and "6dof" in note
    assert page.mgr.pending["environment"]["scenario"] == "dense_obstacles"
    page.reload()                                                              # the form must show the BACKEND's stored values
    page.wait_for_function("() => document.getElementById('selScenario').options.length > 0")
    page.wait_for_function("() => document.getElementById('selScenario').value === 'dense_obstacles'")
    assert page.input_value("#inSeed") == "5" and page.input_value("#inCount") == "6" and page.input_value("#selDyn") == "6dof"
    assert page.input_value("#selShow3d") == "true" and page.mgr.pending["dashboard"]["headless"] is False
    page.select_option("#selShow3d", "false")                                  # ...but do NOT open a GUI window in the test run
    page.click("#btnApply")
    page.wait_for_function("() => document.getElementById('pendingNote').textContent.includes('\"headless\":true')", timeout=15000)
    assert page.mgr.pending["dashboard"]["headless"] is True
    page.click("#btnStart"); wait_state(page, "RUNNING")
    page.wait_for_function("() => document.getElementById('scenario').textContent.includes('dense_obstacles')", timeout=30000)
    assert "seed 5" in page.inner_text("#scenario")
    assert row(page, "envTable", "scenario / seed") == "dense_obstacles / 5"
    # invalid input is shown as an error, not as success
    page.fill("#inSeed", "-3")
    page.click("#btnApply")
    page.wait_for_function("() => document.getElementById('lastAction').className.includes('err')", timeout=15000)
    assert "seed" in page.inner_text("#lastAction").lower()
    assert page.mgr.pending["environment"]["seed"] == 5                        # the rejected value was not stored


def _configure(page, body):
    return page.evaluate("""async (b) => (await fetch('/api/environment/configure', {method: 'POST',
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify(b)})).status""", body)


def test_goal_controls_and_obstacle_count_drive_the_real_running_simulation(page):
    """The obstacle count really changes the generated world (shown by the run), the goal inputs mirror the
    authoritative navigator goal, and Set goal / Replan / Cancel / Start are real actions on that navigator."""
    assert _configure(page, {"scenario": "mapping_validation", "seed": 3, "obstacle_count": 10,
                             "navigation_enabled": True, "navigation_goal": [45.0, 40.0, -5.0]}) == 200
    page.click("#btnStart")
    wait_state(page, "RUNNING")
    # 1. the environment panel reports the count the world was REALLY built with (10 = 2 scenario-fixed + 8 placed)
    page.wait_for_function("""() => { const r = [...document.querySelectorAll('#envTable tr')]
        .find(t => t.textContent.includes('obstacle count requested')); return r && r.lastChild.textContent.trim() === '10 / 10'; }""", timeout=40000)
    # 2. the goal inputs mirror the authoritative goal (navigator.goal) instead of staying empty. x and y are the
    #    goal's world x/y, the third field is its DEPTH (5 m), which is the unit the operator types and reads.
    page.wait_for_function("""() => document.getElementById('goalX').value === '45.0'
        && document.getElementById('goalY').value === '40.0' && document.getElementById('goalZ').value === '5.0'""", timeout=40000)
    assert not page.is_disabled("#btnGoal") and not page.is_disabled("#btnNavReplan")
    # 3. Set goal really changes the mission: the planner re-plans to it and the dashboard follows
    page.fill("#goalX", "20")
    page.fill("#goalY", "30")
    page.click("#btnGoal")
    page.wait_for_function("() => document.getElementById('lastAction').textContent.includes('goal set')", timeout=20000)
    page.wait_for_function("""() => { const n = document.getElementById('navNote').textContent;
        return n.includes('goal 20.0, 30.0, depth 5.0'); }""", timeout=20000)
    page.wait_for_function("() => document.getElementById('replanTable').textContent.includes('goal_changed')", timeout=30000)
    # 4. Replan / Cancel / Start navigation are real state-machine actions
    page.click("#btnNavReplan")
    page.wait_for_function("() => document.getElementById('lastAction').textContent.includes('replan requested')", timeout=20000)
    page.click("#btnNavCancel")
    page.wait_for_function("() => document.getElementById('navState').textContent.trim() === 'IDLE'", timeout=20000)
    assert page.inner_text("#navHistory").strip() != ""                                  # the state history is real too
    page.click("#btnNavStart")
    page.wait_for_function("""() => ['PLAN', 'FOLLOW_PATH'].includes(document.getElementById('navState').textContent.trim())""", timeout=30000)
    # 5. an invalid goal is reported as an error and does NOT change the displayed goal
    page.fill("#goalX", "900")
    page.click("#btnGoal")
    page.wait_for_function("() => document.getElementById('lastAction').className.includes('err')", timeout=20000)
    assert "20.0, 30.0" in page.inner_text("#navNote")
    js_errors = [e for e in page.errors if "422" not in e]      # the 422 above is the expected answer to the bad goal
    assert js_errors == [], js_errors


def test_obstacle_count_field_is_real_and_rejects_the_week4_map_scenario(page):
    page.select_option("#selScenario", "legacy_week4")
    page.click("#btnApply")
    page.wait_for_function("() => document.getElementById('countNote').textContent.includes('does not apply')", timeout=15000)
    assert page.is_disabled("#inCount")                                               # disabled, with the reason shown
    page.select_option("#selScenario", "dense_obstacles")
    page.fill("#inCount", "53")
    page.click("#btnApply")
    page.wait_for_function("() => document.getElementById('pendingNote').textContent.includes('53')", timeout=15000)
    assert page.mgr.pending["environment"]["obstacles"]["count"] == 53                 # stored by the backend
    assert not page.is_disabled("#inCount")


def test_stop_then_reset_buttons_end_the_process_and_clear_the_page(page):
    page.click("#btnStart"); wait_state(page, "RUNNING")
    pid = page.mgr.status()["pid"]
    page.click("#btnStop")
    wait_state(page, "STOPPED", 20)
    assert not page.mgr.alive and page.is_enabled("#btnStart") and page.is_disabled("#btnStop")
    assert page.is_disabled("#btnMapReset")                                    # nothing running -> map controls disabled
    page.click("#btnReset")
    wait_state(page, "IDLE", 20)
    page.wait_for_function("() => document.getElementById('simTime').textContent.trim() === '–'")
    assert page.inner_text("#navTable").strip() == "" and "Start the simulation" in page.inner_text("#thrusters")
    assert page.mgr.status()["state"] == "IDLE"
    page.click("#btnStart"); wait_state(page, "RUNNING")                       # and it can start again
    assert page.mgr.status()["pid"] != pid
    assert page.errors == [], page.errors


def test_page_reports_a_dead_backend_process_as_failed_and_not_running(page):
    page.click("#btnStart"); wait_state(page, "RUNNING")
    terminate_pid(page.mgr.status()["pid"])
    wait_state(page, "FAILED", 20)
    assert page.is_visible("#banner") and "failed" in page.inner_text("#banner").lower()
    assert page.is_enabled("#btnStart") and page.is_disabled("#btnMapReset")


def test_stalled_telemetry_shows_stale_banner(page):
    page.click("#btnStart"); wait_state(page, "RUNNING")
    pid = page.mgr.status()["pid"]
    suspend_pid(pid)
    try:
        wait_state(page, "STALE", 20)
        assert "stalled" in page.inner_text("#banner").lower()
    finally:
        resume_pid(pid)
    wait_state(page, "RUNNING", 20)
    assert not page.is_visible("#banner")


def test_3d_window_selector_is_saved_by_the_backend_and_restored_after_reload(page):
    """Saves 'on' and verifies the stored value; never presses Start with it on (a real-PyBullet run would open a window)."""
    assert page.input_value("#selShow3d") == "false"                        # default: headless
    page.select_option("#selShow3d", "true")
    page.click("#btnApply")
    page.wait_for_function("() => document.getElementById('pendingNote').textContent.includes('headless')", timeout=15000)
    assert page.mgr.pending["dashboard"]["headless"] is False               # the backend really stored GUI-on
    page.reload()
    page.wait_for_function("() => document.getElementById('selScenario').options.length > 0")
    page.wait_for_function("() => document.getElementById('selShow3d').value === 'true'")   # form shows backend state
    page.select_option("#selShow3d", "false")                               # clean up: back to headless before anything starts
    page.click("#btnApply")
    page.wait_for_function("() => document.getElementById('pendingNote').textContent.includes('\"headless\":true')", timeout=15000)
    assert page.mgr.pending["dashboard"]["headless"] is True
    assert page.errors == [], page.errors


def test_3d_window_switch_opens_and_closes_a_live_window_without_stopping_the_run(page):
    """The dashboard control now acts on the RUNNING run: 'on' opens the live 3-D window (the backend starts a viewer
    process that renders this run), 'off' closes only that window. The simulation process is never stopped."""
    wait_state(page, "IDLE")
    assert page.inner_text("#windowNote").startswith("3-D window")
    page.click("#btnStart")
    wait_state(page, "RUNNING")
    pid = page.mgr.status()["pid"]
    page.wait_for_function("() => document.getElementById('sensorTable').textContent.includes('healthy')")
    page.select_option("#selShow3d", "true")
    page.wait_for_function("() => document.getElementById('windowNote').textContent.includes('OPEN')", timeout=60000)
    assert "viewer PID" in page.inner_text("#windowNote")
    assert page.mgr.viewer_alive
    assert page.mgr.status()["pid"] == pid and page.mgr.status()["state"] == "RUNNING"      # not restarted
    page.wait_for_function("() => document.getElementById('events').textContent.includes('Open 3-D window')")
    page.select_option("#selShow3d", "false")
    page.wait_for_function("() => document.getElementById('windowNote').textContent.includes('CLOSED')", timeout=30000)
    assert not page.mgr.viewer_alive
    assert page.mgr.status()["pid"] == pid and page.mgr.status()["state"] == "RUNNING"
    page.wait_for_function("() => document.getElementById('events').textContent.includes('Close 3-D window')")
    assert page.errors == [], page.errors


def test_live_charts_show_real_streaming_telemetry(page):
    """The telemetry charts are driven only by snapshots from the real backend: idle = empty, running = LIVE badge,
    current-value readouts, plotted pixels and a growing sample count; stopping ends the LIVE state."""
    wait_state(page, "IDLE")
    assert page.inner_text("#liveText") == "NO DATA" and page.inner_text("#errNow") == "–" and page.inner_text("#covNow") == "–"
    assert page.get_attribute("#errChart", "data-points") is None            # nothing plotted before any real sample
    page.click("#btnStart")
    wait_state(page, "RUNNING")
    page.wait_for_function("() => document.getElementById('liveText').textContent === 'LIVE'")
    page.wait_for_function("() => +document.getElementById('covChart').dataset.points > 4", timeout=30000)
    n1 = int(page.get_attribute("#covChart", "data-points"))
    assert page.inner_text("#covNow") != "–" and float(page.inner_text("#covNow")) >= 0.0
    assert "min" in page.inner_text("#covStats") and "samples" in page.inner_text("#covStats")
    assert "updated" in page.inner_text("#liveAge")
    assert len(canvas_colors(page, "covChart")) > 3                          # axes, grid, line, marker: really drawn
    time.sleep(2.0)
    assert int(page.get_attribute("#covChart", "data-points")) > n1          # the series keeps growing from live telemetry
    page.select_option("#selWindow", "30")
    page.wait_for_function("() => document.getElementById('covStats').textContent.includes('last 30 s')")
    page.select_option("#selWindow", "0")
    page.wait_for_function("() => document.getElementById('covStats').textContent.includes('all buffered')")
    page.click("#btnStop")
    wait_state(page, "STOPPED")
    page.wait_for_function("() => document.getElementById('liveText').textContent === 'STOPPED'")
    assert page.errors == [], page.errors


def test_adaptive_controller_gains_are_shown_live_and_charted(page):
    """The Adaptive P controller panel shows the EFFECTIVE proportional gains of the running controller (nominal vs
    effective, both multipliers, the errors behind them) and the matching chart accumulates the same real samples.
    Nothing is computed in the browser: the numbers come from the controller telemetry block of each snapshot."""
    wait_state(page, "IDLE")
    # idle: the panel is empty (like every other panel) instead of showing invented numbers
    assert row(page, "adaptTable", "adaptive") is None
    assert page.inner_text("#adaptNow") == "–"
    assert page.get_attribute("#adaptChart", "data-points") is None
    page.click("#btnStart")
    wait_state(page, "RUNNING", timeout=180)                 # the real PyBullet child can be slow to cold-start
    page.wait_for_function("() => (document.getElementById('adaptTable').textContent || '').includes('kp_position')", timeout=60000)
    eff = row(page, "adaptTable", "kp_position nominal")
    assert "→" in eff and eff.split("→")[1].strip() not in ("–", "", None)      # a real effective gain
    mult = float(row(page, "adaptTable", "gain multiplier (active").replace("×", "").strip())
    sched = float(row(page, "adaptTable", "gain schedule").replace("×", "").strip())
    tune = float(row(page, "adaptTable", "self-tuning gain").replace("×", "").strip())
    assert 1.0 <= mult <= 3.0 and mult == pytest.approx(max(sched, tune), rel=1e-6)
    assert row(page, "adaptTable", "position error (m)") not in ("–", "", None)
    assert row(page, "adaptTable", "heading error (°)") not in ("–", "", None)
    # the chart is fed from the same block and keeps growing with live telemetry
    page.wait_for_function("() => +document.getElementById('adaptChart').dataset.points > 2", timeout=30000)
    n1 = int(page.get_attribute("#adaptChart", "data-points"))
    assert page.inner_text("#adaptNow") not in ("–", "")
    assert "samples" in page.inner_text("#adaptStats")
    assert len(canvas_colors(page, "adaptChart")) > 3                        # axes, grid and lines really drawn
    time.sleep(2.0)
    assert int(page.get_attribute("#adaptChart", "data-points")) > n1                     # still growing
    page.click("#btnStop")
    # the short legacy mission may also have finished on its own before Stop lands: both are real terminal states
    page.wait_for_function("() => ['STOPPED', 'COMPLETED'].includes(document.getElementById('state').textContent.trim())", timeout=60000)
    assert page.errors == [], page.errors


def test_the_tracked_adaptive_gain_history_is_shown_and_charted_live(page):
    """The controller's own gain-tracking record - which way it pushed and why, the time budget per adaptation mode,
    the tracked peak / mean multiplier and the episode count - is shown in the panel and plotted over time. Every
    number is read from telemetry; nothing is recomputed in the browser."""
    wait_state(page, "IDLE")
    assert row(page, "adaptTrackTable", "gain tracking") is None            # idle: nothing is invented
    assert page.get_attribute("#adaptTrackChart", "data-points") is None
    page.click("#btnStart")
    wait_state(page, "RUNNING", timeout=180)                                # the PyBullet child can be slow to cold-start
    page.wait_for_function("() => (document.getElementById('adaptTrackTable').textContent || '').includes('adaptation steps')",
                           timeout=60000)
    # what the gain loop did on the last step, and the measured quantity that chose it
    act = row(page, "adaptTrackTable", "adaptation on the last step")
    assert "—" in act and act.split("—")[1].strip() not in ("", "–", None)
    steps0, elapsed0 = [v.strip() for v in row(page, "adaptTrackTable", "adaptation steps").split("/")]
    assert int(steps0) > 0 and float(elapsed0.split()[0]) > 0.0
    # (the panel prints one decimal, so the budget is checked to within that rounding, not bit-exactly)
    ramp, relax, hold = [float(v) for v in row(page, "adaptTrackTable", "time ramping").split("/")]
    elapsed_s = float(elapsed0.split()[0])
    assert ramp + relax + hold == pytest.approx(elapsed_s, abs=0.2)          # the time budget adds up
    boosted, cap = [float(v) for v in row(page, "adaptTrackTable", "time boosted").split("/")]
    assert 0.0 <= cap <= boosted <= elapsed_s + 0.05
    now, mean, peak = [float(v) for v in row(page, "adaptTrackTable", "gain now").split("/")]
    assert 1.0 <= now <= 3.0 and now <= peak + 1e-9 and 1.0 <= mean <= 3.0
    tune_now, tune_peak = [float(v) for v in row(page, "adaptTrackTable", "self-tuning now").split("/")]
    assert 1.0 <= tune_now <= tune_peak <= 3.0
    assert float(row(page, "adaptTrackTable", "peak gain reached at").split()[0]) >= 0.0
    assert row(page, "adaptTrackTable", "adaptation episodes").count("/") == 1
    # ... and the tracked effort is charted from the same telemetry, growing and really drawn
    page.wait_for_function("() => +document.getElementById('adaptTrackChart').dataset.points > 2", timeout=30000)
    assert page.inner_text("#adaptTrackNow") not in ("–", "")
    assert len(canvas_colors(page, "adaptTrackChart")) > 3
    assert len(canvas_colors(page, "adaptChart")) > 3                       # gain + yaw multipliers are drawn too
    time.sleep(2.0)
    steps1 = int(row(page, "adaptTrackTable", "adaptation steps").split("/")[0])
    assert steps1 > int(steps0)                                            # tracked time keeps accumulating
    page.click("#btnStop")
    # the short legacy mission may also have finished on its own before Stop lands: both are real terminal states
    page.wait_for_function("() => ['STOPPED', 'COMPLETED'].includes(document.getElementById('state').textContent.trim())", timeout=60000)
    assert page.errors == [], page.errors


def test_a_second_run_starts_the_gain_charts_clean_instead_of_dropping_its_samples(page):
    """Starting a second run (without passing through IDLE) must clear the chart buffers. A run always begins again at
    sim time ~0, so a buffer still holding the previous run's larger timestamps silently drops EVERY new sample, and
    the gain charts freeze on the old run until the new run's sim time catches up."""
    wait_state(page, "IDLE")
    page.click("#btnStart")
    wait_state(page, "RUNNING", timeout=180)
    # let the first run get well into its mission (or finish it) before stopping it
    page.wait_for_function("""() => {
        const tr = [...document.querySelectorAll('#adaptTrackTable tr')]
            .find(r => r.cells[0] && r.cells[0].textContent.includes('adaptation steps'));
        return document.getElementById('state').textContent.trim() === 'COMPLETED' ||
            (tr && parseFloat(tr.cells[1].textContent.split('/')[1]) > 12); }
        """, timeout=180000)
    first_sim = float(page.inner_text("#simTime").split()[0])
    assert first_sim > 12                                   # the first run really got somewhere
    page.click("#btnStop")
    # the short legacy mission may also have finished on its own before Stop lands: both are real terminal states
    page.wait_for_function("() => ['STOPPED', 'COMPLETED'].includes(document.getElementById('state').textContent.trim())", timeout=60000)
    page.click("#btnStart")
    wait_state(page, "RUNNING", timeout=180)
    page.wait_for_function("() => (document.getElementById('adaptTrackTable').textContent || '').includes('adaptation steps')", timeout=60000)
    page.wait_for_function("() => +document.getElementById('adaptChart').dataset.points > 2", timeout=60000)
    plotted_t = page.evaluate("() => { const m = /t = ([0-9.]+) s/.exec(document.getElementById('adaptStats').textContent); return m ? parseFloat(m[1]) : null; }")
    now_t = float(page.inner_text("#simTime").split()[0])
    assert plotted_t is not None
    assert plotted_t < first_sim - 5.0                      # this is the SECOND run's sim time, not the first run's
    assert abs(now_t - plotted_t) < 10.0                    # the chart is following the run that is live now
    page.click("#btnStop")
    page.wait_for_function("() => ['STOPPED', 'COMPLETED', 'ENDED_BEFORE_GOAL', 'FAILED'].includes(document.getElementById('state').textContent.trim())", timeout=60000)
    assert page.errors == [], page.errors


def test_the_gain_panel_and_charts_follow_the_path_follower_under_autonomous_navigation(page):
    """With autonomous navigation on it is the PATH FOLLOWER that drives the vehicle, so the gain panel and both gain
    charts must show ITS gains and live speed schedule, updating with the telemetry - they used to be dead in exactly
    this mode ("not driving this run"), which is what the operator runs."""
    wait_state(page, "IDLE")
    assert _configure(page, {"scenario": "open_terrain", "seed": 3, "realtime_factor": 4.0,
                             "navigation_enabled": True, "navigation_goal": [30.0, 28.0, 5.0]}) == 200
    page.click("#btnStart")
    wait_state(page, "RUNNING", timeout=120)
    page.wait_for_function("() => (document.getElementById('adaptTable').textContent || '').includes('path follower')", timeout=60000)
    # the follower's own gains, in the units they are applied in - real numbers, not placeholders
    assert "path follower" in row(page, "adaptTable", "gains driving this run")
    for label in ("kv velocity loop", "kp_yaw / kd_yaw", "kp_depth / ki_depth", "station-kp"):
        assert row(page, "adaptTable", label) not in ("–", "", None), label
    assert float(row(page, "adaptTable", "kv velocity loop").strip()) > 0.0
    assert 0.0 <= float(row(page, "adaptTable", "speed schedule now").replace("×", "").strip()) <= 1.0
    assert row(page, "adaptTable", "cross-track (m)") not in ("–", "", None)
    # ... and its tracked schedule history, counted in the follower
    assert row(page, "adaptTrackTable", "schedule factors").count("/") == 2
    steps, elapsed = [v.strip() for v in row(page, "adaptTrackTable", "commands / simulated time").split("/")]
    follow, hold, station = [float(v) for v in row(page, "adaptTrackTable", "time following").split("/")]
    full, slowed, braked = [float(v) for v in row(page, "adaptTrackTable", "time at full speed").split("/")]
    assert int(steps) > 0 and float(elapsed.split()[0]) > 0.0
    assert follow + hold + station == pytest.approx(float(elapsed.split()[0]), abs=0.2)
    assert full + slowed + braked == pytest.approx(float(elapsed.split()[0]), abs=0.2)
    assert row(page, "adaptTrackTable", "slow-down / brake episodes").count("/") == 1
    # both gain charts are fed from the follower's block and keep growing
    page.wait_for_function("() => +document.getElementById('adaptChart').dataset.points > 2", timeout=60000)
    n1 = int(page.get_attribute("#adaptChart", "data-points"))
    assert page.inner_text("#adaptNow") not in ("–", "")
    assert "path follower" in page.inner_text("#adaptChartNote")            # the caption says which controller it is
    assert len(canvas_colors(page, "adaptChart")) > 3
    page.wait_for_function("() => +document.getElementById('adaptTrackChart').dataset.points > 2", timeout=60000)
    assert page.inner_text("#adaptTrackNow") not in ("–", "")
    assert "path follower" in page.inner_text("#adaptTrackNote")
    assert len(canvas_colors(page, "adaptTrackChart")) > 3
    time.sleep(2.0)
    assert int(page.get_attribute("#adaptChart", "data-points")) > n1      # still following the live run
    page.click("#btnStop")
    page.wait_for_function("() => ['STOPPED', 'COMPLETED', 'ENDED_BEFORE_GOAL', 'FAILED'].includes(document.getElementById('state').textContent.trim())", timeout=60000)
    assert page.errors == [], page.errors


def test_the_goal_you_set_is_shown_back_once_the_robot_reaches_it(page):
    """The goal the operator typed is echoed back with its coordinates when the robot reaches it: the status banner,
    the Mission statistics table, the Estimated navigation panel and the Events log all show the numbers that were
    entered (and the map draws the goal label - any page error there would fail the last assertion).

    The z field is a DEPTH below the surface, typed positive (5.0 = 5 m deep) exactly as the Estimated navigation
    panel reports it, so the number that comes back is the number that went in."""
    wait_state(page, "IDLE")
    assert _configure(page, {"scenario": "open_terrain", "seed": 3, "realtime_factor": 4.0,
                             "start_position": [5.0, 5.0, 5.0], "navigation_enabled": False,
                             "navigation_goal": [7.5, 5.0, 5.0]}) == 200
    page.click("#btnStart")
    wait_state(page, "RUNNING")
    wait_state(page, "COMPLETED", timeout=300)
    coords = "7.50, 5.00, depth 5.00"
    banner = page.inner_text("#banner")
    assert "goal reached" in banner.lower() and coords in banner
    assert row(page, "missionTable", "goal reached — the goal you set") == coords
    assert row(page, "missionTable", "goal you entered") == coords                 # the input is displayed explicitly
    # The Estimated navigation panel reports the goal of THIS run in the same units as the estimate, so "did it
    # arrive where I asked?" needs no cross-referencing: the goal x, y / depth are the typed numbers and the
    # remaining difference is the EKF's own answer.
    assert row(page, "navTable", "goal this run x, y") == "7.50 / 5.00"
    assert row(page, "navTable", "goal this run depth") == "5.00"
    left = [float(v) for v in row(page, "navTable", "goal − estimate").split("/")]
    assert all(abs(v) < 0.5 for v in left), left        # the estimate sits ON the goal that was typed
    assert f"goal reached — the goal you set: {coords}" in page.inner_text("#events")
    assert page.errors == [], page.errors
