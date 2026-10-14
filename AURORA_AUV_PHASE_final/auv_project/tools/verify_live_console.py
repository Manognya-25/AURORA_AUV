"""REAL browser verification of a live dashboard console (headless Chromium via Playwright).

Checks the dashboard someone is actually running (default http://127.0.0.1:5000/) as a PAGE, not just as an API:
the failure this guards against is a page that loads but never becomes live (a JS error at boot leaves the charts
empty and stops the polling, while every endpoint still answers 200).

  1 page loads      : no page errors, no console errors, and the registered charts and the served page agree
  2 a run goes live : Start -> RUNNING and the charts really receive points (canvas data-points > 2)
  3 gains follow    : with autonomous navigation on, the controller panel names the controller that is driving and
                      BOTH gain charts plot that run's numbers (adaptive multipliers or follower speed schedule)
  4 still clean     : no page/console error was produced by any of the above

    python tools/verify_live_console.py [--url http://127.0.0.1:5000/] [--timeout 120] [--keep-running]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def chart_points(pg, canvas_id: str):
    return int(pg.get_attribute(f"#{canvas_id}", "data-points") or 0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify a running dashboard in a real browser.")
    ap.add_argument("--url", default="http://127.0.0.1:5000/", help="the console to check")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for a chart to receive points")
    ap.add_argument("--keep-running", action="store_true", help="leave the verified run going instead of stopping it")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:                                                 # pragma: no cover - environment
        print(f"SKIP: Playwright is unavailable ({exc})")
        return 0

    failures: list[str] = []
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:                                             # pragma: no cover - environment
            print(f"SKIP: Chromium is unavailable ({exc})")
            return 0
        pg = browser.new_page(viewport={"width": 1500, "height": 1400})
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        pg.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
        pg.goto(args.url)
        pg.wait_for_function("() => document.getElementById('state') !== null", timeout=20000)

        # 1 -- the page is whole: every chart the JS registers has its canvas on the page
        def defs():
            return pg.evaluate("() => LIVE_CHARTS.charts.map(c => c.canvasId)")
        registered = defs()
        on_page = pg.evaluate("() => [...document.querySelectorAll('canvas[id$=\"Chart\"]')].map(c => c.id)")
        print(f"1 charts registered by the JS: {len(registered)} | chart canvases on the page: {len(on_page)}")
        if sorted(registered) != sorted(on_page):
            failures.append(f"the JS and the page disagree: JS only {sorted(set(registered) - set(on_page))}, "
                            f"page only {sorted(set(on_page) - set(registered))}")
        if errors:
            failures.append(f"errors on load: {errors}")

        # 2 -- a real run makes the charts live (autonomous navigation, so the follower's gains drive them too)
        pg.evaluate("""async () => fetch('/api/environment/configure', {method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({scenario: 'open_terrain', seed: 3, realtime_factor: 4.0,
                                  navigation_enabled: true, navigation_goal: [30.0, 28.0, 5.0], auto_open_3d: false})})""")
        pg.click("#btnStart")
        pg.wait_for_function("() => document.getElementById('state').textContent.trim() === 'RUNNING'", timeout=90000)
        for canvas in ("errChart", "adaptChart", "adaptTrackChart"):
            pg.wait_for_function(f"() => +document.getElementById('{canvas}').dataset.points > 2", timeout=args.timeout * 1000)
        # the charts must KEEP receiving points, not fill once and freeze (the frozen-page failure mode)
        for canvas in ("errChart", "adaptChart", "adaptTrackChart"):
            grew_from = chart_points(pg, canvas)
            pg.wait_for_function(f"() => +document.getElementById('{canvas}').dataset.points > {grew_from}", timeout=30000)
        print(f"2 points plotted and still growing | position error {chart_points(pg, 'errChart')} | "
              f"gain multipliers {chart_points(pg, 'adaptChart')} | tracked gain effort {chart_points(pg, 'adaptTrackChart')}")
        driving = pg.evaluate("""() => { for (const tr of document.querySelectorAll('#adaptTable tr')) {
            const c = tr.querySelectorAll('td');
            if (c.length >= 2 && c[0].textContent.includes('gains driving this run')) return c[1].textContent.trim(); }
            return null; }""")
        tracking = pg.evaluate("() => { const e = document.getElementById('adaptTrackNow'); return e ? e.textContent.trim() : null; }")
        print(f"3 controller panel: {driving!r} | tracked gain effort now: {tracking!r}")
        if not driving or "follower" not in driving.lower():
            failures.append(f"the gain panel does not name the driving follower: {driving!r}")
        if errors:
            failures.append(f"errors during the run: {errors}")

        if not args.keep_running:
            pg.click("#btnStop")
            pg.wait_for_function("() => ['STOPPED', 'COMPLETED', 'ENDED_BEFORE_GOAL', 'FAILED']"
                                 ".includes(document.getElementById('state').textContent.trim())", timeout=90000)
            pg.click("#btnReset")
            pg.wait_for_function("() => document.getElementById('state').textContent.trim() === 'IDLE'", timeout=60000)

        # 4 -- the page is still clean after all of that
        if errors:
            failures.append(f"errors at the end: {errors}")
        print(f"4 page errors in total: {len(errors)}")
        browser.close()

    for f in failures:
        print(f"FAIL: {f}")
    print("PASS: the console loads clean, plots a live run, and follows the controller that is driving"
          if not failures else f"FAILED: {len(failures)} check(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
