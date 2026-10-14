/* AURORA live telemetry charts.

   Rendering only. This module owns NO data: every chart reads the series that dashboard.js accumulates from real
   backend snapshots (getSeries() returns [[sim_time_s, value], ...]). Nothing here generates, smooths, pads or
   extrapolates values. The only animation is cosmetic: the newest point glides from the previous sample to the new
   one over the measured arrival interval, so the time axis scrolls continuously instead of jumping at each poll.

   Chart kinds:
     - single series (cfg.getSeries) - the original eight charts, unchanged output format.
     - multi series (cfg.series = [{label, color, getSeries}] or a function returning that list) - one line per
       component with an in-plot legend; optional stepped rendering (cfg.step) for discrete signals such as sensor
       health, fixed tick labels (cfg.yTicks / cfg.yTickLabels), and a per-chart empty message (cfg.emptyMsg, may
       be a function of info) so a chart can say WHY it is empty (e.g. no thruster layer in legacy dynamics).
     - cfg.sourceOk(info) / cfg.sourceMsg(info): for charts fed by ONE sensor's raw output. While sourceOk() is
       false (that sensor is failed, or its newest sample is invalid/stale) the chart shows NO current value: the
       read-out becomes "–", the stats line carries the real reason from sourceMsg(), and the value tag says
       "no valid sample" instead of a number. The already-plotted line is real recorded history and stays visible.

   Features: rolling time window (10/30/60/120/300 s / all buffered), auto-scaled axis with nice ticks, grid, axis
   titles with units, current-value readout, window min/avg/max, latest-point marker with value tag, LIVE / STALLED /
   ENDED indicator driven by the real run state, hover read-out, HiDPI rendering, reduced-motion support. */
"use strict";

(function (global) {
  const FONT = '11px "Segoe UI","Helvetica Neue",Arial,sans-serif';
  const M = { l: 52, r: 74, t: 12, b: 36 };           // plot margins (css px): room for y labels / value tag / x labels
  const HEAD_PAD = 10;                                 // px between the latest point and the plot's right edge
  const REDUCED = !!(global.matchMedia && global.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const TIME_STEPS = [0.5, 1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600];

  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const num = (v, d) => (v === null || v === undefined || !Number.isFinite(Number(v))) ? "–" : Number(v).toFixed(d);

  function niceStep(span, target) {
    const raw = span / Math.max(1, target), p = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / p;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p;
  }
  function tickValues(lo, hi, step) {
    const out = []; for (let k = Math.ceil(lo / step - 1e-9); k * step <= hi + 1e-9; k++) out.push(+(k * step).toFixed(10)); return out;
  }
  const stepDecimals = (step) => clamp(-Math.floor(Math.log10(step) + 1e-9), 0, 5);

  /* Greedy word wrap for the on-chart empty message: at most `maxLines` lines that each fit `maxWidth`, with an
     ellipsis when the text is longer. Without it a long reason (e.g. "Autonomous controller not driving this run
     (...)") ran over the axes and was cut off by the canvas edge. */
  function wrapText(ctx, text, maxWidth, maxLines) {
    const words = String(text == null ? "" : text).split(/\s+/).filter(Boolean);
    const lines = []; let cur = "", done = 0;
    while (done < words.length && lines.length < maxLines) {
      const probe = cur ? cur + " " + words[done] : words[done];
      if (cur && ctx.measureText(probe).width > maxWidth) { lines.push(cur); cur = ""; continue; }
      cur = probe; done++;
    }
    if (cur) lines.push(cur);
    const clipped = done < words.length;
    lines.forEach((ln, i) => {
      const tail = (clipped && i === lines.length - 1) ? " …" : "";
      let l = ln;
      while (l.length > 1 && ctx.measureText(l + tail).width > maxWidth) l = l.slice(0, -1);
      lines[i] = (l === ln && !tail) ? ln : l.replace(/[\s,;:]+$/, "") + " …";
    });
    return lines.length ? lines : [""];
  }

  /* series definitions: single getSeries or a (static / dynamic) list of {label, color, getSeries} */
  function defsOf(c) {
    if (typeof c.series === "function") return c.series() || [];
    if (Array.isArray(c.series)) return c.series;
    return [{ label: c.label, color: c.color, getSeries: c.getSeries }];
  }
  function emptyOf(c, info) {
    const m = c.emptyMsg;
    return (typeof m === "function") ? m(info) : (m || info.emptyMsg);
  }
  /* True when the chart's single sensor source is delivering a usable sample right now (charts without a
     sourceOk get true: their series are not tied to one sensor's output). */
  function sourceFresh(c, info) {
    return (typeof c.sourceOk !== "function") || !!c.sourceOk(info);
  }
  function sourceMsgOf(c, info) {
    return (typeof c.sourceMsg === "function") ? c.sourceMsg(info) : "no valid data from this chart's source";
  }

  /* ---------------------------------------------------------------- one chart */
  class LiveChart {
    constructor(cfg) {
      this.cfg = cfg;
      this.canvas = document.getElementById(cfg.canvasId);
      this.ctx = this.canvas.getContext("2d");
      this.nowEl = document.getElementById(cfg.nowId);
      this.statsEl = document.getElementById(cfg.statsId);
      this.heads = []; this.froms = []; this.tos = []; this.keys = [];   // animated newest point per series
      this.animStart = 0; this.animDur = 400; this.lastArrival = 0;
      this.yr = null; this.lastFrame = 0; this.hoverX = null;
      this.w = 0; this.h = 0; this.dpr = 1; this.statsKey = "";
      this.layout = null; this.multi = false;
      this._resize();
      if (global.ResizeObserver) new ResizeObserver(() => this._resize()).observe(this.canvas);
      const pos = (e) => { const r = this.canvas.getBoundingClientRect(); return e.clientX - r.left; };
      this.canvas.addEventListener("pointermove", (e) => { this.hoverX = pos(e); });
      this.canvas.addEventListener("pointerleave", () => { this.hoverX = null; });
    }

    _resize() {
      const r = this.canvas.getBoundingClientRect();
      const dpr = Math.max(1, global.devicePixelRatio || 1);
      const w = Math.max(200, Math.round(r.width)), h = Math.max(120, Math.round(r.height));
      if (w === this.w && h === this.h && dpr === this.dpr) return;
      this.w = w; this.h = h; this.dpr = dpr;
      this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
    }

    frame(now, info) {
      const dt = this.lastFrame ? Math.min(250, now - this.lastFrame) : 16; this.lastFrame = now;
      const defs = defsOf(this.cfg);
      this.multi = defs.length > 1;
      const series = defs.map((d) => {
        let data = typeof d.getSeries === "function" ? d.getSeries() : null;
        return { d, data: Array.isArray(data) ? data : [] };
      });
      const primIdx = series.findIndex((s) => s.data.length > 0);
      if (primIdx < 0) {                                             // no real data yet in any series
        this.heads = []; this.froms = []; this.tos = []; this.keys = []; this.yr = null;
        this._setReadouts(null, series, info);
        this._draw(now, info, series, null, 0, dt);
        return;
      }

      // Detect newly arrived real samples (any series) and start gliding the heads toward them.
      const keys = series.map((s) => s.data.length ? (s.data[s.data.length - 1][0] + ":" + s.data[s.data.length - 1][1]) : "none");
      const changed = keys.length !== this.keys.length || keys.some((k, i) => k !== this.keys[i]);
      if (changed) {
        series.forEach((s, i) => {
          const last = s.data.length ? s.data[s.data.length - 1] : null;
          const prev = this.heads[i];
          if (!last) { this.heads[i] = this.froms[i] = this.tos[i] = null; return; }
          if (!prev || last[0] < prev.t - 1e-9 || REDUCED) {          // first sample or a run reset
            this.froms[i] = this.tos[i] = { t: last[0], v: last[1] };
            this.heads[i] = { t: last[0], v: last[1] };
          } else {
            this.froms[i] = { t: prev.t, v: prev.v };
            this.tos[i] = { t: last[0], v: last[1] };
          }
        });
        this.animDur = clamp(now - this.lastArrival, 150, 900); this.animStart = now; this.lastArrival = now;
        this.keys = keys;
        const pl = series[primIdx].data.length;
        this.canvas.dataset.points = String(pl);
        this.canvas.dataset.last = String(series[primIdx].data[pl - 1][1]);
      }
      const p = REDUCED ? 1 : clamp((now - this.animStart) / this.animDur, 0, 1);
      series.forEach((s, i) => {
        const f = this.froms[i], t = this.tos[i];
        this.heads[i] = (f && t) ? { t: f.t + (t.t - f.t) * p, v: f.v + (t.v - f.v) * p } : null;
      });
      const head = this.heads[primIdx];
      this._setReadouts(series[primIdx], series, info);
      this._draw(now, info, series, head, primIdx, dt);
    }

    _visible(data, headT, xl) {
      let i = data.length - 1; while (i > 0 && data[i][0] >= xl) i--;                  // keep one point left of the window
      const pts = [];
      for (; i < data.length; i++) { if (data[i][0] > headT) break; pts.push(data[i]); }
      return pts;
    }

    _setReadouts(prim, series, info) {
      const c = this.cfg;
      const withData = series.filter((s) => s.data.length);
      if (!sourceFresh(c, info)) {                                  // the source sensor is failed / silent: no value
        const msg = sourceMsgOf(c, info);
        if (this.statsKey !== "nosource|" + msg) {
          this.statsKey = "nosource|" + msg;
          this.nowEl.textContent = "–";
          this.statsEl.textContent = msg;
        }
        return;
      }
      const key = withData.map((s) => { const l = s.data[s.data.length - 1]; return l[0] + "|" + l[1]; }).join(";")
        + "|" + info.windowSec + "|" + withData.length;
      if (!prim) {
        if (this.statsKey !== "none|" + info.emptyMsg) { this.statsKey = "none|" + info.emptyMsg; this.nowEl.textContent = "–"; this.statsEl.textContent = emptyOf(c, info); }
        return;
      }
      if (key === this.statsKey) return; this.statsKey = key;
      const lastT = prim.data[prim.data.length - 1][0];
      const xl = info.windowSec > 0 ? lastT - info.windowSec : -Infinity;
      const stat = (data) => {
        let mn = Infinity, mx = -Infinity, sum = 0, n = 0;
        for (let i = data.length - 1; i >= 0 && data[i][0] >= xl; i--) { const v = data[i][1]; mn = Math.min(mn, v); mx = Math.max(mx, v); sum += v; n++; }
        return { mn, mx, mean: n ? sum / n : NaN, n };
      };
      const stats = withData.map((s) => ({ label: s.d.label, ...stat(s.data) }));
      const pq = stat(prim.data);
      const wl = info.windowSec > 0 ? `last ${info.windowSec} s` : "all buffered";
      this.nowEl.textContent = this.multi
        ? withData.map((s) => num(s.data[s.data.length - 1][1], c.decimals)).join(" / ")
        : num(prim.data[prim.data.length - 1][1], c.decimals);
      if (!this.multi) {
        this.statsEl.textContent = `${wl} · min ${num(pq.mn, c.decimals)} · mean ${num(pq.mean, c.decimals)} · max ${num(pq.mx, c.decimals)} ${c.unit} · ${pq.n} samples · t = ${num(lastT, 1)} s`;
        return;
      }
      if (stats.length <= 3) {
        const per = stats.map((q) => `${q.label || ""} ${num(q.mn, c.decimals)}/${num(q.mean, c.decimals)}/${num(q.mx, c.decimals)}`).join(" · ");
        this.statsEl.textContent = `${wl} · min/mean/max: ${per} ${c.unit} · ${pq.n} samples · t = ${num(lastT, 1)} s`;
      } else {
        const allMn = Math.min(...stats.map((q) => q.mn)), allMx = Math.max(...stats.map((q) => q.mx));
        const tot = stats.reduce((a, q) => a + q.n, 0);
        const mean = tot ? stats.reduce((a, q) => a + q.mean * q.n, 0) / tot : NaN;
        this.statsEl.textContent = `${wl} · ${stats.length} series · min ${num(allMn, c.decimals)} · mean ${num(mean, c.decimals)} · max ${num(allMx, c.decimals)} ${c.unit} · ${pq.n} samples · t = ${num(lastT, 1)} s`;
      }
    }

    _yTarget(allPts) {
      const c = this.cfg; let mn = Infinity, mx = -Infinity;
      allPts.forEach((q) => { mn = Math.min(mn, q[1]); mx = Math.max(mx, q[1]); });
      let lo, hi;
      if (c.yMin !== undefined) { lo = c.yMin; hi = mx + Math.max((mx - lo) * 0.15, c.minSpan * 0.1); }
      else { const pad = Math.max((mx - mn) * 0.15, c.minSpan * 0.1); lo = mn - pad; hi = mx + pad; }
      if (hi - lo < c.minSpan) { if (c.yMin !== undefined) hi = lo + c.minSpan; else { const m = (hi + lo) / 2; lo = m - c.minSpan / 2; hi = m + c.minSpan / 2; } }
      if (c.yMax !== undefined) hi = Math.min(hi, c.yMax);
      if (hi <= lo) hi = lo + c.minSpan;
      return { lo, hi };
    }

    _draw(now, info, series, head, primIdx, dt) {
      const ctx = this.ctx, w = this.w, h = this.h, c = this.cfg, col = info.colors;
      const prim = series[primIdx] || series[0];
      const primColor = (prim && prim.d && prim.d.color) || c.color;
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0); ctx.clearRect(0, 0, w, h);
      ctx.font = FONT; ctx.textBaseline = "middle";
      const ml = c.marginLeft || M.l;
      const x0 = ml, x1 = w - M.r, y0 = M.t, y1 = h - M.b;
      let vis = [], xl, xr, lo, hi;

      if (head) {
        // Fixed window: fills left-to-right from the first sample, then scrolls once the run is longer than the window.
        let firstT = Infinity;
        series.forEach((s) => { if (s.data.length) firstT = Math.min(firstT, s.data[0][0]); });
        xl = info.windowSec > 0 ? Math.max(0, head.t - info.windowSec) : Math.min(firstT, head.t - 5);
        xr = info.windowSec > 0 ? xl + info.windowSec : head.t;
        vis = [];
        series.forEach((s, i) => {
          if (!s.data.length) { s.pts = []; return; }
          const hp = this.heads[i] ? this.heads[i].t : s.data[s.data.length - 1][0];
          const pts = this._visible(s.data, hp, xl);
          if (this.heads[i]) pts.push([this.heads[i].t, this.heads[i].v]);
          s.pts = pts;
          vis.push(...pts);
        });
        const t = this._yTarget(vis.length ? vis : [[0, 0]]);
        if (!this.yr || REDUCED) this.yr = t; else { const k = 1 - Math.exp(-dt / 160); this.yr = { lo: this.yr.lo + (t.lo - this.yr.lo) * k, hi: this.yr.hi + (t.hi - this.yr.hi) * k }; }
        lo = this.yr.lo; hi = this.yr.hi;
      } else {                                            // no data yet: draw the empty frame with the nominal scale
        xr = info.windowSec > 0 ? info.windowSec : 60; xl = 0; lo = c.yMin !== undefined ? c.yMin : 0; hi = c.yMin !== undefined ? c.yMin + c.minSpan : c.minSpan;
        series.forEach((s) => { s.pts = []; });
      }
      const tx = (t) => x0 + ((t - xl) / Math.max(xr - xl, 1e-9)) * (x1 - x0 - HEAD_PAD);
      const ty = (v) => y1 - ((v - lo) / Math.max(hi - lo, 1e-12)) * (y1 - y0);
      this.layout = { tx, ty, x0, x1, y0, y1, xl, xr };

      // plot background, grid and axes
      ctx.fillStyle = col.plot; ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      ctx.lineWidth = 1; ctx.textAlign = "right"; ctx.fillStyle = col.muted;
      const ys = niceStep(hi - lo, Math.max(3, Math.floor((y1 - y0) / 38))), yd = stepDecimals(ys);
      const yTicks = Array.isArray(c.yTicks) ? c.yTicks.filter((v) => v >= lo - 1e-9 && v <= hi + 1e-9) : tickValues(lo, hi, ys);
      yTicks.forEach((v) => {
        const y = Math.round(ty(v)) + 0.5; ctx.strokeStyle = col.grid; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
        const lbl = (c.yTickLabels && c.yTickLabels[v] !== undefined) ? c.yTickLabels[v] : v.toFixed(yd);
        ctx.fillText(lbl, x0 - 6, y);
      });
      const span = xr - xl; let xs = TIME_STEPS.find((s) => span / s <= 7) || 900; const xd = xs < 1 ? 1 : 0;
      ctx.textAlign = "center";
      for (let k = Math.ceil(xl / xs); k * xs <= xr + 1e-9; k++) {
        const t = k * xs, x = Math.round(tx(t)) + 0.5; if (x < x0 || x > x1) continue;
        ctx.strokeStyle = col.grid; ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
        ctx.strokeStyle = col.axis; ctx.beginPath(); ctx.moveTo(x, y1); ctx.lineTo(x, y1 + 4); ctx.stroke();
        ctx.fillText(t.toFixed(xd), x, y1 + 12);
      }
      ctx.strokeStyle = col.axis; ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0, y1 - y0);
      ctx.fillStyle = col.muted; ctx.textAlign = "center"; ctx.fillText("Simulation time (s)", (x0 + x1) / 2, h - 7);
      ctx.save(); ctx.translate(11, (y0 + y1) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(c.axisTitle, 0, 0); ctx.restore();

      if (!head) {
        // "why is this chart empty": wrapped to the plot width and clipped to the plot, so it never runs over the
        // axes or the value tag.
        ctx.save(); ctx.beginPath(); ctx.rect(x0, y0, x1 - x0, y1 - y0); ctx.clip();
        ctx.fillStyle = col.muted; ctx.textAlign = "center"; ctx.font = '13px "Segoe UI",Arial,sans-serif';
        const lines = wrapText(ctx, emptyOf(c, info), x1 - x0 - 16, 3), lh = 17;
        const cy = (y0 + y1) / 2 - (lines.length - 1) * lh / 2;
        lines.forEach((ln, i) => ctx.fillText(ln, (x0 + x1) / 2, cy + i * lh));
        ctx.restore(); ctx.font = FONT; return;
      }

      // series: line (+ optional area fill for single-series charts), clipped to the plot
      ctx.save(); ctx.beginPath(); ctx.rect(x0, y0, x1 - x0, y1 - y0); ctx.clip();
      const path = (pts) => {
        ctx.beginPath();
        if (c.step) {                                    // stepped line for discrete signals (sensor health)
          pts.forEach((q, i) => {
            const x = tx(q[0]), y = ty(q[1]);
            if (i === 0) ctx.moveTo(x, y);
            else { ctx.lineTo(x, ty(pts[i - 1][1])); ctx.lineTo(x, y); }
          });
        } else {
          pts.forEach((q, i) => { const x = tx(q[0]), y = ty(q[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
        }
      };
      if (!this.multi && prim && prim.pts && prim.pts.length >= 2) {
        const pts = prim.pts, g = ctx.createLinearGradient(0, y0, 0, y1);
        g.addColorStop(0, c.color + "40"); g.addColorStop(1, c.color + "05");
        path(pts); ctx.lineTo(tx(pts[pts.length - 1][0]), y1); ctx.lineTo(tx(pts[0][0]), y1); ctx.closePath();
        ctx.fillStyle = g; ctx.fill();
      }
      series.forEach((s) => {
        if (!s.pts || s.pts.length < 2) return;
        path(s.pts);
        ctx.strokeStyle = s.d.color || c.color; ctx.lineWidth = 1.8; ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.stroke();
      });
      // navigation event markers (real state transitions / replans / avoidance reported by the navigator)
      const evs = typeof c.getEvents === "function" ? c.getEvents() : [];
      ctx.font = "10px 'Segoe UI',Arial,sans-serif";
      evs.forEach((ev) => {
        if (ev.t < xl || ev.t > xr) return;
        const ex = Math.round(tx(ev.t)) + 0.5;
        ctx.save(); ctx.setLineDash([4, 3]); ctx.strokeStyle = ev.color; ctx.globalAlpha = 0.75; ctx.beginPath(); ctx.moveTo(ex, y0); ctx.lineTo(ex, y1); ctx.stroke(); ctx.restore();
        ctx.fillStyle = ev.color; ctx.textAlign = "left"; ctx.save(); ctx.translate(ex + 3, y1 - 4); ctx.rotate(-Math.PI / 2); ctx.fillText(ev.label, 0, 0); ctx.restore();
      });
      ctx.font = FONT;
      ctx.restore();

      // legend (multi-series): colour + component label
      let legendH = 0;
      if (this.multi) {
        ctx.font = "10px 'Segoe UI',Arial,sans-serif";
        let lx = x0 + 8; const ly = y0 + 9; legendH = 14;
        let hidden = 0;
        series.forEach((s) => {
          const lbl = s.d.label || "";
          const wdt = 14 + ctx.measureText(lbl).width + 8;
          if (lx + wdt > x1 - 6) { hidden++; return; }
          ctx.strokeStyle = s.d.color || c.color; ctx.lineWidth = 3;
          ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 11, ly); ctx.stroke();
          ctx.fillStyle = col.muted; ctx.textAlign = "left"; ctx.fillText(lbl, lx + 15, ly);
          lx += wdt;
        });
        if (hidden) ctx.fillText("+" + hidden, lx, ly);
        ctx.font = FONT;
      }

      // latest-point marker, guide line and value tag (primary series)
      const off = !sourceFresh(c, info);                 // sensor-fed chart with no current sample: never print a number
      const hx = tx(head.t), hy = ty(head.v);
      ctx.save(); ctx.setLineDash([3, 3]); ctx.strokeStyle = primColor; ctx.globalAlpha = 0.6; ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(x1, hy); ctx.stroke(); ctx.restore();
      if (info.live && !REDUCED && !off) {                // pulse only while real samples are arriving
        const ph = (now % 1400) / 1400; ctx.save(); ctx.globalAlpha = 0.45 * (1 - ph); ctx.strokeStyle = primColor; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(hx, hy, 4 + 9 * ph, 0, 6.2832); ctx.stroke(); ctx.restore();
      }
      ctx.beginPath(); ctx.arc(hx, hy, 4.2, 0, 6.2832); ctx.fillStyle = "#fff"; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = off ? col.warn : primColor; ctx.stroke();
      const tagV = this.tos[primIdx] ? this.tos[primIdx].v : head.v;
      const tag = off ? "no valid sample"
        : `${this.multi && prim.d.label ? prim.d.label + " " : ""}${num(tagV, c.decimals)} ${c.unit}`.trim();
      const tw = ctx.measureText(tag).width + 10, th = 18, ty0 = clamp(hy - th / 2, y0, y1 - th);
      ctx.fillStyle = (info.stale || off) ? col.warn : primColor; ctx.fillRect(x1 + 2, ty0, tw, th);
      ctx.beginPath(); ctx.moveTo(x1 + 2, ty0 + th / 2 - 4); ctx.lineTo(x1 - 3, clamp(hy, y0 + 3, y1 - 3)); ctx.lineTo(x1 + 2, ty0 + th / 2 + 4); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.textAlign = "left"; ctx.fillText(tag, x1 + 7, ty0 + th / 2 + 0.5);

      if (info.stale || off) {
        ctx.fillStyle = col.warn; ctx.textAlign = "left";
        ctx.fillText(off ? "NO VALID SAMPLE – no value is shown for this sensor" : "TELEMETRY STALLED – showing last received values", x0 + 8, y0 + 12 + legendH);
      }

      // hover read-out: nearest real sample of each series to the pointer
      if (this.hoverX !== null && this.hoverX >= x0 && this.hoverX <= x1) {
        const nearest = series.map((s) => {
          if (!s.pts || !s.pts.length) return null;
          let best = null, bd = 1e9;
          s.pts.forEach((q) => { const d = Math.abs(tx(q[0]) - this.hoverX); if (d < bd && tx(q[0]) >= x0) { bd = d; best = q; } });
          return best;
        });
        const firstIdx = nearest.findIndex((q) => q);
        if (firstIdx >= 0) {
          const bx = tx(nearest[firstIdx][0]);
          const txt = `t = ${num(nearest[firstIdx][0], 1)} s   ${nearest.map((q, i) => q ? `${series[i].d.label ? series[i].d.label + " " : ""}${num(q[1], c.decimals)}` : null).filter(Boolean).join("  ")} ${c.unit}`.trim();
          ctx.save(); ctx.strokeStyle = col.ink; ctx.globalAlpha = 0.5; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(bx, y0); ctx.lineTo(bx, y1); ctx.stroke(); ctx.restore();
          const dy = ty(nearest[firstIdx][1]);
          ctx.beginPath(); ctx.arc(bx, dy, 3.5, 0, 6.2832); ctx.fillStyle = primColor; ctx.fill();
          const bw = Math.min(ctx.measureText(txt).width + 12, x1 - x0 - 4);
          const bx0 = clamp(bx - bw / 2, x0 + 2, x1 - bw - 2);
          ctx.fillStyle = col.ink; ctx.globalAlpha = 0.92; ctx.fillRect(bx0, y0 + 4, bw, 18); ctx.globalAlpha = 1;
          ctx.fillStyle = "#fff"; ctx.textAlign = "left"; ctx.fillText(txt, bx0 + 6, y0 + 13.5);
        }
      }
    }
  }

  /* ---------------------------------------------------------------- controller */
  const RUN_LABEL = { RUNNING: ["live", "LIVE"], STALE: ["stale", "STALLED"], STARTING: ["wait", "CONNECTING"], STOPPING: ["wait", "STOPPING"],
    STOPPED: ["off", "STOPPED"], COMPLETED: ["off", "COMPLETED"], ENDED_BEFORE_GOAL: ["off", "ENDED"], FAILED: ["bad", "FAILED"], IDLE: ["off", "NO DATA"] };

  function createLiveCharts(opts) {
    const css = getComputedStyle(document.documentElement), v = (n, d) => (css.getPropertyValue(n).trim() || d);
    const colors = { ink: v("--ink", "#16232b"), muted: v("--muted", "#5c6f79"), grid: "#e3e9eb", axis: "#9fb0b8", plot: "#fbfdfd", warn: v("--warn", "#b7791f") };
    /* One unusable chart must never take the whole console down. A page whose DOM does not match this file (an
       older document paired with this JS, or an edited template) used to throw here on the first missing canvas,
       which killed the boot sequence and left a page with no charts AND no polling. Each chart is therefore built
       on its own: a missing canvas is reported and skipped, the rest are drawn, and the live badge still runs. */
    const charts = [];
    opts.charts.forEach((c) => {
      if (!document.getElementById(c.canvasId)) {
        console.error(`chart canvas #${c.canvasId} is missing from the page: reload the page (Ctrl+F5); skipping this chart`);
        return;
      }
      try { charts.push(new LiveChart(c)); } catch (err) { console.error(`chart #${c.canvasId} failed to start:`, err); }
    });
    const sel = document.getElementById(opts.windowSelectId), badge = document.getElementById(opts.badgeId);
    const textEl = document.getElementById(opts.badgeTextId), ageEl = document.getElementById(opts.ageId);
    let cache = "";

    function frame(now) {
      const status = opts.getStatus(), st = status ? status.state : null;
      const [cls, label] = status ? (RUN_LABEL[st] || ["off", st]) : ["bad", "OFFLINE"];
      const info = { windowSec: Number(sel && sel.value) || 0, live: cls === "live", stale: cls === "stale", colors,
        emptyMsg: !status ? "Backend unreachable" : st === "IDLE" ? "No data – start the simulation" : "Waiting for telemetry…" };
      const age = status && status.telemetry_age_s !== null && status.telemetry_age_s !== undefined ? `updated ${num(status.telemetry_age_s, 1)} s ago` : "";
      const k = cls + label + age;
      if (k !== cache) {
        cache = k;
        if (badge) badge.className = "live-badge lb-" + cls;
        if (textEl) textEl.textContent = label;
        if (ageEl) ageEl.textContent = age;
      }
      charts.forEach((c) => c.frame(now, info));
      global.requestAnimationFrame(frame);
    }
    global.requestAnimationFrame(frame);
    return { charts };
  }

  global.createLiveCharts = createLiveCharts;
})(window);
