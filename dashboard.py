"""
Expert dashboard generation for Classical BAP+QCA solver v9.0
Rich interactive HTML with Gantt chart, cost waterfall, SA convergence,
and detailed metrics — pure HTML/CSS/JS (no external dependencies).
"""
import os
import json
import logging
from solver_helpers import _iso_to_hours

logger = logging.getLogger("qcentroid-user-log")


def _generate_expert_dashboard(assignments, berths, vessels, cost_breakdown,
                                optimization_convergence, berth_utilization,
                                priority_analysis, gantt_data, schedule_metrics,
                                computation_metrics, crane_allocation):
    """Generate rich interactive expert dashboard HTML."""
    os.makedirs("additional_output", exist_ok=True)

    # ── Prepare Gantt data ───────────────────────────────────────────
    gantt_json = []
    for g in gantt_data:
        gantt_json.append({
            "vessel": g.get("vessel", "Unknown"),
            "berth": str(g.get("berth", "?")),
            "start_h": round(_iso_to_hours(g.get("start", "")), 2),
            "end_h": round(_iso_to_hours(g.get("end", "")), 2),
            "cranes": g.get("cranes", 1),
            "priority": g.get("priority", 3)
        })

    # ── Prepare vessel table rows ────────────────────────────────────
    vessel_rows = ""
    for a in assignments:
        if a.get("berth_id") is None:
            continue
        p = a.get("priority", 3)
        p_class = "p-high" if p <= 2 else "p-normal"
        vessel_rows += (
            f'<tr class="{p_class}">'
            f'<td>{a.get("vessel_name", a["vessel_id"])}</td>'
            f'<td>{a["berth_id"]}</td>'
            f'<td class="num">{a.get("cranes_assigned", 0)}</td>'
            f'<td class="num">{a.get("handling_hours", 0):.1f}h</td>'
            f'<td class="num">{a.get("waiting_hours", 0):.1f}h</td>'
            f'<td class="num">{a.get("delay_hours", 0):.1f}h</td>'
            f'<td class="num">${a.get("cost", 0):,.0f}</td></tr>\n'
        )

    # ── Prepare berth utilization rows ───────────────────────────────
    berth_rows = ""
    for b in berth_utilization:
        util = b.get("utilization_pct", 0)
        bar_color = "#64ffda" if util > 30 else "#ff6b6b" if util < 10 else "#ffd93d"
        berth_rows += (
            f'<tr><td>{b.get("berth_id", "?")}</td>'
            f'<td class="num">{b.get("vessels_served", 0)}</td>'
            f'<td class="num">{b.get("occupied_hours", 0):.1f}h</td>'
            f'<td><div class="bar-bg"><div class="bar-fill" style="width:{min(util, 100):.0f}%;background:{bar_color}"></div>'
            f'<span class="bar-label">{util:.1f}%</span></div></td>'
            f'<td class="num">{b.get("total_teu_handled", 0):,.0f}</td></tr>\n'
        )

    # ── Prepare priority rows ────────────────────────────────────────
    priority_rows = ""
    for key in sorted(priority_analysis.keys()):
        pa = priority_analysis[key]
        priority_rows += (
            f'<tr><td><span class="priority-badge {"priority-high" if key in ("P1","P2") else "priority-low"}">{key}</span></td>'
            f'<td class="num">{pa.get("count", 0)}</td>'
            f'<td class="num">${pa.get("avg_cost", 0):,.0f}</td>'
            f'<td class="num">{pa.get("avg_wait_h", 0):.1f}h</td>'
            f'<td class="num">{pa.get("avg_delay_h", 0):.1f}h</td>'
            f'<td class="num">${pa.get("total_cost", 0):,.0f}</td></tr>\n'
        )

    # ── Cost evolution JSON ──────────────────────────────────────────
    cost_evolution = optimization_convergence.get("cost_evolution", [])
    cost_evolution_json = json.dumps(cost_evolution)

    # ── Key metrics ──────────────────────────────────────────────────
    tc = cost_breakdown.get("total_cost", 0)
    gc = optimization_convergence.get("greedy_initial_cost", 0)
    fc = optimization_convergence.get("final_optimized_cost", 0)
    imp = optimization_convergence.get("improvement_pct", 0)
    two_opt = optimization_convergence.get("two_opt_cost", 0)
    crane_reopt = optimization_convergence.get("crane_reopt_cost", 0)
    sa_cost = optimization_convergence.get("sa_cost", fc)

    # Cost breakdown for waterfall
    crane_cost = cost_breakdown.get("crane_handling_cost", 0)
    wait_cost = cost_breakdown.get("waiting_cost", 0)
    delay_cost = cost_breakdown.get("delay_penalty_cost", 0)

    # SA stats
    sa_iters = optimization_convergence.get("sa_iterations", 0)
    sa_accepted = optimization_convergence.get("sa_accepted", 0)
    sa_improved = optimization_convergence.get("sa_improved", 0)
    sa_t_final = optimization_convergence.get("sa_temperature_final", 0)
    sa_accept_rate = round(sa_accepted / max(sa_iters, 1) * 100, 1)

    version = computation_metrics.get("solver_version", "9.0")
    algorithm = computation_metrics.get("algorithm", "Unknown")
    wall_time = computation_metrics.get("wall_time_s", 0)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BAP+QCA Classical Solver v{version} — Expert Dashboard</title>
<style>
:root {{
  --bg-primary: #0a192f;
  --bg-secondary: #112240;
  --bg-card: rgba(17, 34, 64, 0.8);
  --accent: #64ffda;
  --accent-dim: rgba(100, 255, 218, 0.15);
  --text-primary: #ccd6f6;
  --text-secondary: #8892b0;
  --border: #233554;
  --red: #ff6b6b;
  --yellow: #ffd93d;
  --blue: #4fc3f7;
  --purple: #bb86fc;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: var(--bg-primary); color: var(--text-primary); font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; padding: 24px; line-height: 1.5; }}
.header {{ display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 2px solid var(--accent); padding-bottom: 16px; margin-bottom: 24px; }}
.header h1 {{ color: var(--accent); font-size: 22px; font-weight: 600; }}
.header .meta {{ color: var(--text-secondary); font-size: 12px; text-align: right; }}
.header .algo {{ background: var(--accent-dim); color: var(--accent); padding: 2px 10px; border-radius: 12px; font-size: 11px; display: inline-block; margin-top: 4px; }}

/* KPI Cards */
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }}
.kpi {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; border-left: 4px solid var(--accent); transition: transform 0.2s; }}
.kpi:hover {{ transform: translateY(-2px); border-left-color: var(--blue); }}
.kpi .val {{ font-size: 22px; font-weight: 700; color: var(--accent); }}
.kpi .lbl {{ font-size: 10px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }}
.kpi.negative .val {{ color: var(--red); }}
.kpi.highlight .val {{ color: var(--yellow); }}

/* Section */
.section {{ margin-bottom: 28px; }}
.section h2 {{ color: var(--accent); font-size: 15px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }}
.section h2::before {{ content: ''; width: 4px; height: 16px; background: var(--accent); border-radius: 2px; }}

/* Tabs */
.tabs {{ display: flex; gap: 0; margin-bottom: 0; border-bottom: 2px solid var(--border); }}
.tab {{ padding: 8px 18px; cursor: pointer; color: var(--text-secondary); font-size: 12px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 2px solid transparent; margin-bottom: -2px; transition: all 0.2s; }}
.tab:hover {{ color: var(--text-primary); }}
.tab.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
.tab-content {{ display: none; background: var(--bg-card); border: 1px solid var(--border); border-top: none; border-radius: 0 0 10px 10px; padding: 16px; }}
.tab-content.active {{ display: block; }}

/* Tables */
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ background: rgba(100, 255, 218, 0.08); color: var(--accent); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; padding: 10px 12px; text-align: left; border-bottom: 2px solid var(--border); position: sticky; top: 0; }}
td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); }}
tr:hover {{ background: rgba(100, 255, 218, 0.03); }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.p-high {{ background: rgba(255, 107, 107, 0.06); }}
.p-high td:first-child {{ border-left: 3px solid var(--red); }}

/* Priority badges */
.priority-badge {{ padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }}
.priority-high {{ background: rgba(255, 107, 107, 0.2); color: var(--red); }}
.priority-low {{ background: rgba(100, 255, 218, 0.15); color: var(--accent); }}

/* Bars */
.bar-bg {{ background: var(--bg-primary); border-radius: 6px; height: 20px; position: relative; overflow: hidden; min-width: 100px; }}
.bar-fill {{ height: 100%; border-radius: 6px; transition: width 0.5s; }}
.bar-label {{ position: absolute; right: 6px; top: 1px; font-size: 11px; font-weight: 600; color: var(--text-primary); }}

/* Canvas containers */
.chart-container {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
canvas {{ width: 100% !important; }}

/* Gantt */
#gantt-chart {{ overflow-x: auto; }}
.gantt-row {{ display: flex; align-items: center; margin: 3px 0; }}
.gantt-label {{ width: 90px; font-size: 11px; color: var(--text-secondary); flex-shrink: 0; text-align: right; padding-right: 10px; }}
.gantt-track {{ flex: 1; height: 28px; position: relative; background: rgba(255,255,255,0.02); border-radius: 4px; }}
.gantt-bar {{ position: absolute; height: 24px; top: 2px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 600; color: #0a192f; cursor: pointer; transition: opacity 0.2s; min-width: 20px; }}
.gantt-bar:hover {{ opacity: 0.85; filter: brightness(1.1); }}
.gantt-tooltip {{ display: none; position: fixed; background: var(--bg-secondary); border: 1px solid var(--accent); padding: 10px; border-radius: 8px; font-size: 12px; z-index: 100; pointer-events: none; min-width: 180px; }}

/* Convergence flow */
.conv-flow {{ display: flex; align-items: center; gap: 0; flex-wrap: wrap; justify-content: center; margin: 16px 0; }}
.conv-step {{ text-align: center; padding: 10px 14px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; min-width: 110px; }}
.conv-step .phase {{ font-size: 10px; color: var(--text-secondary); text-transform: uppercase; }}
.conv-step .cost {{ font-size: 16px; font-weight: 700; color: var(--accent); }}
.conv-step .delta {{ font-size: 10px; color: var(--red); }}
.conv-arrow {{ font-size: 20px; color: var(--accent); margin: 0 4px; }}
.conv-step.best {{ border-color: var(--accent); box-shadow: 0 0 12px rgba(100,255,218,0.2); }}

/* Waterfall */
.waterfall-bar {{ display: flex; align-items: center; margin: 6px 0; }}
.waterfall-label {{ width: 130px; font-size: 12px; color: var(--text-secondary); text-align: right; padding-right: 12px; }}
.waterfall-track {{ flex: 1; height: 28px; position: relative; }}
.waterfall-segment {{ position: absolute; height: 28px; border-radius: 4px; display: flex; align-items: center; padding: 0 8px; font-size: 11px; font-weight: 600; }}

/* SA stats */
.sa-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }}
.sa-stat {{ background: var(--bg-primary); border-radius: 8px; padding: 12px; }}
.sa-stat .label {{ font-size: 10px; color: var(--text-secondary); text-transform: uppercase; }}
.sa-stat .value {{ font-size: 18px; font-weight: 700; color: var(--blue); }}

/* Responsive */
@media (max-width: 768px) {{
  .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
  .conv-flow {{ flex-direction: column; }}
  .conv-arrow {{ transform: rotate(90deg); }}
}}
</style>
</head><body>

<div class="header">
  <div>
    <h1>BAP+QCA Classical Solver v{version}</h1>
    <p style="color:var(--text-secondary);font-size:13px">Berth Allocation + Quay Crane Assignment | Expert Dashboard</p>
  </div>
  <div class="meta">
    <div>{computation_metrics.get('wall_time_s', 0):.2f}s runtime</div>
    <div class="algo">{algorithm}</div>
  </div>
</div>

<!-- ── KPI Cards ─────────────────────────────────────────────────── -->
<div class="kpi-grid">
  <div class="kpi"><div class="val">${tc:,.0f}</div><div class="lbl">Total Cost</div></div>
  <div class="kpi"><div class="val">${cost_breakdown.get('cost_per_vessel', 0):,.0f}</div><div class="lbl">Cost / Vessel</div></div>
  <div class="kpi highlight"><div class="val">{imp:.1f}%</div><div class="lbl">Improvement</div></div>
  <div class="kpi"><div class="val">{wall_time:.1f}s</div><div class="lbl">Runtime</div></div>
  <div class="kpi"><div class="val">{crane_allocation.get('avg_cranes_per_vessel', 0):.1f}</div><div class="lbl">Avg Cranes</div></div>
  <div class="kpi"><div class="val">{schedule_metrics.get('makespan', 0):.1f}h</div><div class="lbl">Makespan</div></div>
  <div class="kpi"><div class="val">{schedule_metrics.get('total_waiting_time', 0):.1f}h</div><div class="lbl">Total Wait</div></div>
  <div class="kpi"><div class="val">{schedule_metrics.get('feasible_assignments', 0)}/{schedule_metrics.get('feasible_assignments', 0) + schedule_metrics.get('infeasible_assignments', 0)}</div><div class="lbl">Feasible</div></div>
</div>

<!-- ── Optimization Convergence ───────────────────────────────────── -->
<div class="section">
  <h2>Optimization Convergence</h2>
  <div class="conv-flow">
    <div class="conv-step"><div class="phase">Greedy</div><div class="cost">${gc:,.0f}</div><div class="delta">baseline</div></div>
    <div class="conv-arrow">&#8594;</div>
    <div class="conv-step"><div class="phase">2-Opt</div><div class="cost">${two_opt:,.0f}</div><div class="delta">-${max(0, gc - two_opt):,.0f}</div></div>
    <div class="conv-arrow">&#8594;</div>
    <div class="conv-step"><div class="phase">Crane Reopt</div><div class="cost">${crane_reopt:,.0f}</div><div class="delta">-${max(0, two_opt - crane_reopt):,.0f}</div></div>
    <div class="conv-arrow">&#8594;</div>
    <div class="conv-step{"  best" if sa_cost <= crane_reopt else ""}"><div class="phase">Simulated Annealing</div><div class="cost">${sa_cost:,.0f}</div><div class="delta">-${max(0, crane_reopt - sa_cost):,.0f}</div></div>
    <div class="conv-arrow">&#8594;</div>
    <div class="conv-step best"><div class="phase">Final</div><div class="cost">${fc:,.0f}</div><div class="delta">-${max(0, gc - fc):,.0f} ({imp:.1f}%)</div></div>
  </div>

  <!-- Cost evolution chart (canvas) -->
  <div class="chart-container">
    <canvas id="costEvolutionChart" height="180"></canvas>
  </div>
</div>

<!-- ── Cost Breakdown Waterfall ───────────────────────────────────── -->
<div class="section">
  <h2>Cost Breakdown</h2>
  <div class="chart-container">
    <div class="waterfall-bar">
      <div class="waterfall-label">Crane Handling</div>
      <div class="waterfall-track">
        <div class="waterfall-segment" style="left:0;width:{crane_cost/max(tc,1)*100:.1f}%;background:var(--blue);">${crane_cost:,.0f}</div>
      </div>
    </div>
    <div class="waterfall-bar">
      <div class="waterfall-label">Waiting Cost</div>
      <div class="waterfall-track">
        <div class="waterfall-segment" style="left:0;width:{wait_cost/max(tc,1)*100:.1f}%;background:var(--yellow);">${wait_cost:,.0f}</div>
      </div>
    </div>
    <div class="waterfall-bar">
      <div class="waterfall-label">Delay Penalty</div>
      <div class="waterfall-track">
        <div class="waterfall-segment" style="left:0;width:{delay_cost/max(tc,1)*100:.1f}%;background:var(--red);">${delay_cost:,.0f}</div>
      </div>
    </div>
    <div class="waterfall-bar">
      <div class="waterfall-label" style="font-weight:700;color:var(--accent)">Total</div>
      <div class="waterfall-track">
        <div class="waterfall-segment" style="left:0;width:100%;background:var(--accent);color:var(--bg-primary);">${tc:,.0f}</div>
      </div>
    </div>
  </div>
</div>

<!-- ── Tabbed Details ─────────────────────────────────────────────── -->
<div class="section">
  <div class="tabs">
    <div class="tab active" onclick="switchTab(event, 'tab-gantt')">Gantt Chart</div>
    <div class="tab" onclick="switchTab(event, 'tab-vessels')">Vessels</div>
    <div class="tab" onclick="switchTab(event, 'tab-berths')">Berths</div>
    <div class="tab" onclick="switchTab(event, 'tab-priority')">Priority</div>
    <div class="tab" onclick="switchTab(event, 'tab-sa')">SA Stats</div>
  </div>

  <!-- Gantt Tab -->
  <div id="tab-gantt" class="tab-content active">
    <div id="gantt-chart"></div>
    <div id="gantt-legend" style="margin-top:10px;font-size:11px;color:var(--text-secondary)">
      Hover over bars for details. Colors by berth. Bar width = handling duration.
    </div>
  </div>

  <!-- Vessels Tab -->
  <div id="tab-vessels" class="tab-content">
    <div style="max-height:400px;overflow-y:auto">
    <table>
      <tr><th>Vessel</th><th>Berth</th><th>Cranes</th><th>Handling</th><th>Wait</th><th>Delay</th><th>Cost</th></tr>
      {vessel_rows}
    </table>
    </div>
  </div>

  <!-- Berths Tab -->
  <div id="tab-berths" class="tab-content">
    <table>
      <tr><th>Berth</th><th>Vessels</th><th>Occupied</th><th>Utilization</th><th>TEU</th></tr>
      {berth_rows}
    </table>
  </div>

  <!-- Priority Tab -->
  <div id="tab-priority" class="tab-content">
    <table>
      <tr><th>Priority</th><th>Count</th><th>Avg Cost</th><th>Avg Wait</th><th>Avg Delay</th><th>Total Cost</th></tr>
      {priority_rows}
    </table>
  </div>

  <!-- SA Stats Tab -->
  <div id="tab-sa" class="tab-content">
    <div class="sa-grid">
      <div class="sa-stat"><div class="label">SA Iterations</div><div class="value">{sa_iters:,}</div></div>
      <div class="sa-stat"><div class="label">Accepted Moves</div><div class="value">{sa_accepted:,}</div></div>
      <div class="sa-stat"><div class="label">Improving Moves</div><div class="value">{sa_improved:,}</div></div>
      <div class="sa-stat"><div class="label">Acceptance Rate</div><div class="value">{sa_accept_rate}%</div></div>
      <div class="sa-stat"><div class="label">Final Temperature</div><div class="value">{sa_t_final:,.0f}</div></div>
      <div class="sa-stat"><div class="label">Pre-SA Cost</div><div class="value">${crane_reopt:,.0f}</div></div>
      <div class="sa-stat"><div class="label">Post-SA Cost</div><div class="value">${sa_cost:,.0f}</div></div>
      <div class="sa-stat"><div class="label">SA Savings</div><div class="value">${max(0, crane_reopt - sa_cost):,.0f}</div></div>
    </div>
    <div style="margin-top:16px;font-size:12px;color:var(--text-secondary)">
      Move types: swap_berth, relocate, crane_adjust, swap_order. Metropolis criterion with exponential cooling.
    </div>
  </div>
</div>

<!-- ── Performance Footer ─────────────────────────────────────────── -->
<div style="margin-top:24px;padding:14px;background:var(--bg-card);border:1px solid var(--border);border-radius:10px;font-size:12px;color:var(--text-secondary)">
  <strong style="color:var(--accent)">Algorithm Pipeline:</strong> {algorithm} |
  Makespan: {schedule_metrics.get('makespan', 0):.1f}h |
  Total Wait: {schedule_metrics.get('total_waiting_time', 0):.1f}h |
  Or-Opt: {computation_metrics.get('or_opt_improvements', 0)} moves |
  Crane Rebal: {computation_metrics.get('crane_rebalance_improvements', 0)} adj |
  SA: {sa_iters} iters ({sa_improved} improved) |
  Cost/TEU: ${cost_breakdown.get('cost_per_teu', 0):.2f}
</div>

<div style="margin-top:12px;text-align:center;color:var(--border);font-size:10px">
  Classical Solver v{version} | QCentroid Platform | Generated dynamically
</div>

<!-- ── Tooltip ────────────────────────────────────────────────────── -->
<div class="gantt-tooltip" id="tooltip"></div>

<script>
// ── Tab switching ───────────────────────────────────────────────────
function switchTab(e, tabId) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  e.target.classList.add('active');
  document.getElementById(tabId).classList.add('active');
}}

// ── Gantt Chart ─────────────────────────────────────────────────────
(function() {{
  const data = {json.dumps(gantt_json)};
  const berths = [...new Set(data.map(d => d.berth))].sort();
  const colors = ['#64ffda', '#4fc3f7', '#bb86fc', '#ffd93d', '#ff6b6b', '#a5d6a7', '#ffab91', '#80deea'];
  const berthColor = {{}};
  berths.forEach((b, i) => berthColor[b] = colors[i % colors.length]);

  // Find time range
  let minH = Infinity, maxH = -Infinity;
  data.forEach(d => {{ if (d.start_h < minH) minH = d.start_h; if (d.end_h > maxH) maxH = d.end_h; }});
  const range = maxH - minH || 1;

  const container = document.getElementById('gantt-chart');

  // Time axis
  const axisDiv = document.createElement('div');
  axisDiv.style.cssText = 'display:flex;margin-left:90px;margin-bottom:4px;position:relative;height:20px;';
  const steps = 8;
  for (let i = 0; i <= steps; i++) {{
    const h = minH + (range / steps) * i;
    const tick = document.createElement('div');
    tick.style.cssText = `position:absolute;left:${{(i / steps * 100)}}%;font-size:10px;color:#8892b0;transform:translateX(-50%)`;
    tick.textContent = h.toFixed(0) + 'h';
    axisDiv.appendChild(tick);
  }}
  container.appendChild(axisDiv);

  // Sort data by berth then start
  data.sort((a, b) => a.berth === b.berth ? a.start_h - b.start_h : a.berth.localeCompare(b.berth));

  const tooltip = document.getElementById('tooltip');

  data.forEach(d => {{
    const row = document.createElement('div');
    row.className = 'gantt-row';

    const label = document.createElement('div');
    label.className = 'gantt-label';
    label.textContent = d.vessel.length > 12 ? d.vessel.substring(0, 10) + '..' : d.vessel;
    row.appendChild(label);

    const track = document.createElement('div');
    track.className = 'gantt-track';

    const bar = document.createElement('div');
    bar.className = 'gantt-bar';
    const left = ((d.start_h - minH) / range) * 100;
    const width = Math.max(((d.end_h - d.start_h) / range) * 100, 1.5);
    bar.style.left = left + '%';
    bar.style.width = width + '%';
    bar.style.background = berthColor[d.berth];
    bar.textContent = d.cranes + 'c';

    // Priority indicator
    if (d.priority <= 2) {{
      bar.style.boxShadow = '0 0 6px rgba(255,107,107,0.6)';
      bar.style.border = '1px solid #ff6b6b';
    }}

    bar.addEventListener('mouseenter', function(e) {{
      tooltip.style.display = 'block';
      tooltip.innerHTML = `<strong style="color:var(--accent)">${{d.vessel}}</strong><br>`
        + `Berth: ${{d.berth}}<br>`
        + `Time: ${{d.start_h.toFixed(1)}}h — ${{d.end_h.toFixed(1)}}h (${{(d.end_h - d.start_h).toFixed(1)}}h)<br>`
        + `Cranes: ${{d.cranes}}<br>`
        + `Priority: P${{d.priority}}`;
    }});
    bar.addEventListener('mousemove', function(e) {{
      tooltip.style.left = (e.clientX + 12) + 'px';
      tooltip.style.top = (e.clientY - 10) + 'px';
    }});
    bar.addEventListener('mouseleave', function() {{
      tooltip.style.display = 'none';
    }});

    track.appendChild(bar);
    row.appendChild(track);
    container.appendChild(row);
  }});

  // Legend
  const legend = document.getElementById('gantt-legend');
  berths.forEach(b => {{
    const span = document.createElement('span');
    span.style.cssText = `display:inline-block;margin-right:12px;`;
    span.innerHTML = `<span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:${{berthColor[b]}};vertical-align:middle;margin-right:4px"></span>${{b}}`;
    legend.appendChild(span);
  }});
}})();

// ── Cost Evolution Chart (Canvas) ────────────────────────────────────
(function() {{
  const data = {cost_evolution_json};
  if (!data.length) return;

  const canvas = document.getElementById('costEvolutionChart');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = 180 * dpr;
  canvas.style.height = '180px';
  ctx.scale(dpr, dpr);

  const W = rect.width, H = 180;
  const pad = {{ left: 80, right: 20, top: 20, bottom: 35 }};
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const values = data.map(d => d.objective_value);
  const minV = Math.min(...values) * 0.95;
  const maxV = Math.max(...values) * 1.05;
  const rangeV = maxV - minV || 1;

  // Grid
  ctx.strokeStyle = '#233554';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const y = pad.top + plotH * (1 - i / 4);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    ctx.fillStyle = '#8892b0';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    const val = minV + rangeV * (i / 4);
    ctx.fillText('$' + Math.round(val).toLocaleString(), pad.left - 8, y + 3);
  }}

  // Line + points
  const phaseColors = {{ greedy: '#ff6b6b', '2-opt': '#ffd93d', 'or-opt': '#bb86fc', crane_reopt: '#4fc3f7', simulated_annealing: '#64ffda', final: '#64ffda' }};

  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(100,255,218,0.5)';
  ctx.beginPath();
  data.forEach((d, i) => {{
    const x = pad.left + (i / Math.max(data.length - 1, 1)) * plotW;
    const y = pad.top + plotH * (1 - (d.objective_value - minV) / rangeV);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }});
  ctx.stroke();

  // Points + labels
  data.forEach((d, i) => {{
    const x = pad.left + (i / Math.max(data.length - 1, 1)) * plotW;
    const y = pad.top + plotH * (1 - (d.objective_value - minV) / rangeV);
    const color = phaseColors[d.phase] || '#64ffda';

    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#0a192f';
    ctx.beginPath(); ctx.arc(x, y, 2, 0, Math.PI * 2); ctx.fill();

    // Phase label
    ctx.fillStyle = '#8892b0';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(d.phase, x, H - pad.bottom + 14);

    // Value label
    ctx.fillStyle = color;
    ctx.font = '10px sans-serif';
    ctx.fillText('$' + Math.round(d.objective_value).toLocaleString(), x, y - 10);
  }});
}})();
</script>
</body></html>"""

    with open("additional_output/01_expert_dashboard.html", "w") as f:
        f.write(html)
    logger.info("Expert dashboard v%s written to additional_output/01_expert_dashboard.html", version)
