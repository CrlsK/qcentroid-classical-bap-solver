"""
Expert dashboard generation for Classical BAP+QCA solver v6.0
Compact version - generates summary HTML without Plotly dependency.
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
    """Generate compact expert dashboard HTML with summary tables and metrics."""
    os.makedirs("additional_output", exist_ok=True)

    # Build vessel assignment rows
    vessel_rows = ""
    for a in assignments:
        if a.get("berth_id") is None:
            continue
        vessel_rows += (
            f'<tr><td>{a.get("vessel_name", a["vessel_id"])}</td>'
            f'<td>{a["berth_id"]}</td>'
            f'<td>{a.get("cranes_assigned", 0)}</td>'
            f'<td>{a.get("handling_hours", 0):.1f}</td>'
            f'<td>{a.get("waiting_hours", 0):.1f}</td>'
            f'<td>{a.get("delay_hours", 0):.1f}</td>'
            f'<td>${a.get("cost", 0):,.0f}</td></tr>\n'
        )

    # Build berth utilization rows
    berth_rows = ""
    for b in berth_utilization:
        berth_rows += (
            f'<tr><td>{b.get("berth_id", "?")}</td>'
            f'<td>{b.get("vessels_served", 0)}</td>'
            f'<td>{b.get("utilization_pct", 0):.1f}%</td></tr>\n'
        )

    # Build priority analysis rows
    priority_rows = ""
    for key in sorted(priority_analysis.keys()):
        pa = priority_analysis[key]
        priority_rows += (
            f'<tr><td>{key}</td>'
            f'<td>{pa.get("count", 0)}</td>'
            f'<td>${pa.get("avg_cost", 0):,.0f}</td>'
            f'<td>{pa.get("avg_wait_h", 0):.1f}</td>'
            f'<td>{pa.get("avg_delay_h", 0):.1f}</td></tr>\n'
        )

    # Build gantt data JSON for optional Plotly rendering
    gantt_json = []
    for g in gantt_data:
        gantt_json.append({
            "vessel": g.get("vessel", "Unknown"),
            "berth": g.get("berth", "Unknown"),
            "start_h": _iso_to_hours(g.get("start", "")),
            "end_h": _iso_to_hours(g.get("end", "")),
            "priority": g.get("priority", 3)
        })

    tc = cost_breakdown.get("total_cost", 0)
    gc = optimization_convergence.get("greedy_initial_cost", 0)
    fc = optimization_convergence.get("final_optimized_cost", 0)
    imp = optimization_convergence.get("improvement_pct", 0)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Classical Solver v6.0 - Expert Dashboard</title>
<style>
body{{background:#0a192f;color:#fff;font-family:sans-serif;padding:20px}}
h1{{color:#64ffda;border-bottom:2px solid #2c74b3;padding-bottom:10px}}
h2{{color:#64ffda;margin-top:30px}}
table{{border-collapse:collapse;width:100%;margin:10px 0}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #2c74b3}}
th{{background:rgba(44,116,179,0.3);color:#64ffda;font-size:12px;text-transform:uppercase}}
tr:hover{{background:rgba(100,255,218,0.05)}}
.kpi{{display:inline-block;background:rgba(44,116,179,0.2);padding:15px 25px;margin:5px;border-radius:8px;border-left:4px solid #64ffda}}
.kpi .val{{font-size:24px;font-weight:bold;color:#64ffda}}
.kpi .lbl{{font-size:11px;color:#aaa;text-transform:uppercase}}
</style></head><body>
<h1>Classical Solver v6.0 - Expert Dashboard</h1>
<p style="color:#aaa">Berth Allocation + Quay Crane Assignment | Barcelona Peak Dataset</p>

<div style="margin:20px 0">
<div class="kpi"><div class="val">${tc:,.0f}</div><div class="lbl">Total Cost</div></div>
<div class="kpi"><div class="val">${cost_breakdown.get('cost_per_vessel', 0):,.0f}</div><div class="lbl">Cost/Vessel</div></div>
<div class="kpi"><div class="val">{imp:.1f}%</div><div class="lbl">Improvement</div></div>
<div class="kpi"><div class="val">{computation_metrics.get('wall_time_s', 0):.1f}s</div><div class="lbl">Runtime</div></div>
<div class="kpi"><div class="val">{crane_allocation.get('avg_cranes_per_vessel', 0):.1f}</div><div class="lbl">Avg Cranes</div></div>
<div class="kpi"><div class="val">{schedule_metrics.get('feasible_assignments', 0)}/{schedule_metrics.get('feasible_assignments', 0) + schedule_metrics.get('infeasible_assignments', 0)}</div><div class="lbl">Feasible</div></div>
</div>

<h2>Optimization Convergence</h2>
<table>
<tr><th>Phase</th><th>Cost</th><th>Savings</th></tr>
<tr><td>Greedy Initial</td><td>${gc:,.0f}</td><td>Baseline</td></tr>
<tr><td>2-Opt Search</td><td>${optimization_convergence.get('two_opt_cost', 0):,.0f}</td><td>${max(0, gc - optimization_convergence.get('two_opt_cost', 0)):,.0f}</td></tr>
<tr><td>Crane Reopt</td><td>${optimization_convergence.get('crane_reopt_cost', fc):,.0f}</td><td>-</td></tr>
<tr><td>Final</td><td>${fc:,.0f}</td><td>${max(0, gc - fc):,.0f} ({imp:.1f}%)</td></tr>
</table>

<h2>Vessel Assignments</h2>
<table>
<tr><th>Vessel</th><th>Berth</th><th>Cranes</th><th>Handling (h)</th><th>Wait (h)</th><th>Delay (h)</th><th>Cost</th></tr>
{vessel_rows}
</table>

<h2>Berth Utilization</h2>
<table>
<tr><th>Berth</th><th>Vessels Served</th><th>Utilization</th></tr>
{berth_rows}
</table>

<h2>Priority Analysis</h2>
<table>
<tr><th>Priority</th><th>Count</th><th>Avg Cost</th><th>Avg Wait (h)</th><th>Avg Delay (h)</th></tr>
{priority_rows}
</table>

<div style="margin-top:30px;padding:15px;background:rgba(44,116,179,0.1);border-radius:8px">
<strong style="color:#64ffda">Performance:</strong>
Algorithm=Greedy-2Opt-OrOpt-3Opt-CraneRebalance |
Makespan={schedule_metrics.get('makespan', 0):.1f}h |
Total Wait={schedule_metrics.get('total_waiting_time', 0):.1f}h |
Iterations={computation_metrics.get('iterations', 0)} |
Or-Opt={computation_metrics.get('or_opt_improvements', 0)} |
3-Opt={computation_metrics.get('three_opt_improvements', 0)} |
Crane-Rebal={computation_metrics.get('crane_rebalance_improvements', 0)}
</div>

<div style="margin-top:20px;color:#666;font-size:11px;text-align:center">
Classical Solver v6.0 | Generated dynamically
</div>
</body></html>"""

    with open("additional_output/01_expert_dashboard.html", "w") as f:
        f.write(html)
