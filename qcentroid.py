"""
Classical BAP+QCA Greedy-2Opt Solver v3.1
Berth Allocation + Quay Crane Assignment using greedy construction + 2-opt local search.

v3.1 changes:
- Added additional_output visualizations with interactive Plotly.js charts
- 5 HTML dashboards: Gantt timeline, cost analysis, optimization convergence, berth utilization, metrics summary

v3.0 changes:
- Fixed _try_swap: now explores all crane levels during 2-opt (was hardcoded to min_cranes)
- Added crane re-optimization pass after 2-opt berth swaps
- Improved cost breakdown with waiting/delay separation
- Enhanced rich visual output for benchmarking dashboards
"""
import logging
import time
import itertools
import os

logger = logging.getLogger("qcentroid-user-log")


def run(input_data: dict, solver_params: dict, extra_arguments: dict) -> dict:
    start_time = time.time()
    logger.info("=== Classical BAP+QCA Greedy-2Opt Solver v3.1 ===")

    # ── 1. Parse inputs ──────────────────────────────────────────────
    vessels = input_data.get("vessels", [])
    berths = input_data.get("berths", [])
    cranes_cfg = input_data.get("cranes", {})
    cost_weights = input_data.get("cost_weights", {})

    total_cranes = cranes_cfg.get("total_available", 10)
    min_cranes = cranes_cfg.get("min_per_vessel", 1)
    max_cranes = cranes_cfg.get("max_per_vessel", 4)

    w_wait = cost_weights.get("waiting_cost_per_hour", 500)
    w_handle = cost_weights.get("handling_cost_per_crane_hour", 150)
    w_delay = cost_weights.get("delay_penalty_per_hour", 1000)
    w_priority = cost_weights.get("priority_multiplier", 1.5)

    n_vessels = len(vessels)
    n_berths = len(berths)
    logger.info(f"Problem: {n_vessels} vessels, {n_berths} berths, {total_cranes} cranes")

    # Sort vessels by priority (ascending = higher priority first), then arrival
    sorted_vessels = sorted(vessels, key=lambda v: (v.get("priority", 5), v.get("arrival_time", "")))

    # ── 2. Greedy construction ───────────────────────────────────────
    assignments = []
    berth_end_times = {}
    cost_evolution = []

    for v in sorted_vessels:
        v_id = v["id"]
        v_len = v.get("length_m", 200)
        v_draft = v.get("draft_m", 12.0)
        v_teu = v.get("handling_volume_teu", 1000)
        v_arrival = v.get("arrival_time", "2025-01-01T00:00:00Z")
        v_deadline = v.get("max_departure_time", "2025-01-02T00:00:00Z")
        v_priority = v.get("priority", 3)
        v_name = v.get("name", f"Vessel-{v_id}")

        best_berth = None
        best_cost = float("inf")
        best_start = None
        best_cranes_assigned = min_cranes
        best_handling_hours = 0
        best_wait_hours = 0
        best_delay_hours = 0

        for b in berths:
            b_id = b["id"]
            b_len = b.get("length_m", 300)
            b_depth = b.get("depth_m", 15.0)
            b_prod = b.get("productivity_teu_per_crane_hour", 25)

            if v_len > b_len or v_draft > b_depth:
                continue

            berth_free = berth_end_times.get(b_id, v_arrival)
            actual_start = max(v_arrival, berth_free)

            for nc in range(min_cranes, min(max_cranes, total_cranes) + 1):
                handling_hours = v_teu / (b_prod * nc) if b_prod * nc > 0 else 999
                end_time_h = _iso_to_hours(actual_start) + handling_hours
                deadline_h = _iso_to_hours(v_deadline)

                wait_hours = max(0, _iso_to_hours(actual_start) - _iso_to_hours(v_arrival))
                delay_hours = max(0, end_time_h - deadline_h)
                crane_cost = handling_hours * nc * w_handle
                wait_cost = wait_hours * w_wait * (w_priority if v_priority <= 2 else 1.0)
                delay_cost = delay_hours * w_delay * (w_priority if v_priority <= 2 else 1.0)

                total_cost = crane_cost + wait_cost + delay_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_berth = b_id
                    best_start = actual_start
                    best_cranes_assigned = nc
                    best_handling_hours = handling_hours
                    best_wait_hours = wait_hours
                    best_delay_hours = delay_hours

        if best_berth is not None:
            end_time_str = _hours_to_iso(
                _iso_to_hours(best_start) + best_handling_hours,
                best_start
            )
            berth_end_times[best_berth] = end_time_str

            assignments.append({
                "vessel_id": v_id,
                "vessel_name": v_name,
                "berth_id": best_berth,
                "start_time": best_start,
                "end_time": end_time_str,
                "cranes_assigned": best_cranes_assigned,
                "handling_hours": round(best_handling_hours, 2),
                "waiting_hours": round(best_wait_hours, 2),
                "delay_hours": round(best_delay_hours, 2),
                "cost": round(best_cost, 2),
                "priority": v_priority,
                "teu_volume": v_teu
            })
        else:
            assignments.append({
                "vessel_id": v_id,
                "vessel_name": v_name,
                "berth_id": None,
                "start_time": None,
                "end_time": None,
                "cranes_assigned": 0,
                "handling_hours": 0,
                "cost": 0,
                "status": "infeasible"
            })
            logger.warning(f"Vessel {v_id}: no feasible berth found!")

    greedy_cost = sum(a["cost"] for a in assignments)
    cost_evolution.append({"iteration": 0, "phase": "greedy", "objective_value": round(greedy_cost, 2)})
    logger.info(f"Greedy phase complete: cost={greedy_cost:.2f}")

    # ── 3. 2-opt local search improvement (v3: with crane optimization) ──
    max_iterations = solver_params.get("max_2opt_iterations", 100)
    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(len(assignments)):
            for j in range(i + 1, len(assignments)):
                a1, a2 = assignments[i], assignments[j]
                if a1.get("berth_id") is None or a2.get("berth_id") is None:
                    continue
                old_cost = a1["cost"] + a2["cost"]
                new_a1, new_a2 = _try_swap(a1, a2, berths, vessels, cost_weights, cranes_cfg)
                if new_a1 is not None:
                    new_cost = new_a1["cost"] + new_a2["cost"]
                    if new_cost < old_cost * 0.99:
                        assignments[i] = new_a1
                        assignments[j] = new_a2
                        improved = True

        current_cost = sum(a["cost"] for a in assignments)
        cost_evolution.append({
            "iteration": iteration,
            "phase": "2-opt",
            "objective_value": round(current_cost, 2)
        })

    two_opt_cost = sum(a["cost"] for a in assignments)
    logger.info(f"2-opt completed after {iteration} iterations: cost={two_opt_cost:.2f}")

    # ── 3b. Crane re-optimization pass (v3 new) ─────────────────────
    crane_opt_improved = 0
    for idx, a in enumerate(assignments):
        if a.get("berth_id") is None:
            continue
        v_data = next((v for v in vessels if v["id"] == a["vessel_id"]), None)
        b_data = next((b for b in berths if b["id"] == a["berth_id"]), None)
        if not v_data or not b_data:
            continue

        v_teu = v_data.get("handling_volume_teu", 1000)
        v_priority = v_data.get("priority", 3)
        pm = w_priority if v_priority <= 2 else 1.0
        b_prod = b_data.get("productivity_teu_per_crane_hour", 25)

        best_nc = a["cranes_assigned"]
        best_cost = a["cost"]

        for nc in range(min_cranes, min(max_cranes, total_cranes) + 1):
            handling_h = v_teu / (b_prod * nc) if b_prod * nc > 0 else 999
            end_h = _iso_to_hours(a["start_time"]) + handling_h
            deadline_h = _iso_to_hours(v_data.get("max_departure_time", "2025-12-31T23:59:00Z"))
            arr_h = _iso_to_hours(v_data.get("arrival_time", a["start_time"]))
            wait_h = max(0, _iso_to_hours(a["start_time"]) - arr_h)
            delay_h = max(0, end_h - deadline_h)

            cost = (handling_h * nc * w_handle +
                    wait_h * w_wait * pm +
                    delay_h * w_delay * pm)

            if cost < best_cost:
                best_cost = cost
                best_nc = nc

        if best_nc != a["cranes_assigned"]:
            handling_h = v_teu / (b_prod * best_nc) if b_prod * best_nc > 0 else 999
            end_h = _iso_to_hours(a["start_time"]) + handling_h
            deadline_h = _iso_to_hours(v_data.get("max_departure_time", "2025-12-31T23:59:00Z"))
            delay_h = max(0, end_h - deadline_h)
            wait_h = a.get("waiting_hours", 0)

            assignments[idx] = dict(a)
            assignments[idx]["cranes_assigned"] = best_nc
            assignments[idx]["handling_hours"] = round(handling_h, 2)
            assignments[idx]["delay_hours"] = round(delay_h, 2)
            assignments[idx]["cost"] = round(best_cost, 2)
            assignments[idx]["end_time"] = _hours_to_iso(end_h, a["start_time"])
            crane_opt_improved += 1

    final_cost_after_crane_opt = sum(a["cost"] for a in assignments)
    if crane_opt_improved > 0:
        cost_evolution.append({
            "iteration": iteration + 1,
            "phase": "crane_reopt",
            "objective_value": round(final_cost_after_crane_opt, 2)
        })
        logger.info(f"Crane re-optimization improved {crane_opt_improved} assignments: "
                     f"cost={final_cost_after_crane_opt:.2f}")

    # ── 4. Compute metrics ───────────────────────────────────────────
    total_cost = sum(a["cost"] for a in assignments)
    feasible_count = sum(1 for a in assignments if a.get("berth_id") is not None)
    total_wait = sum(a.get("waiting_hours", 0) for a in assignments)
    total_handling = sum(a.get("handling_hours", 0) for a in assignments)
    total_teu = sum(a.get("teu_volume", 0) for a in assignments)

    makespan = 0
    if assignments:
        end_hours = [_iso_to_hours(a["end_time"]) for a in assignments if a.get("end_time")]
        start_hours = [_iso_to_hours(a["start_time"]) for a in assignments if a.get("start_time")]
        if end_hours and start_hours:
            makespan = max(end_hours) - min(start_hours)

    status = "optimal" if feasible_count == n_vessels else (
        "feasible" if feasible_count > 0 else "infeasible"
    )

    # ── 5. Build rich visual output ──────────────────────────────────
    # Berth utilization breakdown
    berth_utilization = []
    for b in berths:
        b_id = b["id"]
        b_assignments = [a for a in assignments if a.get("berth_id") == b_id]
        occupied_hours = sum(a.get("handling_hours", 0) for a in b_assignments)
        util_pct = round(occupied_hours / max(makespan, 1) * 100, 1)
        berth_utilization.append({
            "berth_id": b_id,
            "vessels_served": len(b_assignments),
            "occupied_hours": round(occupied_hours, 2),
            "utilization_pct": util_pct,
            "total_teu_handled": sum(a.get("teu_volume", 0) for a in b_assignments)
        })

    # Cost breakdown by category
    total_crane_cost = sum(
        a.get("handling_hours", 0) * a.get("cranes_assigned", 1) * w_handle
        for a in assignments if a.get("berth_id")
    )
    total_wait_cost = sum(
        a.get("waiting_hours", 0) * w_wait * (w_priority if a.get("priority", 3) <= 2 else 1.0)
        for a in assignments if a.get("berth_id")
    )
    total_delay_cost = sum(
        a.get("delay_hours", 0) * w_delay * (w_priority if a.get("priority", 3) <= 2 else 1.0)
        for a in assignments if a.get("berth_id")
    )

    # Crane allocation distribution
    crane_distribution = {}
    for a in assignments:
        nc = a.get("cranes_assigned", 0)
        crane_distribution[str(nc)] = crane_distribution.get(str(nc), 0) + 1

    # Gantt chart data
    gantt_data = []
    for a in assignments:
        if a.get("berth_id") is not None:
            gantt_data.append({
                "vessel": a.get("vessel_name", a["vessel_id"]),
                "berth": a["berth_id"],
                "start": a["start_time"],
                "end": a["end_time"],
                "cranes": a["cranes_assigned"],
                "priority": a.get("priority", 3)
            })

    # Priority analysis
    priority_analysis = {}
    for a in assignments:
        p = a.get("priority", 3)
        key = f"P{p}"
        if key not in priority_analysis:
            priority_analysis[key] = {"count": 0, "total_cost": 0, "total_wait_h": 0, "total_delay_h": 0}
        priority_analysis[key]["count"] += 1
        priority_analysis[key]["total_cost"] += a.get("cost", 0)
        priority_analysis[key]["total_wait_h"] += a.get("waiting_hours", 0)
        priority_analysis[key]["total_delay_h"] += a.get("delay_hours", 0)
    for key in priority_analysis:
        pa = priority_analysis[key]
        pa["avg_cost"] = round(pa["total_cost"] / max(pa["count"], 1), 2)
        pa["total_cost"] = round(pa["total_cost"], 2)
        pa["avg_wait_h"] = round(pa["total_wait_h"] / max(pa["count"], 1), 2)
        pa["avg_delay_h"] = round(pa["total_delay_h"] / max(pa["count"], 1), 2)

    improvement_pct = round((1 - total_cost / max(greedy_cost, 1)) * 100, 2) if greedy_cost > 0 else 0

    elapsed = round(time.time() - start_time, 3)
    logger.info(f"Total cost: {total_cost:.2f}, Status: {status}, Time: {elapsed}s")
    logger.info(f"Improvement: {improvement_pct}% over greedy (greedy={greedy_cost:.2f}, "
                f"2-opt={two_opt_cost:.2f}, crane-reopt={final_cost_after_crane_opt:.2f})")

    # ── 6. Generate additional output visualizations ─────────────────
    try:
        _generate_additional_output(
            assignments=assignments,
            berths=berths,
            vessels=vessels,
            cost_breakdown={
                "total_cost": round(total_cost, 2),
                "crane_handling_cost": round(total_crane_cost, 2),
                "waiting_cost": round(total_wait_cost, 2),
                "delay_penalty_cost": round(total_delay_cost, 2),
                "cost_per_vessel": round(total_cost / max(n_vessels, 1), 2),
                "cost_per_teu": round(total_cost / max(total_teu, 1), 4)
            },
            optimization_convergence={
                "greedy_initial_cost": round(greedy_cost, 2),
                "two_opt_cost": round(two_opt_cost, 2),
                "crane_reopt_cost": round(final_cost_after_crane_opt, 2),
                "final_optimized_cost": round(total_cost, 2),
                "improvement_pct": improvement_pct,
                "iterations_used": iteration,
                "crane_adjustments": crane_opt_improved,
                "cost_evolution": cost_evolution
            },
            berth_utilization=berth_utilization,
            priority_analysis=priority_analysis,
            gantt_data=gantt_data,
            schedule_metrics={
                "total_waiting_time": round(total_wait, 2),
                "avg_waiting_time": round(total_wait / max(n_vessels, 1), 2),
                "makespan": round(makespan, 2),
                "utilization": round(total_handling / max(makespan * n_berths, 1), 4),
                "total_teu_processed": total_teu,
                "feasible_assignments": feasible_count,
                "infeasible_assignments": n_vessels - feasible_count
            },
            computation_metrics={
                "wall_time_s": elapsed,
                "algorithm": "Greedy_2Opt_CraneReopt_BAP_QCA",
                "iterations": iteration,
                "crane_reopt_improvements": crane_opt_improved,
                "search_space_explored": n_vessels * n_berths * (max_cranes - min_cranes + 1),
                "solver_version": "3.1"
            },
            crane_allocation={
                "distribution": crane_distribution,
                "avg_cranes_per_vessel": round(
                    sum(a.get("cranes_assigned", 0) for a in assignments) / max(feasible_count, 1), 2
                ),
                "total_crane_hours": round(
                    sum(a.get("handling_hours", 0) * a.get("cranes_assigned", 0) for a in assignments), 2
                )
            }
        )
        logger.info("Additional output visualizations generated successfully")
    except Exception as e:
        logger.warning(f"Failed to generate additional output: {e}")

    return {
        # ── Core assignment result ──
        "assignments": assignments,
        "objective_value": round(total_cost, 2),
        "solution_status": status,

        # ── Input size metrics ──
        "num_vessels": n_vessels,
        "num_berths": n_berths,
        "total_cranes": total_cranes,

        # ── Schedule metrics ──
        "schedule_metrics": {
            "total_waiting_time": round(total_wait, 2),
            "avg_waiting_time": round(total_wait / max(n_vessels, 1), 2),
            "makespan": round(makespan, 2),
            "utilization": round(total_handling / max(makespan * n_berths, 1), 4),
            "total_teu_processed": total_teu,
            "feasible_assignments": feasible_count,
            "infeasible_assignments": n_vessels - feasible_count
        },

        # ── Visual: Cost breakdown (pie/bar chart ready) ──
        "cost_breakdown": {
            "total_cost": round(total_cost, 2),
            "crane_handling_cost": round(total_crane_cost, 2),
            "waiting_cost": round(total_wait_cost, 2),
            "delay_penalty_cost": round(total_delay_cost, 2),
            "cost_per_vessel": round(total_cost / max(n_vessels, 1), 2),
            "cost_per_teu": round(total_cost / max(total_teu, 1), 4)
        },

        # ── Visual: Optimization convergence (line chart ready) ──
        "optimization_convergence": {
            "greedy_initial_cost": round(greedy_cost, 2),
            "two_opt_cost": round(two_opt_cost, 2),
            "crane_reopt_cost": round(final_cost_after_crane_opt, 2),
            "final_optimized_cost": round(total_cost, 2),
            "improvement_pct": improvement_pct,
            "iterations_used": iteration,
            "crane_adjustments": crane_opt_improved,
            "cost_evolution": cost_evolution
        },

        # ── Visual: Berth utilization (bar chart ready) ──
        "berth_utilization": berth_utilization,

        # ── Visual: Crane allocation distribution (histogram ready) ──
        "crane_allocation": {
            "distribution": crane_distribution,
            "avg_cranes_per_vessel": round(
                sum(a.get("cranes_assigned", 0) for a in assignments) / max(feasible_count, 1), 2
            ),
            "total_crane_hours": round(
                sum(a.get("handling_hours", 0) * a.get("cranes_assigned", 0) for a in assignments), 2
            )
        },

        # ── Visual: Gantt chart data (timeline ready) ──
        "gantt_schedule": gantt_data,

        # ── Visual: Priority analysis (grouped bar chart ready) ──
        "priority_analysis": priority_analysis,

        # ── Computation metrics ──
        "computation_metrics": {
            "wall_time_s": elapsed,
            "algorithm": "Greedy_2Opt_CraneReopt_BAP_QCA",
            "iterations": iteration,
            "crane_reopt_improvements": crane_opt_improved,
            "search_space_explored": n_vessels * n_berths * (max_cranes - min_cranes + 1),
            "solver_version": "3.1"
        },

        # ── Platform benchmark contract ──
        "benchmark": {
            "execution_cost": {"value": 1.0, "unit": "credits"},
            "time_elapsed": f"{elapsed}s",
            "energy_consumption": 0.0
        }
    }


# ── Helper functions ─────────────────────────────────────────────────

def _iso_to_hours(iso_str):
    """Convert ISO timestamp to hours since epoch (simplified)."""
    if not iso_str or not isinstance(iso_str, str):
        return 0
    try:
        parts = iso_str.replace("Z", "").split("T")
        date_parts = parts[0].split("-")
        time_parts = parts[1].split(":") if len(parts) > 1 else ["0", "0", "0"]
        day_of_year = int(date_parts[1]) * 30 + int(date_parts[2])
        return day_of_year * 24 + int(time_parts[0]) + int(time_parts[1]) / 60
    except (IndexError, ValueError):
        return 0


def _hours_to_iso(hours, reference_iso):
    """Convert hours back to ISO string (approximate, same date base)."""
    if not reference_iso:
        return "2025-01-01T00:00:00Z"
    try:
        parts = reference_iso.replace("Z", "").split("T")
        date_parts = parts[0].split("-")
        total_h = int(hours) % 24
        total_m = int((hours - int(hours)) * 60)
        day_offset = int(hours) // 24
        month = day_offset // 30
        day = day_offset % 30
        if month < 1:
            month = 1
        if day < 1:
            day = 1
        return f"{date_parts[0]}-{month:02d}-{day:02d}T{total_h:02d}:{total_m:02d}:00Z"
    except Exception:
        return reference_iso


def _try_swap(a1, a2, berths, vessels, cost_weights, cranes_cfg):
    """Try swapping berth assignments of two vessels.
    v3 FIX: explores all crane levels instead of hardcoding min_cranes.
    """
    b1_id, b2_id = a1["berth_id"], a2["berth_id"]
    if b1_id == b2_id:
        return None, None

    v1 = next((v for v in vessels if v["id"] == a1["vessel_id"]), None)
    v2 = next((v for v in vessels if v["id"] == a2["vessel_id"]), None)
    b1 = next((b for b in berths if b["id"] == b1_id), None)
    b2 = next((b for b in berths if b["id"] == b2_id), None)

    if not all([v1, v2, b1, b2]):
        return None, None

    if v1.get("length_m", 200) > b2.get("length_m", 300):
        return None, None
    if v2.get("length_m", 200) > b1.get("length_m", 300):
        return None, None
    if v1.get("draft_m", 12) > b2.get("depth_m", 15):
        return None, None
    if v2.get("draft_m", 12) > b1.get("depth_m", 15):
        return None, None

    w_handle = cost_weights.get("handling_cost_per_crane_hour", 150)
    w_wait = cost_weights.get("waiting_cost_per_hour", 500)
    w_delay = cost_weights.get("delay_penalty_per_hour", 1000)
    w_priority = cost_weights.get("priority_multiplier", 1.5)
    min_cranes = cranes_cfg.get("min_per_vessel", 1)
    max_cranes = cranes_cfg.get("max_per_vessel", 4)
    total_cranes = cranes_cfg.get("total_available", 10)

    def calc_best_cost(v, b, start_time):
        """v3 FIX: Try all crane levels, return best cost + crane count."""
        prod = b.get("productivity_teu_per_crane_hour", 25)
        teu = v.get("handling_volume_teu", 1000)
        p = v.get("priority", 3)
        pm = w_priority if p <= 2 else 1.0
        arr_h = _iso_to_hours(v.get("arrival_time", start_time))
        deadline_h = _iso_to_hours(v.get("max_departure_time", "2025-12-31T23:59:00Z"))
        start_h = _iso_to_hours(start_time)
        wait_h = max(0, start_h - arr_h)

        best_nc = min_cranes
        best_cost = float("inf")
        best_handling = 0

        for nc in range(min_cranes, min(max_cranes, total_cranes) + 1):
            handling_h = teu / (prod * nc) if prod * nc > 0 else 999
            end_h = start_h + handling_h
            delay_h = max(0, end_h - deadline_h)
            cost = (handling_h * nc * w_handle +
                    wait_h * w_wait * pm +
                    delay_h * w_delay * pm)
            if cost < best_cost:
                best_cost = cost
                best_nc = nc
                best_handling = handling_h

        return best_cost, best_handling, best_nc, wait_h

    c1, h1, nc1, w1 = calc_best_cost(v1, b2, a1["start_time"])
    c2, h2, nc2, w2 = calc_best_cost(v2, b1, a2["start_time"])

    # Compute delay hours for output
    end1_h = _iso_to_hours(a1["start_time"]) + h1
    deadline1_h = _iso_to_hours(v1.get("max_departure_time", "2025-12-31T23:59:00Z"))
    delay1_h = max(0, end1_h - deadline1_h)

    end2_h = _iso_to_hours(a2["start_time"]) + h2
    deadline2_h = _iso_to_hours(v2.get("max_departure_time", "2025-12-31T23:59:00Z"))
    delay2_h = max(0, end2_h - deadline2_h)

    new_a1 = dict(a1)
    new_a1["berth_id"] = b2_id
    new_a1["cost"] = round(c1, 2)
    new_a1["handling_hours"] = round(h1, 2)
    new_a1["cranes_assigned"] = nc1
    new_a1["waiting_hours"] = round(w1, 2)
    new_a1["delay_hours"] = round(delay1_h, 2)
    new_a1["end_time"] = _hours_to_iso(end1_h, a1["start_time"])

    new_a2 = dict(a2)
    new_a2["berth_id"] = b1_id
    new_a2["cost"] = round(c2, 2)
    new_a2["handling_hours"] = round(h2, 2)
    new_a2["cranes_assigned"] = nc2
    new_a2["waiting_hours"] = round(w2, 2)
    new_a2["delay_hours"] = round(delay2_h, 2)
    new_a2["end_time"] = _hours_to_iso(end2_h, a2["start_time"])

    return new_a1, new_a2


def _generate_additional_output(assignments, berths, vessels, cost_breakdown,
                                 optimization_convergence, berth_utilization,
                                 priority_analysis, gantt_data, schedule_metrics,
                                 computation_metrics, crane_allocation):
    """Generate 5 interactive HTML visualization files using Plotly.js CDN."""

    # Create output directory
    os.makedirs("additional_output", exist_ok=True)

    # Color scheme: dark teal/blue theme
    bg_color = "#0a192f"
    accent_color = "#2c74b3"
    highlight_color = "#64ffda"
    text_color = "#ffffff"

    # ── 1. Berth Gantt Timeline ──────────────────────────────────────
    gantt_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Berth Gantt Timeline</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ background-color: {bg_color}; color: {text_color}; margin: 0; padding: 20px; font-family: 'Segoe UI', Arial; }}
        h1 {{ color: {highlight_color}; }}
        #gantt-chart {{ width: 100%; height: 600px; }}
    </style>
</head>
<body>
    <h1>Berth Gantt Timeline</h1>
    <div id="gantt-chart"></div>
    <script>
        var ganttData = [
"""

    # Build gantt trace
    vessel_names = []
    start_times = []
    durations = []
    colors = []
    hover_texts = []

    for g in gantt_data:
        vessel_names.append(g.get("vessel", "Unknown"))
        start_times.append(g.get("start", "2025-01-01T00:00:00Z"))
        start_h = _iso_to_hours(g.get("start", "2025-01-01T00:00:00Z"))
        end_h = _iso_to_hours(g.get("end", "2025-01-01T00:00:00Z"))
        duration = max(0.1, end_h - start_h)
        durations.append(duration)

        # Color based on priority
        priority = g.get("priority", 3)
        if priority <= 1:
            colors.append("#ff6b6b")
        elif priority <= 2:
            colors.append(highlight_color)
        else:
            colors.append(accent_color)

        cranes = g.get("cranes", 0)
        hover_texts.append(f"Vessel: {g.get('vessel', 'Unknown')}<br>Berth: {g.get('berth', 'N/A')}<br>Cranes: {cranes}<br>Priority: P{priority}")

    gantt_html += f"""
            {{
                x: {durations},
                y: {vessel_names},
                mode: 'bars',
                type: 'bar',
                orientation: 'h',
                marker: {{ color: {colors} }},
                text: {hover_texts},
                hovertemplate: '%{{text}}<extra></extra>',
                name: 'Vessels'
            }}
        ];

        var layout = {{
            title: 'Vessel-to-Berth Assignments Over Time',
            barmode: 'group',
            xaxis: {{ title: 'Duration (hours)', color: '{text_color}' }},
            yaxis: {{ title: 'Vessel / Berth', color: '{text_color}' }},
            plot_bgcolor: '{bg_color}',
            paper_bgcolor: '{bg_color}',
            font: {{ color: '{text_color}' }},
            hovermode: 'closest'
        }};

        Plotly.newPlot('gantt-chart', ganttData, layout, {{ responsive: true }});
    </script>
</body>
</html>
"""

    with open("additional_output/01_berth_gantt_timeline.html", "w") as f:
        f.write(gantt_html)

    # ── 2. Cost Analysis Dashboard ───────────────────────────────────
    cost_labels = ["Crane Handling", "Waiting", "Delay Penalty"]
    cost_values = [
        cost_breakdown.get("crane_handling_cost", 0),
        cost_breakdown.get("waiting_cost", 0),
        cost_breakdown.get("delay_penalty_cost", 0)
    ]

    cost_per_vessel_list = [a.get("cost", 0) for a in assignments if a.get("berth_id") is not None]
    vessel_names_cost = [a.get("vessel_name", a["vessel_id"]) for a in assignments if a.get("berth_id") is not None]

    cost_analysis_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Cost Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ background-color: {bg_color}; color: {text_color}; margin: 0; padding: 20px; font-family: 'Segoe UI', Arial; }}
        h1 {{ color: {highlight_color}; }}
        .kpi-section {{ display: flex; gap: 20px; margin: 20px 0; }}
        .kpi-card {{ background: {accent_color}; padding: 15px; border-radius: 8px; flex: 1; }}
        .kpi-value {{ font-size: 24px; font-weight: bold; color: {highlight_color}; }}
        .kpi-label {{ font-size: 12px; color: {bg_color}; }}
        #pie-chart, #bar-chart {{ width: 48%; height: 400px; display: inline-block; }}
        .chart-container {{ display: flex; gap: 20px; }}
    </style>
</head>
<body>
    <h1>Cost Analysis Dashboard</h1>
    <div class="kpi-section">
        <div class="kpi-card">
            <div class="kpi-label">TOTAL COST</div>
            <div class="kpi-value">${cost_breakdown.get('total_cost', 0):.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">COST PER VESSEL</div>
            <div class="kpi-value">${cost_breakdown.get('cost_per_vessel', 0):.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">COST PER TEU</div>
            <div class="kpi-value">${cost_breakdown.get('cost_per_teu', 0):.4f}</div>
        </div>
    </div>
    <div class="chart-container">
        <div id="pie-chart"></div>
        <div id="bar-chart"></div>
    </div>
    <script>
        var pieData = [{{
            values: {cost_values},
            labels: {cost_labels},
            type: 'pie',
            marker: {{ colors: ['#64ffda', '#2c74b3', '#ff6b6b'] }}
        }}];

        var pieLayout = {{
            title: 'Cost Breakdown by Category',
            plot_bgcolor: '{bg_color}',
            paper_bgcolor: '{bg_color}',
            font: {{ color: '{text_color}' }}
        }};

        Plotly.newPlot('pie-chart', pieData, pieLayout, {{ responsive: true }});

        var barData = [{{
            x: {vessel_names_cost},
            y: {cost_per_vessel_list},
            type: 'bar',
            marker: {{ color: '{accent_color}' }}
        }}];

        var barLayout = {{
            title: 'Cost Per Vessel',
            xaxis: {{ title: 'Vessel', color: '{text_color}' }},
            yaxis: {{ title: 'Cost ($)', color: '{text_color}' }},
            plot_bgcolor: '{bg_color}',
            paper_bgcolor: '{bg_color}',
            font: {{ color: '{text_color}' }},
            xaxis: {{ tickangle: -45 }}
        }};

        Plotly.newPlot('bar-chart', barData, barLayout, {{ responsive: true }});
    </script>
</body>
</html>
"""

    with open("additional_output/02_cost_analysis_dashboard.html", "w") as f:
        f.write(cost_analysis_html)

    # ── 3. Optimization Convergence ──────────────────────────────────
    iterations = [e.get("iteration", 0) for e in optimization_convergence.get("cost_evolution", [])]
    objectives = [e.get("objective_value", 0) for e in optimization_convergence.get("cost_evolution", [])]
    phases = [e.get("phase", "unknown") for e in optimization_convergence.get("cost_evolution", [])]

    convergence_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Optimization Convergence</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ background-color: {bg_color}; color: {text_color}; margin: 0; padding: 20px; font-family: 'Segoe UI', Arial; }}
        h1 {{ color: {highlight_color}; }}
        #convergence-chart {{ width: 100%; height: 600px; }}
        .metric {{ margin: 15px 0; padding: 10px; background: {accent_color}; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Optimization Convergence</h1>
    <div class="metric">Improvement: <strong>{optimization_convergence.get('improvement_pct', 0):.2f}%</strong> over greedy</div>
    <div class="metric">Greedy Cost: ${optimization_convergence.get('greedy_initial_cost', 0):.2f}</div>
    <div class="metric">Final Cost: ${optimization_convergence.get('final_optimized_cost', 0):.2f}</div>
    <div id="convergence-chart"></div>
    <script>
        var convData = [{{
            x: {iterations},
            y: {objectives},
            type: 'scatter',
            mode: 'lines+markers',
            line: {{ color: '{highlight_color}', width: 3 }},
            marker: {{ size: 8, color: '{accent_color}' }}
        }}];

        var convLayout = {{
            title: 'Cost Evolution Across Optimization Phases',
            xaxis: {{ title: 'Iteration / Phase', color: '{text_color}' }},
            yaxis: {{ title: 'Objective Value ($)', color: '{text_color}' }},
            plot_bgcolor: '{bg_color}',
            paper_bgcolor: '{bg_color}',
            font: {{ color: '{text_color}' }},
            hovermode: 'closest'
        }};

        Plotly.newPlot('convergence-chart', convData, convLayout, {{ responsive: true }});
    </script>
</body>
</html>
"""

    with open("additional_output/03_optimization_convergence.html", "w") as f:
        f.write(convergence_html)

    # ── 4. Berth Utilization Heatmap ─────────────────────────────────
    berth_ids = [b.get("berth_id", "B?") for b in berth_utilization]
    util_pcts = [b.get("utilization_pct", 0) for b in berth_utilization]
    vessels_served = [b.get("vessels_served", 0) for b in berth_utilization]
    teu_handled = [b.get("total_teu_handled", 0) for b in berth_utilization]

    utilization_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Berth Utilization</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ background-color: {bg_color}; color: {text_color}; margin: 0; padding: 20px; font-family: 'Segoe UI', Arial; }}
        h1 {{ color: {highlight_color}; }}
        #util-chart {{ width: 100%; height: 600px; }}
    </style>
</head>
<body>
    <h1>Berth Utilization Analysis</h1>
    <div id="util-chart"></div>
    <script>
        var utilData = [
            {{
                x: {berth_ids},
                y: {util_pcts},
                name: 'Utilization %',
                type: 'bar',
                marker: {{ color: '{highlight_color}' }},
                yaxis: 'y'
            }},
            {{
                x: {berth_ids},
                y: {vessels_served},
                name: 'Vessels Served',
                type: 'scatter',
                mode: 'lines+markers',
                line: {{ color: '{accent_color}', width: 3 }},
                marker: {{ size: 10 }},
                yaxis: 'y2'
            }}
        ];

        var utilLayout = {{
            title: 'Berth Utilization and Vessel Distribution',
            xaxis: {{ title: 'Berth', color: '{text_color}' }},
            yaxis: {{ title: 'Utilization %', color: '{text_color}' }},
            yaxis2: {{ title: 'Vessels Served', overlaying: 'y', side: 'right', color: '{text_color}' }},
            plot_bgcolor: '{bg_color}',
            paper_bgcolor: '{bg_color}',
            font: {{ color: '{text_color}' }},
            hovermode: 'x unified'
        }};

        Plotly.newPlot('util-chart', utilData, utilLayout, {{ responsive: true }});
    </script>
</body>
</html>
"""

    with open("additional_output/04_berth_utilization_heatmap.html", "w") as f:
        f.write(utilization_html)

    # ── 5. Classical Metrics Summary ──────────────────────────────────
    crane_dist_labels = list(crane_allocation.get("distribution", {}).keys())
    crane_dist_values = list(crane_allocation.get("distribution", {}).values())

    priority_keys = list(priority_analysis.keys())
    priority_costs = [priority_analysis[k].get("avg_cost", 0) for k in priority_keys]
    priority_counts = [priority_analysis[k].get("count", 0) for k in priority_keys]

    metrics_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Classical Metrics Summary</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ background-color: {bg_color}; color: {text_color}; margin: 0; padding: 20px; font-family: 'Segoe UI', Arial; }}
        h1 {{ color: {highlight_color}; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: {accent_color}; padding: 15px; border-radius: 8px; border-left: 4px solid {highlight_color}; }}
        .metric-value {{ font-size: 20px; font-weight: bold; color: {highlight_color}; }}
        .metric-label {{ font-size: 11px; text-transform: uppercase; color: {bg_color}; margin-top: 5px; }}
        .charts-row {{ display: flex; gap: 20px; margin: 20px 0; }}
        .chart-box {{ flex: 1; }}
        #crane-chart, #priority-chart {{ width: 100%; height: 400px; }}
        .table-section {{ margin: 30px 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid {accent_color}; }}
        th {{ background: {accent_color}; color: {bg_color}; }}
        tr:hover {{ background: rgba(44, 116, 179, 0.2); }}
    </style>
</head>
<body>
    <h1>Classical Metrics Summary</h1>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{schedule_metrics.get('total_waiting_time', 0):.2f}</div>
            <div class="metric-label">Total Waiting (hrs)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{schedule_metrics.get('makespan', 0):.2f}</div>
            <div class="metric-label">Makespan (hrs)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{schedule_metrics.get('utilization', 0):.2%}</div>
            <div class="metric-label">Utilization</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{crane_allocation.get('avg_cranes_per_vessel', 0):.2f}</div>
            <div class="metric-label">Avg Cranes/Vessel</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{computation_metrics.get('wall_time_s', 0):.3f}</div>
            <div class="metric-label">Runtime (s)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{computation_metrics.get('iterations', 0)}</div>
            <div class="metric-label">2-Opt Iterations</div>
        </div>
    </div>

    <div class="charts-row">
        <div class="chart-box">
            <div id="crane-chart"></div>
        </div>
        <div class="chart-box">
            <div id="priority-chart"></div>
        </div>
    </div>

    <div class="table-section">
        <h2 style="color: {highlight_color};">Priority Analysis</h2>
        <table>
            <tr>
                <th>Priority</th>
                <th>Count</th>
                <th>Avg Cost</th>
                <th>Avg Wait (hrs)</th>
                <th>Avg Delay (hrs)</th>
            </tr>
"""

    for key in priority_keys:
        pa = priority_analysis[key]
        metrics_html += f"""
            <tr>
                <td>{key}</td>
                <td>{pa.get('count', 0)}</td>
                <td>${pa.get('avg_cost', 0):.2f}</td>
                <td>{pa.get('avg_wait_h', 0):.2f}</td>
                <td>{pa.get('avg_delay_h', 0):.2f}</td>
            </tr>
"""

    metrics_html += f"""
        </table>
    </div>

    <script>
        var craneData = [{{
            x: {crane_dist_labels},
            y: {crane_dist_values},
            type: 'bar',
            marker: {{ color: '{highlight_color}' }}
        }}];

        var craneLayout = {{
            title: 'Crane Allocation Distribution',
            xaxis: {{ title: 'Cranes per Vessel', color: '{text_color}' }},
            yaxis: {{ title: 'Count', color: '{text_color}' }},
            plot_bgcolor: '{bg_color}',
            paper_bgcolor: '{bg_color}',
            font: {{ color: '{text_color}' }}
        }};

        Plotly.newPlot('crane-chart', craneData, craneLayout, {{ responsive: true }});

        var priorityData = [
            {{
                x: {priority_keys},
                y: {priority_costs},
                name: 'Avg Cost',
                type: 'bar',
                marker: {{ color: '{accent_color}' }}
            }},
            {{
                x: {priority_keys},
                y: {priority_counts},
                name: 'Count',
                type: 'scatter',
                mode: 'lines+markers',
                line: {{ color: '{highlight_color}', width: 3 }},
                marker: {{ size: 10 }},
                yaxis: 'y2'
            }}
        ];

        var priorityLayout = {{
            title: 'Cost and Distribution by Priority Level',
            xaxis: {{ title: 'Priority', color: '{text_color}' }},
            yaxis: {{ title: 'Avg Cost ($)', color: '{text_color}' }},
            yaxis2: {{ title: 'Vessel Count', overlaying: 'y', side: 'right', color: '{text_color}' }},
            plot_bgcolor: '{bg_color}',
            paper_bgcolor: '{bg_color}',
            font: {{ color: '{text_color}' }},
            hovermode: 'x unified'
        }};

        Plotly.newPlot('priority-chart', priorityData, priorityLayout, {{ responsive: true }});
    </script>
</body>
</html>
"""

    with open("additional_output/05_classical_metrics_summary.html", "w") as f:
        f.write(metrics_html)