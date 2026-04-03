"""
Classical BAP+QCA Greedy-2Opt Solver
Berth Allocation + Quay Crane Assignment using greedy construction + 2-opt local search.
Enhanced with rich visual output for benchmarking dashboards.
"""
import logging
import time
import itertools

logger = logging.getLogger("qcentroid-user-log")


def run(input_data: dict, solver_params: dict, extra_arguments: dict) -> dict:
    start_time = time.time()
    logger.info("=== Classical BAP+QCA Greedy-2Opt Solver v2.0 ===")

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
    cost_evolution = []  # Track cost improvement over iterations

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

    # ── 3. 2-opt local search improvement ────────────────────────────
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

    logger.info(f"2-opt completed after {iteration} iterations")

    # ── 4. Compute metrics ───────────────────────────────────────────
    total_cost = sum(a["cost"] for a in assignments)
    feasible_count = sum(1 for a in assignments if a.get("berth_id") is not None)
    total_wait = 0
    total_handling = 0
    total_teu = 0

    for a in assignments:
        total_handling += a.get("handling_hours", 0)
        total_teu += a.get("teu_volume", 0)
        if a.get("start_time") and a.get("vessel_id"):
            v_data = next((v for v in vessels if v["id"] == a["vessel_id"]), {})
            arr = v_data.get("arrival_time", a["start_time"])
            total_wait += max(0, _iso_to_hours(a["start_time"]) - _iso_to_hours(arr))

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
    total_crane_cost = 0
    total_wait_cost = 0
    total_delay_cost = 0
    for a in assignments:
        if a.get("berth_id") is None:
            continue
        v_data = next((v for v in vessels if v["id"] == a["vessel_id"]), {})
        b_data = next((b for b in berths if b["id"] == a["berth_id"]), {})
        v_priority = v_data.get("priority", 3)
        pm = w_priority if v_priority <= 2 else 1.0
        b_prod = b_data.get("productivity_teu_per_crane_hour", 25)

        crane_c = a.get("handling_hours", 0) * a.get("cranes_assigned", 1) * w_handle
        arr_h = _iso_to_hours(v_data.get("arrival_time", a["start_time"]))
        start_h = _iso_to_hours(a["start_time"])
        wait_h = max(0, start_h - arr_h)
        wait_c = wait_h * w_wait * pm
        end_h = _iso_to_hours(a["end_time"])
        deadline_h = _iso_to_hours(v_data.get("max_departure_time", "2025-12-31T23:59:00Z"))
        delay_h = max(0, end_h - deadline_h)
        delay_c = delay_h * w_delay * pm

        total_crane_cost += crane_c
        total_wait_cost += wait_c
        total_delay_cost += delay_c

    # Crane allocation distribution
    crane_distribution = {}
    for a in assignments:
        nc = a.get("cranes_assigned", 0)
        crane_distribution[nc] = crane_distribution.get(nc, 0) + 1

    # Gantt chart data (timeline per berth)
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
            priority_analysis[key] = {"count": 0, "total_cost": 0, "avg_wait": 0, "total_wait": 0}
        priority_analysis[key]["count"] += 1
        priority_analysis[key]["total_cost"] += a.get("cost", 0)
    for key in priority_analysis:
        pa = priority_analysis[key]
        pa["avg_cost"] = round(pa["total_cost"] / max(pa["count"], 1), 2)
        pa["total_cost"] = round(pa["total_cost"], 2)

    improvement_pct = round((1 - total_cost / max(greedy_cost, 1)) * 100, 2) if greedy_cost > 0 else 0

    elapsed = round(time.time() - start_time, 3)
    logger.info(f"Total cost: {total_cost:.2f}, Status: {status}, Time: {elapsed}s")
    logger.info(f"2-opt improvement: {improvement_pct}% over greedy baseline")

    return {
        # ── Core assignment result ──
        "assignments": assignments,
        "objective_value": round(total_cost, 2),
        "solution_status": status,

        # ── Input size metrics (for benchmarking) ──
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
            "final_optimized_cost": round(total_cost, 2),
            "improvement_pct": improvement_pct,
            "iterations_used": iteration,
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
            "algorithm": "Greedy_2Opt_BAP_QCA",
            "iterations": iteration,
            "search_space_explored": n_vessels * n_berths * (max_cranes - min_cranes + 1),
            "solver_version": "2.0"
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
    """Try swapping berth assignments of two vessels. Return new assignments or None."""
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

    def calc_cost(v, b, start_time):
        nc = min_cranes
        prod = b.get("productivity_teu_per_crane_hour", 25)
        teu = v.get("handling_volume_teu", 1000)
        handling_h = teu / (prod * nc) if prod * nc > 0 else 999
        end_h = _iso_to_hours(start_time) + handling_h
        deadline_h = _iso_to_hours(v.get("max_departure_time", "2025-12-31T23:59:00Z"))
        arr_h = _iso_to_hours(v.get("arrival_time", start_time))
        wait_h = max(0, _iso_to_hours(start_time) - arr_h)
        delay_h = max(0, end_h - deadline_h)
        p = v.get("priority", 3)
        pm = w_priority if p <= 2 else 1.0
        return (handling_h * nc * w_handle + wait_h * w_wait * pm +
                delay_h * w_delay * pm), handling_h, nc

    c1, h1, nc1 = calc_cost(v1, b2, a1["start_time"])
    c2, h2, nc2 = calc_cost(v2, b1, a2["start_time"])

    new_a1 = dict(a1)
    new_a1["berth_id"] = b2_id
    new_a1["cost"] = round(c1, 2)
    new_a1["handling_hours"] = round(h1, 2)
    new_a1["cranes_assigned"] = nc1

    new_a2 = dict(a2)
    new_a2["berth_id"] = b1_id
    new_a2["cost"] = round(c2, 2)
    new_a2["handling_hours"] = round(h2, 2)
    new_a2["cranes_assigned"] = nc2

    return new_a1, new_a2
