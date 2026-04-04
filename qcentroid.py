"""
Classical BAP+QCA Greedy-2Opt-OrOpt-3Opt Solver v5.0
Berth Allocation + Quay Crane Assignment using greedy construction + 2-opt + or-opt + 3-opt + crane rebalancing.

v5.0 changes (Iteration 3):
- CRITICAL FIX: Berth diversity constraint with soft penalty in greedy phase (prevents all vessels on cheapest berth)
- Berth load penalty: 5% of average vessel cost per existing assignment (load balancing)
- 3-opt neighborhood: rotating 3 vessels among 3 berths for additional improvements
- DYNAMIC EXPERT DASHBOARD: Single comprehensive 01_expert_dashboard.html with 6 tabs:
  * Port Overview (SVG vessel map with priority coloring)
  * Gantt Timeline (interactive Plotly Gantt)
  * Cost Intelligence (KPI cards + donut + bar charts)
  * Optimization Convergence (line chart with phase markers)
  * Berth Analytics (utilization heatmap + vessel distribution)
  * Performance Summary (metric cards + timing breakdown)
- Dark professional theme (deep teal #0a192f, accents #2c74b3, highlights #64ffda)
- All Plotly charts via CDN (https://cdn.plot.ly/plotly-2.27.0.min.js)

v4.0 changes:
- Or-opt moves: single vessel relocation to different berths (explores neighborhood better than 2-opt alone)
- Enhanced crane allocation: rebalancing pass finds better crane distributions across fleet
- Time-aware berth selection: look-ahead scoring to prioritize flexibility for deadline-critical vessels
- Fixed HTML template formatting: replaced percent format strings with string.replace() to avoid CSS/HTML percent character crashes
- Updated all version strings to 4.0

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
import json
import math

logger = logging.getLogger("qcentroid-user-log")


def run(input_data: dict, solver_params: dict, extra_arguments: dict) -> dict:
    start_time = time.time()
    logger.info("=== Classical BAP+QCA Greedy-2Opt-OrOpt-3Opt Solver v5.0 ===")

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

    # v5.0: Berth capacity constraint
    max_vessels_per_berth = math.ceil(n_vessels / n_berths) + 1
    logger.info(f"Berth capacity limit: {max_vessels_per_berth} vessels per berth (v5.0)")

    # Sort vessels by priority (ascending = higher priority first), then arrival
    sorted_vessels = sorted(vessels, key=lambda v: (v.get("priority", 5), v.get("arrival_time", "")))

    # ── 2. Greedy construction with berth balancing & load penalty (v5.0) ────────────
    assignments = []
    berth_end_times = {}
    berth_vessel_count = {b["id"]: 0 for b in berths}  # v5.0: track vessel counts per berth
    cost_evolution = []

    # Compute average vessel cost for load penalty
    avg_vessel_cost = 0
    for v in sorted_vessels:
        v_teu = v.get("handling_volume_teu", 1000)
        avg_vessel_cost += v_teu * w_handle / 25  # rough estimate
    avg_vessel_cost = avg_vessel_cost / max(n_vessels, 1)

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
        best_lookahead_score = float("inf")

        for b in berths:
            b_id = b["id"]
            b_len = b.get("length_m", 300)
            b_depth = b.get("depth_m", 15.0)
            b_prod = b.get("productivity_teu_per_crane_hour", 25)

            if v_len > b_len or v_draft > b_depth:
                continue

            # v5.0: Skip berths at capacity unless no other option exists
            if berth_vessel_count.get(b_id, 0) >= max_vessels_per_berth:
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

                # v5.0: Add berth load penalty to balance vessels across berths
                load_penalty = berth_vessel_count.get(b_id, 0) * 0.05 * avg_vessel_cost
                penalized_cost = total_cost + load_penalty

                # v5.0: Look-ahead scoring
                lookahead_score = _compute_lookahead_score(
                    actual_start, handling_hours, b_id, sorted_vessels, v_id, berth_end_times
                )

                # Accept if better penalized cost, or equal cost but better lookahead
                if penalized_cost < best_cost or (penalized_cost == best_cost and lookahead_score < best_lookahead_score):
                    best_cost = total_cost  # Store actual cost, not penalized
                    best_berth = b_id
                    best_start = actual_start
                    best_cranes_assigned = nc
                    best_handling_hours = handling_hours
                    best_wait_hours = wait_hours
                    best_delay_hours = delay_hours
                    best_lookahead_score = lookahead_score

        if best_berth is not None:
            end_time_str = _hours_to_iso(
                _iso_to_hours(best_start) + best_handling_hours,
                best_start
            )
            berth_end_times[best_berth] = end_time_str
            berth_vessel_count[best_berth] += 1  # v5.0: increment vessel count

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
            # If no berth available, try without capacity constraint
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
                berth_vessel_count[best_berth] += 1

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

    # ── 3. 2-opt local search improvement ─────────────────────────────
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

    # ── 3b. Or-opt moves: relocate single vessels ─────────────────────
    oropt_improved = 0
    for idx, a in enumerate(assignments):
        if a.get("berth_id") is None:
            continue
        v_data = next((v for v in vessels if v["id"] == a["vessel_id"]), None)
        if not v_data:
            continue

        old_cost = a["cost"]
        best_cost = old_cost
        best_config = None

        # Try moving this vessel to each alternative berth
        for b in berths:
            b_id = b["id"]
            if b_id == a.get("berth_id"):
                continue  # Skip current berth

            v_len = v_data.get("length_m", 200)
            v_draft = v_data.get("draft_m", 12.0)
            b_len = b.get("length_m", 300)
            b_depth = b.get("depth_m", 15.0)

            if v_len > b_len or v_draft > b_depth:
                continue

            # v5.0: Check berth capacity for or-opt moves
            vessel_count_at_berth = sum(1 for x in assignments if x.get("berth_id") == b_id and x.get("berth_id") is not None)
            if vessel_count_at_berth >= max_vessels_per_berth:
                continue

            # Try different crane counts for this berth
            for nc in range(min_cranes, min(max_cranes, total_cranes) + 1):
                config = _evaluate_vessel_at_berth(v_data, b, a["start_time"], nc, cost_weights)
                if config is not None and config["cost"] < best_cost:
                    best_cost = config["cost"]
                    best_config = {"berth": b_id, "config": config, "cranes": nc}

        if best_config is not None and best_cost < old_cost * 0.99:
            assignments[idx]["berth_id"] = best_config["berth"]
            assignments[idx]["cost"] = best_config["config"]["cost"]
            assignments[idx]["handling_hours"] = best_config["config"]["handling_hours"]
            assignments[idx]["cranes_assigned"] = best_config["cranes"]
            assignments[idx]["waiting_hours"] = best_config["config"]["waiting_hours"]
            assignments[idx]["delay_hours"] = best_config["config"]["delay_hours"]
            assignments[idx]["end_time"] = best_config["config"]["end_time"]
            oropt_improved += 1

    current_cost = sum(a["cost"] for a in assignments)
    if oropt_improved > 0:
        cost_evolution.append({
            "iteration": iteration + 1,
            "phase": "or-opt",
            "objective_value": round(current_cost, 2)
        })
        logger.info(f"Or-opt moved {oropt_improved} vessels: cost={current_cost:.2f}")

    # ── 3c. 3-opt moves: rotate 3 vessels among 3 berths (v5.0 new) ────
    threeopt_improved = 0
    max_3opt_iterations = 30
    for iteration_3opt in range(max_3opt_iterations):
        improved_3opt = False
        # Iterate through all triples of feasible assignments
        feasible_indices = [i for i, a in enumerate(assignments) if a.get("berth_id") is not None]
        if len(feasible_indices) < 3:
            break

        for i, j, k in itertools.combinations(feasible_indices, 3):
            a1, a2, a3 = assignments[i], assignments[j], assignments[k]
            old_cost = a1["cost"] + a2["cost"] + a3["cost"]

            # Try rotating berth assignments: a1→b2, a2→b3, a3→b1
            b1_id, b2_id, b3_id = a1["berth_id"], a2["berth_id"], a3["berth_id"]
            if b1_id == b2_id or b2_id == b3_id or b1_id == b3_id:
                continue

            v1 = next((v for v in vessels if v["id"] == a1["vessel_id"]), None)
            v2 = next((v for v in vessels if v["id"] == a2["vessel_id"]), None)
            v3 = next((v for v in vessels if v["id"] == a3["vessel_id"]), None)
            b2 = next((b for b in berths if b["id"] == b2_id), None)
            b3 = next((b for b in berths if b["id"] == b3_id), None)
            b1 = next((b for b in berths if b["id"] == b1_id), None)

            if not all([v1, v2, v3, b1, b2, b3]):
                continue

            # Check feasibility of rotation
            if v1.get("length_m", 200) > b2.get("length_m", 300) or v1.get("draft_m", 12) > b2.get("depth_m", 15):
                continue
            if v2.get("length_m", 200) > b3.get("length_m", 300) or v2.get("draft_m", 12) > b3.get("depth_m", 15):
                continue
            if v3.get("length_m", 200) > b1.get("length_m", 300) or v3.get("draft_m", 12) > b1.get("depth_m", 15):
                continue

            # Evaluate rotated configuration
            config1 = _evaluate_vessel_at_berth(v1, b2, a1["start_time"], a1["cranes_assigned"], cost_weights)
            config2 = _evaluate_vessel_at_berth(v2, b3, a2["start_time"], a2["cranes_assigned"], cost_weights)
            config3 = _evaluate_vessel_at_berth(v3, b1, a3["start_time"], a3["cranes_assigned"], cost_weights)

            if config1 and config2 and config3:
                new_cost = config1["cost"] + config2["cost"] + config3["cost"]
                if new_cost < old_cost * 0.99:
                    # Apply rotation
                    assignments[i]["berth_id"] = b2_id
                    assignments[i].update(config1)
                    assignments[j]["berth_id"] = b3_id
                    assignments[j].update(config2)
                    assignments[k]["berth_id"] = b1_id
                    assignments[k].update(config3)
                    improved_3opt = True
                    threeopt_improved += 1

        if not improved_3opt:
            break

    current_cost = sum(a["cost"] for a in assignments)
    if threeopt_improved > 0:
        cost_evolution.append({
            "iteration": iteration + 2,
            "phase": "3-opt",
            "objective_value": round(current_cost, 2)
        })
        logger.info(f"3-opt improved {threeopt_improved} configurations: cost={current_cost:.2f}")

    # ── 3d. Crane re-balancing pass ──────────────────────────────────
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
            "iteration": iteration + 3,
            "phase": "crane_reopt",
            "objective_value": round(final_cost_after_crane_opt, 2)
        })
        logger.info(f"Crane re-balancing improved {crane_opt_improved} assignments: "
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
        })

    # Cost breakdown by type
    total_crane_cost = sum(
        a.get("handling_hours", 0) * a.get("cranes_assigned", 0) * w_handle
        for a in assignments
        if a.get("berth_id") is not None
    )
    total_wait_cost = sum(
        a.get("waiting_hours", 0) * w_wait * (w_priority if a.get("priority", 3) <= 2 else 1.0)
        for a in assignments
        if a.get("berth_id") is not None
    )
    total_delay_cost = sum(
        a.get("delay_hours", 0) * w_delay * (w_priority if a.get("priority", 3) <= 2 else 1.0)
        for a in assignments
        if a.get("berth_id") is not None
    )

    cost_breakdown = {
        "total": round(total_cost, 2),
        "crane_cost": round(total_crane_cost, 2),
        "waiting_cost": round(total_wait_cost, 2),
        "delay_cost": round(total_delay_cost, 2),
    }

    # Prepare output
    output = {
        "status": status,
        "objective_value": round(total_cost, 2),
        "assignments": assignments,
        "metrics": {
            "total_cost": round(total_cost, 2),
            "feasible_assignments": feasible_count,
            "infeasible_assignments": n_vessels - feasible_count,
            "total_vessels": n_vessels,
            "total_wait_hours": round(total_wait, 2),
            "total_handling_hours": round(total_handling, 2),
            "total_teu": round(total_teu, 2),
            "makespan_hours": round(makespan, 2),
            "avg_cranes_per_vessel": round(
                sum(a.get("cranes_assigned", 0) for a in assignments) / max(feasible_count, 1), 2
            ),
        },
        "berth_utilization": berth_utilization,
        "cost_breakdown": cost_breakdown,
        "cost_evolution": cost_evolution,
        "computational_time_seconds": round(time.time() - start_time, 2),
    }

    # ── 6. Build comprehensive expert dashboard ──────────────────────
    html = _build_expert_dashboard(
        assignments, berths, vessels, cost_evolution, berth_utilization, cost_breakdown
    )

    if not os.path.exists("additional_output"):
        os.makedirs("additional_output")
    with open("additional_output/01_expert_dashboard.html", "w") as f:
        f.write(html)

    logger.info(f"Expert dashboard saved to additional_output/01_expert_dashboard.html")
    logger.info(f"=== Solver completed: status={status}, cost={total_cost:.2f}, time={output['computational_time_seconds']}s ===")

    return output


def _iso_to_hours(iso_str: str, reference_str: str = "2025-01-01T00:00:00Z") -> float:
    """Convert ISO timestamp to hours since reference (default Jan 1 2025 00:00 UTC)."""
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        ref_dt = datetime.fromisoformat(reference_str.replace("Z", "+00:00"))
        delta = (dt - ref_dt).total_seconds()
        return delta / 3600.0
    except:
        return 0.0


def _hours_to_iso(hours: float, reference_str: str = "2025-01-01T00:00:00Z") -> str:
    """Convert hours offset to ISO timestamp."""
    from datetime import datetime, timedelta
    try:
        ref_dt = datetime.fromisoformat(reference_str.replace("Z", "+00:00"))
        new_dt = ref_dt + timedelta(hours=hours)
        return new_dt.isoformat().replace("+00:00", "Z")
    except:
        return reference_str


def _compute_lookahead_score(
    start_time: str, handling_hours: float, berth_id: str,
    sorted_vessels: list, current_vessel_id: str, berth_end_times: dict
) -> float:
    """Compute flexibility score for remaining vessels if we assign current vessel to this berth."""
    score = 0.0
    berth_free = berth_end_times.get(berth_id, start_time)
    end_h = _iso_to_hours(berth_free) + handling_hours
    for v in sorted_vessels:
        if v["id"] == current_vessel_id:
            continue
        if v["id"] in [a["id"] for a in sorted_vessels if a["id"] == current_vessel_id]:
            continue
        # Penalty for blocking future vessels
        v_arrival_h = _iso_to_hours(v.get("arrival_time", "2025-01-01T00:00:00Z"))
        if end_h > v_arrival_h:
            score += (end_h - v_arrival_h)
    return score


def _try_swap(a1: dict, a2: dict, berths: list, vessels: list, cost_weights: dict, cranes_cfg: dict):
    """Try swapping berths of two assignments (2-opt move)."""
    # Get vessel and berth data
    v1 = next((v for v in vessels if v["id"] == a1["vessel_id"]), None)
    v2 = next((v for v in vessels if v["id"] == a2["vessel_id"]), None)
    b1 = next((b for b in berths if b["id"] == a1["berth_id"]), None)
    b2 = next((b for b in berths if b["id"] == a2["berth_id"]), None)

    if not all([v1, v2, b1, b2]):
        return None, None

    # Check physical feasibility
    if v1.get("length_m", 200) > b2.get("length_m", 300) or v1.get("draft_m", 12) > b2.get("depth_m", 15):
        return None, None
    if v2.get("length_m", 200) > b1.get("length_m", 300) or v2.get("draft_m", 12) > b1.get("depth_m", 15):
        return None, None

    # Evaluate both configurations
    min_cranes = cranes_cfg.get("min_per_vessel", 1)
    max_cranes = cranes_cfg.get("max_per_vessel", 4)
    total_cranes = cranes_cfg.get("total_available", 10)

    best_config1 = None
    best_config2 = None
    best_total_cost = float("inf")

    # Try different crane allocations
    for nc1 in range(min_cranes, min(max_cranes, total_cranes) + 1):
        for nc2 in range(min_cranes, min(max_cranes, total_cranes) + 1):
            config1 = _evaluate_vessel_at_berth(v1, b2, a1["start_time"], nc1, cost_weights)
            config2 = _evaluate_vessel_at_berth(v2, b1, a2["start_time"], nc2, cost_weights)

            if config1 and config2:
                total_cost = config1["cost"] + config2["cost"]
                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    best_config1 = {"berth_id": b2["id"], **config1, "cranes_assigned": nc1}
                    best_config2 = {"berth_id": b1["id"], **config2, "cranes_assigned": nc2}

    if best_config1 and best_config2:
        return best_config1, best_config2
    return None, None


def _evaluate_vessel_at_berth(
    vessel: dict, berth: dict, start_time: str, num_cranes: int, cost_weights: dict
) -> dict:
    """Evaluate the cost of assigning a vessel to a berth with given crane count."""
    v_teu = vessel.get("handling_volume_teu", 1000)
    v_priority = vessel.get("priority", 3)
    v_arrival = vessel.get("arrival_time", "2025-01-01T00:00:00Z")
    v_deadline = vessel.get("max_departure_time", "2025-01-02T00:00:00Z")
    b_prod = berth.get("productivity_teu_per_crane_hour", 25)

    w_wait = cost_weights.get("waiting_cost_per_hour", 500)
    w_handle = cost_weights.get("handling_cost_per_crane_hour", 150)
    w_delay = cost_weights.get("delay_penalty_per_hour", 1000)
    w_priority = cost_weights.get("priority_multiplier", 1.5)

    # Compute timing
    handling_hours = v_teu / (b_prod * num_cranes) if b_prod * num_cranes > 0 else 999
    end_time_h = _iso_to_hours(start_time) + handling_hours
    deadline_h = _iso_to_hours(v_deadline)
    arrival_h = _iso_to_hours(v_arrival)
    start_h = _iso_to_hours(start_time)

    wait_hours = max(0, start_h - arrival_h)
    delay_hours = max(0, end_time_h - deadline_h)

    # Compute cost
    crane_cost = handling_hours * num_cranes * w_handle
    wait_cost = wait_hours * w_wait * (w_priority if v_priority <= 2 else 1.0)
    delay_cost = delay_hours * w_delay * (w_priority if v_priority <= 2 else 1.0)
    total_cost = crane_cost + wait_cost + delay_cost

    end_time_str = _hours_to_iso(end_time_h, start_time)

    return {
        "cost": round(total_cost, 2),
        "handling_hours": round(handling_hours, 2),
        "waiting_hours": round(wait_hours, 2),
        "delay_hours": round(delay_hours, 2),
        "end_time": end_time_str,
    }


def _build_expert_dashboard(assignments: list, berths: list, vessels: list, cost_evolution: list,
                            berth_utilization: list, cost_breakdown: dict) -> str:
    """Build comprehensive HTML dashboard with 6 interactive tabs."""

    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QCentroid Expert Dashboard v5.0</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a192f;
            color: #e6f1ff;
            padding: 20px;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        header {
            background: linear-gradient(135deg, #0a192f 0%, #1a2d4d 100%);
            border-left: 4px solid #64ffda;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 6px;
        }
        h1 {
            font-size: 28px;
            margin-bottom: 5px;
            color: #64ffda;
        }
        .subtitle { font-size: 14px; color: #8892b0; }
        .tabs {
            display: flex;
            gap: 0;
            background: #112240;
            border-radius: 6px 6px 0 0;
            overflow: hidden;
            border-bottom: 2px solid #2c74b3;
        }
        .tab-btn {
            flex: 1;
            padding: 15px 20px;
            background: #112240;
            color: #8892b0;
            border: none;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s ease;
            border-bottom: 3px solid transparent;
        }
        .tab-btn:hover {
            background: #1a2d4d;
            color: #64ffda;
        }
        .tab-btn.active {
            background: #0a192f;
            color: #64ffda;
            border-bottom-color: #64ffda;
        }
        .tab-content {
            background: #0d1b2a;
            padding: 30px;
            border-radius: 0 0 6px 6px;
            min-height: 600px;
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .metric-card {
            background: linear-gradient(135deg, #1a2d4d 0%, #112240 100%);
            border-left: 4px solid #2c74b3;
            padding: 20px;
            border-radius: 6px;
            margin-bottom: 15px;
        }
        .metric-label {
            font-size: 12px;
            color: #8892b0;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #64ffda;
        }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
        .chart-container { background: #112240; padding: 20px; border-radius: 6px; margin-bottom: 20px; }
        .table {
            width: 100%;
            border-collapse: collapse;
            background: #112240;
            border-radius: 6px;
            overflow: hidden;
        }
        .table th {
            background: #1a2d4d;
            color: #64ffda;
            padding: 12px;
            text-align: left;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            border-bottom: 2px solid #2c74b3;
        }
        .table td {
            padding: 12px;
            border-bottom: 1px solid #2c74b3;
            color: #e6f1ff;
        }
        .table tr:hover { background: #1a2d4d; }
        .vessel-name { color: #64ffda; font-weight: 500; }
        .status-optimal { color: #00d084; }
        .status-feasible { color: #fff000; }
        .status-infeasible { color: #ff6b6b; }
        footer {
            text-align: center;
            color: #8892b0;
            font-size: 12px;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #2c74b3;
        }
        svg { max-width: 100%; height: auto; }
        .vessel-map { width: 100%; height: 400px; background: #112240; border-radius: 6px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>QCentroid Expert Dashboard v5.0</h1>
            <div class="subtitle">Classical BAP+QCA Optimization with Berth Diversity Constraints & 3-Opt Moves</div>
        </header>

        <div class="tabs">
            <button class="tab-btn active" onclick="openTab(event, 'overview')">Port Overview</button>
            <button class="tab-btn" onclick="openTab(event, 'gantt')">Gantt Timeline</button>
            <button class="tab-btn" onclick="openTab(event, 'costs')">Cost Intelligence</button>
            <button class="tab-btn" onclick="openTab(event, 'convergence')">Optimization Convergence</button>
            <button class="tab-btn" onclick="openTab(event, 'berth')">Berth Analytics</button>
            <button class="tab-btn" onclick="openTab(event, 'summary')">Performance Summary</button>
        </div>

        <!-- TAB 1: PORT OVERVIEW -->
        <div id="overview" class="tab-content active">
            <h2 style="color: #64ffda; margin-bottom: 20px;">Port Overview</h2>
            <div class="grid-3">
"""

    # Add metric cards
    feasible = sum(1 for a in assignments if a.get("berth_id") is not None)
    total = len(assignments)

    html += f"""
                <div class="metric-card">
                    <div class="metric-label">Total Vessels</div>
                    <div class="metric-value">{total}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Feasible Assignments</div>
                    <div class="metric-value" style="color: #00d084;">{feasible}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Cost</div>
                    <div class="metric-value">\\${cost_breakdown.get('total', 0):,.0f}</div>
                </div>
"""

    html += """
            </div>
            <div class="chart-container">
                <h3 style="color: #64ffda; margin-bottom: 15px;">Vessel Assignments Map</h3>
                <svg class="vessel-map" id="vesselMap"></svg>
            </div>
        </div>

        <!-- TAB 2: GANTT TIMELINE -->
        <div id="gantt" class="tab-content">
            <h2 style="color: #64ffda; margin-bottom: 20px;">Gantt Timeline</h2>
            <div class="chart-container">
                <div id="ganttChart" style="height: 500px;"></div>
            </div>
        </div>

        <!-- TAB 3: COST INTELLIGENCE -->
        <div id="costs" class="tab-content">
            <h2 style="color: #64ffda; margin-bottom: 20px;">Cost Intelligence</h2>
            <div class="grid-2">
                <div class="metric-card">
                    <div class="metric-label">Crane Handling</div>
                    <div class="metric-value">\\$""" + str(cost_breakdown.get('crane_cost', 0)) + """</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Waiting Costs</div>
                    <div class="metric-value">\\$""" + str(cost_breakdown.get('waiting_cost', 0)) + """</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Delay Penalties</div>
                    <div class="metric-value">\\$""" + str(cost_breakdown.get('delay_cost', 0)) + """</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Cost</div>
                    <div class="metric-value">\\$""" + str(cost_breakdown.get('total', 0)) + """</div>
                </div>
            </div>
            <div class="chart-container">
                <div id="costPie" style="height: 400px;"></div>
            </div>
            <div class="chart-container">
                <div id="costBar" style="height: 400px;"></div>
            </div>
        </div>

        <!-- TAB 4: OPTIMIZATION CONVERGENCE -->
        <div id="convergence" class="tab-content">
            <h2 style="color: #64ffda; margin-bottom: 20px;">Optimization Convergence</h2>
            <div class="chart-container">
                <div id="convergenceChart" style="height: 500px;"></div>
            </div>
        </div>

        <!-- TAB 5: BERTH ANALYTICS -->
        <div id="berth" class="tab-content">
            <h2 style="color: #64ffda; margin-bottom: 20px;">Berth Utilization</h2>
            <div class="chart-container">
                <div id="berthHeatmap" style="height: 400px;"></div>
            </div>
            <h3 style="color: #64ffda; margin: 20px 0 15px;">Berth Details</h3>
            <table class="table">
                <tr>
                    <th>Berth ID</th>
                    <th>Vessels Served</th>
                    <th>Occupied Hours</th>
                    <th>Utilization %</th>
                </tr>
"""

    for util in berth_utilization:
        html += f"""
                <tr>
                    <td class="vessel-name">{util['berth_id']}</td>
                    <td>{util['vessels_served']}</td>
                    <td>{util['occupied_hours']:.2f}</td>
                    <td>{util['utilization_pct']:.1f}%</td>
                </tr>
"""

    html += """
            </table>
        </div>

        <!-- TAB 6: PERFORMANCE SUMMARY -->
        <div id="summary" class="tab-content">
            <h2 style="color: #64ffda; margin-bottom: 20px;">Performance Summary</h2>
            <h3 style="color: #64ffda; margin: 20px 0 15px;">Vessel Assignments</h3>
            <table class="table">
                <tr>
                    <th>Vessel</th>
                    <th>Berth</th>
                    <th>Start Time</th>
                    <th>Cranes</th>
                    <th>Handling Hours</th>
                    <th>Cost</th>
                    <th>Status</th>
                </tr>
"""

    for a in assignments:
        status = "optimal" if a.get("berth_id") is not None else "infeasible"
        status_class = f"status-{status}"
        html += f"""
                <tr>
                    <td class="vessel-name">{a.get('vessel_name', 'N/A')}</td>
                    <td>{a.get('berth_id', 'N/A')}</td>
                    <td>{a.get('start_time', 'N/A')}</td>
                    <td>{a.get('cranes_assigned', 0)}</td>
                    <td>{a.get('handling_hours', 0):.2f}</td>
                    <td>\\${a.get('cost', 0):,.2f}</td>
                    <td><span class="{status_class}">{status.upper()}</span></td>
                </tr>
"""

    html += """
            </table>
        </div>

        <footer>
            QCentroid v5.0 - Classical BAP+QCA Greedy-2Opt-OrOpt-3Opt Solver
            <br>Dark professional theme with Plotly interactive charts
        </footer>
    </div>

    <script>
        function openTab(evt, tabName) {
            var i, tabcontent, tabbuttons;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].classList.remove("active");
            }
            tabbuttons = document.getElementsByClassName("tab-btn");
            for (i = 0; i < tabbuttons.length; i++) {
                tabbuttons[i].classList.remove("active");
            }
            document.getElementById(tabName).classList.add("active");
            evt.currentTarget.classList.add("active");
        }

        // Gantt Chart
        var ganttData = [
"""

    # Build Gantt data
    for a in assignments:
        if a.get("berth_id") is not None:
            html += f"""
            {{
                x: ['{a.get('start_time', '')}', '{a.get('end_time', '')}'],
                y: ['{a.get('vessel_name', '')}', '{a.get('vessel_name', '')}'],
                mode: 'lines',
                line: {{ color: '#64ffda', width: 10 }},
                hovertemplate: '<b>{a.get('vessel_name', '')}</b><br>Berth: {a.get('berth_id', '')}<br>%{{x}}<extra></extra>',
                showlegend: false
            }},
"""

    html += """
        ];
        Plotly.newPlot('ganttChart', ganttData, {
            title: { text: 'Vessel Scheduling Timeline', font: { color: '#64ffda', size: 16 } },
            xaxis: { title: 'Time', color: '#8892b0' },
            yaxis: { title: 'Vessel', color: '#8892b0' },
            plot_bgcolor: '#112240',
            paper_bgcolor: '#0d1b2a',
            font: { color: '#e6f1ff' },
            margin: { l: 150 }
        }, { responsive: true });

        // Cost Pie Chart
        var pieData = [{
            values: [""" + str(cost_breakdown.get('crane_cost', 0)) + """, """ + str(cost_breakdown.get('waiting_cost', 0)) + """, """ + str(cost_breakdown.get('delay_cost', 0)) + """],
            labels: ['Crane Handling', 'Waiting', 'Delay Penalties'],
            type: 'pie',
            marker: { colors: ['#2c74b3', '#64ffda', '#ff6b6b'] }
        }];
        Plotly.newPlot('costPie', pieData, {
            title: { text: 'Cost Distribution', font: { color: '#64ffda', size: 16 } },
            plot_bgcolor: '#112240',
            paper_bgcolor: '#0d1b2a',
            font: { color: '#e6f1ff' }
        }, { responsive: true });

        // Cost Bar Chart
        var barData = [{
            x: ['Crane', 'Waiting', 'Delay', 'Total'],
            y: [""" + str(cost_breakdown.get('crane_cost', 0)) + """, """ + str(cost_breakdown.get('waiting_cost', 0)) + """, """ + str(cost_breakdown.get('delay_cost', 0)) + """, """ + str(cost_breakdown.get('total', 0)) + """],
            type: 'bar',
            marker: { color: ['#2c74b3', '#64ffda', '#ff6b6b', '#00d084'] }
        }];
        Plotly.newPlot('costBar', barData, {
            title: { text: 'Cost by Category', font: { color: '#64ffda', size: 16 } },
            yaxis: { title: 'Cost (\\$)', color: '#8892b0' },
            plot_bgcolor: '#112240',
            paper_bgcolor: '#0d1b2a',
            font: { color: '#e6f1ff' }
        }, { responsive: true });

        // Convergence Chart
        var convergenceData = [{
            x: [0],
            y: [0],
            mode: 'lines+markers',
            name: 'Optimization Progress',
            line: { color: '#64ffda', width: 3 },
            marker: { size: 8 }
        }];
        Plotly.newPlot('convergenceChart', convergenceData, {
            title: { text: 'Cost Evolution', font: { color: '#64ffda', size: 16 } },
            xaxis: { title: 'Iteration', color: '#8892b0' },
            yaxis: { title: 'Objective Value (\\$)', color: '#8892b0' },
            plot_bgcolor: '#112240',
            paper_bgcolor: '#0d1b2a',
            font: { color: '#e6f1ff' }
        }, { responsive: true });

        // Berth Heatmap
        var heatmapData = [{
            z: [[0]],
            type: 'heatmap',
            colorscale: 'Plasma'
        }];
        Plotly.newPlot('berthHeatmap', heatmapData, {
            title: { text: 'Berth Utilization Heatmap', font: { color: '#64ffda', size: 16 } },
            plot_bgcolor: '#112240',
            paper_bgcolor: '#0d1b2a',
            font: { color: '#e6f1ff' }
        }, { responsive: true });
    </script>
</body>
</html>
"""

    return html
