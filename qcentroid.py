"""
Classical BAP+QCA Greedy-2Opt-OrOpt-3Opt Solver v6.0
Berth Allocation + Quay Crane Assignment using greedy construction + 2-opt + or-opt + 3-opt + crane rebalancing.

v6.0 changes (Iteration 2 - Barcelona):
- CRITICAL FIX: Global crane budget enforcement (18 cranes max at any time t)
- Priority-weighted crane allocation across simultaneously active berths
- Re-sequencing of all berths after crane adjustments
- Proper sequential scheduling enforcement

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
    logger.info("=== Classical BAP+QCA Greedy-2Opt-OrOpt-3Opt Solver v6.0 ===")

    # ── 1. Parse inputs ──────────────────────────────────────────────
    vessels = input_data.get("vessels", [])
    berths = input_data.get("berths", [])
    cranes_cfg = input_data.get("cranes", {})
    cost_weights = input_data.get("cost_weights", {})

    total_cranes = cranes_cfg.get("total_available", 10)
    min_cranes = cranes_cfg.get("min_per_vessel", 1)
    max_cranes = cranes_cfg.get("max_per_vessel", 4)
    # v6.0: Allow full crane range in greedy — the crane budget enforcement
    # handles time-slot-aware allocation after assignment

    w_wait = cost_weights.get("waiting_cost_per_hour", 500)
    w_handle = cost_weights.get("handling_cost_per_crane_hour", 150)
    w_delay = cost_weights.get("delay_penalty_per_hour", 1000)
    w_priority = cost_weights.get("priority_multiplier", 1.5)

    n_vessels = len(vessels)
    n_berths = len(berths)
    logger.info(f"Problem: {n_vessels} vessels, {n_berths} berths, {total_cranes} cranes")

    # v6.0: Berth capacity constraint (tighter)
    max_vessels_per_berth = math.ceil(n_vessels / n_berths) + 1
    logger.info(f"Berth capacity limit: {max_vessels_per_berth} vessels per berth")

    # v6.0: Crane budget awareness — with N berths active, each gets ~total_cranes/N
    crane_budget_per_active = max(min_cranes, total_cranes // max(n_berths, 1))
    logger.info(f"Crane budget per berth (all active): ~{crane_budget_per_active}")

    # Sort vessels by priority (ascending = higher priority first), then arrival
    sorted_vessels = sorted(vessels, key=lambda v: (v.get("priority", 5), v.get("arrival_time", "")))

    # ── 2. Greedy construction with crane-budget-aware berth balancing (v6.0) ────────
    assignments = []
    berth_end_times = {}
    berth_vessel_count = {b["id"]: 0 for b in berths}  # track vessel counts per berth
    cost_evolution = []

    # v6.0: Compute realistic load penalty based on delay cost (much stronger)
    avg_vessel_cost = 0
    for v in sorted_vessels:
        v_teu = v.get("handling_volume_teu", 1000)
        avg_vessel_cost += v_teu * w_handle / 25
    avg_vessel_cost = avg_vessel_cost / max(n_vessels, 1)
    # v6.0: Much stronger load penalty (50% not 5%) to force berth spreading
    load_penalty_factor = 0.50

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
        best_cost = float("inf")          # v6.0: now stores PENALIZED cost for comparison
        best_actual_cost = float("inf")   # v6.0: actual cost (without penalties) for assignment
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

            for nc in range(min_cranes, max_cranes + 1):
                handling_hours = v_teu / (b_prod * nc) if b_prod * nc > 0 else 999
                end_time_h = _iso_to_hours(actual_start) + handling_hours
                deadline_h = _iso_to_hours(v_deadline)

                wait_hours = max(0, _iso_to_hours(actual_start) - _iso_to_hours(v_arrival))
                delay_hours = max(0, end_time_h - deadline_h)
                crane_cost = handling_hours * nc * w_handle
                wait_cost = wait_hours * w_wait * (w_priority if v_priority <= 2 else 1.0)
                delay_cost = delay_hours * w_delay * (w_priority if v_priority <= 2 else 1.0)

                total_cost = crane_cost + wait_cost + delay_cost

                # v6.0: Strong berth load penalty to force spreading across all berths
                load_penalty = berth_vessel_count.get(b_id, 0) * load_penalty_factor * avg_vessel_cost
                # v6.0: Extra penalty for queuing delay — estimate cascading delay impact
                queue_penalty = 0
                if berth_vessel_count.get(b_id, 0) > 0:
                    # Estimate how much queuing at this berth delays future vessels
                    queue_penalty = handling_hours * w_delay * 0.3  # 30% of delay cost of our handling time
                penalized_cost = total_cost + load_penalty + queue_penalty

                # v5.0: Look-ahead scoring
                lookahead_score = _compute_lookahead_score(
                    actual_start, handling_hours, b_id, sorted_vessels, v_id, berth_end_times
                )

                # v6.0 FIX: Compare penalized vs penalized (was comparing penalized vs actual — penalties had no effect!)
                if penalized_cost < best_cost or (penalized_cost == best_cost and lookahead_score < best_lookahead_score):
                    best_cost = penalized_cost      # v6.0: store PENALIZED cost for correct comparison
                    best_actual_cost = total_cost   # v6.0: actual cost for the assignment record
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
                "cost": round(best_actual_cost, 2),  # v6.0: use actual cost, not penalized
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

                for nc in range(min_cranes, max_cranes + 1):
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
                        best_actual_cost = total_cost
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

    # ── 2b. EARLY crane budget enforcement + resequencing (v6.0) ─────
    # Apply crane budget BEFORE local search so 2-opt/or-opt see realistic costs
    logger.info("Early crane budget enforcement (before local search)...")
    assignments, early_cb_changes = _enforce_crane_budget(
        assignments, vessels, berths, total_cranes, min_cranes, max_cranes, cost_weights
    )
    if early_cb_changes > 0:
        logger.info(f"Early crane budget adjusted {early_cb_changes} assignments")
    assignments, early_rs_changes = _resequence_all_berths(
        assignments, vessels, berths, cost_weights, w_priority
    )
    if early_rs_changes > 0:
        logger.info(f"Early resequencing updated {early_rs_changes} assignments")
    post_early_cost = sum(a["cost"] for a in assignments)
    cost_evolution.append({"iteration": 0.5, "phase": "early_crane_budget", "objective_value": round(post_early_cost, 2)})
    logger.info(f"Post-early-enforcement cost: {post_early_cost:.2f}")

    # ── 3. 2-opt local search improvement ─────────────────────────────
    max_iterations = solver_params.get("max_2opt_iterations", 50)
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

            for nc in range(min_cranes, max_cranes + 1):
                config = _evaluate_vessel_at_berth(v_data, b, a["start_time"], nc, cost_weights)
                if config is not None:
                    # v6.0: Add load penalty to or-opt to prevent undoing greedy spreading
                    oropt_load_penalty = vessel_count_at_berth * load_penalty_factor * avg_vessel_cost
                    oropt_penalized = config["cost"] + oropt_load_penalty
                    if oropt_penalized < best_cost:
                        best_cost = oropt_penalized
                        best_config = {"berth": b_id, "config": config, "cranes": nc}

        # v6.0: Also apply penalty to current berth for fair comparison
        current_berth_count = sum(1 for x in assignments if x.get("berth_id") == a.get("berth_id") and x.get("berth_id") is not None) - 1
        old_penalized_cost = old_cost + current_berth_count * load_penalty_factor * avg_vessel_cost
        if best_config is not None and best_cost < old_penalized_cost * 0.99:
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

        for nc in range(min_cranes, max_cranes + 1):
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

    # ── 3e. ITERATIVE CRANE BUDGET + RESEQUENCING (v6.0) ────────────
    # Iterate until convergent: crane budget → resequence → check again
    for cb_round in range(5):  # max 5 rounds
        logger.info(f"Crane budget enforcement round {cb_round+1}...")
        assignments, crane_budget_changes = _enforce_crane_budget(
            assignments, vessels, berths, total_cranes, min_cranes, max_cranes, cost_weights
        )
        assignments, reseq_changes = _resequence_all_berths(
            assignments, vessels, berths, cost_weights, w_priority
        )
        round_cost = sum(a["cost"] for a in assignments)
        cost_evolution.append({
            "iteration": iteration + 4 + cb_round,
            "phase": f"crane_budget_r{cb_round+1}",
            "objective_value": round(round_cost, 2)
        })
        logger.info(f"  Round {cb_round+1}: {crane_budget_changes} crane adjustments, "
                     f"{reseq_changes} resequenced, cost={round_cost:.2f}")
        if crane_budget_changes == 0:
            break
    post_reseq_cost = sum(a["cost"] for a in assignments)
    logger.info(f"Final post-enforcement cost: {post_reseq_cost:.2f}")

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
                f"2-opt={two_opt_cost:.2f}, or-opt={current_cost:.2f}, crane-reopt={final_cost_after_crane_opt:.2f})")

    # ── 6. Generate additional output visualizations ─────────────────
    try:
        _generate_expert_dashboard(
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
                "algorithm": "Greedy_2Opt_OrOpt_3Opt_CraneRebalance_BAP_QCA",
                "iterations": iteration,
                "or_opt_improvements": oropt_improved,
                "three_opt_improvements": threeopt_improved,
                "crane_rebalance_improvements": crane_opt_improved,
                "search_space_explored": n_vessels * n_berths * (max_cranes - min_cranes + 1),
                "solver_version": "6.0"
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
        logger.info("Expert dashboard generated successfully (v6.0)")
    except Exception as e:
        logger.warning(f"Failed to generate expert dashboard: {e}")

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
            "algorithm": "Greedy_2Opt_OrOpt_3Opt_CraneRebalance_BAP_QCA",
            "iterations": iteration,
            "or_opt_improvements": oropt_improved,
            "three_opt_improvements": threeopt_improved,
            "crane_rebalance_improvements": crane_opt_improved,
            "search_space_explored": n_vessels * n_berths * (max_cranes - min_cranes + 1),
            "solver_version": "6.0"
        },

        # ── Platform benchmark contract ──
        "benchmark": {
            "execution_cost": {"value": 1.0, "unit": "credits"},
            "time_elapsed": f"{elapsed}s",
            "energy_consumption": 0.0
        }
    }


# ── v6.0 Helper functions ───────────────────────────────────────────

def _enforce_crane_budget(assignments, vessels, berths, total_cranes, min_cranes, max_cranes, cost_weights):
    """
    v6.0: Enforce global crane constraint.
    At any time t, sum of cranes across ALL simultaneously active vessels <= total_cranes.
    Uses priority-weighted allocation: P1 vessels keep cranes, P5 gets reduced first.
    After crane reduction, recalculates handling time, end time, delay, and cost.
    """
    w_handle = cost_weights.get("handling_cost_per_crane_hour", 150)
    w_wait = cost_weights.get("waiting_cost_per_hour", 500)
    w_delay = cost_weights.get("delay_penalty_per_hour", 1000)
    w_priority_mult = cost_weights.get("priority_multiplier", 1.5)
    
    changes = 0
    if not assignments:
        return assignments, changes
    
    # Build time intervals for active assignments
    active_assignments = []
    for idx, a in enumerate(assignments):
        if a.get("berth_id") is None:
            continue
        start_h = _iso_to_hours(a["start_time"])
        end_h = _iso_to_hours(a["end_time"])
        if end_h <= start_h:
            end_h = start_h + 1  # safety
        active_assignments.append((start_h, end_h, idx))
    
    if not active_assignments:
        return assignments, changes
    
    # Find all critical time points
    events = set()
    for s, e, _ in active_assignments:
        events.add(s)
        events.add(e)
    events = sorted(events)
    
    # For each time point, check crane usage
    # Track max allowed cranes per vessel (minimum across all time slots)
    max_allowed = {}
    for _, _, idx in active_assignments:
        max_allowed[idx] = assignments[idx]["cranes_assigned"]
    
    for t_idx in range(len(events) - 1):
        t = events[t_idx]
        # Find vessels active at time t
        active_at_t = [(s, e, idx) for s, e, idx in active_assignments if s <= t < e]
        total_used = sum(assignments[idx]["cranes_assigned"] for _, _, idx in active_at_t)
        
        if total_used <= total_cranes:
            continue  # No violation at this time
        
        # Violation! Allocate cranes by priority
        # Sort: highest priority (P1=1) first, then by TEU volume desc
        active_sorted = []
        for s, e, idx in active_at_t:
            a = assignments[idx]
            v_data = next((v for v in vessels if v["id"] == a["vessel_id"]), None)
            priority = v_data.get("priority", 5) if v_data else 5
            teu = a.get("teu_volume", 0)
            active_sorted.append((priority, -teu, idx))
        active_sorted.sort()
        
        n_active = len(active_sorted)
        budget = total_cranes
        
        # Allocate: give min_cranes to everyone first
        slot_alloc = {}
        for _, _, idx in active_sorted:
            slot_alloc[idx] = min_cranes
            budget -= min_cranes
        
        # Distribute remaining by priority order (highest priority first)
        for _, _, idx in active_sorted:
            if budget <= 0:
                break
            wanted = min(assignments[idx]["cranes_assigned"] - min_cranes, budget)
            if wanted > 0:
                slot_alloc[idx] += wanted
                budget -= wanted
        
        # Apply minimum across all time slots
        for idx, nc in slot_alloc.items():
            if nc < max_allowed.get(idx, 999):
                max_allowed[idx] = nc
    
    # Apply crane reductions and recalculate
    for idx, new_nc in max_allowed.items():
        a = assignments[idx]
        if new_nc >= a["cranes_assigned"]:
            continue  # No change needed
        
        v_data = next((v for v in vessels if v["id"] == a["vessel_id"]), None)
        b_data = next((b for b in berths if b["id"] == a["berth_id"]), None)
        if not v_data or not b_data:
            continue
        
        v_teu = v_data.get("handling_volume_teu", 1000)
        v_priority = v_data.get("priority", 3)
        pm = w_priority_mult if v_priority <= 2 else 1.0
        b_prod = b_data.get("productivity_teu_per_crane_hour", 25)
        
        new_handling_h = v_teu / (b_prod * new_nc) if b_prod * new_nc > 0 else 999
        start_h = _iso_to_hours(a["start_time"])
        arr_h = _iso_to_hours(v_data.get("arrival_time", a["start_time"]))
        deadline_h = _iso_to_hours(v_data.get("max_departure_time", "2025-12-31T23:59:00Z"))
        
        wait_h = max(0, start_h - arr_h)
        new_end_h = start_h + new_handling_h
        new_delay_h = max(0, new_end_h - deadline_h)
        
        new_cost = (new_handling_h * new_nc * w_handle +
                    wait_h * w_wait * pm +
                    new_delay_h * w_delay * pm)
        
        assignments[idx] = dict(a)
        assignments[idx]["cranes_assigned"] = new_nc
        assignments[idx]["handling_hours"] = round(new_handling_h, 2)
        assignments[idx]["delay_hours"] = round(new_delay_h, 2)
        assignments[idx]["cost"] = round(new_cost, 2)
        assignments[idx]["end_time"] = _hours_to_iso(new_end_h, a["start_time"])
        assignments[idx]["waiting_hours"] = round(wait_h, 2)
        changes += 1
    
    return assignments, changes


def _resequence_all_berths(assignments, vessels, berths, cost_weights, w_priority_mult):
    """
    v6.0: After crane changes, re-sequence vessels at each berth.
    Vessels at the same berth must be sequential (no overlap).
    When handling times change, subsequent vessels' start times shift.
    """
    w_handle = cost_weights.get("handling_cost_per_crane_hour", 150)
    w_wait = cost_weights.get("waiting_cost_per_hour", 500)
    w_delay = cost_weights.get("delay_penalty_per_hour", 1000)
    
    changes = 0
    
    # Group assignments by berth
    berth_groups = {}
    for idx, a in enumerate(assignments):
        b_id = a.get("berth_id")
        if b_id is None:
            continue
        if b_id not in berth_groups:
            berth_groups[b_id] = []
        berth_groups[b_id].append(idx)
    
    # For each berth, sort by start time and re-sequence
    for b_id, indices in berth_groups.items():
        # Sort by current start time
        indices.sort(key=lambda idx: _iso_to_hours(assignments[idx]["start_time"]))
        
        berth_free_h = None
        for pos, idx in enumerate(indices):
            a = assignments[idx]
            v_data = next((v for v in vessels if v["id"] == a["vessel_id"]), None)
            b_data = next((b for b in berths if b["id"] == b_id), None)
            if not v_data or not b_data:
                continue
            
            arr_h = _iso_to_hours(v_data.get("arrival_time", a["start_time"]))
            deadline_h = _iso_to_hours(v_data.get("max_departure_time", "2025-12-31T23:59:00Z"))
            v_priority = v_data.get("priority", 3)
            pm = w_priority_mult if v_priority <= 2 else 1.0
            v_teu = v_data.get("handling_volume_teu", 1000)
            b_prod = b_data.get("productivity_teu_per_crane_hour", 25)
            nc = a["cranes_assigned"]
            
            # Calculate correct start time
            if berth_free_h is None:
                new_start_h = arr_h
            else:
                new_start_h = max(arr_h, berth_free_h)
            
            handling_h = v_teu / (b_prod * nc) if b_prod * nc > 0 else 999
            new_end_h = new_start_h + handling_h
            wait_h = max(0, new_start_h - arr_h)
            delay_h = max(0, new_end_h - deadline_h)
            
            new_cost = (handling_h * nc * w_handle +
                        wait_h * w_wait * pm +
                        delay_h * w_delay * pm)
            
            # Check if anything changed
            old_start_h = _iso_to_hours(a["start_time"])
            if abs(new_start_h - old_start_h) > 0.01 or abs(new_cost - a["cost"]) > 0.01:
                ref_iso = v_data.get("arrival_time", a["start_time"])
                assignments[idx] = dict(a)
                assignments[idx]["start_time"] = _hours_to_iso(new_start_h, ref_iso)
                assignments[idx]["end_time"] = _hours_to_iso(new_end_h, ref_iso)
                assignments[idx]["handling_hours"] = round(handling_h, 2)
                assignments[idx]["waiting_hours"] = round(wait_h, 2)
                assignments[idx]["delay_hours"] = round(delay_h, 2)
                assignments[idx]["cost"] = round(new_cost, 2)
                changes += 1
            
            berth_free_h = new_end_h
    
    return assignments, changes


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


def _compute_lookahead_score(start_time, duration, berth_id, remaining_vessels, current_v_id, berth_end_times):
    """
    v5.0: Compute look-ahead score to prefer berths that leave flexibility for deadline-critical vessels.
    Returns penalty: higher = worse choice for this berth.
    """
    score = 0
    end_h = _iso_to_hours(start_time) + duration

    # Penalize berths that would delay many high-priority vessels
    for v in remaining_vessels:
        if v["id"] == current_v_id:
            continue
        v_priority = v.get("priority", 3)
        v_deadline = _iso_to_hours(v.get("max_departure_time", "2025-12-31T23:59:00Z"))

        # If this vessel is high-priority and its deadline is near end_h, penalize
        if v_priority <= 2 and v_deadline < end_h + 48:  # 48 hours lookahead
            score += (v_priority == 1) * 100 + (v_priority == 2) * 50

    return score


def _evaluate_vessel_at_berth(vessel, berth, start_time, num_cranes, cost_weights):
    """
    v5.0: Evaluate cost of assigning a vessel to a berth with a specific crane count.
    Returns dict with cost, handling_hours, waiting_hours, delay_hours, end_time.
    Returns None if infeasible.
    """
    w_handle = cost_weights.get("handling_cost_per_crane_hour", 150)
    w_wait = cost_weights.get("waiting_cost_per_hour", 500)
    w_delay = cost_weights.get("delay_penalty_per_hour", 1000)
    w_priority = cost_weights.get("priority_multiplier", 1.5)

    v_teu = vessel.get("handling_volume_teu", 1000)
    v_priority = vessel.get("priority", 3)
    v_arrival = vessel.get("arrival_time", start_time)
    v_deadline = vessel.get("max_departure_time", "2025-12-31T23:59:00Z")

    pm = w_priority if v_priority <= 2 else 1.0
    b_prod = berth.get("productivity_teu_per_crane_hour", 25)

    handling_h = v_teu / (b_prod * num_cranes) if b_prod * num_cranes > 0 else 999
    start_h = _iso_to_hours(start_time)
    arr_h = _iso_to_hours(v_arrival)
    deadline_h = _iso_to_hours(v_deadline)

    wait_h = max(0, start_h - arr_h)
    end_h = start_h + handling_h
    delay_h = max(0, end_h - deadline_h)

    cost = (handling_h * num_cranes * w_handle +
            wait_h * w_wait * pm +
            delay_h * w_delay * pm)

    return {
        "cost": round(cost, 2),
        "handling_hours": round(handling_h, 2),
        "waiting_hours": round(wait_h, 2),
        "delay_hours": round(delay_h, 2),
        "end_time": _hours_to_iso(end_h, start_time)
    }


def _try_swap(a1, a2, berths, vessels, cost_weights, cranes_cfg):
    """Try swapping berth assignments of two vessels.
    v5.0: maintains full crane level exploration from v3.
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
        """Try all crane levels, return best cost + crane count."""
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

        for nc in range(min_cranes, max_cranes + 1):
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


def _generate_expert_dashboard(assignments, berths, vessels, cost_breakdown,
                                optimization_convergence, berth_utilization,
                                priority_analysis, gantt_data, schedule_metrics,
                                computation_metrics, crane_allocation):
    """
    v5.0: Generate SINGLE comprehensive expert dashboard with 6 tabs using Plotly.js CDN.
    Replaces 5 separate HTML files with one professional single-page app.
    Dark theme: deep teal #0a192f, accents #2c74b3, highlights #64ffda
    """
    os.makedirs("additional_output", exist_ok=True)

    # Color scheme
    bg_color = "#0a192f"
    accent_color = "#2c74b3"
    highlight_color = "#64ffda"
    text_color = "#ffffff"

    # Prepare data for injection
    priority_colors = {1: "#ff6b6b", 2: "#ffa500", 3: "#2c74b3", 4: "#4caf50", 5: "#999999"}

    # Build gantt chart data for Plotly
    gantt_vessels = []
    gantt_berths = []
    gantt_starts = []
    gantt_ends = []
    gantt_colors = []
    for g in gantt_data:
        gantt_vessels.append(g.get("vessel", "Unknown"))
        gantt_berths.append(g.get("berth", "Unknown"))
        gantt_starts.append(g.get("start", ""))
        gantt_ends.append(g.get("end", ""))
        priority = g.get("priority", 3)
        gantt_colors.append(priority_colors.get(priority, "#999999"))

    # Cost per vessel list
    cost_per_vessel_list = [a.get("cost", 0) for a in assignments if a.get("berth_id") is not None]
    vessel_names_cost = [a.get("vessel_name", a["vessel_id"]) for a in assignments if a.get("berth_id") is not None]

    # Berth info
    berth_ids = [b.get("berth_id", "B?") for b in berth_utilization]
    util_pcts = [b.get("utilization_pct", 0) for b in berth_utilization]
    vessels_served = [b.get("vessels_served", 0) for b in berth_utilization]

    # Cost breakdown
    cost_labels = ["Crane Handling", "Waiting", "Delay Penalty"]
    cost_values = [
        cost_breakdown.get("crane_handling_cost", 0),
        cost_breakdown.get("waiting_cost", 0),
        cost_breakdown.get("delay_penalty_cost", 0)
    ]

    # Convergence data
    iterations = [e.get("iteration", 0) for e in optimization_convergence.get("cost_evolution", [])]
    objectives = [e.get("objective_value", 0) for e in optimization_convergence.get("cost_evolution", [])]
    phases = [e.get("phase", "unknown") for e in optimization_convergence.get("cost_evolution", [])]

    # Crane distribution
    crane_dist_labels = list(crane_allocation.get("distribution", {}).keys())
    crane_dist_values = list(crane_allocation.get("distribution", {}).values())

    # Priority analysis
    priority_keys = list(priority_analysis.keys())
    priority_costs = [priority_analysis[k].get("avg_cost", 0) for k in priority_keys]
    priority_counts = [priority_analysis[k].get("count", 0) for k in priority_keys]

    # Build priority rows HTML
    priority_rows = ""
    for key in priority_keys:
        pa = priority_analysis[key]
        priority_rows += f"""            <tr>
                <td>{key}</td>
                <td>{pa.get('count', 0)}</td>
                <td>${pa.get('avg_cost', 0):.2f}</td>
                <td>{pa.get('avg_wait_h', 0):.2f}</td>
                <td>{pa.get('avg_delay_h', 0):.2f}</td>
            </tr>
"""

    # Prepare SVG vessel map for Port Overview
    svg_vessels = ""
    berth_y_pos = {b["id"]: 100 + i * 80 for i, b in enumerate(berths)}
    for i, a in enumerate(assignments):
        if a.get("berth_id") is None:
            continue
        berth_id = a["berth_id"]
        y = berth_y_pos.get(berth_id, 100)
        x = 50 + i * 120
        priority = a.get("priority", 3)
        color = priority_colors.get(priority, "#999999")
        vessel_name = a.get("vessel_name", a["vessel_id"])[:10]
        svg_vessels += f'<rect x="{x}" y="{y}" width="100" height="60" fill="{color}" stroke="{highlight_color}" stroke-width="2" rx="5"><title>{vessel_name} (P{priority})</title></rect>'
        svg_vessels += f'<text x="{x+50}" y="{y+30}" text-anchor="middle" fill="{text_color}" font-size="12" font-weight="bold">{vessel_name}</text>'

    # Main HTML template with 6 tabs
    dashboard_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classical Solver v5.0 - Expert Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, __BG_COLOR__ 0%, #0f2847 100%);
            color: __TEXT_COLOR__;
            font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
            min-height: 100vh;
        }
        .header {
            background: rgba(44, 116, 179, 0.1);
            padding: 30px 20px;
            border-bottom: 2px solid __ACCENT_COLOR__;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        .header h1 {
            color: __HIGHLIGHT_COLOR__;
            font-size: 28px;
            margin-bottom: 10px;
        }
        .header p {
            font-size: 14px;
            color: #aaa;
        }
        .tabs {
            display: flex;
            gap: 2px;
            padding: 20px;
            background: rgba(44, 116, 179, 0.05);
            overflow-x: auto;
            border-bottom: 1px solid __ACCENT_COLOR__;
        }
        .tab-button {
            padding: 12px 24px;
            background: transparent;
            border: none;
            color: #aaa;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s ease;
            border-bottom: 3px solid transparent;
            white-space: nowrap;
        }
        .tab-button:hover {
            color: __TEXT_COLOR__;
            background: rgba(100, 255, 218, 0.05);
        }
        .tab-button.active {
            color: __HIGHLIGHT_COLOR__;
            border-bottom-color: __HIGHLIGHT_COLOR__;
            background: rgba(100, 255, 218, 0.1);
        }
        .tab-content {
            display: none;
            padding: 30px 20px;
            animation: fadeIn 0.3s ease;
        }
        .tab-content.active {
            display: block;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .kpi-card {
            background: linear-gradient(135deg, __ACCENT_COLOR__ 0%, rgba(44, 116, 179, 0.5) 100%);
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid __HIGHLIGHT_COLOR__;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .kpi-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(100, 255, 218, 0.2);
        }
        .kpi-value {
            font-size: 28px;
            font-weight: bold;
            color: __HIGHLIGHT_COLOR__;
            margin-bottom: 8px;
        }
        .kpi-label {
            font-size: 12px;
            text-transform: uppercase;
            color: #ccc;
            letter-spacing: 0.5px;
        }
        .chart-container {
            background: rgba(44, 116, 179, 0.05);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 1px solid __ACCENT_COLOR__;
        }
        .chart-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .chart-box {
            background: rgba(44, 116, 179, 0.05);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid __ACCENT_COLOR__;
        }
        .chart-box h3 {
            color: __HIGHLIGHT_COLOR__;
            font-size: 14px;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        .plotly-chart {
            width: 100%;
            height: 400px;
        }
        .svg-map {
            background: rgba(10, 25, 47, 0.6);
            border: 1px solid __ACCENT_COLOR__;
            border-radius: 8px;
            overflow-x: auto;
            margin-bottom: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid __ACCENT_COLOR__;
        }
        th {
            background: rgba(44, 116, 179, 0.2);
            color: __HIGHLIGHT_COLOR__;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 12px;
        }
        tr:hover {
            background: rgba(100, 255, 218, 0.05);
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 12px;
            background: rgba(44, 116, 179, 0.05);
            margin-bottom: 8px;
            border-radius: 4px;
        }
        .metric-label {
            font-weight: 600;
            color: #ccc;
        }
        .metric-value {
            color: __HIGHLIGHT_COLOR__;
            font-weight: bold;
        }
        .footer {
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 12px;
            border-top: 1px solid __ACCENT_COLOR__;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Classical Solver v5.0 - Expert Dashboard</h1>
        <p>Port Berth Allocation & Quay Crane Assignment | Real-time Optimization Insights</p>
    </div>

    <div class="tabs">
        <button class="tab-button active" onclick="switchTab(event, 'overview')">Port Overview</button>
        <button class="tab-button" onclick="switchTab(event, 'gantt')">Gantt Timeline</button>
        <button class="tab-button" onclick="switchTab(event, 'costs')">Cost Intelligence</button>
        <button class="tab-button" onclick="switchTab(event, 'convergence')">Opt. Convergence</button>
        <button class="tab-button" onclick="switchTab(event, 'berths')">Berth Analytics</button>
        <button class="tab-button" onclick="switchTab(event, 'performance')">Performance</button>
    </div>

    <!-- TAB 1: PORT OVERVIEW -->
    <div id="overview" class="tab-content active">
        <h2 style="color: __HIGHLIGHT_COLOR__; margin-bottom: 20px;">Port Overview - Vessel Map</h2>
        <div class="chart-container">
            <svg class="svg-map" width="100%" height="400">
                <defs>
                    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:__BG_COLOR__;stop-opacity:1" />
                        <stop offset="100%" style="stop-color:#0f2847;stop-opacity:1" />
                    </linearGradient>
                </defs>
                <rect width="100%" height="100%" fill="url(#bgGrad)"/>
                <text x="20" y="30" fill="__HIGHLIGHT_COLOR__" font-size="16" font-weight="bold">Berth Layout (Priority Color-Coded)</text>
                __SVG_VESSELS__
            </svg>
        </div>
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-value">__TOTAL_COST__</div>
                <div class="kpi-label">Total Cost</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">__COST_PER_VESSEL__</div>
                <div class="kpi-label">Cost Per Vessel</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">__COST_PER_TEU__</div>
                <div class="kpi-label">Cost Per TEU</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">__FEASIBLE_COUNT__/__TOTAL_VESSELS__</div>
                <div class="kpi-label">Feasible Assignments</div>
            </div>
        </div>
        <div class="metric-row">
            <span class="metric-label">P1 (Red) = Highest Priority | P5 (Gray) = Lowest</span>
        </div>
    </div>

    <!-- TAB 2: GANTT TIMELINE -->
    <div id="gantt" class="tab-content">
        <h2 style="color: __HIGHLIGHT_COLOR__; margin-bottom: 20px;">Gantt Timeline - Vessel-to-Berth Schedule</h2>
        <div class="chart-container">
            <div id="gantt-chart" class="plotly-chart"></div>
        </div>
    </div>

    <!-- TAB 3: COST INTELLIGENCE -->
    <div id="costs" class="tab-content">
        <h2 style="color: __HIGHLIGHT_COLOR__; margin-bottom: 20px;">Cost Intelligence Dashboard</h2>
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-value">$__TOTAL_COST__</div>
                <div class="kpi-label">Total Cost</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">$__COST_PER_VESSEL__</div>
                <div class="kpi-label">Avg Cost Per Vessel</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">$__COST_PER_TEU__</div>
                <div class="kpi-label">Cost Per TEU</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">__UTILIZATION__</div>
                <div class="kpi-label">Port Utilization</div>
            </div>
        </div>
        <div class="chart-row">
            <div class="chart-box">
                <h3>Cost Breakdown</h3>
                <div id="cost-pie" class="plotly-chart"></div>
            </div>
            <div class="chart-box">
                <h3>Cost Per Vessel</h3>
                <div id="cost-bar" class="plotly-chart"></div>
            </div>
        </div>
    </div>

    <!-- TAB 4: OPTIMIZATION CONVERGENCE -->
    <div id="convergence" class="tab-content">
        <h2 style="color: __HIGHLIGHT_COLOR__; margin-bottom: 20px;">Optimization Convergence</h2>
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-value">__IMPROVEMENT_PCT__</div>
                <div class="kpi-label">Improvement Over Greedy</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">$__GREEDY_COST__</div>
                <div class="kpi-label">Greedy Initial Cost</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">$__FINAL_COST__</div>
                <div class="kpi-label">Final Optimized Cost</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">__ITERATIONS__</div>
                <div class="kpi-label">2-Opt Iterations</div>
            </div>
        </div>
        <div class="chart-container">
            <div id="convergence-chart" class="plotly-chart"></div>
        </div>
        <table>
            <tr>
                <th>Phase</th>
                <th>Cost Reduction</th>
                <th>Improvements Made</th>
            </tr>
            <tr>
                <td>Greedy Initial</td>
                <td>$__GREEDY_COST__</td>
                <td>Baseline</td>
            </tr>
            <tr>
                <td>2-Opt Search</td>
                <td>-$__TWOOPT_SAVINGS__</td>
                <td>Berth swaps</td>
            </tr>
            <tr>
                <td>Or-Opt Moves</td>
                <td>-$__OROPT_SAVINGS__</td>
                <td>__OROPT_IMPROVEMENTS__</td>
            </tr>
            <tr>
                <td>3-Opt Rotations</td>
                <td>-$__THREEOPT_SAVINGS__</td>
                <td>__THREEOPT_IMPROVEMENTS__</td>
            </tr>
            <tr>
                <td>Crane Rebalance</td>
                <td>-$__CRANE_SAVINGS__</td>
                <td>__CRANE_IMPROVEMENTS__</td>
            </tr>
        </table>
    </div>

    <!-- TAB 5: BERTH ANALYTICS -->
    <div id="berths" class="tab-content">
        <h2 style="color: __HIGHLIGHT_COLOR__; margin-bottom: 20px;">Berth Utilization Analytics</h2>
        <div class="chart-row">
            <div class="chart-box">
                <h3>Utilization & Capacity</h3>
                <div id="util-chart" class="plotly-chart"></div>
            </div>
            <div class="chart-box">
                <h3>Crane Distribution</h3>
                <div id="crane-chart" class="plotly-chart"></div>
            </div>
        </div>
    </div>

    <!-- TAB 6: PERFORMANCE SUMMARY -->
    <div id="performance" class="tab-content">
        <h2 style="color: __HIGHLIGHT_COLOR__; margin-bottom: 20px;">Performance Summary</h2>
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-value">__WALL_TIME__s</div>
                <div class="kpi-label">Total Runtime</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">__MAKESPAN__h</div>
                <div class="kpi-label">Makespan (hours)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">__AVG_CRANES__</div>
                <div class="kpi-label">Avg Cranes Per Vessel</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">__TOTAL_WAIT__h</div>
                <div class="kpi-label">Total Waiting Time</div>
            </div>
        </div>
        <div class="chart-row">
            <div class="chart-box">
                <h3>Priority Analysis</h3>
                <div id="priority-chart" class="plotly-chart"></div>
            </div>
        </div>
        <table>
            <tr>
                <th>Priority</th>
                <th>Count</th>
                <th>Avg Cost</th>
                <th>Avg Wait (hrs)</th>
                <th>Avg Delay (hrs)</th>
            </tr>
            __PRIORITY_ROWS__
        </table>
        <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid __ACCENT_COLOR__;"></div>
        <div class="metric-row">
            <span class="metric-label">Algorithm</span>
            <span class="metric-value">Greedy-2Opt-OrOpt-3Opt-CraneRebalance</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Solver Version</span>
            <span class="metric-value">5.0</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Problem Size</span>
            <span class="metric-value">__TOTAL_VESSELS__ vessels × __TOTAL_BERTHS__ berths</span>
        </div>
    </div>

    <div class="footer">
        Classical Solver v5.0 | Berth Allocation + Quay Crane Assignment | Generated with dynamic expert dashboard
    </div>

    <script>
        const DATA = __DATA_JSON__;

        function switchTab(evt, tabName) {
            const tabs = document.querySelectorAll('.tab-content');
            tabs.forEach(tab => tab.classList.remove('active'));

            const buttons = document.querySelectorAll('.tab-button');
            buttons.forEach(btn => btn.classList.remove('active'));

            document.getElementById(tabName).classList.add('active');
            evt.currentTarget.classList.add('active');
        }

        // Gantt Chart
        if (DATA.gantt_data && DATA.gantt_data.length > 0) {
            const ganttTrace = {
                x: DATA.gantt_data.map(g => g.end_time_hours - g.start_time_hours),
                y: DATA.gantt_data.map(g => g.vessel + ' @ ' + g.berth),
                base: DATA.gantt_data.map(g => g.start_time_hours),
                type: 'bar',
                orientation: 'h',
                marker: { color: DATA.gantt_data.map(g => DATA.priority_colors[g.priority]) },
                hovertemplate: '%{y}<br>Duration: %{x}h<extra></extra>'
            };
            const ganttLayout = {
                title: 'Vessel-to-Berth Assignments',
                xaxis: { title: 'Time (hours)' },
                yaxis: { title: 'Vessel @ Berth' },
                plot_bgcolor: 'rgba(44, 116, 179, 0.1)',
                paper_bgcolor: '__BG_COLOR__',
                font: { color: '__TEXT_COLOR__' },
                margin: { l: 200 }
            };
            Plotly.newPlot('gantt-chart', [ganttTrace], ganttLayout, { responsive: true });
        }

        // Cost Pie Chart
        const pieData = [{
            values: DATA.cost_values,
            labels: DATA.cost_labels,
            type: 'pie',
            marker: { colors: ['__HIGHLIGHT_COLOR__', '__ACCENT_COLOR__', '#ff6b6b'] }
        }];
        const pieLayout = {
            title: 'Cost Breakdown',
            plot_bgcolor: 'rgba(44, 116, 179, 0.1)',
            paper_bgcolor: '__BG_COLOR__',
            font: { color: '__TEXT_COLOR__' }
        };
        Plotly.newPlot('cost-pie', pieData, pieLayout, { responsive: true });

        // Cost Bar Chart
        const barData = [{
            x: DATA.vessel_names_cost,
            y: DATA.cost_per_vessel_list,
            type: 'bar',
            marker: { color: '__ACCENT_COLOR__' }
        }];
        const barLayout = {
            title: 'Cost Distribution by Vessel',
            xaxis: { title: 'Vessel' },
            yaxis: { title: 'Cost ($)' },
            plot_bgcolor: 'rgba(44, 116, 179, 0.1)',
            paper_bgcolor: '__BG_COLOR__',
            font: { color: '__TEXT_COLOR__' },
            margin: { b: 100 }
        };
        Plotly.newPlot('cost-bar', barData, barLayout, { responsive: true });

        // Convergence Chart
        const convData = [{
            x: DATA.iterations,
            y: DATA.objectives,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '__HIGHLIGHT_COLOR__', width: 3 },
            marker: { size: 8, color: '__ACCENT_COLOR__' }
        }];
        const convLayout = {
            title: 'Cost Evolution Across Optimization Phases',
            xaxis: { title: 'Iteration / Phase' },
            yaxis: { title: 'Objective Value ($)' },
            plot_bgcolor: 'rgba(44, 116, 179, 0.1)',
            paper_bgcolor: '__BG_COLOR__',
            font: { color: '__TEXT_COLOR__' }
        };
        Plotly.newPlot('convergence-chart', convData, convLayout, { responsive: true });

        // Utilization Chart
        const utilData = [
            {
                x: DATA.berth_ids,
                y: DATA.util_pcts,
                name: 'Utilization (%)',
                type: 'bar',
                marker: { color: '__HIGHLIGHT_COLOR__' },
                yaxis: 'y'
            },
            {
                x: DATA.berth_ids,
                y: DATA.vessels_served,
                name: 'Vessels Served',
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '__ACCENT_COLOR__', width: 3 },
                marker: { size: 10 },
                yaxis: 'y2'
            }
        ];
        const utilLayout = {
            title: 'Berth Utilization & Capacity',
            xaxis: { title: 'Berth' },
            yaxis: { title: 'Utilization (%)' },
            yaxis2: { title: 'Vessels Served', overlaying: 'y', side: 'right' },
            plot_bgcolor: 'rgba(44, 116, 179, 0.1)',
            paper_bgcolor: '__BG_COLOR__',
            font: { color: '__TEXT_COLOR__' }
        };
        Plotly.newPlot('util-chart', utilData, utilLayout, { responsive: true });

        // Crane Distribution Chart
        const craneData = [{
            x: DATA.crane_dist_labels,
            y: DATA.crane_dist_values,
            type: 'bar',
            marker: { color: '__HIGHLIGHT_COLOR__' }
        }];
        const craneLayout = {
            title: 'Crane Allocation Distribution',
            xaxis: { title: 'Cranes per Vessel' },
            yaxis: { title: 'Count' },
            plot_bgcolor: 'rgba(44, 116, 179, 0.1)',
            paper_bgcolor: '__BG_COLOR__',
            font: { color: '__TEXT_COLOR__' }
        };
        Plotly.newPlot('crane-chart', craneData, craneLayout, { responsive: true });

        // Priority Analysis Chart
        const priorityData = [
            {
                x: DATA.priority_keys,
                y: DATA.priority_costs,
                name: 'Avg Cost',
                type: 'bar',
                marker: { color: '__ACCENT_COLOR__' },
                yaxis: 'y'
            },
            {
                x: DATA.priority_keys,
                y: DATA.priority_counts,
                name: 'Vessel Count',
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '__HIGHLIGHT_COLOR__', width: 3 },
                marker: { size: 10 },
                yaxis: 'y2'
            }
        ];
        const priorityLayout = {
            title: 'Cost & Distribution by Priority',
            xaxis: { title: 'Priority Level' },
            yaxis: { title: 'Avg Cost ($)' },
            yaxis2: { title: 'Vessel Count', overlaying: 'y', side: 'right' },
            plot_bgcolor: 'rgba(44, 116, 179, 0.1)',
            paper_bgcolor: '__BG_COLOR__',
            font: { color: '__TEXT_COLOR__' }
        };
        Plotly.newPlot('priority-chart', priorityData, priorityLayout, { responsive: true });
    </script>
</body>
</html>
"""

    # Prepare data dictionary for injection
    # Convert times to hours for Gantt
    gantt_data_with_hours = []
    for g in gantt_data:
        gantt_data_with_hours.append({
            "vessel": g.get("vessel", "Unknown"),
            "berth": g.get("berth", "Unknown"),
            "start_time_hours": _iso_to_hours(g.get("start", "")),
            "end_time_hours": _iso_to_hours(g.get("end", "")),
            "priority": g.get("priority", 3)
        })

    data_dict = {
        "gantt_data": gantt_data_with_hours,
        "priority_colors": {1: "#ff6b6b", 2: "#ffa500", 3: "#2c74b3", 4: "#4caf50", 5: "#999999"},
        "cost_values": cost_values,
        "cost_labels": cost_labels,
        "vessel_names_cost": vessel_names_cost,
        "cost_per_vessel_list": cost_per_vessel_list,
        "iterations": iterations,
        "objectives": objectives,
        "phases": phases,
        "berth_ids": berth_ids,
        "util_pcts": util_pcts,
        "vessels_served": vessels_served,
        "crane_dist_labels": crane_dist_labels,
        "crane_dist_values": crane_dist_values,
        "priority_keys": priority_keys,
        "priority_costs": priority_costs,
        "priority_counts": priority_counts
    }

    # Perform all template replacements
    html = dashboard_template
    html = html.replace("__BG_COLOR__", bg_color)
    html = html.replace("__TEXT_COLOR__", text_color)
    html = html.replace("__ACCENT_COLOR__", accent_color)
    html = html.replace("__HIGHLIGHT_COLOR__", highlight_color)

    # KPI values
    html = html.replace("__TOTAL_COST__", f"{cost_breakdown.get('total_cost', 0):.2f}")
    html = html.replace("__COST_PER_VESSEL__", f"{cost_breakdown.get('cost_per_vessel', 0):.2f}")
    html = html.replace("__COST_PER_TEU__", f"{cost_breakdown.get('cost_per_teu', 0):.4f}")
    html = html.replace("__UTILIZATION__", f"{schedule_metrics.get('utilization', 0):.2%}")
    html = html.replace("__FEASIBLE_COUNT__", str(schedule_metrics.get('feasible_assignments', 0)))
    html = html.replace("__TOTAL_VESSELS__", str(schedule_metrics.get('feasible_assignments', 0) + schedule_metrics.get('infeasible_assignments', 0)))

    # Convergence metrics
    html = html.replace("__IMPROVEMENT_PCT__", f"{optimization_convergence.get('improvement_pct', 0):.2f}%")
    html = html.replace("__GREEDY_COST__", f"{optimization_convergence.get('greedy_initial_cost', 0):.2f}")
    html = html.replace("__FINAL_COST__", f"{optimization_convergence.get('final_optimized_cost', 0):.2f}")
    html = html.replace("__ITERATIONS__", str(computation_metrics.get('iterations', 0)))

    # Savings calculations
    twoopt_savings = optimization_convergence.get('greedy_initial_cost', 0) - optimization_convergence.get('two_opt_cost', 0)
    oropt_savings = optimization_convergence.get('two_opt_cost', 0) - (optimization_convergence.get('final_optimized_cost', 0) if computation_metrics.get('or_opt_improvements', 0) > 0 else optimization_convergence.get('two_opt_cost', 0))
    threeopt_savings = 0  # Placeholder
    crane_savings = optimization_convergence.get('final_optimized_cost', 0) if computation_metrics.get('crane_rebalance_improvements', 0) > 0 else 0

    html = html.replace("__TWOOPT_SAVINGS__", f"{max(0, twoopt_savings):.2f}")
    html = html.replace("__OROPT_SAVINGS__", f"{max(0, oropt_savings):.2f}")
    html = html.replace("__THREEOPT_SAVINGS__", f"{max(0, threeopt_savings):.2f}")
    html = html.replace("__CRANE_SAVINGS__", f"{max(0, crane_savings):.2f}")
    html = html.replace("__OROPT_IMPROVEMENTS__", str(computation_metrics.get('or_opt_improvements', 0)))
    html = html.replace("__THREEOPT_IMPROVEMENTS__", str(computation_metrics.get('three_opt_improvements', 0)))
    html = html.replace("__CRANE_IMPROVEMENTS__", str(computation_metrics.get('crane_rebalance_improvements', 0)))

    # Performance metrics
    html = html.replace("__WALL_TIME__", f"{computation_metrics.get('wall_time_s', 0):.3f}")
    html = html.replace("__MAKESPAN__", f"{schedule_metrics.get('makespan', 0):.2f}")
    html = html.replace("__AVG_CRANES__", f"{crane_allocation.get('avg_cranes_per_vessel', 0):.2f}")
    html = html.replace("__TOTAL_WAIT__", f"{schedule_metrics.get('total_waiting_time', 0):.2f}")

    # Vessel and berth counts
    html = html.replace("__TOTAL_BERTHS__", str(len(berths)))

    # SVG and priority rows
    html = html.replace("__SVG_VESSELS__", svg_vessels)
    html = html.replace("__PRIORITY_ROWS__", priority_rows)

    # Data injection as JSON
    html = html.replace("__DATA_JSON__", json.dumps(data_dict))

    # Write the single comprehensive dashboard
    with open("additional_output/01_expert_dashboard.html", "w") as f:
        f.write(html)


Claude is active in this tab group
Open chat
Dismiss
