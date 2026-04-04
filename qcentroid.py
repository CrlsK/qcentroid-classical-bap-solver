"""
Classical BAP+QCA Greedy-2Opt-OrOpt-3Opt Solver v6.0
Berth Allocation + Quay Crane Assignment.
v6.0: Global crane budget enforcement, priority-weighted crane allocation, re-sequencing.
"""
import logging
import time
import itertools
import os
import json
import math

from solver_helpers import (_enforce_crane_budget, _resequence_all_berths, _try_swap,
                            _hours_to_iso, _iso_to_hours, _compute_lookahead_score,
                            _evaluate_vessel_at_berth)
from dashboard import _generate_expert_dashboard

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

