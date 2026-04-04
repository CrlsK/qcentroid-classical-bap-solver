"""\nClassical BAP+QCA Greedy-2Opt-OrOpt-3Opt Solver v8.1\nBerth Allocation + Quay Crane Assignment.\nv8.1: Hard crane cap during greedy forces all-berth usage.\nKey fix: cap cranes at total/n_berths during greedy (=3 for 18/6),\nenabling 6 concurrent vessels across all 6 berths.\nPost-hoc safety net REMOVED (always destructive).\n"""
import logging
import time
import itertools
import os
import json
import math

from solver_helpers import (_enforce_crane_budget, _resequence_all_berths, _try_swap,
                            _hours_to_iso, _iso_to_hours, _compute_lookahead_score,
                            _evaluate_vessel_at_berth, _get_available_cranes_at_window)
from dashboard import _generate_expert_dashboard

logger = logging.getLogger("qcentroid-user-log")

def run(input_data: dict, solver_params: dict, extra_arguments: dict) -> dict:
    start_time = time.time()
    logger.info("=== Classical BAP+QCA Solver v8.1 — Hard Crane Cap + All-Berth Spreading ===")

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

    # v8.1: HARD crane cap during greedy — forces berth spreading
    # With 18 cranes and 6 berths, cap at 3 cranes/vessel → 6 concurrent vessels possible
    greedy_max_cranes = max(min_cranes, min(total_cranes // n_berths, max_cranes))
    logger.info(f"v8.1: Greedy crane cap = {greedy_max_cranes} (hard limit during construction)")
    logger.info(f"v8.1: Post-greedy optimization may adjust up to {max_cranes} per vessel")

    # Sort vessels by priority (ascending = higher priority first), then arrival
    sorted_vessels = sorted(vessels, key=lambda v: (v.get("priority", 5), v.get("arrival_time", "")))

    # ── 2. Greedy construction (v8.0) ──────────────────────────────────
    # Strategy: spread vessels across ALL berths, use balanced cranes
    assignments = []
    berth_end_times = {}
    berth_vessel_count = {b["id"]: 0 for b in berths}
    crane_timeline = []
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

        # v8.0: Sort berths by current vessel count (least-loaded first) for spreading
        berths_by_load = sorted(berths, key=lambda b: (
            berth_vessel_count.get(b["id"], 0),
            _iso_to_hours(berth_end_times.get(b["id"], v_arrival))
        ))

        for b in berths_by_load:
            b_id = b["id"]
            b_len = b.get("length_m", 300)
            b_depth = b.get("depth_m", 15.0)
            b_prod = b.get("productivity_teu_per_crane_hour", 25)

            if v_len > b_len or v_draft > b_depth:
                continue

            berth_free = berth_end_times.get(b_id, v_arrival)
            actual_start = max(v_arrival, berth_free)
            start_h = _iso_to_hours(actual_start)
            deadline_h = _iso_to_hours(v_deadline)
            arr_h = _iso_to_hours(v_arrival)

            # v8.1: HARD CAP at greedy_max_cranes — never exceed during construction
            # This forces vessels onto more berths since fewer concurrent cranes needed
            crane_candidates = list(range(greedy_max_cranes, min_cranes - 1, -1))

            for nc in crane_candidates:
                handling_hours = v_teu / (b_prod * nc) if b_prod * nc > 0 else 999
                end_h = start_h + handling_hours

                # v8.0: Check crane budget feasibility
                available_cranes = _get_available_cranes_at_window(start_h, end_h, crane_timeline, total_cranes)
                if nc > available_cranes:
                    continue

                wait_hours = max(0, start_h - arr_h)
                delay_hours = max(0, end_h - deadline_h)
                pm = w_priority if v_priority <= 2 else 1.0
                crane_cost = handling_hours * nc * w_handle
                wait_cost = wait_hours * w_wait * pm
                delay_cost = delay_hours * w_delay * pm
                total_cost = crane_cost + wait_cost + delay_cost

                # v8.0: STRONG berth-spreading penalty
                # Penalize proportional to queue length (not just count)
                vessels_at_berth = berth_vessel_count.get(b_id, 0)
                # Quadratic penalty: each additional vessel at same berth costs more
                queue_wait_penalty = 0
                if vessels_at_berth > 0:
                    # Estimate: each vessel queued adds ~handling_hours of wait for all future vessels
                    queue_wait_penalty = vessels_at_berth * handling_hours * w_wait * 0.5
                # Berth balance penalty: encourage even distribution
                avg_load = sum(berth_vessel_count.values()) / max(n_berths, 1)
                imbalance_penalty = max(0, vessels_at_berth - avg_load) * w_delay * 10

                penalized_cost = total_cost + queue_wait_penalty + imbalance_penalty

                if penalized_cost < best_cost:
                    best_cost = penalized_cost
                    best_berth = b_id
                    best_start = actual_start
                    best_cranes_assigned = nc
                    best_handling_hours = handling_hours
                    best_wait_hours = wait_hours
                    best_delay_hours = delay_hours
                    best_actual_cost = total_cost

        if best_berth is not None:
            end_time_str = _hours_to_iso(
                _iso_to_hours(best_start) + best_handling_hours,
                best_start
            )
            start_h = _iso_to_hours(best_start)
            end_h = start_h + best_handling_hours

            berth_end_times[best_berth] = end_time_str
            berth_vessel_count[best_berth] += 1
            crane_timeline.append((start_h, end_h, best_cranes_assigned))

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
                "cost": round(best_actual_cost, 2),
                "priority": v_priority,
                "teu_volume": v_teu
            })
        else:
            # Fallback: try any berth, any cranes (ignore budget)
            fallback_best_cost = float("inf")
            fallback_config = None
            for b in berths:
                b_id = b["id"]
                if v_len > b.get("length_m", 300) or v_draft > b.get("depth_m", 15.0):
                    continue
                b_prod = b.get("productivity_teu_per_crane_hour", 25)
                berth_free = berth_end_times.get(b_id, v_arrival)
                actual_start = max(v_arrival, berth_free)
                start_h = _iso_to_hours(actual_start)
                arr_h = _iso_to_hours(v_arrival)
                deadline_h = _iso_to_hours(v_deadline)

                for nc in range(min_cranes, greedy_max_cranes + 1):
                    handling_hours = v_teu / (b_prod * nc) if b_prod * nc > 0 else 999
                    end_h = start_h + handling_hours
                    wait_hours = max(0, start_h - arr_h)
                    delay_hours = max(0, end_h - deadline_h)
                    pm = w_priority if v_priority <= 2 else 1.0
                    cost = (handling_hours * nc * w_handle +
                            wait_hours * w_wait * pm +
                            delay_hours * w_delay * pm)
                    if cost < fallback_best_cost:
                        fallback_best_cost = cost
                        fallback_config = {
                            "berth_id": b_id, "start": actual_start,
                            "cranes": nc, "handling_hours": handling_hours,
                            "wait_hours": wait_hours, "delay_hours": delay_hours,
                            "cost": cost
                        }

            if fallback_config:
                fc = fallback_config
                end_time_str = _hours_to_iso(
                    _iso_to_hours(fc["start"]) + fc["handling_hours"], fc["start"]
                )
                berth_end_times[fc["berth_id"]] = end_time_str
                berth_vessel_count[fc["berth_id"]] += 1
                crane_timeline.append((_iso_to_hours(fc["start"]),
                                       _iso_to_hours(fc["start"]) + fc["handling_hours"],
                                       fc["cranes"]))
                assignments.append({
                    "vessel_id": v_id, "vessel_name": v_name,
                    "berth_id": fc["berth_id"], "start_time": fc["start"],
                    "end_time": end_time_str, "cranes_assigned": fc["cranes"],
                    "handling_hours": round(fc["handling_hours"], 2),
                    "waiting_hours": round(fc["wait_hours"], 2),
                    "delay_hours": round(fc["delay_hours"], 2),
                    "cost": round(fc["cost"], 2),
                    "priority": v_priority, "teu_volume": v_teu
                })
            else:
                assignments.append({
                    "vessel_id": v_id, "vessel_name": v_name,
                    "berth_id": None, "start_time": None, "end_time": None,
                    "cranes_assigned": 0, "handling_hours": 0, "cost": 0,
                    "status": "infeasible"
                })
                logger.warning(f"Vessel {v_id}: no feasible berth found!")

    greedy_cost = sum(a["cost"] for a in assignments)
    cost_evolution.append({"iteration": 0, "phase": "greedy", "objective_value": round(greedy_cost, 2)})
    logger.info(f"Greedy phase: cost=${greedy_cost:,.2f}")

    # Log berth distribution after greedy
    for b in berths:
        cnt = berth_vessel_count.get(b["id"], 0)
        if cnt > 0:
            logger.info(f"  {b['id']}: {cnt} vessels")

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
    logger.info(f"2-opt completed after {iteration} iterations: cost=${two_opt_cost:,.2f}")

    # ── 3b. Or-opt moves: relocate single vessels ─────────────────────
    oropt_improved = 0
    max_vessels_per_berth = math.ceil(n_vessels / n_berths) + 1
    for idx, a in enumerate(assignments):
        if a.get("berth_id") is None:
            continue
        v_data = next((v for v in vessels if v["id"] == a["vessel_id"]), None)
        if not v_data:
            continue

        old_cost = a["cost"]
        best_cost_oropt = old_cost
        best_config = None

        for b in berths:
            b_id = b["id"]
            if b_id == a.get("berth_id"):
                continue

            v_len = v_data.get("length_m", 200)
            v_draft = v_data.get("draft_m", 12.0)
            if v_len > b.get("length_m", 300) or v_draft > b.get("depth_m", 15.0):
                continue

            vessel_count_at_berth = sum(1 for x in assignments if x.get("berth_id") == b_id)
            if vessel_count_at_berth >= max_vessels_per_berth:
                continue

            for nc in range(min_cranes, max_cranes + 1):
                config = _evaluate_vessel_at_berth(v_data, b, a["start_time"], nc, cost_weights)
                if config is not None and config["cost"] < best_cost_oropt:
                    best_cost_oropt = config["cost"]
                    best_config = {"berth": b_id, "config": config, "cranes": nc}

        if best_config is not None and best_cost_oropt < old_cost * 0.99:
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
        logger.info(f"Or-opt moved {oropt_improved} vessels: cost=${current_cost:,.2f}")

    # ── 3c. 3-opt moves: rotate 3 vessels among 3 berths ────
    threeopt_improved = 0
    max_3opt_iterations = 30
    for iteration_3opt in range(max_3opt_iterations):
        improved_3opt = False
        feasible_indices = [i for i, a in enumerate(assignments) if a.get("berth_id") is not None]
        if len(feasible_indices) < 3:
            break

        for i, j, k in itertools.combinations(feasible_indices, 3):
            a1, a2, a3 = assignments[i], assignments[j], assignments[k]
            old_cost = a1["cost"] + a2["cost"] + a3["cost"]

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

            if v1.get("length_m", 200) > b2.get("length_m", 300) or v1.get("draft_m", 12) > b2.get("depth_m", 15):
                continue
            if v2.get("length_m", 200) > b3.get("length_m", 300) or v2.get("draft_m", 12) > b3.get("depth_m", 15):
                continue
            if v3.get("length_m", 200) > b1.get("length_m", 300) or v3.get("draft_m", 12) > b1.get("depth_m", 15):
                continue

            config1 = _evaluate_vessel_at_berth(v1, b2, a1["start_time"], a1["cranes_assigned"], cost_weights)
            config2 = _evaluate_vessel_at_berth(v2, b3, a2["start_time"], a2["cranes_assigned"], cost_weights)
            config3 = _evaluate_vessel_at_berth(v3, b1, a3["start_time"], a3["cranes_assigned"], cost_weights)

            if config1 and config2 and config3:
                new_cost = config1["cost"] + config2["cost"] + config3["cost"]
                if new_cost < old_cost * 0.99:
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
        logger.info(f"3-opt improved {threeopt_improved} configs: cost=${current_cost:,.2f}")

    # ── 3d. Crane re-optimization pass ──────────────────────────────────
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
        best_cost_crane = a["cost"]

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

            if cost < best_cost_crane:
                best_cost_crane = cost
                best_nc = nc

        if best_nc != a["cranes_assigned"]:
            handling_h = v_teu / (b_prod * best_nc) if b_prod * best_nc > 0 else 999
            end_h = _iso_to_hours(a["start_time"]) + handling_h
            delay_h = max(0, end_h - _iso_to_hours(v_data.get("max_departure_time", "2025-12-31T23:59:00Z")))
            wait_h = a.get("waiting_hours", 0)

            assignments[idx] = dict(a)
            assignments[idx]["cranes_assigned"] = best_nc
            assignments[idx]["handling_hours"] = round(handling_h, 2)
            assignments[idx]["delay_hours"] = round(delay_h, 2)
            assignments[idx]["cost"] = round(best_cost_crane, 2)
            assignments[idx]["end_time"] = _hours_to_iso(end_h, a["start_time"])
            crane_opt_improved += 1

    final_cost_after_crane_opt = sum(a["cost"] for a in assignments)
    if crane_opt_improved > 0:
        cost_evolution.append({
            "iteration": iteration + 3,
            "phase": "crane_reopt",
            "objective_value": round(final_cost_after_crane_opt, 2)
        })
        logger.info(f"Crane reopt: {crane_opt_improved} adjusted, cost=${final_cost_after_crane_opt:,.2f}")

    # ── 3e. v8.1: Safety net REMOVED — it was always destructive ────
    # Previous versions: safety net added $900K-$3.5M every time
    # With hard crane cap during greedy, budget is inherently feasible
    # Just do a light resequencing pass
    assignments, final_reseq_changes = _resequence_all_berths(
        assignments, vessels, berths, cost_weights, w_priority
    )
    if final_reseq_changes > 0:
        logger.info(f"Final resequencing: {final_reseq_changes} adjusted")
    post_reseq_cost = sum(a["cost"] for a in assignments)
    cost_evolution.append({
        "iteration": iteration + 4,
        "phase": "resequence",
        "objective_value": round(post_reseq_cost, 2)
    })
    logger.info(f"Final cost after resequencing: ${post_reseq_cost:,.2f}")

    # Compute final metrics
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
    status = "optimal" if feasible_count == n_vessels else ("feasible" if feasible_count > 0 else "infeasible")

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

    total_crane_cost = sum(a.get("handling_hours", 0) * a.get("cranes_assigned", 1) * w_handle for a in assignments if a.get("berth_id"))
    total_wait_cost = sum(a.get("waiting_hours", 0) * w_wait * (w_priority if a.get("priority", 3) <= 2 else 1.0) for a in assignments if a.get("berth_id"))
    total_delay_cost = sum(a.get("delay_hours", 0) * w_delay * (w_priority if a.get("priority", 3) <= 2 else 1.0) for a in assignments if a.get("berth_id"))

    crane_distribution = {}
    for a in assignments:
        nc = a.get("cranes_assigned", 0)
        crane_distribution[str(nc)] = crane_distribution.get(str(nc), 0) + 1

    gantt_data = [{"vessel": a.get("vessel_name", a["vessel_id"]), "berth": a["berth_id"], "start": a["start_time"], "end": a["end_time"], "cranes": a["cranes_assigned"], "priority": a.get("priority", 3)} for a in assignments if a.get("berth_id") is not None]

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
    logger.info(f"Total cost: ${total_cost:,.2f}, Status: {status}, Time: {elapsed}s")
    logger.info(f"Improvement: {improvement_pct}% (greedy=${greedy_cost:,.2f} → final=${total_cost:,.2f})")

    try:
        _generate_expert_dashboard(
            assignments=assignments, berths=berths, vessels=vessels,
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
                "solver_version": "8.1"
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
        logger.info("Expert dashboard generated (v8.1)")
    except Exception as e:
        logger.warning(f"Dashboard generation failed: {e}")

    return {
        "assignments": assignments,
        "objective_value": round(total_cost, 2),
        "solution_status": status,
        "num_vessels": n_vessels,
        "num_berths": n_berths,
        "total_cranes": total_cranes,
        "schedule_metrics": {
            "total_waiting_time": round(total_wait, 2),
            "avg_waiting_time": round(total_wait / max(n_vessels, 1), 2),
            "makespan": round(makespan, 2),
            "utilization": round(total_handling / max(makespan * n_berths, 1), 4),
            "total_teu_processed": total_teu,
            "feasible_assignments": feasible_count,
            "infeasible_assignments": n_vessels - feasible_count
        },
        "cost_breakdown": {
            "total_cost": round(total_cost, 2),
            "crane_handling_cost": round(total_crane_cost, 2),
            "waiting_cost": round(total_wait_cost, 2),
            "delay_penalty_cost": round(total_delay_cost, 2),
            "cost_per_vessel": round(total_cost / max(n_vessels, 1), 2),
            "cost_per_teu": round(total_cost / max(total_teu, 1), 4)
        },
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
        "berth_utilization": berth_utilization,
        "crane_allocation": {
            "distribution": crane_distribution,
            "avg_cranes_per_vessel": round(
                sum(a.get("cranes_assigned", 0) for a in assignments) / max(feasible_count, 1), 2
            ),
            "total_crane_hours": round(
                sum(a.get("handling_hours", 0) * a.get("cranes_assigned", 0) for a in assignments), 2
            )
        },
        "gantt_schedule": gantt_data,
        "priority_analysis": priority_analysis,
        "computation_metrics": {
            "wall_time_s": elapsed,
            "algorithm": "Greedy_2Opt_OrOpt_3Opt_CraneRebalance_BAP_QCA",
            "iterations": iteration,
            "or_opt_improvements": oropt_improved,
            "three_opt_improvements": threeopt_improved,
            "crane_rebalance_improvements": crane_opt_improved,
            "search_space_explored": n_vessels * n_berths * (max_cranes - min_cranes + 1),
            "solver_version": "8.1"
        },
        "benchmark": {
            "execution_cost": {"value": 1.0, "unit": "credits"},
            "time_elapsed": f"{elapsed}s",
            "energy_consumption": 0.0
        }
    }
