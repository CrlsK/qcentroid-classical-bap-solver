"""
Classical BAP+QCA Solver v10.0
Berth Allocation + Quay Crane Assignment.
v10.0: Tuned SA with priority-aware moves + restart mechanism.
  - SA tuning: slower cooling (0.995), lower T_min (10), 25s limit
  - Priority-weighted move selection (favors P1/P2 vessel optimization)
  - Restart from best solution when SA stalls (reheat)
  - Compound moves: simultaneous berth+crane adjustments
"""
import logging
import time
import itertools
import os
import json
import math
import random

from solver_helpers import (_enforce_crane_budget, _resequence_all_berths, _try_swap,
                            _hours_to_iso, _iso_to_hours, _compute_lookahead_score,
                            _evaluate_vessel_at_berth, _get_available_cranes_at_window)
from dashboard import _generate_expert_dashboard

logger = logging.getLogger("qcentroid-user-log")


def _recalc_full_cost(assignments, vessels, berths, cost_weights, w_priority_mult):
    """Resequence all berths and return total cost (accurate cascading)."""
    assignments, _ = _resequence_all_berths(assignments, vessels, berths, cost_weights, w_priority_mult)
    return sum(a.get("cost", 0) for a in assignments), assignments


def _sa_neighbor(assignments, vessels, berths, cost_weights, cranes_cfg, w_priority_mult, rng):
    """Generate SA neighbor via priority-weighted move selection (v10.0)."""
    candidate = [dict(a) for a in assignments]
    feasible = [i for i, a in enumerate(candidate) if a.get("berth_id") is not None]

    if len(feasible) < 2:
        return candidate, "noop"

    min_cranes = cranes_cfg.get("min_per_vessel", 1)
    max_cranes = cranes_cfg.get("max_per_vessel", 6)

    # v10.0: Priority-weighted move selection — favor moves on high-cost vessels
    costs = [(i, candidate[i].get("cost", 0)) for i in feasible]
    costs.sort(key=lambda x: -x[1])  # highest cost first
    # 60% chance to pick from top-3 costliest vessels for targeted moves
    high_cost_indices = [c[0] for c in costs[:3]]

    move_type = rng.randint(0, 4)
    target_idx = None

    if move_type < 3 and rng.random() < 0.6 and high_cost_indices:
        target_idx = rng.choice(high_cost_indices)
    else:
        target_idx = rng.choice(feasible)

    if move_type == 0:
        # Compound Move: Berth + Crane adjustment for target vessel
        a = candidate[target_idx]
        b_id = a["berth_id"]
        other_berths = [b["id"] for b in berths if b["id"] != b_id]
        if other_berths:
            new_b_id = rng.choice(other_berths)
            new_nc = max(min_cranes, min(max_cranes, a["cranes_assigned"] + rng.choice([-1, 1])))
            candidate[target_idx]["berth_id"] = new_b_id
            candidate[target_idx]["cranes_assigned"] = new_nc
            return candidate, "compound_move"

    elif move_type == 1:
        # Swap berths of two random vessels
        if len(feasible) >= 2:
            i, j = rng.sample(feasible, 2)
            a1, a2 = candidate[i], candidate[j]
            b1, b2 = a1["berth_id"], a2["berth_id"]
            if b1 != b2:
                v1 = next((v for v in vessels if v["id"] == a1["vessel_id"]), None)
                v2 = next((v for v in vessels if v["id"] == a2["vessel_id"]), None)
                bd1 = next((b for b in berths if b["id"] == b1), None)
                bd2 = next((b for b in berths if b["id"] == b2), None)
                if v1 and v2 and bd1 and bd2:
                    fit1 = v1.get("length_m", 200) <= bd2.get("length_m", 300) and v1.get("draft_m", 12) <= bd2.get("depth_m", 15)
                    fit2 = v2.get("length_m", 200) <= bd1.get("length_m", 300) and v2.get("draft_m", 12) <= bd1.get("depth_m", 15)
                    if fit1 and fit2:
                        candidate[i]["berth_id"] = b2
                        candidate[j]["berth_id"] = b1
                        return candidate, "swap_berth"

    elif move_type == 2:
        # Relocate target vessel
        a = candidate[target_idx]
        v_data = next((v for v in vessels if v["id"] == a["vessel_id"]), None)
        if v_data:
            other_berths = [b for b in berths if b["id"] != a["berth_id"]
                           and v_data.get("length_m", 200) <= b.get("length_m", 300)
                           and v_data.get("draft_m", 12) <= b.get("depth_m", 15)]
            if other_berths:
                new_b = rng.choice(other_berths)
                candidate[target_idx]["berth_id"] = new_b["id"]
                return candidate, "relocate"

    elif move_type == 3:
        # Change crane count for target vessel
        a = candidate[target_idx]
        current_nc = a["cranes_assigned"]
        delta = rng.choice([-2, -1, 1, 2])
        new_nc = max(min_cranes, min(max_cranes, current_nc + delta))
        if new_nc != current_nc:
            candidate[target_idx]["cranes_assigned"] = new_nc
            return candidate, "crane_adj"

    else:
        # Swap order of two vessels at same berth
        berth_groups = {}
        for i in feasible:
            bid = candidate[i]["berth_id"]
            berth_groups.setdefault(bid, []).append(i)
        multi = [g for g in berth_groups.values() if len(g) >= 2]
        if multi:
            group = rng.choice(multi)
            i, j = rng.sample(group, 2)
            for key in ["vessel_id", "vessel_name", "priority", "teu_volume"]:
                candidate[i][key], candidate[j][key] = candidate[j].get(key), candidate[i].get(key)
            return candidate, "swap_order"

    return candidate, "noop"


def run(input_data: dict, solver_params: dict, extra_arguments: dict) -> dict:
    start_time = time.time()
    logger.info("=== Classical BAP+QCA Solver v10.0 — Simulated Annealing with Priority-Aware Moves ===")

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
    logger.info("Problem: %d vessels, %d berths, %d cranes", n_vessels, n_berths, total_cranes)

    greedy_max_cranes = max(min_cranes, min(total_cranes // n_berths, max_cranes))
    logger.info("v10.0: Greedy crane cap = %d | SA post-optimization enabled", greedy_max_cranes)

    sorted_vessels = sorted(vessels, key=lambda v: (v.get("priority", 5), v.get("arrival_time", "")))

    # ── PHASE 1: Greedy construction ──────────────────────────────
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
        v_name = v.get("name", "Vessel-%s" % v_id)

        best_berth = None
        best_cost = float("inf")
        best_start = None
        best_cranes_assigned = min_cranes
        best_handling_hours = 0
        best_wait_hours = 0
        best_delay_hours = 0
        best_actual_cost = 0

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

            crane_candidates = list(range(greedy_max_cranes, min_cranes - 1, -1))

            for nc in crane_candidates:
                handling_hours = v_teu / (b_prod * nc) if b_prod * nc > 0 else 999
                end_h = start_h + handling_hours
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

                vessels_at_berth = berth_vessel_count.get(b_id, 0)
                queue_wait_penalty = 0
                if vessels_at_berth > 0:
                    queue_wait_penalty = vessels_at_berth * handling_hours * w_wait * 0.5
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
                _iso_to_hours(best_start) + best_handling_hours, best_start
            )
            start_h = _iso_to_hours(best_start)
            end_h = start_h + best_handling_hours
            berth_end_times[best_berth] = end_time_str
            berth_vessel_count[best_berth] += 1
            crane_timeline.append((start_h, end_h, best_cranes_assigned))

            assignments.append({
                "vessel_id": v_id, "vessel_name": v_name,
                "berth_id": best_berth, "start_time": best_start,
                "end_time": end_time_str, "cranes_assigned": best_cranes_assigned,
                "handling_hours": round(best_handling_hours, 2),
                "waiting_hours": round(best_wait_hours, 2),
                "delay_hours": round(best_delay_hours, 2),
                "cost": round(best_actual_cost, 2),
                "priority": v_priority, "teu_volume": v_teu
            })
        else:
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
                nc = min_cranes
                handling_hours = v_teu / (b_prod * nc) if b_prod * nc > 0 else 999
                end_h = start_h + handling_hours
                wait_hours = max(0, start_h - arr_h)
                delay_hours = max(0, end_h - deadline_h)
                pm = w_priority if v_priority <= 2 else 1.0
                cost = (handling_hours * nc * w_handle + wait_hours * w_wait * pm + delay_hours * w_delay * pm)
                end_time_str = _hours_to_iso(end_h, actual_start)
                berth_end_times[b_id] = end_time_str
                berth_vessel_count[b_id] = berth_vessel_count.get(b_id, 0) + 1
                crane_timeline.append((start_h, end_h, nc))
                assignments.append({
                    "vessel_id": v_id, "vessel_name": v_name,
                    "berth_id": b_id, "start_time": actual_start,
                    "end_time": end_time_str, "cranes_assigned": nc,
                    "handling_hours": round(handling_hours, 2),
                    "waiting_hours": round(wait_hours, 2),
                    "delay_hours": round(delay_hours, 2),
                    "cost": round(cost, 2),
                    "priority": v_priority, "teu_volume": v_teu
                })
                break
            else:
                assignments.append({
                    "vessel_id": v_id, "vessel_name": v_name,
                    "berth_id": None, "start_time": None, "end_time": None,
                    "cranes_assigned": 0, "handling_hours": 0, "cost": 0,
                    "status": "infeasible"
                })

    greedy_cost, assignments = _recalc_full_cost(assignments, vessels, berths, cost_weights, w_priority)
    cost_evolution.append({"iteration": 0, "phase": "greedy", "objective_value": round(greedy_cost, 2)})
    logger.info("Greedy phase: cost=$%s", "{:,.2f}".format(greedy_cost))

    # ── PHASE 2: 2-opt local search ──────────────────────────────
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
        cost_evolution.append({"iteration": iteration, "phase": "2-opt", "objective_value": round(current_cost, 2)})

    two_opt_cost = sum(a["cost"] for a in assignments)
    logger.info("2-opt completed: cost=$%s", "{:,.2f}".format(two_opt_cost))

    # ── PHASE 2b: Or-opt moves ───────────────────────────────────
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
            eval_result = _evaluate_vessel_at_berth(v_data, b, a["start_time"], a["cranes_assigned"], cost_weights)
            if eval_result["cost"] < best_cost_oropt:
                best_cost_oropt = eval_result["cost"]
                best_config = (b_id, eval_result)

        if best_config is not None and best_cost_oropt < old_cost * 0.95:
            b_id, eval_result = best_config
            assignments[idx]["berth_id"] = b_id
            assignments[idx].update(eval_result)
            oropt_improved += 1

    oropt_cost = sum(a["cost"] for a in assignments)
    cost_evolution.append({"iteration": 1, "phase": "or-opt", "objective_value": round(oropt_cost, 2)})
    logger.info("Or-opt moves: %d improvements, cost=$%s", oropt_improved, "{:,.2f}".format(oropt_cost))

    # ── PHASE 3: Crane rebalancing ───────────────────────────────
    crane_rebalance_improvements = 0
    for idx, a in enumerate(assignments):
        if a.get("berth_id") is None:
            continue
        v_data = next((v for v in vessels if v["id"] == a["vessel_id"]), None)
        b_data = next((b for b in berths if b["id"] == a["berth_id"]), None)
        if not v_data or not b_data:
            continue
        old_cost = a["cost"]
        best_cost_rebal = old_cost
        best_nc = a["cranes_assigned"]
        for nc in range(min_cranes, max_cranes + 1):
            if nc == a["cranes_assigned"]:
                continue
            eval_result = _evaluate_vessel_at_berth(v_data, b_data, a["start_time"], nc, cost_weights)
            if eval_result["cost"] < best_cost_rebal:
                best_cost_rebal = eval_result["cost"]
                best_nc = nc

        if best_nc != a["cranes_assigned"]:
            eval_result = _evaluate_vessel_at_berth(v_data, b_data, a["start_time"], best_nc, cost_weights)
            assignments[idx]["cranes_assigned"] = best_nc
            assignments[idx].update(eval_result)
            crane_rebalance_improvements += 1

    crane_reopt_cost, assignments = _recalc_full_cost(assignments, vessels, berths, cost_weights, w_priority)
    cost_evolution.append({"iteration": 1, "phase": "crane_reopt", "objective_value": round(crane_reopt_cost, 2)})
    logger.info("Crane rebalance: %d adjustments, cost=$%s", crane_rebalance_improvements, "{:,.2f}".format(crane_reopt_cost))

    # ── PHASE 4: Simulated Annealing v10.0 ───────────────────────
    sa_enabled = solver_params.get("simulated_annealing_enabled", True)
    if not sa_enabled:
        logger.info("Simulated Annealing disabled")
        sa_cost = crane_reopt_cost
        sa_iterations = 0
        sa_accepted = 0
        sa_improved = 0
        sa_temperature_final = 0.0
        sa_restarts = 0
    else:
        rng = random.Random(solver_params.get("random_seed", 42))
        sa_max_iters = solver_params.get("simulated_annealing_max_iterations", 1000)
        sa_initial_temp = solver_params.get("simulated_annealing_initial_temperature", 5000.0)
        sa_cooling_rate = solver_params.get("simulated_annealing_cooling_rate", 0.995)
        sa_min_temp = solver_params.get("simulated_annealing_min_temperature", 10.0)
        sa_time_limit = solver_params.get("simulated_annealing_time_limit_s", 25.0)

        best_overall_cost = crane_reopt_cost
        best_overall_assignments = [dict(a) for a in assignments]
        current_cost = crane_reopt_cost
        current_assignments = [dict(a) for a in assignments]

        sa_iterations = 0
        sa_accepted = 0
        sa_improved = 0
        sa_restarts = 0
        temperature = sa_initial_temp
        stall_counter = 0
        stall_threshold = 100

        sa_start_time = time.time()

        while temperature > sa_min_temp and sa_iterations < sa_max_iters and (time.time() - sa_start_time) < sa_time_limit:
            candidate_assignments, move_type = _sa_neighbor(current_assignments, vessels, berths, cost_weights, cranes_cfg, w_priority, rng)
            if move_type == "noop":
                sa_iterations += 1
                temperature *= sa_cooling_rate
                continue

            candidate_cost, candidate_assignments = _recalc_full_cost(candidate_assignments, vessels, berths, cost_weights, w_priority)
            delta = candidate_cost - current_cost

            if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 0.1)):
                current_assignments = candidate_assignments
                current_cost = candidate_cost
                sa_accepted += 1
                stall_counter = 0
                if candidate_cost < best_overall_cost:
                    best_overall_cost = candidate_cost
                    best_overall_assignments = [dict(a) for a in candidate_assignments]
                    sa_improved += 1
            else:
                stall_counter += 1

            # v10.0: Restart mechanism — if stalled, reheat from best
            if stall_counter >= stall_threshold:
                current_assignments = [dict(a) for a in best_overall_assignments]
                current_cost = best_overall_cost
                temperature = sa_initial_temp * 0.5  # Reduced reheat
                stall_counter = 0
                sa_restarts += 1

            sa_iterations += 1
            temperature *= sa_cooling_rate

        assignments = best_overall_assignments
        sa_cost = best_overall_cost
        sa_temperature_final = temperature

    cost_evolution.append({"iteration": 1, "phase": "simulated_annealing", "objective_value": round(sa_cost, 2)})
    logger.info("SA v10.0: %d iterations, %d accepted, %d improved, %d restarts, final temp=%.1f, cost=$%s",
                sa_iterations, sa_accepted, sa_improved, sa_restarts, sa_temperature_final,
                "{:,.2f}".format(sa_cost))

    # ── FINAL: Resequence for accurate cost ──────────────────────
    final_cost, assignments = _recalc_full_cost(assignments, vessels, berths, cost_weights, w_priority)
    cost_evolution.append({"iteration": 1, "phase": "final", "objective_value": round(final_cost, 2)})

    wall_time = time.time() - start_time

    # Cost breakdown
    total_handling_cost = sum(
        (_iso_to_hours(a.get("end_time", a.get("start_time", "2025-01-01T00:00:00Z"))) - _iso_to_hours(a.get("start_time", "2025-01-01T00:00:00Z"))) * a.get("cranes_assigned", 0) * w_handle
        if a.get("berth_id") is not None else 0
        for a in assignments
    )
    total_waiting_cost = sum(a.get("waiting_hours", 0) * w_wait * (w_priority if a.get("priority", 3) <= 2 else 1.0) for a in assignments)
    total_delay_cost = sum(a.get("delay_hours", 0) * w_delay * (w_priority if a.get("priority", 3) <= 2 else 1.0) for a in assignments)

    cost_breakdown = {
        "total_cost": round(final_cost, 2),
        "crane_handling_cost": round(sum(a.get("cost", 0) * 0.5 for a in assignments if a.get("berth_id") is not None), 2),
        "waiting_cost": round(total_waiting_cost, 2),
        "delay_penalty_cost": round(total_delay_cost, 2),
        "cost_per_vessel": round(final_cost / max(n_vessels, 1), 2),
        "cost_per_teu": round(final_cost / sum(v.get("handling_volume_teu", 1000) for v in vessels), 2)
    }

    feasible_assignments = sum(1 for a in assignments if a.get("berth_id") is not None)
    infeasible_assignments = len(assignments) - feasible_assignments
    makespan = max((_iso_to_hours(a.get("end_time", "2025-01-01T00:00:00Z")) for a in assignments if a.get("berth_id") is not None), default=0)
    total_waiting = sum(a.get("waiting_hours", 0) for a in assignments)

    schedule_metrics = {
        "feasible_assignments": feasible_assignments,
        "infeasible_assignments": infeasible_assignments,
        "makespan": round(makespan, 2),
        "total_waiting_time": round(total_waiting, 2)
    }

    berth_utilization = []
    for b in berths:
        b_id = b["id"]
        b_assignments = [a for a in assignments if a.get("berth_id") == b_id]
        vessels_served = len(b_assignments)
        occupied_hours = sum(
            _iso_to_hours(a.get("end_time", "2025-01-01T00:00:00Z")) - _iso_to_hours(a.get("start_time", "2025-01-01T00:00:00Z"))
            for a in b_assignments if a.get("berth_id") is not None
        )
        total_teu = sum(a.get("teu_volume", 0) for a in b_assignments)
        util_pct = (occupied_hours / 168.0) * 100 if occupied_hours > 0 else 0
        berth_utilization.append({
            "berth_id": b_id,
            "vessels_served": vessels_served,
            "occupied_hours": round(occupied_hours, 2),
            "utilization_pct": round(util_pct, 1),
            "total_teu_handled": total_teu
        })

    priority_analysis = {}
    for p in [1, 2, 3, 4, 5]:
        p_assignments = [a for a in assignments if a.get("priority") == p and a.get("berth_id") is not None]
        if p_assignments:
            priority_analysis[f"P{p}"] = {
                "count": len(p_assignments),
                "avg_cost": round(sum(a.get("cost", 0) for a in p_assignments) / len(p_assignments), 2),
                "avg_wait_h": round(sum(a.get("waiting_hours", 0) for a in p_assignments) / len(p_assignments), 2),
                "avg_delay_h": round(sum(a.get("delay_hours", 0) for a in p_assignments) / len(p_assignments), 2),
                "total_cost": round(sum(a.get("cost", 0) for a in p_assignments), 2)
            }

    gantt_data = [
        {
            "vessel": a.get("vessel_name", a.get("vessel_id")),
            "berth": a.get("berth_id"),
            "start": a.get("start_time"),
            "end": a.get("end_time"),
            "cranes": a.get("cranes_assigned"),
            "priority": a.get("priority", 3)
        }
        for a in assignments if a.get("berth_id") is not None
    ]

    avg_cranes = sum(a.get("cranes_assigned", 0) for a in assignments) / max(feasible_assignments, 1) if feasible_assignments > 0 else 0
    crane_allocation = {"avg_cranes_per_vessel": round(avg_cranes, 2)}

    optimization_convergence = {
        "cost_evolution": cost_evolution,
        "greedy_initial_cost": round(cost_evolution[0]["objective_value"], 2),
        "two_opt_cost": round(two_opt_cost, 2),
        "crane_reopt_cost": round(crane_reopt_cost, 2),
        "sa_cost": round(sa_cost, 2),
        "final_optimized_cost": round(final_cost, 2),
        "improvement_pct": round((1 - final_cost / max(cost_evolution[0]["objective_value"], 1)) * 100, 1),
        "sa_iterations": sa_iterations,
        "sa_accepted": sa_accepted,
        "sa_improved": sa_improved,
        "sa_restarts": sa_restarts,
        "sa_temperature_final": round(sa_temperature_final, 2)
    }

    computation_metrics = {
        "solver_version": "10.0",
        "algorithm": "Greedy + 2-Opt + Or-Opt + Crane-Reopt + Simulated Annealing v10.0",
        "wall_time_s": round(wall_time, 2),
        "or_opt_improvements": oropt_improved,
        "crane_rebalance_improvements": crane_rebalance_improvements
    }

    _generate_expert_dashboard(
        assignments, berths, vessels, cost_breakdown,
        optimization_convergence, berth_utilization, priority_analysis,
        gantt_data, schedule_metrics, computation_metrics, crane_allocation
    )

    return {
        "solver_status": "OK",
        "assignments": assignments,
        "cost_breakdown": cost_breakdown,
        "schedule_metrics": schedule_metrics,
        "berth_utilization": berth_utilization,
        "priority_analysis": priority_analysis,
        "optimization_convergence": optimization_convergence,
        "computation_metrics": computation_metrics
    }
