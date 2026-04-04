"""
Helper functions for Classical BAP+QCA solver v6.0
Split from qcentroid.py for deployment compatibility.
"""
import logging
import math

logger = logging.getLogger("qcentroid-user-log")

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

