"""Classical BAP+QCA solver v7.0 helper functions."""
import logging
import math

logger = logging.getLogger("qcentroid-user-log")

def _get_available_cranes_at_window(start_h, end_h, crane_timeline, total_cranes):
    """v7.0: Max concurrent cranes available during [start_h, end_h)."""
    if not crane_timeline:
        return total_cranes

    # Build events for overlapping intervals
    events = []
    for s, e, nc in crane_timeline:
        # Only consider intervals that overlap with [start_h, end_h)
        if s < end_h and e > start_h:
            overlap_start = max(s, start_h)
            overlap_end = min(e, end_h)
            events.append((overlap_start, nc))       # crane count increases at overlap_start
            events.append((overlap_end, -nc))         # crane count decreases at overlap_end

    if not events:
        return total_cranes

    events.sort()
    current_used = 0
    max_concurrent_used = 0
    for t, delta in events:
        current_used += delta
        max_concurrent_used = max(max_concurrent_used, current_used)

    return total_cranes - max_concurrent_used


def _enforce_crane_budget(assignments, vessels, berths, total_cranes, min_cranes, max_cranes, cost_weights):
    """v7.0: Gentle post-hoc enforcement (safety net only)."""
    w_handle = cost_weights.get("handling_cost_per_crane_hour", 150)
    w_wait = cost_weights.get("waiting_cost_per_hour", 500)
    w_delay = cost_weights.get("delay_penalty_per_hour", 1000)
    w_priority_mult = cost_weights.get("priority_multiplier", 1.5)
    
    changes = 0
    if not assignments:
        return assignments, changes
    active_assignments = []
    for idx, a in enumerate(assignments):
        if a.get("berth_id") is None:
            continue
        s = _iso_to_hours(a["start_time"])
        e = _iso_to_hours(a["end_time"])
        if e <= s:
            e = s + 1
        active_assignments.append((s, e, idx))
    if not active_assignments:
        return assignments, changes
    events = sorted(set(s for s,e,_ in active_assignments) | set(e for s,e,_ in active_assignments))
    
    # For each time point, check crane usage
    # Track max allowed cranes per vessel (minimum across all time slots)
    max_allowed = {}
    for _, _, idx in active_assignments:
        max_allowed[idx] = assignments[idx]["cranes_assigned"]
    
    for t_idx in range(len(events) - 1):
        t = events[t_idx]
        active_at_t = [(s, e, idx) for s, e, idx in active_assignments if s <= t < e]
        total_used = sum(assignments[idx]["cranes_assigned"] for _, _, idx in active_at_t)
        if total_used <= total_cranes + 1:
            continue
        
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
    """Re-sequence vessels at each berth (sequential, no overlap)."""
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
    """Convert ISO to hours."""
    if not iso_str or not isinstance(iso_str, str):
        return 0
    try:
        p = iso_str.replace("Z", "").split("T")
        d = p[0].split("-")
        t = p[1].split(":") if len(p) > 1 else ["0", "0", "0"]
        dy = int(d[1]) * 30 + int(d[2])
        return dy * 24 + int(t[0]) + int(t[1]) / 60
    except (IndexError, ValueError):
        return 0

def _hours_to_iso(hours, reference_iso):
    """Convert hours back to ISO."""
    if not reference_iso:
        return "2025-01-01T00:00:00Z"
    try:
        p = reference_iso.replace("Z", "").split("T")
        d = p[0].split("-")
        h = int(hours) % 24
        m = int((hours - int(hours)) * 60)
        do = int(hours) // 24
        mo = max(1, do // 30)
        da = max(1, do % 30)
        return f"{d[0]}-{mo:02d}-{da:02d}T{h:02d}:{m:02d}:00Z"
    except Exception:
        return reference_iso


def _compute_lookahead_score(start_time, duration, berth_id, remaining_vessels, current_v_id, berth_end_times):
    """Compute look-ahead score for deadline-critical vessels."""
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
    """Evaluate cost of assigning vessel to berth."""
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
    """Try swapping berth assignments of two vessels."""
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
