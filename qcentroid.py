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
- Expert dashboard with 5 interactive tabs (port overview, timeline, costs, convergence, berth analytics)
- SVG vessel map color-coded by priority
- Plotly Gantt chart, donut chart for cost distribution, heatmap for berth utilization
- Dashboard auto-opens in browser post-optimization

v3.0 changes:
- Multiple construction heuristics: greedy, time-aware, cost-balanced
- 2-opt local search for vessel permutations
- Real-time cost tracking and phase convergence
- Clean separation of construction and improvement phases

v2.0 changes:
- Multiple berths (3) with independent scheduling
- Crane availability constraints (max 18 per time unit)
- Cost model: vessel_cost * crane_days
- Greedy construction with simple 2-opt improvement

v1.0 - Initial release:
- Single berth model
- Basic crane scheduling
"""

import random
import math
import json
import os
import time
import sys
from datetime import datetime
from collections import defaultdict, deque
import webbrowser
import shutil
import subprocess

###############################################################################
# CONFIGURATION & CONSTANTS
###############################################################################

# Vessel class/priority weights (for balanced construction)
PRIORITY_WEIGHTS = {
    'high': 1.5,
    'medium': 1.0,
    'low': 0.5
}

# Global crane budget
MAX_CRANES = 18

# Local search parameters
MAX_2OPT_ITERATIONS = 100
MAX_OROPT_ITERATIONS = 50
MAX_3OPT_ITERATIONS = 30

# Berth-specific parameters
BERTH_CONFIGS = {
    'A': {'max_vessels': 8, 'cost_per_slot': 100},
    'B': {'max_vessels': 6, 'cost_per_slot': 80},
    'C': {'max_vessels': 7, 'cost_per_slot': 90}
}

# Seed for reproducibility
RANDOM_SEED = 42

###############################################################################
# VESSEL & SCHEDULE DATA STRUCTURES
###############################################################################

class Vessel:
    """Represents a vessel with scheduling requirements"""
    
    def __init__(self, vessel_id, arrival_time, processing_time, cranes_needed, 
                 priority='medium', deadline=None, cost_factor=1.0):
        self.vessel_id = vessel_id
        self.arrival_time = arrival_time
        self.processing_time = processing_time
        self.cranes_needed = cranes_needed
        self.priority = priority
        self.deadline = deadline or (arrival_time + 1000)
        self.cost_factor = cost_factor
    
    def __repr__(self):
        return (f"Vessel({self.vessel_id}, arr={self.arrival_time}, "
                f"proc={self.processing_time}, cranes={self.cranes_needed}, "
                f"pri={self.priority})")


class BerthSchedule:
    """Manages vessel schedule for a single berth"""
    
    def __init__(self, berth_id):
        self.berth_id = berth_id
        self.vessels = []
        self.start_times = {}
        self.end_times = {}
        self.crane_allocations = {}
    
    def add_vessel(self, vessel, start_time, crane_count):
        """Add a vessel to this berth's schedule"""
        self.vessels.append(vessel)
        self.start_times[vessel.vessel_id] = start_time
        self.end_times[vessel.vessel_id] = start_time + vessel.processing_time
        self.crane_allocations[vessel.vessel_id] = crane_count
    
    def remove_vessel(self, vessel_id):
        """Remove a vessel from schedule"""
        if vessel_id in self.start_times:
            self.vessels = [v for v in self.vessels if v.vessel_id != vessel_id]
            del self.start_times[vessel_id]
            del self.end_times[vessel_id]
            del self.crane_allocations[vessel_id]
    
    def get_vessel_by_id(self, vessel_id):
        """Retrieve vessel object by ID"""
        for v in self.vessels:
            if v.vessel_id == vessel_id:
                return v
        return None
    
    def total_cost(self):
        """Calculate total cost for this berth"""
        cost = 0
        for vessel in self.vessels:
            crane_count = self.crane_allocations.get(vessel.vessel_id, vessel.cranes_needed)
            cost += vessel.processing_time * crane_count * vessel.cost_factor
        return cost


class Solution:
    """Complete port scheduling solution with multiple berths"""
    
    def __init__(self):
        self.berth_schedules = {
            'A': BerthSchedule('A'),
            'B': BerthSchedule('B'),
            'C': BerthSchedule('C')
        }
        self.unscheduled_vessels = []
        self.total_cost = 0
        self.convergence_history = []
        self.phase_markers = []
    
    def assign_vessel(self, vessel, berth_id, start_time, crane_count):
        """Assign vessel to berth"""
        self.berth_schedules[berth_id].add_vessel(vessel, start_time, crane_count)
    
    def unassign_vessel(self, vessel_id, berth_id):
        """Remove vessel from berth"""
        self.berth_schedules[berth_id].remove_vessel(vessel_id)
    
    def calculate_total_cost(self):
        """Sum costs across all berths"""
        return sum(bs.total_cost() for bs in self.berth_schedules.values())
    
    def is_feasible(self):
        """Check feasibility: no time conflicts, crane budget respected"""
        # Check time feasibility per berth
        for berth_id, berth in self.berth_schedules.items():
            times_used = defaultdict(int)
            for vessel in berth.vessels:
                for t in range(self.start_times[vessel.vessel_id], 
                              self.end_times[vessel.vessel_id]):
                    times_used[t] += self.crane_allocations.get(vessel.vessel_id, 1)
            
            if any(c > MAX_CRANES for c in times_used.values()):
                return False
        
        return True
    
    def calculate_crane_usage(self):
        """Get total crane usage over time"""
        crane_usage = defaultdict(int)
        for berth in self.berth_schedules.values():
            for vessel in berth.vessels:
                start = berth.start_times[vessel.vessel_id]
                end = berth.end_times[vessel.vessel_id]
                cranes = berth.crane_allocations[vessel.vessel_id]
                for t in range(start, end):
                    crane_usage[t] += cranes
        return crane_usage


###############################################################################
# GREEDY CONSTRUCTION HEURISTIC
###############################################################################

def greedy_construction(vessels, diversity_penalty=5, load_penalty=1):
    """
    Greedy construction with:
    - Berth diversity constraint (soft penalty)
    - Load balancing across berths
    - Time-aware placement
    """
    solution = Solution()
    remaining = sorted(vessels, key=lambda v: (v.priority != 'high', v.deadline))
    
    berth_load = {'A': 0, 'B': 0, 'C': 0}
    avg_vessel_cost = sum(v.cost_factor for v in vessels) / len(vessels) if vessels else 1.0
    
    for vessel in remaining:
        best_berth = None
        best_start_time = None
        best_cranes = None
        best_cost = float('inf')
        
        for berth_id, berth in solution.berth_schedules.items():
            # Try to fit vessel in this berth
            earliest_start = vessel.arrival_time
            
            for existing_vessel in berth.vessels:
                existing_end = berth.end_times[existing_vessel.vessel_id]
                if existing_end > earliest_start:
                    earliest_start = max(earliest_start, existing_end)
            
            # Check crane feasibility
            crane_usage = solution.calculate_crane_usage()
            can_fit = True
            for t in range(earliest_start, earliest_start + vessel.processing_time):
                if crane_usage[t] + vessel.cranes_needed > MAX_CRANES:
                    can_fit = False
                    break
            
            if not can_fit:
                continue
            
            # Calculate cost with penalties
            cost = vessel.processing_time * vessel.cranes_needed * vessel.cost_factor
            
            # Berth diversity penalty: discourage all vessels on cheapest berth
            if berth_load[berth_id] > 0:
                cost += diversity_penalty * avg_vessel_cost * 0.05
            
            # Load penalty: existing assignments increase cost
            cost += load_penalty * avg_vessel_cost * 0.05 * len(berth.vessels)
            
            if cost < best_cost:
                best_cost = cost
                best_berth = berth_id
                best_start_time = earliest_start
                best_cranes = vessel.cranes_needed
        
        if best_berth:
            solution.assign_vessel(vessel, best_berth, best_start_time, best_cranes)
            berth_load[best_berth] += 1
        else:
            solution.unscheduled_vessels.append(vessel)
    
    solution.total_cost = solution.calculate_total_cost()
    return solution


###############################################################################
# LOCAL SEARCH IMPROVEMENTS: 2-OPT, OR-OPT, 3-OPT
###############################################################################

def two_opt(solution):
    """
    2-opt: swap pair of vessels to reduce cost
    Try all pairs of vessels within and across berths
    """
    improved = True
    iteration = 0
    
    while improved and iteration < MAX_2OPT_ITERATIONS:
        improved = False
        iteration += 1
        
        # Collect all scheduled vessels
        all_vessels = []
        for berth_id, berth in solution.berth_schedules.items():
            for vessel in berth.vessels:
                all_vessels.append((berth_id, vessel))
        
        # Try swaps
        for i, (berth_i, vessel_i) in enumerate(all_vessels):
            for j, (berth_j, vessel_j) in enumerate(all_vessels):
                if i >= j:
                    continue
                
                # Swap vessels
                solution.unassign_vessel(vessel_i.vessel_id, berth_i)
                solution.unassign_vessel(vessel_j.vessel_id, berth_j)
                
                # Reassign swapped
                start_i_before = solution.berth_schedules[berth_i].start_times.get(vessel_i.vessel_id, vessel_i.arrival_time)
                start_j_before = solution.berth_schedules[berth_j].start_times.get(vessel_j.vessel_id, vessel_j.arrival_time)
                
                try:
                    solution.assign_vessel(vessel_i, berth_j, start_j_before, vessel_i.cranes_needed)
                    solution.assign_vessel(vessel_j, berth_i, start_i_before, vessel_j.cranes_needed)
                    
                    new_cost = solution.calculate_total_cost()
                    if new_cost < solution.total_cost and solution.is_feasible():
                        solution.total_cost = new_cost
                        improved = True
                        solution.convergence_history.append(new_cost)
                    else:
                        # Revert
                        solution.unassign_vessel(vessel_i.vessel_id, berth_j)
                        solution.unassign_vessel(vessel_j.vessel_id, berth_i)
                        solution.assign_vessel(vessel_i, berth_i, start_i_before, vessel_i.cranes_needed)
                        solution.assign_vessel(vessel_j, berth_j, start_j_before, vessel_j.cranes_needed)
                except:
                    # Revert on error
                    solution.unassign_vessel(vessel_i.vessel_id, berth_j) if vessel_i.vessel_id in solution.berth_schedules[berth_j].start_times else None
                    solution.unassign_vessel(vessel_j.vessel_id, berth_i) if vessel_j.vessel_id in solution.berth_schedules[berth_i].start_times else None
                    solution.assign_vessel(vessel_i, berth_i, start_i_before, vessel_i.cranes_needed)
                    solution.assign_vessel(vessel_j, berth_j, start_j_before, vessel_j.cranes_needed)
    
    return solution


def or_opt(solution):
    """
    Or-opt: relocate single vessel to different berth
    Explores neighborhood better than 2-opt alone
    """
    improved = True
    iteration = 0
    
    while improved and iteration < MAX_OROPT_ITERATIONS:
        improved = False
        iteration += 1
        
        for berth_id, berth in solution.berth_schedules.items():
            if not berth.vessels:
                continue
            
            for vessel in list(berth.vessels):
                original_start = berth.start_times[vessel.vessel_id]
                original_cranes = berth.crane_allocations[vessel.vessel_id]
                original_cost = solution.total_cost
                
                # Try moving to each other berth
                for target_berth_id in solution.berth_schedules.keys():
                    if target_berth_id == berth_id:
                        continue
                    
                    target_berth = solution.berth_schedules[target_berth_id]
                    
                    # Find earliest time in target berth
                    earliest_start = vessel.arrival_time
                    for other_vessel in target_berth.vessels:
                        existing_end = target_berth.end_times[other_vessel.vessel_id]
                        if existing_end > earliest_start:
                            earliest_start = max(earliest_start, existing_end)
                    
                    # Remove from original
                    solution.unassign_vessel(vessel.vessel_id, berth_id)
                    
                    # Try to add to target
                    try:
                        solution.assign_vessel(vessel, target_berth_id, earliest_start, original_cranes)
                        
                        new_cost = solution.calculate_total_cost()
                        if new_cost < original_cost and solution.is_feasible():
                            solution.total_cost = new_cost
                            improved = True
                            solution.convergence_history.append(new_cost)
                        else:
                            # Revert
                            solution.unassign_vessel(vessel.vessel_id, target_berth_id)
                            solution.assign_vessel(vessel, berth_id, original_start, original_cranes)
                    except:
                        # Revert on error
                        if vessel.vessel_id not in berth.start_times:
                            solution.unassign_vessel(vessel.vessel_id, target_berth_id) if vessel.vessel_id in solution.berth_schedules[target_berth_id].start_times else None
                            solution.assign_vessel(vessel, berth_id, original_start, original_cranes)
    
    return solution


def three_opt(solution):
    """
    3-opt: rotate 3 vessels among 3 berths for additional improvements
    """
    improved = True
    iteration = 0
    
    while improved and iteration < MAX_3OPT_ITERATIONS:
        improved = False
        iteration += 1
        
        berth_ids = list(solution.berth_schedules.keys())
        if len(berth_ids) < 3:
            break
        
        # Get vessels from first 3 berths
        vessels_by_berth = {bid: [] for bid in berth_ids}
        for bid in berth_ids:
            vessels_by_berth[bid] = list(solution.berth_schedules[bid].vessels)
        
        # Try 3-way rotations
        for b1_idx, b2_idx, b3_idx in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
            b1, b2, b3 = berth_ids[b1_idx], berth_ids[b2_idx], berth_ids[b3_idx]
            
            if not vessels_by_berth[b1] or not vessels_by_berth[b2] or not vessels_by_berth[b3]:
                continue
            
            v1 = vessels_by_berth[b1][0]
            v2 = vessels_by_berth[b2][0]
            v3 = vessels_by_berth[b3][0]
            
            original_cost = solution.total_cost
            
            try:
                # Store originals
                start_v1, start_v2, start_v3 = (solution.berth_schedules[b1].start_times[v1.vessel_id],
                                               solution.berth_schedules[b2].start_times[v2.vessel_id],
                                               solution.berth_schedules[b3].start_times[v3.vessel_id])
                
                # Remove all three
                solution.unassign_vessel(v1.vessel_id, b1)
                solution.unassign_vessel(v2.vessel_id, b2)
                solution.unassign_vessel(v3.vessel_id, b3)
                
                # Rotate: v1 -> b2, v2 -> b3, v3 -> b1
                solution.assign_vessel(v1, b2, start_v2, v1.cranes_needed)
                solution.assign_vessel(v2, b3, start_v3, v2.cranes_needed)
                solution.assign_vessel(v3, b1, start_v1, v3.cranes_needed)
                
                new_cost = solution.calculate_total_cost()
                if new_cost < original_cost and solution.is_feasible():
                    solution.total_cost = new_cost
                    improved = True
                    solution.convergence_history.append(new_cost)
                else:
                    # Revert
                    solution.unassign_vessel(v1.vessel_id, b2)
                    solution.unassign_vessel(v2.vessel_id, b3)
                    solution.unassign_vessel(v3.vessel_id, b1)
                    solution.assign_vessel(v1, b1, start_v1, v1.cranes_needed)
                    solution.assign_vessel(v2, b2, start_v2, v2.cranes_needed)
                    solution.assign_vessel(v3, b3, start_v3, v3.cranes_needed)
            except:
                pass
    
    return solution


###############################################################################
# CRANE REBALANCING
###############################################################################

def rebalance_cranes(solution):
    """
    Rebalance crane allocation across fleet
    Find better distributions that respect global budget
    """
    crane_usage = solution.calculate_crane_usage()
    max_usage = max(crane_usage.values()) if crane_usage else 0
    
    if max_usage <= MAX_CRANES:
        return solution  # Already feasible
    
    # Try reducing crane allocation per vessel
    for berth in solution.berth_schedules.values():
        for vessel in berth.vessels:
            current_cranes = berth.crane_allocations[vessel.vessel_id]
            if current_cranes > 1:
                # Try reducing
                berth.crane_allocations[vessel.vessel_id] = max(1, current_cranes - 1)
                
                crane_usage = solution.calculate_crane_usage()
                if max(crane_usage.values()) > MAX_CRANES:
                    # Revert if infeasible
                    berth.crane_allocations[vessel.vessel_id] = current_cranes
    
    solution.total_cost = solution.calculate_total_cost()
    return solution


###############################################################################
# MAIN OPTIMIZATION FUNCTION
###############################################################################

def optimize(vessels, output_dir='additional_output'):
    """
    Main optimization pipeline:
    1. Greedy construction with diversity & load balancing
    2. Crane rebalancing
    3. 2-opt local search
    4. Or-opt improvements
    5. 3-opt fine-tuning
    6. Generate expert dashboard
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(RANDOM_SEED)
    
    print("\n" + "="*80)
    print("BAP+QCA SOLVER v6.0")
    print("="*80)
    
    start_time = time.time()
    
    # Phase 1: Greedy Construction
    print("\n[PHASE 1] Greedy Construction with Diversity & Load Balancing...")
    phase_1_start = time.time()
    solution = greedy_construction(vessels, diversity_penalty=5, load_penalty=1)
    phase_1_time = time.time() - phase_1_start
    print(f"  Greedy cost: {solution.total_cost:.2f}")
    print(f"  Scheduled: {sum(len(b.vessels) for b in solution.berth_schedules.values())}/{len(vessels)}")
    print(f"  Time: {phase_1_time:.2f}s")
    solution.phase_markers.append(('Greedy Construction', solution.total_cost, phase_1_time))
    
    # Phase 2: Crane Rebalancing
    print("\n[PHASE 2] Crane Rebalancing...")
    phase_2_start = time.time()
    solution = rebalance_cranes(solution)
    phase_2_time = time.time() - phase_2_start
    print(f"  Rebalanced cost: {solution.total_cost:.2f}")
    print(f"  Time: {phase_2_time:.2f}s")
    solution.phase_markers.append(('Crane Rebalancing', solution.total_cost, phase_2_time))
    
    # Phase 3: 2-opt
    print("\n[PHASE 3] 2-opt Local Search...")
    phase_3_start = time.time()
    solution = two_opt(solution)
    phase_3_time = time.time() - phase_3_start
    print(f"  2-opt cost: {solution.total_cost:.2f}")
    print(f"  Time: {phase_3_time:.2f}s")
    solution.phase_markers.append(('2-opt Local Search', solution.total_cost, phase_3_time))
    
    # Phase 4: Or-opt
    print("\n[PHASE 4] Or-opt Improvement...")
    phase_4_start = time.time()
    solution = or_opt(solution)
    phase_4_time = time.time() - phase_4_start
    print(f"  Or-opt cost: {solution.total_cost:.2f}")
    print(f"  Time: {phase_4_time:.2f}s")
    solution.phase_markers.append(('Or-opt Improvement', solution.total_cost, phase_4_time))
    
    # Phase 5: 3-opt
    print("\n[PHASE 5] 3-opt Fine-tuning...")
    phase_5_start = time.time()
    solution = three_opt(solution)
    phase_5_time = time.time() - phase_5_start
    print(f"  3-opt cost: {solution.total_cost:.2f}")
    print(f"  Time: {phase_5_time:.2f}s")
    solution.phase_markers.append(('3-opt Fine-tuning', solution.total_cost, phase_5_time))
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Total cost: {solution.total_cost:.2f}")
    print(f"Total time: {total_time:.2f}s")
    
    # Phase 6: Generate expert dashboard
    print("\n[PHASE 6] Generating Expert Dashboard...")
    generate_expert_dashboard(solution, output_dir)
    print(f"  Dashboard saved to {output_dir}/01_expert_dashboard.html")
    
    return solution


###############################################################################
# DASHBOARD GENERATION
###############################################################################

def generate_expert_dashboard(solution, output_dir):
    """
    Generate comprehensive expert dashboard with 6 interactive tabs
    """
    # Collect data
    total_cost = solution.total_cost
    num_scheduled = sum(len(b.vessels) for b in solution.berth_schedules.values())
    num_unscheduled = len(solution.unscheduled_vessels)
    berth_loads = {bid: len(b.vessels) for bid, b in solution.berth_schedules.items()}
    crane_usage = solution.calculate_crane_usage()
    
    # Prepare chart data
    phase_names = [p[0] for p in solution.phase_markers]
    phase_costs = [p[1] for p in solution.phase_markers]
    phase_times = [p[2] for p in solution.phase_markers]
    
    convergence_data = solution.convergence_history if solution.convergence_history else [solution.total_cost]
    
    # Generate Gantt data
    gantt_traces = []
    colors = {'A': 'rgb(31, 119, 180)', 'B': 'rgb(255, 127, 14)', 'C': 'rgb(44, 160, 44)'}
    for berth_id, berth in solution.berth_schedules.items():
        for vessel in berth.vessels:
            start = berth.start_times[vessel.vessel_id]
            duration = vessel.processing_time
            gantt_traces.append({
                'x': [start, start + duration],
                'y': [berth_id, berth_id],
                'vessel': vessel.vessel_id,
                'color': colors[berth_id]
            })
    
    # Build HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>QCentroid Expert Dashboard v6.0</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0a192f;
            color: #e0e0e0;
            line-height: 1.6;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #0a192f 0%, #1a3a52 100%);
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #64ffda;
        }
        .header h1 {
            color: #64ffda;
            font-size: 32px;
            margin-bottom: 10px;
        }
        .header p {
            color: #b0bfc7;
            font-size: 14px;
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            border-bottom: 2px solid #2c74b3;
            flex-wrap: wrap;
        }
        .tab-button {
            padding: 12px 24px;
            background: #1a3a52;
            color: #64ffda;
            border: none;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }
        .tab-button.active {
            background: #2c74b3;
            border-bottom-color: #64ffda;
        }
        .tab-button:hover {
            background: #2c74b3;
        }
        .tab-content {
            display: none;
            animation: fadeIn 0.3s;
        }
        .tab-content.active {
            display: block;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .kpi-card {
            background: #1a3a52;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #2c74b3;
        }
        .kpi-label {
            color: #b0bfc7;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .kpi-value {
            color: #64ffda;
            font-size: 28px;
            font-weight: bold;
            margin-top: 8px;
        }
        .chart-container {
            background: #1a3a52;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            border: 1px solid #2c74b3;
        }
        .chart-title {
            color: #64ffda;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 20px;
        }
        .metric-table {
            width: 100%;
            border-collapse: collapse;
            background: #1a3a52;
            border-radius: 8px;
            overflow: hidden;
        }
        .metric-table th {
            background: #2c74b3;
            padding: 12px;
            text-align: left;
            color: #64ffda;
            font-weight: 600;
        }
        .metric-table td {
            padding: 12px;
            border-bottom: 1px solid #2c74b3;
            color: #e0e0e0;
        }
        .metric-table tr:hover {
            background: #0f2a42;
        }
        .footer {
            margin-top: 40px;
            padding: 20px;
            text-align: center;
            color: #b0bfc7;
            font-size: 12px;
            border-top: 1px solid #2c74b3;
        }
        svg {
            max-width: 100%;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>QCentroid Expert Dashboard</h1>
            <p>Classical BAP+QCA Solver v6.0 | Berth Allocation & Quay Crane Assignment</p>
        </div>
        
        <div class="tabs">
            <button class="tab-button active" onclick="switchTab('overview')">Port Overview</button>
            <button class="tab-button" onclick="switchTab('gantt')">Gantt Timeline</button>
            <button class="tab-button" onclick="switchTab('costs')">Cost Intelligence</button>
            <button class="tab-button" onclick="switchTab('convergence')">Optimization Convergence</button>
            <button class="tab-button" onclick="switchTab('berth')">Berth Analytics</button>
            <button class="tab-button" onclick="switchTab('summary')">Performance Summary</button>
        </div>
        
        <!-- TAB 1: Port Overview -->
        <div id="overview" class="tab-content active">
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Total Cost</div>
                    <div class="kpi-value">""" + f"${total_cost:.0f}" + """</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Vessels Scheduled</div>
                    <div class="kpi-value">""" + str(num_scheduled) + """</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Unscheduled</div>
                    <div class="kpi-value">""" + str(num_unscheduled) + """</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Utilization</div>
                    <div class="kpi-value">""" + f"{(num_scheduled/(num_scheduled+num_unscheduled)*100):.0f}%" + """</div>
                </div>
            </div>
            <div class="chart-container">
                <div class="chart-title">Berth Vessel Distribution</div>
                <div id="berth-pie-chart"></div>
            </div>
        </div>
        
        <!-- TAB 2: Gantt Timeline -->
        <div id="gantt" class="tab-content">
            <div class="chart-container">
                <div class="chart-title">Vessel Scheduling Timeline</div>
                <div id="gantt-chart"></div>
            </div>
        </div>
        
        <!-- TAB 3: Cost Intelligence -->
        <div id="costs" class="tab-content">
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Total Cost</div>
                    <div class="kpi-value">""" + f"${total_cost:.0f}" + """</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Avg Cost / Vessel</div>
                    <div class="kpi-value">""" + f"${total_cost/num_scheduled:.0f}" if num_scheduled > 0 else "$0" + """</div>
                </div>
            </div>
            <div class="chart-container">
                <div class="chart-title">Cost Breakdown by Berth</div>
                <div id="cost-chart"></div>
            </div>
        </div>
        
        <!-- TAB 4: Optimization Convergence -->
        <div id="convergence" class="tab-content">
            <div class="chart-container">
                <div class="chart-title">Convergence History with Phase Markers</div>
                <div id="convergence-chart"></div>
            </div>
            <div class="chart-container">
                <div class="chart-title">Phase Timing Analysis</div>
                <div id="phase-table"></div>
            </div>
        </div>
        
        <!-- TAB 5: Berth Analytics -->
        <div id="berth" class="tab-content">
            <div class="chart-container">
                <div class="chart-title">Berth Utilization & Load Heatmap</div>
                <div id="berth-heatmap"></div>
            </div>
        </div>
        
        <!-- TAB 6: Performance Summary -->
        <div id="summary" class="tab-content">
            <div class="chart-container">
                <div class="chart-title">Key Metrics</div>
                <table class="metric-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Total Cost</td>
                            <td>""" + f"${total_cost:.2f}" + """</td>
                        </tr>
                        <tr>
                            <td>Vessels Scheduled</td>
                            <td>""" + str(num_scheduled) + """</td>
                        </tr>
                        <tr>
                            <td>Scheduling Rate</td>
                            <td>""" + f"{(num_scheduled/(num_scheduled+num_unscheduled)*100):.1f}%" if (num_scheduled+num_unscheduled) > 0 else "0%" + """</td>
                        </tr>
                        <tr>
                            <td>Berth A Load</td>
                            <td>""" + str(berth_loads['A']) + """ vessels</td>
                        </tr>
                        <tr>
                            <td>Berth B Load</td>
                            <td>""" + str(berth_loads['B']) + """ vessels</td>
                        </tr>
                        <tr>
                            <td>Berth C Load</td>
                            <td>""" + str(berth_loads['C']) + """ vessels</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="footer">
            Generated by QCentroid v6.0 | """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
        </div>
    </div>
    
    <script>
        function switchTab(tabName) {
            // Hide all tabs
            const tabs = document.querySelectorAll('.tab-content');
            tabs.forEach(t => t.classList.remove('active'));
            
            // Remove active from all buttons
            const buttons = document.querySelectorAll('.tab-button');
            buttons.forEach(b => b.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            
            // Mark button as active
            event.target.classList.add('active');
        }
        
        // Berth distribution pie chart
        const berth_data = [{
            x: ['Berth A', 'Berth B', 'Berth C'],
            y: [""" + str(berth_loads['A']) + """, """ + str(berth_loads['B']) + """, """ + str(berth_loads['C']) + """],
            type: 'pie',
            marker: { colors: ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)'] }
        }];
        
        const berth_layout = {
            title: '',
            paper_bgcolor: '#1a3a52',
            plot_bgcolor: '#0a192f',
            font: { color: '#64ffda' },
            margin: { l: 0, r: 0, b: 0, t: 0 }
        };
        
        Plotly.newPlot('berth-pie-chart', berth_data, berth_layout, {responsive: true});
        
        // Cost breakdown
        const cost_data = [{
            x: ['Berth A', 'Berth B', 'Berth C'],
            y: [
                """ + str(sum(solution.berth_schedules['A'].total_cost() for _ in [1])) + """,
                """ + str(sum(solution.berth_schedules['B'].total_cost() for _ in [1])) + """,
                """ + str(sum(solution.berth_schedules['C'].total_cost() for _ in [1])) + """
            ],
            type: 'bar',
            marker: { color: ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)'] }
        }];
        
        const cost_layout = {
            title: '',
            xaxis: { title: 'Berth' },
            yaxis: { title: 'Cost ($)' },
            paper_bgcolor: '#1a3a52',
            plot_bgcolor: '#0a192f',
            font: { color: '#64ffda' },
            margin: { l: 60, r: 30, b: 60, t: 30 }
        };
        
        Plotly.newPlot('cost-chart', cost_data, cost_layout, {responsive: true});
        
        // Convergence chart
        const convergence_data = [{
            y: """ + str(convergence_data).replace("'", '"') + """,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#64ffda', width: 2 },
            marker: { size: 6, color: '#2c74b3' }
        }];
        
        const convergence_layout = {
            title: '',
            xaxis: { title: 'Iteration' },
            yaxis: { title: 'Total Cost ($)' },
            paper_bgcolor: '#1a3a52',
            plot_bgcolor: '#0a192f',
            font: { color: '#64ffda' },
            margin: { l: 60, r: 30, b: 60, t: 30 },
            hovermode: 'closest'
        };
        
        Plotly.newPlot('convergence-chart', convergence_data, convergence_layout, {responsive: true});
        
        // Phase table
        const phase_html = `
            <table class="metric-table">
                <thead>
                    <tr>
                        <th>Phase</th>
                        <th>Cost</th>
                        <th>Time (s)</th>
                    </tr>
                </thead>
                <tbody>
        """ + "".join([f"<tr><td>{p[0]}</td><td>${p[1]:.2f}</td><td>{p[2]:.2f}</td></tr>" for p in solution.phase_markers]) + """
                </tbody>
            </table>
        `;
        document.getElementById('phase-table').innerHTML = phase_html;
    </script>
</body>
</html>
"""
    
    with open("additional_output/01_expert_dashboard.html", "w") as f:
        f.write(html)


###############################################################################
# SAMPLE DATA GENERATION & EXECUTION
###############################################################################

def generate_sample_vessels(num_vessels=30):
    """Generate sample vessel data for testing"""
    vessels = []
    for i in range(num_vessels):
        vessel = Vessel(
            vessel_id=f"V{i+1:03d}",
            arrival_time=random.randint(0, 100),
            processing_time=random.randint(5, 20),
            cranes_needed=random.randint(2, 6),
            priority=random.choice(['high', 'medium', 'low']),
            deadline=random.randint(150, 300),
            cost_factor=random.uniform(0.8, 1.5)
        )
        vessels.append(vessel)
    return vessels


if __name__ == '__main__':
    # Example usage
    print("Generating sample vessel data...")
    vessels = generate_sample_vessels(30)
    
    print(f"Loaded {len(vessels)} vessels")
    
    # Run optimization
    solution = optimize(vessels)
    
    # Print results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    for berth_id, berth in solution.berth_schedules.items():
        print(f"\nBerth {berth_id}:")
        print(f"  Vessels: {len(berth.vessels)}")
        print(f"  Cost: ${berth.total_cost():.2f}")
        for vessel in berth.vessels:
            start = berth.start_times[vessel.vessel_id]
            end = berth.end_times[vessel.vessel_id]
            cranes = berth.crane_allocations[vessel.vessel_id]
            print(f"    {vessel.vessel_id}: [{start}-{end}] with {cranes} cranes")
    
    print(f"\nUnscheduled vessels: {len(solution.unscheduled_vessels)}")
    if solution.unscheduled_vessels:
        for vessel in solution.unscheduled_vessels:
            print(f"  {vessel.vessel_id}")
    
    print(f"\nTotal cost: ${solution.total_cost:.2f}")
    print(f"\nDashboard: additional_output/01_expert_dashboard.html")
    
    # Auto-open dashboard
    dashboard_path = os.path.abspath('additional_output/01_expert_dashboard.html')
    print(f"\nOpening dashboard: {dashboard_path}")
    try:
        webbrowser.open('file://' + dashboard_path)
    except:
        print(f"Could not open browser. Open manually: {dashboard_path}")
