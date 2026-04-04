"""
Classical BAP+QCA Greedy-2Opt-OrOpt-3Opt Solver v7.0
Berth Allocation + Quay Crane Assignment.
v7.0: Crane-budget-aware greedy, deadline-aware target crane allocation, gentler post-hoc enforcement.
"""
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