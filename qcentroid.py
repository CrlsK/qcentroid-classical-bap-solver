"""
Classical BAP+QCA Greedy-2Opt Solver v3.1
Berth Allocation + Quay Crane Assignment using greedy construction + 2-opt local search.

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

logger = logging.getLogger("qcentroid-user-log")