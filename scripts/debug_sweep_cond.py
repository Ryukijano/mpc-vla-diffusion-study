import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('mpc_baselines_repo'))
import numpy as np

from scripts.run_pareto_cpu_low_latency import make_benchmark, get_u_bounds
from src.collision_free_mpc import CollisionFreeMPC

bench = make_benchmark('reaching', seed=0)
bench.u_bounds = get_u_bounds(bench)
stage, term, Q, R, Qf = bench.make_costs()

for it in [1, 3, 5, 10, 15]:
    mpc = CollisionFreeMPC(bench.dyn.dynamics, stage, term, bench.world,
                           horizon=16, u_bounds=bench.u_bounds, collision_weight=100.0, ilqr_iters=it)
    s = bench.reset()
    success = False
    state = s.copy()
    for t in range(60):
        U = mpc.solve(state)
        a = U[0]
        state = bench.step(state, a)
        if bench.is_success(state):
            success = True
            break
    print(f'iters={it} success={success} steps={t+1} final={state[:2]}')
