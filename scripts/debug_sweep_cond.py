import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('mpc_baselines_repo'))
import numpy as np

from scripts.run_pareto_cpu_low_latency import make_benchmark, get_u_bounds
from src.nonlinear_mpc import NonlinearMPC

bench = make_benchmark('reaching', seed=0)
bench.u_bounds = get_u_bounds(bench)
stage, term, Q, R, Qf = bench.make_costs()

for it in [1, 3, 5, 10, 15]:
    mpc = NonlinearMPC(bench.dyn.dynamics, stage, term, horizon=16, u_bounds=bench.u_bounds)
    s = bench.reset()
    success = False
    state = s.copy()
    for t in range(60):
        a = mpc.solve(state, max_iter=it)['action']
        if t < 5:
            print(f'  iters={it} step={t} state={state[:2]} action={a}')
        state = bench.step(state, a)
        if bench.is_success(state):
            success = True
            break
    print(f'iters={it} success={success} steps={t+1} final={state[:2]}')
