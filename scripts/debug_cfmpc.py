import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('mpc_baselines_repo'))
from src.utils.dynamics import PointMass2D
from src.utils.obstacles import CircleObstacle, is_in_collision
from src.linear_mpc import LinearMPC
from src.nonlinear_mpc import NonlinearMPC
from src.collision_free_mpc import CollisionFreeMPC, SDFWorld
import numpy as np

from scripts.run_pareto_cpu_low_latency import _FallbackReaching

bench = _FallbackReaching(seed=0)
stage, term, Q, R, Qf = bench.make_costs()
u_bounds = (-5*np.ones(2), 5*np.ones(2))
mpc = CollisionFreeMPC(bench.dyn.dynamics, stage, term, bench.world,
                        horizon=16, u_bounds=u_bounds, collision_weight=100.0, ilqr_iters=15)

s = bench.reset()
print('start', s, 'goal', bench.goal)
for t in range(60):
    U = mpc.solve(s)
    a = U[0]
    print(t, s[:2], a)
    s = bench.step(s, a)
    if bench.is_success(s):
        print('success at', t, s)
        break
    if bench.is_collision(s):
        print('collision at', t, s)
        break
else:
    print('timeout', s)
