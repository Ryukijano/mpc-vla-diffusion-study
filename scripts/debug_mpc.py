import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('mpc_baselines_repo'))
from src.utils.dynamics import PointMass2D
from src.utils.obstacles import CircleObstacle, is_in_collision
from src.nonlinear_mpc import NonlinearMPC
from src.linear_mpc import LinearMPC
from src.collision_free_mpc import CollisionFreeMPC, SDFWorld
import numpy as np

from scripts.run_pareto_cpu_low_latency import _FallbackReaching

bench = _FallbackReaching(seed=0)
stage, term, Q, R, Qf = bench.make_costs()
u_bounds = (-5*np.ones(2), 5*np.ones(2))
A, B = bench.dyn.linearize(np.zeros(4), np.zeros(2))

print('A', A, 'B', B)
print('start', bench.reset(), 'goal', bench.goal)

nmpc = NonlinearMPC(bench.dyn.dynamics, stage, term, horizon=16, u_bounds=u_bounds)
for it in [1,3,5,10,15]:
    res = nmpc.solve(bench.reset(), max_iter=it)
    print('it', it, 'action', res['action'], 'cost', res['cost'])

lmpc = LinearMPC(A, B, Q, R, Qf, horizon=10, u_bounds=u_bounds)
ref = np.tile(bench.goal, (11, 1))
res = lmpc.solve(bench.reset(), ref)
print('lmpc action', res.control)
