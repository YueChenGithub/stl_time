import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from stl import *
from predicate import predicate
import os
from find_max_area import G_left, G_right

if __name__ == "__main__":
    # set time horizon
    T = 50

    # create model
    m = gp.Model("exp3")

    # system variable, simple system
    d = 1
    I = np.eye(d)
    z = np.zeros((d, d))
    A = np.block(I)
    B = np.block(I)
    C = np.block(I)
    D = np.block(z)

    u = m.addMVar(shape=(T + 1, 1), vtype=GRB.CONTINUOUS, name="u", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    x = m.addMVar(shape=(T + 1, 1), vtype=GRB.CONTINUOUS, name="x", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    y = m.addMVar(shape=(T + 1, 1), vtype=GRB.CONTINUOUS, name="y", lb=-GRB.INFINITY, ub=GRB.INFINITY)

    x0 = np.array([0])

    # dynamic constraints
    m.addConstr(x[0, :] == x0)
    for t in range(T):
        m.addConstr(x[t + 1, :] == A @ x[t, :] + B @ u[t, :])
        m.addConstr(y[t, :] == C @ x[t, :] + D @ u[t, :])
    m.addConstr(y[T, :] == C @ x[T, :] + D @ u[T, :])

    # constraints
    robustness = 'c_min'
    a = 1
    b = 10
    t_left = 15
    t_right = 25
    r1 = predicate(m, y, a, b, robustness)
    r = globally_min(m, r1, 0, 15, 25)
    # r = globally_sum(m, r1, 0, t_left, t_right)
    # r = finally_max(m, r1, 0, 15, 25)

    # goal constraint
    m.addConstr(y[T] == 0)
    m.addConstr(r >= 0)

    # add system constraints
    u_max = 1
    m.addConstr(u <= u_max)
    m.addConstr(u >= -u_max)

    # cost function
    cost = 0
    Q = np.eye(1) * 10e-6
    R = np.eye(1) * 10e-6
    for t in range(0, T + 1):
        cost += x[t, :] @ Q @ x[t, :] + u[t, :] @ R @ u[t, :]

    A = 1
    cost = cost - A * r
    m.setObjective(cost, GRB.MINIMIZE)

    # solve
    m.optimize()
    assert m.status == GRB.OPTIMAL, "Optimization was stopped with status %d" % m.status

    t = np.arange(0, T + 1)
    u = u.X.flatten()
    x = x.X.flatten()
    y = y.X.flatten()
    r_p = r1.X.flatten()
    r = r.X.flatten()

    # create a figure that contains three subplots
    fig, axs = plt.subplots(4, 1, figsize=(6, 10), constrained_layout=True)
    # plot y
    axs[0].plot(t, y, '-')
    axs[0].set_title(f"Position, r={r[0]:.01f}")
    axs[0].fill_between(np.arange(t_left, t_right + 1), b / a, 80, color='springgreen', alpha=0.5)
    axs[0].set_ylim(-2, 32)
    axs[0].axvline(x=t_left, color='gray', linestyle='--', alpha=0.5)
    axs[0].axvline(x=t_right, color='gray', linestyle='--', alpha=0.5)
    axs[0].axhline(y=b / a, color='gray', linestyle='--', alpha=0.5)

    # plot u
    axs[1].plot(t, u, '-')
    axs[1].set_title(f"Control Input")
    axs[1].axvline(x=t_left, color='gray', linestyle='--', alpha=0.5)
    axs[1].axvline(x=t_right, color='gray', linestyle='--', alpha=0.5)
    axs[1].axhline(y=u_max, color='gray', linestyle='--', alpha=0.5)
    axs[1].text(0, u_max, 'u_max', ha='left', va='bottom', fontsize=10)
    axs[1].axhline(y=-u_max, color='gray', linestyle='--', alpha=0.5)
    axs[1].text(0, -u_max, 'u_min', ha='left', va='top', fontsize=10)
    max_value = np.max(np.abs(u))
    axs[1].set_ylim(-1.2 * max_value, 1.2 * max_value)

    # plot r
    axs[2].plot(t, r_p, '-')
    axs[2].set_title(f"Robustness")
    axs[2].set_xlabel("Time (steps)")
    axs[2].axvline(x=t_left, color='gray', linestyle='--', alpha=0.5)
    axs[2].axvline(x=t_right, color='gray', linestyle='--', alpha=0.5)
    max_value = np.max(np.abs(r_p))
    axs[2].set_ylim(-1.2 * max_value, 1.2 * max_value)

    # plot robustness
    axs[3].plot(t, y, '-')
    axs[3].set_title(f"Robustness")
    axs[3].fill_between(np.arange(t_left, t_right + 1), b / a, 80, color='springgreen', alpha=0.5)
    axs[3].set_ylim(-2, 32)
    axs[3].axvline(x=t_left, color='gray', linestyle='--', alpha=0.5)
    axs[3].axvline(x=t_right, color='gray', linestyle='--', alpha=0.5)
    axs[3].axhline(y=b / a, color='gray', linestyle='--', alpha=0.5)
    robustness_right, rectangle_right = G_right(a, b, t_left, t_right, y)
    axs[3].fill_between([rectangle_right[0], rectangle_right[1]], [rectangle_right[2]], [rectangle_right[3]],
                        color='coral', alpha=0.5, label=f'Robustness Right={robustness_right:.01f}')
    robustness_left, rectangle_left = G_left(a, b, t_left, t_right, y)
    axs[3].fill_between([rectangle_left[0], rectangle_left[1]], [rectangle_left[2]], [rectangle_left[3]],
                        color='lightcoral', alpha=0.5, label=f'Robustness Left={robustness_left:.01f}')
    axs[3].legend()

    output_dir = './simple_g/'
    # create directory if not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir + robustness + '.png', dpi=300)
    plt.show()

    # save data
    np.savez(output_dir + robustness + '.npz',
             u=u,
             y=y,
             x=x,
             r=r,
             r_p=r_p,
             t=t,
             u_max=u_max,
             t_left=t_left,
             t_right=t_right,
             b=b,
             a=a,
             rectangle_right=rectangle_right,
             rectangle_left=rectangle_left,
             robustness_right=robustness_right,
             robustness_left=robustness_left,
             Q=Q,
             R=R,
             variables=m.NumVars,
             continuous_variables=m.NumVars - m.NumIntVars,
             integer_variables=m.NumIntVars,
             binary_variables=m.NumBinVars,
             constraints=m.NumConstrs,
             running_time=m.Runtime
             )
    # save variables number and running time
    with open(output_dir + robustness + '.txt', 'w') as f:
        f.write('variables: %d\n' % m.NumVars)
        f.write('continuous variables: %d\n' % (m.NumVars - m.NumIntVars))
        f.write('integer variables: %d\n' % m.NumIntVars)
        f.write('binary variables: %d\n' % m.NumBinVars)
        f.write('constraints: %d\n' % m.NumConstrs)
        f.write('running time: %f\n' % m.Runtime)
