import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from stl import *
from predicate import predicate
import os

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
    # ['s', 't_min', 't_sum', 't_left', 't_right', 'c_min', 'c_sum', 'c_left', 'c_right']
    robustness = 'c_sum'
    a1 = 1
    b1 = 10
    t_left1 = 15
    t_right1 = 25
    r1 = predicate(m, y, a1, b1, robustness)
    r1s = globally_min(m, r1, 0, t_left1, t_right1)

    a2 = -1
    b2 = -20
    t_left2 = 15
    t_right2 = 25
    r2 = predicate(m, y, a2, b2, robustness)
    r2s = globally_min(m, r2, 0, t_left2, t_right2)

    r = quant_and(m, [r1s, r2s])

    # goal constraint
    m.addConstr(y[T] == 0)
    m.addConstr(r >= 0)

    # add system constraints
    u_max = 3
    m.addConstr(u <= u_max)
    m.addConstr(u >= -u_max)

    # cost function
    cost = 0
    B = 10e-5
    Q = np.eye(1) * B
    R = np.eye(1) * B
    for t in range(0, T + 1):
        cost += x[t, :] @ Q @ x[t, :] + u[t, :] @ R @ u[t, :]

    A = 1
    cost = cost - A * r
    m.setObjective(cost, GRB.MINIMIZE)

    # solve
    m.optimize()
    assert m.status == GRB.OPTIMAL, "Optimization was stopped with status %d" % m.status

    save_data = True

    t = np.arange(0, T + 1)
    u = u.X.flatten()
    x = x.X.flatten()
    y = y.X.flatten()
    r_p1 = r1.X.flatten()
    r_p2 = r2.X.flatten()
    r = r.X.flatten()

    # create a figure that contains three subplots
    fig, axs = plt.subplots(3, 1, figsize=(6, 9), constrained_layout=True)
    # plot y
    axs[0].plot(t, y, '-')
    axs[0].set_title(f"Position, r={r[0]:.01f}")
    axs[0].fill_between(np.arange(t_left1, t_right1 + 1), b1 / a1, b2 / a2, color='springgreen', alpha=0.5)
    axs[0].set_xlim(-2, 52)
    axs[0].set_ylim(-2, 22)
    axs[0].axvline(x=t_left1, color='gray', linestyle='--', alpha=0.5)
    axs[0].axvline(x=t_right2, color='gray', linestyle='--', alpha=0.5)
    axs[0].axhline(y=b1 / a1, color='gray', linestyle='--', alpha=0.5)
    axs[0].axhline(y=b2 / a2, color='gray', linestyle='--', alpha=0.5)
    axs[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # plot u
    axs[1].plot(t, u, '-')
    axs[1].set_title(f"Control Input")
    axs[1].axvline(x=t_left1, color='gray', linestyle='--', alpha=0.5)
    axs[1].axvline(x=t_right2, color='gray', linestyle='--', alpha=0.5)
    axs[1].axhline(y=u_max, color='gray', linestyle='--', alpha=0.5)
    axs[1].text(0, u_max, 'u_max', ha='left', va='bottom', fontsize=10)
    axs[1].axhline(y=-u_max, color='gray', linestyle='--', alpha=0.5)
    axs[1].text(0, -u_max, 'u_min', ha='left', va='top', fontsize=10)
    max_value = np.max(np.abs(u))
    axs[1].set_ylim(-1.2 * max_value, 1.2 * max_value)
    axs[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # plot r
    axs[2].plot(t, r_p1, '-', label=r'$\mu_1$')
    axs[2].plot(t, r_p2, '-', label=r'$\mu_2$')
    axs[2].set_title(f"Predicate Robustness")
    axs[2].set_xlabel("Time (steps)")
    axs[2].axvline(x=t_left1, color='gray', linestyle='--', alpha=0.5)
    axs[2].axvline(x=t_right2, color='gray', linestyle='--', alpha=0.5)
    max_value = max(np.max(np.abs(r_p1)), np.max(np.abs(r_p2)))
    axs[2].set_ylim(-1.2 * max_value, 1.2 * max_value)
    axs[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axs[2].legend()

    if save_data:
        output_dir = './simple_gg/'
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
                 r_p1=r_p1,
                 r_p2=r_p2,
                 t=t,
                 u_max=u_max,
                 t_left1=t_left1,
                 t_right2=t_right2,
                 t_right1=t_right1,
                 t_left2=t_left2,
                 b1=b1,
                 a1=a1,
                 b2=b2,
                 a2=a2,
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
    else:
        plt.show()
