import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from stl import *
from predicate import predicate
import os

colors = plt.rcParams['axes.prop_cycle']

if __name__ == "__main__":
    # set time horizon
    T = 50

    # create model
    m = gp.Model("exp3")

    # system variable, double integral system
    d = 1
    dt = 1
    I = np.eye(d)
    z = np.zeros((d,d))
    A = np.block([[I, I * dt], [z, I]])
    B = np.block([[I * 0.5 * dt * dt], [I * dt]])
    C = np.block([I, z])
    D = np.block(z)

    u = m.addMVar(shape=(T + 1, 1), vtype=GRB.CONTINUOUS, name="u", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    x = m.addMVar(shape=(T + 1, 2), vtype=GRB.CONTINUOUS, name="x", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    y = m.addMVar(shape=(T + 1, 1), vtype=GRB.CONTINUOUS, name="y", lb=-GRB.INFINITY, ub=GRB.INFINITY)

    x0 = np.array([[0], [0]])

    # dynamic constraints
    m.addConstr(x[0, :] == x0)
    for t in range(T):
        m.addConstr(x[t + 1, :] == A @ x[t, :] + B @ u[t, :])
        m.addConstr(y[t, :] == C @ x[t, :] + D @ u[t, :])
    m.addConstr(y[T, :] == C @ x[T, :] + D @ u[T, :])

    # constraints
    # ['sum', 'mul']
    robustness = 'c_sum'
    output_dir = './g_2/'
    a = 1
    b = 10
    t_left = 15
    t_right = 25
    r1_s = predicate(m, y, a, b, robustness='s')
    r_s = globally_min(m, r1_s, 0, t_left, t_right)

    r1_t = predicate(m, y, a, b, robustness='t_min')
    r_t = globally_min(m, r1_t, 0, t_left, t_right)

    r = quant_and(m, [r_s, r_t])

    # goal constraint
    m.addConstr(y[T] == 0)  # reach the goal
    # m.addConstr(x[T, 1] == 0)  # zero velocity at the goal
    m.addConstr(r >= 0)

    # add system constraints
    u_max = 0.2
    m.addConstr(u <= u_max)
    m.addConstr(u >= -u_max)

    # cost function
    cost = 0
    B = 10e-6
    Q = np.diag([0, 1]) * B  # just penalize high velocities
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
    x = x[:,1].X.flatten()
    y = y.X.flatten()
    r_p = r1.X.flatten()
    r = r.X.flatten()

    # create a figure that contains three subplots
    fig, axs = plt.subplots(4, 1, figsize=(6, 12), constrained_layout=True)
    # plot y
    axs[0].plot(t, y, '-')
    axs[0].set_title(f"Position")
    axs[0].fill_between(np.arange(t_left, t_right + 1), b / a, 80, color='springgreen', alpha=0.5)
    axs[0].set_xlim(-2, 52)
    axs[0].set_ylim(-2, 52)
    axs[0].axvline(x=t_left, color='gray', linestyle='--', alpha=0.5)
    axs[0].axvline(x=t_right, color='gray', linestyle='--', alpha=0.5)
    axs[0].axhline(y=b / a, color='gray', linestyle='--', alpha=0.5)
    axs[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # plot u
    axs[2].plot(t, u, '-')
    axs[2].set_title(f"Acceleration (Control Input)")
    axs[2].axvline(x=t_left, color='gray', linestyle='--', alpha=0.5)
    axs[2].axvline(x=t_right, color='gray', linestyle='--', alpha=0.5)
    axs[2].axhline(y=u_max, color='gray', linestyle='--', alpha=0.5)
    axs[2].text(0, u_max, 'u_max', ha='left', va='bottom', fontsize=10)
    axs[2].axhline(y=-u_max, color='gray', linestyle='--', alpha=0.5)
    axs[2].text(0, -u_max, 'u_min', ha='left', va='top', fontsize=10)
    max_value = np.max(np.abs(u))
    axs[2].set_ylim(-1.2 * max_value, 1.2 * max_value)
    axs[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # plot v
    axs[1].plot(t, x, '-')
    axs[1].set_title(f"Velocity")
    axs[1].axvline(x=t_left, color='gray', linestyle='--', alpha=0.5)
    axs[1].axvline(x=t_right, color='gray', linestyle='--', alpha=0.5)
    axs[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    v_max = 5
    axs[1].set_ylim(-v_max, v_max)


    # plot r
    axs[3].plot(t, r_p, '-', label=r'$\mu_1$')
    axs[3].set_title(f"Predicate Robustness")
    axs[3].set_xlabel("Time (steps)")
    axs[3].axvline(x=t_left, color='gray', linestyle='--', alpha=0.5)
    axs[3].axvline(x=t_right, color='gray', linestyle='--', alpha=0.5)
    max_value = np.max(np.abs(r_p))
    axs[3].set_ylim(-1.2 * max_value, 1.2 * max_value)
    axs[3].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axs[3].legend()


    if save_data:
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
