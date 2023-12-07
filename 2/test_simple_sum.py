import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
from stl.stl import *
from stl.predicate import predicate

if __name__ == "__main__":
    # set time horizon
    T = 50

    # create model
    m = gp.Model("exp3")


    # system variable, simple system
    d = 1
    I = np.eye(d)
    z = np.zeros((d,d))
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
    robustness_spatial = 's'
    r1_spatial = predicate(m, y, 1, 10, robustness_spatial)
    r1s_spatial = globally_min(m, r1_spatial, 0, 15, 25)
    # r1s_spatial = globally_sum(m, r1_spatial, 0, 15, 25)

    robustness_temporal = 't_min'
    r1_temporal = predicate(m, y, 1, 10, robustness_temporal)
    r1s_temporal = globally_min(m, r1_temporal, 0, 15, 25)
    # r1s_temporal = globally_sum(m, r1_temporal, 0, 15, 25)

    c_temporal = 0.5
    c_spatial = 0.5
    r = m.addMVar(shape=(1, 1), vtype=GRB.CONTINUOUS, name="r", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    m.addConstr(r == c_temporal * r1s_temporal + c_spatial * r1s_spatial)




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


    # plot y
    plt.plot(np.arange(0, T + 1), y.X.flatten(), '-', label=f"y, r={r.X[0][0]:.01f}")
    # plt.plot(np.arange(0, T + 1), u.X.flatten() / u_max * 5, '-', label=f"u, scaled")

    r = c_temporal * r1_temporal.X.flatten() + c_spatial * r1_spatial.X.flatten()
    plt.plot(np.arange(0, T + 1), r / max(np.abs(r)) * 10, '-', label=f"r, scaled")
    plt.legend()
    plt.xlabel("timestep (t)")
    plt.ylabel("y")
    plt.fill_between(np.arange(15, 26), 10, 80, color='gray', alpha=0.5)
    plt.ylim(-11, 35)
    plt.title(f"robustness: {c_spatial} + {robustness_spatial} * {c_temporal} + {robustness_temporal}")
    plt.show()


