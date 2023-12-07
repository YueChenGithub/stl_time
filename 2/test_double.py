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

    x0 = np.array([[0, 0]])

    # dynamic constraints
    m.addConstr(x[0, :] == x0)
    for t in range(T):
        m.addConstr(x[t + 1, :] == A @ x[t, :] + B @ u[t, :])
        m.addConstr(y[t, :] == C @ x[t, :] + D @ u[t, :])
    m.addConstr(y[T, :] == C @ x[T, :] + D @ u[T, :])


    # constraints
    robustness = 'c_sum'
    r1 = predicate(m, y, 1, 10, robustness)
    r1s = globally_min(m, r1, 0, 10, 15)
    # r1 = globally_halder2022(m, r1, 0, 10, 15)

    r2 = predicate(m, y, -1, 10, robustness)
    r2s = globally_min(m, r2, 0, 35, 40)
    # r2 = globally_halder2022(m, r2, 0, 35, 40)

    r = quant_and(m, [r1s, r2s])

    # goal constraint
    m.addConstr(y[T] == 0)
    m.addConstr(x[T, 1] == 0)
    m.addConstr(r >= 0)

    # add system constraints
    u_max = 0.5
    m.addConstr(u <= u_max)
    m.addConstr(u >= -u_max)


    # cost function
    cost = 0
    Q = np.diag([0, 1]) * 10e-6  # just penalize high velocities
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
    # plot u
    plt.plot(np.arange(0, T + 1), u.X.flatten() / u_max * 10, '-', label=f"u, scaled")

    # plot r
    r = np.maximum(r1.X.flatten(), r2.X.flatten())
    plt.plot(np.arange(0, T + 1), r / max(np.abs(r)) * 10, '-', label=f"r, scaled")

    plt.legend()
    plt.xlabel("timestep (t)")
    plt.ylabel("y")
    plt.fill_between(np.arange(10, 16), 10, 40, color='gray', alpha=0.5)
    plt.fill_between(np.arange(35, 41), -10, -40, color='gray', alpha=0.5)
    plt.ylim(-35, 35)
    plt.title(f"robustness: {robustness}")
    plt.show()


