import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from stl import *
from predicate import predicate
import os
from eval.eval import *
from ggee_graph import draw_and_save
import argparse

def main(robustness):
    # set time horizon
    T = 50

    # create model
    m = gp.Model("exp3")

    # system variable, double integral system
    d = 1
    dt = 1
    I = np.eye(d)
    z = np.zeros((d, d))
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
    # ['s', 'c_min']
    output_dir = './ggee_1/'
    a1 = 1
    b1 = 10
    t_left1 = 15
    t_right1 = 25
    r1 = predicate(m, y, a1, b1, robustness)
    r1s = globally_sum(m, r1, 0, t_left1, t_right1)
    a3 = -1
    b3 = -20
    r3 = predicate(m, y, a3, b3, robustness)
    r3s = globally_sum(m, r3, 0, t_left1, t_right1)

    a2 = -1
    b2 = 10
    t_left2 = 25
    t_right2 = 50
    r2 = predicate(m, y, a2, b2, robustness)
    r2s = finally_max(m, r2, 0, t_left2, t_right2)
    a4 = 1
    b4 = -20
    r4 = predicate(m, y, a4, b4, robustness)
    r4s = finally_max(m, r4, 0, t_left2, t_right2)

    r = quant_and(m, [r1s, 10*r2s, r3s, 10*r4s])

    # goal constraint
    m.addConstr(x[T, 1] == 0)  # zero velocity at the goal
    m.addConstr(r >= 0)

    # add system constraints
    u_max = 1
    m.addConstr(u <= u_max)
    m.addConstr(u >= -u_max)
    m.addConstr(y <= 30)
    m.addConstr(y >= -30)

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
    draw_and_save(Q, R, T, a1, a2, a3, a4, b1, b2, b3, b4, m, output_dir, r, robustness, save_data, t_left1, t_left2,
                  t_right1, t_right2, u, u_max, x, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run main function with variable x.")
    parser.add_argument("-robustness", type=str, default='s')
    args = parser.parse_args()
    main(args.robustness)