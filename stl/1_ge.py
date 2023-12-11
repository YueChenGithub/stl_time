import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from stl import *
from predicate import predicate
import os
from eval.eval import *
from draw_graph1 import draw_and_save
import argparse
import scipy

def main(method, x0):
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

    x0 = np.array([x0, 0])

    # dynamic constraints
    m.addConstr(x[0, :] == x0)
    for t in range(T):
        m.addConstr(x[t + 1, :] == A @ x[t, :] + B @ u[t, :])
        m.addConstr(y[t, :] == C @ x[t, :] + D @ u[t, :])
    m.addConstr(y[T, :] == C @ x[T, :] + D @ u[T, :])

    # constraints
    a1 = 1
    b1 = 10
    t_left1 = 10
    t_right1 = 15
    a2 = -1
    b2 = 10
    t_left2 = 35
    t_right2 = 40
    if method not in ['mul', 'sum', 'xi']:
        # predicate level or operator level
        global_operator = globally_min
        robustness = method
        weight_F = 1
        if method == '0':
            robustness = 's'
        if 'G' in method:
            # operator level
            robustness = method[2:]
            global_operator = globally_sum
            weight_F = 10

        # define predicate and specification
        r1 = predicate(m, y, a1, b1, robustness)
        r1s = global_operator(m, r1, 0, t_left1, t_right1)

        r2 = predicate(m, y, a2, b2, robustness)
        r2s = finally_max(m, r2, 0, t_left2, t_right2) * weight_F
        r = quant_and(m, [r1s, r2s])

    elif method in ['mul', 'sum']:
        # specification level
        r1_1 = predicate(m, y, a1, b1, robustness='s')
        r1s_1 = globally_min(m, r1_1, 0, t_left1, t_right1)
        r1_2 = predicate(m, y, a1, b1, robustness='t_min')
        r1s_2 = globally_min(m, r1_2, 0, t_left1, t_right1)

        r2_1 = predicate(m, y, a2, b2, robustness='s')
        r2s_1 = finally_max(m, r2_1, 0, t_left2, t_right2)
        r2_2 = predicate(m, y, a2, b2, robustness='t_min')
        r2s_2 = finally_max(m, r2_2, 0, t_left2, t_right2)

        r_1 = quant_and(m, [r1s_1, r2s_1])
        r_2 = quant_and(m, [r1s_2, r2s_2])

        r = m.addMVar(shape=(1, 1), vtype=GRB.CONTINUOUS, name="r", lb=-GRB.INFINITY, ub=GRB.INFINITY)
        if method == 'mul':
            m.setParam('NonConvex', 2)
            m.addConstr(r == r_1 * r_2)
            m.addConstr(r_1 >= 0)
            m.addConstr(r_2 >= 0)
        elif method == 'sum':
            m.addConstr(r == r_1 + r_2)
            m.addConstr(r_1 >= 0)
            m.addConstr(r_2 >= 0)
    elif method == 'xi':
        def get_rho(t0):
            r1_1 = predicate(m, y, a1, b1, robustness='s')
            r1s_1 = globally_min(m, r1_1, t0, t_left1, t_right1)
            r2_1 = predicate(m, y, a2, b2, robustness='s')
            r2s_1 = finally_max(m, r2_1, t0, t_left2, t_right2)
            return quant_and(m, [r1s_1, r2s_1])

        t_total = 5
        r_list = m.addMVar(shape=(t_total, 1), vtype=GRB.CONTINUOUS, name="r", lb=-GRB.INFINITY, ub=GRB.INFINITY)
        for t0 in range(t_total):
            r0 = quant_and(m, [get_rho(t0), get_rho(-t0)])
            m.addConstr(r_list[t0, 0] == r0 * scipy.stats.multivariate_normal.pdf(t0, mean=0, cov=5))
        r = m.addMVar(shape=(1, 1), vtype=GRB.CONTINUOUS, name="r", lb=-GRB.INFINITY, ub=GRB.INFINITY)
        m.addConstr(r == r_list.sum())





    # goal constraint
    # m.addConstr(y[T] == 0)  # reach the goal
    # m.addConstr(x[T, 1] == 0)  # zero velocity at the goal
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
    if method =='0': A = 0
    cost = cost - A * r
    m.setObjective(cost, GRB.MINIMIZE)

    # solve
    m.optimize()
    assert m.status == GRB.OPTIMAL, "Optimization was stopped with status %d" % m.status


    # save data and graph
    save_data = True
    output_dir = f'./output/1_ge_{x0[0]}/'
    draw_and_save(Q, R, T, a1, a2, b1, b2, m, output_dir, r, method, save_data, t_left1, t_left2, t_right1,
                  t_right2, u, u_max, x, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run main function with variable x.")
    # ['s', 't_min', 'c_min', 'c_sum2', 'G_s', 'G_c_min', 'mul', 'sum', '0', 'xi']
    parser.add_argument("-method", type=str, default='xi')
    parser.add_argument("-x0", type=float, default=0)
    args = parser.parse_args()
    main(args.method, args.x0)
