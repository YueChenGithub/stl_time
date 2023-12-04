import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from stl_1d import *
import os

# start main function if name is main
if __name__ == "__main__":
    # set time horizon
    T = 50

    # create model
    m = gp.Model("exp2")

    # system variable
    u = m.addMVar(shape=(T + 1, 1), vtype=GRB.CONTINUOUS, name="u", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    x = m.addMVar(shape=(T + 1, 1), vtype=GRB.CONTINUOUS, name="x", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    y = m.addMVar(shape=(T + 1, 1), vtype=GRB.CONTINUOUS, name="y", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    A = 1
    B = 1
    C = 1
    D = 0
    x0 = 0

    # dynamic
    m.addConstr(x[0, :] == x0)
    for t in range(T):
        m.addConstr(x[t + 1, :] == A * x[t, :] + B * u[t, :])
        m.addConstr(y[t, :] == C * x[t, :] + D * u[t, :])
    m.addConstr(y[T, :] == C * x[T, :] + D * u[T, :])

    exp = 'F|F_cs'

    rho1, z1 = predicate(m, y, 1, 20, robustness='spatial')  # a*y - b > 0
    r1 = quant_finally(m, rho1, 0, 10, 20)
    rho2, z2 = predicate(m, y, -1, 20, robustness='spatial')  # a*y - b > 0
    r2 = quant_finally(m, rho2, 0, 30, 40)
    rs = quant_or(m, [r1, r2])
    r = rs

    # rho1, z1 = predicate(m, y, 1, 20, robustness='temporal')  # a*y - b > 0
    # r1 = quant_finally(m, rho1, 0, 10, 20)
    # rho2, z2 = predicate(m, y, -1, 20, robustness='temporal')  # a*y - b > 0
    # r2 = quant_finally(m, rho2, 0, 30, 40)
    # rt = quant_or(m, [r1, r2])
    # r = rt

    # rho1, z1 = predicate(m, y, 1, 20, robustness='combined')  # a*y - b > 0
    # r1 = quant_finally(m, rho1, 0, 10, 20)
    # rho2, z2 = predicate(m, y, -1, 20, robustness='combined')  # a*y - b > 0
    # r2 = quant_finally(m, rho2, 0, 30, 40)
    # rt = quant_or(m, [r1, r2])
    # r = rt

    # r = m.addMVar(shape=(1, 1), vtype=GRB.CONTINUOUS, name="r", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    # m.setParam("NonConvex", 2)
    # m.addConstr(r == rs * rt)
    # m.addConstr(rs >= 0)
    # m.addConstr(rt >= 0)

    # force r >= 0
    m.addConstr(r >= 0)

    # objective quadratic cost
    m.addConstr(u <= 10)
    m.addConstr(u >= -10)
    Q = 0.03
    R = 0.03
    cost = x.transpose() @ x * Q + u.transpose() @ u * R
    m.setObjective(r - cost, GRB.MAXIMIZE)

    # solve
    m.optimize()

    assert m.status == GRB.OPTIMAL, "Optimization was stopped with status %d" % m.status

    # plot solution u and y
    # plt.plot(np.arange(0, T + 1), u.X.flatten(), '-', label="u")
    plt.plot(np.arange(0, T + 1), y.X.flatten(), '-', label="y")
    plt.legend()
    plt.xlabel("timestep (t)")
    # plot a region where y >= 20 and 10 <= t <= 20
    plt.fill_between(np.arange(10, 21), 20, 80, color='gray', alpha=0.5)
    # plot a region where y <= -20 and 30 <= t <= 40
    plt.fill_between(np.arange(30, 41), -80, -20, color='gray', alpha=0.5)
    # set the y-axis range to [-60, 60]
    plt.ylim(-60, 60)


    # add title
    plt.title(exp)
    # save figure
    output_dir = './exp2/'
    # create directory if not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir + exp + '.png', dpi=300)
    # save data
    np.savez(output_dir + exp + '.npz', u=u.X, y=y.X, x=x.X, r=r.X)
    # save variables number and running time
    with open(output_dir + exp + '.txt', 'w') as f:
        f.write('variables: %d\n' % m.NumVars)
        f.write('continuous variables: %d\n' % (m.NumVars - m.NumIntVars))
        f.write('integer variables: %d\n' % m.NumIntVars)
        f.write('binary variables: %d\n' % m.NumBinVars)
        f.write('constraints: %d\n' % m.NumConstrs)
        f.write('running time: %f\n' % m.Runtime)

    plt.show()

    # print r
    print("r: ", r.X)

    # print r1 r2
    print("r1: ", r1.X)
    print("r2: ", r2.X)