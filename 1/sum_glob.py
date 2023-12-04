import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from stl_1d import *
import os


def sum_globally(m, r, t, t1, t2):
    '''
    add constraints for specification globally_[a,b] spec, calculate robustness r_out for the spec
    :param m: Gurobi Model
    :param r: robustness of the specification
    :param t: current timestep
    :param t1: start timestep
    :param t2: end timestep
    :return:
    '''
    atN = t1 + t
    btN = t2 + t + 1

    assert btN <= r.shape[0], "end timestep is larger than the horizon"
    r_out = sum_and(m, r[atN:btN])

    return r_out



def sum_and(m, r):
    '''
    sum up the robustness, force each r >= 0
    :param N: Gurobi Model
    :param r: robustness of the specification
    :return:
    '''


    if type(r) == list:
        N = len(r)
    else:
        N = r.shape[0]
    r_out = m.addMVar(shape=(1, 1), vtype=GRB.CONTINUOUS, name="r_out", lb=-GRB.INFINITY, ub=GRB.INFINITY)


    for i in range(N):
        m.addConstr(r[i] >= 0)
    m.addConstr(r_out == r.sum())
    return r_out


# start main function if name is main
if __name__ == "__main__":
    # set time horizon
    T = 50

    # create model
    m = gp.Model("exp1")

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

    exp = 'sum'

    rho, z = predicate(m, y, 1, 20, robustness='spatial')  # a*y - b > 0
    rs = sum_globally(m, rho, 0, 10, 20)
    r = rs

    # force r >= 0
    m.addConstr(r >= 0)

    # objective quadratic cost
    m.addConstr(u <= 5)
    m.addConstr(u >= -5)
    Q = 0.01
    R = 0.01
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
    # plt.fill_between(np.arange(30, 41), -80, -20, color='gray', alpha=0.5)
    # set the y-axis range to [-60, 60]
    plt.ylim(-60, 60)


    # add title
    plt.title(exp)
    # save figure
    # output_dir = './exp1/'
    # create directory if not exist
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # plt.savefig(output_dir + exp + '.png', dpi=300)
    # save data
    # np.savez(output_dir + exp + '.npz', u=u.X, y=y.X, x=x.X, r=r.X)
    # save variables number and running time
    # with open(output_dir + exp + '.txt', 'w') as f:
    #     f.write('variables: %d\n' % m.NumVars)
    #     f.write('continuous variables: %d\n' % (m.NumVars - m.NumIntVars))
    #     f.write('integer variables: %d\n' % m.NumIntVars)
    #     f.write('binary variables: %d\n' % m.NumBinVars)
    #     f.write('constraints: %d\n' % m.NumConstrs)
    #     f.write('running time: %f\n' % m.Runtime)

    plt.show()

    # print r
    print("r: ", r.X)
