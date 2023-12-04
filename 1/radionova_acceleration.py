import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt


def qual_mu(m, y, a, b):
    '''
    add constraints for the predicate a*y - b > 0, calculate satisfaction z for each timestep
    :param m: Gurobi Model
    :param y: system output
    :param a:
    :param b:
    :return:
    '''
    M = 1e6
    eps = 1e-4

    N = y.shape[0]
    z = m.addMVar(shape=(N, d), vtype=GRB.BINARY, name="z")

    # radionova 2021
    for t in range(N):
        m.addConstr(M * (z[t, :] - 1) + eps <= a @ y[t,:] - b)
        m.addConstr(a @ y[t,:] - b <= (M + eps) * z[t, :] - eps)

    # raman donze 2014
    # m.addConstr(-(a * y - b) <= M * (1-z) - eps)
    # m.addConstr(a * y - b <= M * z - eps)
    return z


def quant_globally(m, r, t, t1, t2):
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
    r_out = quant_and(m, r[atN:btN, :])

    return r_out


def quant_and(m, r):
    '''
    add constraints for conjunction, calculate robustness r_out for the spec
    :param N: Gurobi Model
    :param r: robustness of the specification
    :return:
    '''
    M = 1e6

    if type(r) == list:
        N = len(r)
    else:
        N = r.shape[0]
    r_out = m.addMVar(shape=(1, d), vtype=GRB.CONTINUOUS, name="r_out", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    b = m.addMVar(shape=(N, d), vtype=GRB.BINARY, name="b")

    for i in range(N):
        m.addConstr(r[i, :] - M * (1 - b[i, :]) <= r_out)
        m.addConstr(r_out <= r[i, :])
    m.addConstr(b.sum() == 1)  # todo check
    return r_out


def quant_finally(m, r, t, t1, t2):
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
    r_out = quant_or(m, r[atN:btN, :])

    return r_out


def quant_or(m, r):
    '''
    add constraints for conjunction, calculate robustness r_out for the spec
    :param N: Gurobi Model
    :param r: robustness of the specification
    :return:
    '''
    M = 1e6

    if type(r) == list:
        N = len(r)
    else:
        N = r.shape[0]
    r_out = m.addMVar(shape=(1, d), vtype=GRB.CONTINUOUS, name="r_out", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    b = m.addMVar(shape=(N, d), vtype=GRB.BINARY, name="b")

    for i in range(N):
        m.addConstr(r[i, :] <= r_out)
        m.addConstr(r_out <= r[i, :] + M * (1 - b[i, :]))
    m.addConstr(b.sum() == 1)  # todo check
    return r_out




def predicate(m, y, a, b, robustness='spatial'):
    '''
    add constraints for the predicate a*y - b > 0, calculate the robustness
    :param m: Gurobi Model
    :param y: system output
    :param a:
    :param b:
    :return:
    '''
    # predicate constraints
    z = qual_mu(m, y, a, b)

    r = None
    # robustness
    assert robustness in ['spatial', 'temporal', 'combined'], "robustness type not supported"
    if robustness == 'spatial':
        rho = m.addMVar(shape=(y.shape[0], d), vtype=GRB.CONTINUOUS, name="rho", lb=-GRB.INFINITY, ub=GRB.INFINITY)
        for t in range(y.shape[0]):
            m.addConstr(rho[t, :] == a @ y[t, :] - b)
        r = rho
    elif robustness == 'temporal':
        theta = temporal_robustness(m, z)
        r = theta
    elif robustness == 'combined':
        pass
    return r, z

def temporal_robustness(m, z):
    '''
    calculate the temporal robustness
    :param m: Gurobi Model
    :param z: satisfaction of the predicate
    :return:
    '''
    N = z.shape[0]
    theta = m.addMVar(shape=(N, d), vtype=GRB.INTEGER, name="theta", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    c1 = m.addMVar(shape=(N+1, d), vtype=GRB.INTEGER, name="c1", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    c0 = m.addMVar(shape=(N+1, d), vtype=GRB.INTEGER, name="c2", lb=-GRB.INFINITY, ub=GRB.INFINITY)

    m.addConstr(c1[-1, :] == 0)  # todo
    m.addConstr(c0[-1, :] == 0)

    for t in range(N-1,-1,-1):
        m.addConstr(c1[t, :] == (c1[t+1, :]+1) * z[t, :])  # todo +1 or array
        m.addConstr(c0[t, :] == (c0[t+1, :]-1) * (1-z[t, :]))

    c1_shift = c1[0:N, :] - z  # todo shift has no meaning?
    c0_shift = c0[0:N, :] + (1-z)  # todo 1 or array
    m.addConstr(theta == c1_shift + c0_shift)
    return theta


# start main function if name is main
if __name__ == "__main__":
    # set time horizon
    T = 50

    # create model
    m = gp.Model("test")

    # system variable
    global d
    d = 1
    I = np.eye(d)
    z = np.zeros((d,d))
    A = np.block([[I, I], [z, I]])
    B = np.block([[z], [I]])
    C = np.block([I, z])
    D = np.block(z)


    u = m.addMVar(shape=(T + 1, d), vtype=GRB.CONTINUOUS, name="u", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    x = m.addMVar(shape=(T + 1, 2*d), vtype=GRB.CONTINUOUS, name="x", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    y = m.addMVar(shape=(T + 1, d), vtype=GRB.CONTINUOUS, name="y", lb=-GRB.INFINITY, ub=GRB.INFINITY)

    x0 = np.array([[10, 0]])

    # dynamic
    m.addConstr(x[0, :] == x0)
    for t in range(T):
        m.addConstr(x[t + 1, :] == A @ x[t, :] + B @ u[t, :])
        m.addConstr(y[t, :] == C @ x[t, :] + D @ u[t, :])
    m.addConstr(y[T, :] == C @ x[T, :] + D @ u[T, :])

    a = np.array([[1]])
    b = np.array([[20]])
    rho, z = predicate(m, y, a, b, robustness='temporal')  # a @ y - b > 0
    r = quant_globally(m, rho, 0, 10, 20)


    # force r >= 0
    m.addConstr(r >= 0)

    # m.setParam('InfUnbdInfo', 1)

    # objective quadratic cost
    m.addConstr(u <= 1)
    m.addConstr(u >= -1)
    cost = 0
    Q = np.diag([0,1])  # just penalize high velocities
    R = np.eye(1)
    for t in range(0, T+1):
        cost += x[t, :] @ Q @ x[t, :] + u[t, :] @ R @ u[t, :]
    m.setObjective(r - cost * 0.1, GRB.MAXIMIZE)


    # solve
    m.optimize()

    assert m.status == GRB.OPTIMAL, "Optimization was stopped with status %d" % m.status

    # print u
    # print(u.X)

    # print r
    print(r.X)

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
    # plt.ylim(-60, 60)
    plt.show()
