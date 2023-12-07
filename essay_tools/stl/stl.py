from gurobipy import GRB

def globally_min(m, r, t, t1, t2):
    '''
    define specification G_[a,b] spec1, return its robustness
    :param m: Gurobi Model
    :param r: robustness of the spec1
    :param t: current timestep
    :param t1: start timestep
    :param t2: end timestep
    :return:
    '''
    atN = t1 + t
    btN = min(t2 + t + 1, r.shape[0])
    r_out = quant_and(m, r[atN:btN])  # return the min
    return r_out


def quant_and(m, r):
    '''
    return the min of r
    :param N: Gurobi Model
    :param r: robustness of the specification
    :return:
    '''
    M = 1e6

    if type(r) == list:
        N = len(r)
    else:
        N = r.shape[0]
    r_out = m.addMVar(shape=(1, 1), vtype=GRB.CONTINUOUS, name="r_out", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    b = m.addMVar(shape=(N, 1), vtype=GRB.BINARY, name="b")

    for i in range(N):
        m.addConstr(r[i] - M * (1 - b[i]) <= r_out)
        m.addConstr(r_out <= r[i])
        # m.addConstr(r[i] >= 0)  # force all r >= 0
    m.addConstr(b.sum() == 1)
    return r_out

def finally_max(m, r, t, t1, t2):
    '''
    define specification F_[a,b] spec1, return its robustness
    :param m: Gurobi Model
    :param r: robustness of the specification
    :param t: current timestep
    :param t1: start timestep
    :param t2: end timestep
    :return:
    '''
    atN = t1 + t
    btN = min(t2 + t + 1, r.shape[0])
    r_out = quant_or(m, r[atN:btN])

    return r_out


def quant_or(m, r):
    '''
    return the max of r
    :param N: Gurobi Model
    :param r: robustness of the specification
    :return:
    '''
    M = 1e6

    if type(r) == list:
        N = len(r)
    else:
        N = r.shape[0]
    r_out = m.addMVar(shape=(1, 1), vtype=GRB.CONTINUOUS, name="r_out", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    b = m.addMVar(shape=(N, 1), vtype=GRB.BINARY, name="b")

    for i in range(N):
        m.addConstr(r[i] <= r_out)
        m.addConstr(r_out <= r[i] + M * (1 - b[i]))
    m.addConstr(b.sum() == 1)
    # m.addConstr(r_out >= 0)  # force at least one r >= 0
    return r_out

def globally_sum(m, r, t, t1, t2):
    '''
    define specification G_[a,b] spec1, return its robustness
    :param m: Gurobi Model
    :param r: robustness of the specification
    :param t: current timestep
    :param t1: start timestep
    :param t2: end timestep
    :return:
    '''

    atN = t1 + t
    btN = min(t2 + t + 1, r.shape[0])
    r_out = m.addMVar(shape=(1, 1), vtype=GRB.CONTINUOUS, name="r_out", lb=-GRB.INFINITY, ub=GRB.INFINITY)

    for t in range(atN, btN):
        m.addConstr(r[t] >= 0)  # force all r >= 0

    m.addConstr(r_out == r[atN:btN].sum())
    return r_out

