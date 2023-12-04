from gurobipy import GRB
from stl import quant_and, quant_or

def predicate(m, y, a, b, robustness='s'):
    '''
    define a predicate a*y - b > 0, return the robustness
    :param m: Gurobi Model
    :param y: system output
    :param a:
    :param b:
    :return:
    '''

    r = None
    '''robustness list
    s: spatial robustness
    left_t: left hand temporal robustness
    right_t: right hand temporal robustness
    combined_t: combined temporal robustness
    '''
    robustness_list = ['s', 't_min', 't_sum', 't_left', 't_right', 'c_min', 'c_sum', 'c_left', 'c_right']
    assert robustness in robustness_list, f"robustness type must be in {str(robustness_list)}"
    if robustness == 's':
        r = spatial_robustness(m, y, a, b)
    if robustness == 't_left':
        z = satification(m, y, a, b)
        r = temporal_robustness_left(m, z, abs=False, zero=False)
    if robustness == 't_right':
        z = satification(m, y, a, b)
        r = temporal_robustness_right(m, z, abs=False, zero=False)
    if robustness == 't_min':
        z = satification(m, y, a, b)
        r = temporal_robustness_min(m, z, abs=False, zero=False)
    if robustness == 't_sum':
        z = satification(m, y, a, b)
        r = temporal_robustness_sum(m, z, abs=False, zero=False)
    if robustness == 'c_min':
        m.setParam("NonConvex", 2)
        rho = spatial_robustness(m, y, a, b)
        z = satification(m, y, a, b)
        theta = temporal_robustness_min(m, z, abs=True, zero=False)
        r = m.addMVar(shape=(y.shape[0], 1), vtype=GRB.CONTINUOUS, name="r", lb=-GRB.INFINITY, ub=GRB.INFINITY)
        m.addConstr(r == rho * theta)
    if robustness == 'c_sum':
        m.setParam("NonConvex", 2)
        rho = spatial_robustness(m, y, a, b)
        z = satification(m, y, a, b)
        theta = temporal_robustness_sum(m, z, abs=True, zero=False)
        r = m.addMVar(shape=(y.shape[0], 1), vtype=GRB.CONTINUOUS, name="r", lb=-GRB.INFINITY, ub=GRB.INFINITY)
        m.addConstr(r == rho * theta)
    if robustness == 'c_left':
        m.setParam("NonConvex", 2)
        rho = spatial_robustness(m, y, a, b)
        z = satification(m, y, a, b)
        theta = temporal_robustness_left(m, z, abs=True, zero=False)
        r = m.addMVar(shape=(y.shape[0], 1), vtype=GRB.CONTINUOUS, name="r", lb=-GRB.INFINITY, ub=GRB.INFINITY)
        m.addConstr(r == rho * theta)
    if robustness == 'c_right':
        m.setParam("NonConvex", 2)
        rho = spatial_robustness(m, y, a, b)
        z = satification(m, y, a, b)
        theta = temporal_robustness_right(m, z, abs=True, zero=False)
        r = m.addMVar(shape=(y.shape[0], 1), vtype=GRB.CONTINUOUS, name="r", lb=-GRB.INFINITY, ub=GRB.INFINITY)
        m.addConstr(r == rho * theta)
    return r

def satification(m, y, a, b):
    '''
    calculate satisfaction z for the predicate a*y - b > 0 at each timestep
    :param m: Gurobi Model
    :param y: system output
    :param a:
    :param b:
    :return:
    '''
    M = 1e6
    eps = 1e-4

    N = y.shape[0]
    z = m.addMVar(shape=(N, 1), vtype=GRB.BINARY, name="z")

    # radionova 2021
    m.addConstr(M * (z - 1) + eps <= a * y - b)
    m.addConstr(a * y - b <= (M + eps) * z - eps)

    # raman donze 2014
    # m.addConstr(-(a * y - b) <= M * (1-z) - eps)
    # m.addConstr(a * y - b <= M * z - eps)
    return z

def spatial_robustness(m, y, a, b):
    '''
    calculate the spatial robustness of the predicate a*y - b > 0
    :param m: Gurobi Model
    :param y: system output
    :param a:
    :param b:
    :return:
    '''
    rho = m.addMVar(shape=(y.shape[0], 1), vtype=GRB.CONTINUOUS, name="rho", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    m.addConstr(rho == a * y - b)
    return rho

def temporal_robustness_right(m, z, abs=False, zero=False):
    '''
    calculate the right hand temporal robustness of the predicate a*y - b > 0
    :param m: Gurobi Model
    :param y: system output
    :param a:
    :param b:
    :param abs: if true, return the absolute value of the robustness
    :param zero: if true, return 0 if the robustness switches its sign, otherwise return -1 or 1
    :return:
    '''

    N = z.shape[0]
    theta = m.addMVar(shape=(N, 1), vtype=GRB.INTEGER, name="theta", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    c1 = m.addMVar(shape=(N + 1, 1), vtype=GRB.INTEGER, name="c1", lb=-GRB.INFINITY,
                   ub=GRB.INFINITY)
    c0 = m.addMVar(shape=(N + 1, 1), vtype=GRB.INTEGER, name="c2", lb=-GRB.INFINITY, ub=GRB.INFINITY)

    m.addConstr(c1[-1] == 0)
    m.addConstr(c0[-1] == 0)

    for t in range(N-1,-1,-1):
        m.addConstr(c1[t] == (c1[t+1]+1) * z[t])
        if abs:
            m.addConstr(c0[t] == (c0[t+1]+1) * (1-z[t]))  # c0 >= 0
        else:
            m.addConstr(c0[t] == (c0[t+1]-1) * (1-z[t]))  # c0 <= 0

    if zero:
        # follow donze 2014 and radionova 2021
        c1_shift = c1[0:N] - z
        c0_shift = c0[0:N] + (1 - z)
        m.addConstr(theta == c1_shift + c0_shift)
    else:
        # not consider zero
        m.addConstr(theta == c1[:N] + c0[:N])

    return theta


def temporal_robustness_left(m, z, abs=False, zero=False):
    '''
    calculate the right hand temporal robustness of the predicate a*y - b > 0
    :param m: Gurobi Model
    :param y: system output
    :param a:
    :param b:
    :param abs: if true, return the absolute value of the robustness
    :param zero: if true, return 0 if the robustness switches its sign, otherwise return -1 or 1
    :return:
    '''

    N = z.shape[0]
    theta = m.addMVar(shape=(N, 1), vtype=GRB.INTEGER, name="theta", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    c1 = m.addMVar(shape=(N + 1, 1), vtype=GRB.INTEGER, name="c1", lb=-GRB.INFINITY,
                   ub=GRB.INFINITY)
    c0 = m.addMVar(shape=(N + 1, 1), vtype=GRB.INTEGER, name="c2", lb=-GRB.INFINITY, ub=GRB.INFINITY)

    m.addConstr(c1[0] == 0)
    m.addConstr(c0[0] == 0)

    for t in range(N):
        m.addConstr(c1[t+1] == (c1[t]+1) * z[t])
        if abs:
            m.addConstr(c0[t+1] == (c0[t]+1) * (1-z[t]))  # c0 >= 0
        else:
            m.addConstr(c0[t+1] == (c0[t]-1) * (1-z[t]))  # c0 <= 0

    if zero:
        # follow donze 2014 and radionova 2021
        c1_shift = c1[1:N+1] - z
        c0_shift = c0[1:N+1] + (1 - z)
        m.addConstr(theta == c1_shift + c0_shift)
    else:
        # not consider zero
        m.addConstr(theta == c1[1:N+1] + c0[1:N+1])
    return theta

def temporal_robustness_min(m, z, abs=False, zero=False):
    '''
    calculate the combined left and right hand temporal robustness of the predicate a*y - b > 0
    :param m: Gurobi Model
    :param y: system output
    :param a:
    :param b:
    :param abs: if true, return the absolute value of the robustness
    :param zero: if true, return 0 if the robustness switches its sign, otherwise return -1 or 1
    :return:
    '''
    N = z.shape[0]
    theta_left = temporal_robustness_left(m, z, abs, zero)
    theta_right = temporal_robustness_right(m, z, abs, zero)

    theta = m.addMVar(shape=(N, 1), vtype=GRB.INTEGER, name="theta", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    for t in range(N):
        v = quant_and(m, [theta_left[t], theta_right[t]])  # v = min(theta_left, theta_right)
        w = quant_or(m, [theta_left[t], theta_right[t]])  # w = max(theta_left, theta_right)
        m.addConstr(theta[t] == z[t]*v + (1-z[t])*w)
    return theta


def temporal_robustness_sum(m, z, abs=False, zero=False):
    '''
    calculate the combined left and right hand temporal robustness of the predicate a*y - b > 0
    :param m: Gurobi Model
    :param y: system output
    :param a:
    :param b:
    :param abs: if true, return the absolute value of the robustness
    :param zero: if true, return 0 if the robustness switches its sign, otherwise return -1 or 1
    :return:
    '''
    N = z.shape[0]
    theta_left = temporal_robustness_left(m, z, abs, zero)
    theta_right = temporal_robustness_right(m, z, abs, zero)

    theta = m.addMVar(shape=(N, 1), vtype=GRB.INTEGER, name="theta", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    m.addConstr(theta == theta_left + theta_right)
    return theta

