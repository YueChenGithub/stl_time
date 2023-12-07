import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from stl import *
from predicate import predicate
import os
from eval.eval import *

def draw_and_save(Q, R, T, a1, a2, b1, b2, m, output_dir, r, robustness, save_data, t_left1, t_left2, t_right1,
                  t_right2, u, u_max, x, y):
    t = np.arange(0, T + 1)
    # u = u.X.flatten()
    # x = x[:, 1].X.flatten()
    # y = y.X.flatten()
    # r = r.X.flatten()
    # create a figure that contains three subplots
    fig, axs = plt.subplots(2, 1, figsize=(5, 6), gridspec_kw={'height_ratios': [1.5, 1]}, constrained_layout=True)
    axs[0].set_xlabel('Time (steps)')
    # plot y
    y_position = 0
    axs[y_position].plot(t, y, '-o')
    axs[y_position].set_title(f"Position", fontweight='bold')
    axs[y_position].fill_between(np.arange(t_left1, t_right1 + 1), b1 / a1, 80, color='springgreen', alpha=0.5,
                                 label='Globally')
    axs[y_position].fill_between(np.arange(t_left2, t_right2 + 1), -80, b2 / a2, color='skyblue', alpha=0.5,
                                 label='Eventually')
    axs[y_position].set_xlim(-2, 52)
    axs[y_position].set_ylim(-32, 32)
    axs[y_position].axvline(x=t_left1, color='gray', linestyle='--', alpha=0.5)
    axs[y_position].axvline(x=t_right1, color='gray', linestyle='--', alpha=0.5)
    # axs[y_position].axvline(x=t_left2, color='gray', linestyle='--', alpha=0.5)
    axs[y_position].axvline(x=t_right2, color='gray', linestyle='--', alpha=0.5)
    axs[y_position].axhline(y=b1 / a1, color='gray', linestyle='--', alpha=0.5)
    axs[y_position].axhline(y=b2 / a2, color='gray', linestyle='--', alpha=0.5)
    axs[y_position].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axs[y_position].legend()
    # # plot u
    # u_position = 3
    # axs[u_position].plot(t, u, '-')
    # axs[u_position].set_title(f"Acceleration (Control Input)")
    # axs[u_position].axvline(x=t_left1, color='gray', linestyle='--', alpha=0.5)
    # axs[u_position].axvline(x=t_right1, color='gray', linestyle='--', alpha=0.5)
    # # axs[u_position].axvline(x=t_left2, color='gray', linestyle='--', alpha=0.5)
    # axs[u_position].axvline(x=t_right2, color='gray', linestyle='--', alpha=0.5)
    # axs[u_position].axhline(y=u_max, color='gray', linestyle='--', alpha=0.5)
    # axs[u_position].text(0, u_max, 'u_max', ha='left', va='bottom', fontsize=10)
    # axs[u_position].axhline(y=-u_max, color='gray', linestyle='--', alpha=0.5)
    # axs[u_position].text(0, -u_max, 'u_min', ha='left', va='top', fontsize=10)
    # axs[u_position].set_ylim(-1.3 * u_max, 1.3 * u_max)
    # axs[u_position].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    # # plot v
    # v_position = 2
    # v_max = 8
    # axs[v_position].plot(t, x, '-')
    # axs[v_position].set_title(f"Velocity")
    # axs[v_position].axvline(x=t_left1, color='gray', linestyle='--', alpha=0.5)
    # axs[v_position].axvline(x=t_right1, color='gray', linestyle='--', alpha=0.5)
    # # axs[v_position].axvline(x=t_left2, color='gray', linestyle='--', alpha=0.5)
    # axs[v_position].axvline(x=t_right2, color='gray', linestyle='--', alpha=0.5)
    # axs[v_position].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    # axs[v_position].set_ylim(-v_max, v_max)
    # plot xi
    rhos1 = cal_shifted_rho(a1, b1, t_left1, t_right1, y, 'G')
    rhos2 = cal_shifted_rho(a2, b2, t_left2, t_right2, y, 'F')
    l = np.minimum(len(rhos1), len(rhos2))
    rhos = np.minimum(rhos1[:l], rhos2[:l])  # conjunction: min, disjunction: max
    xi = np.arange(0, len(rhos)) * rhos
    xi_plus = np.sum(rhos)
    xi_minus = np.max(xi)
    theta_optimal = np.argmax(xi)
    rho_optimal = rhos[theta_optimal]
    fps = len(rhos)
    print('fps: ', fps)
    print('xi_plus: ', xi_plus)
    print('xi_minus: ', xi_minus)
    print('theta_optimal: ', theta_optimal)
    print('rho_optimal: ', rho_optimal)
    # visualization
    # plot the robustness graph
    robustness_position = 1
    x_lim_xi = 22
    t_xi_minus, y_xi_minus = curve_xi_minus(fps, rho_optimal, theta_optimal)
    t_xi_plus, y_xi_plus = curve_xi_plus(fps, rhos)
    draw_robustness_graph(axs[robustness_position], xi_minus, t_xi_minus, y_xi_minus, xi_plus, t_xi_plus, y_xi_plus,
                          x_lim_xi)
    if save_data:
        # create directory if not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + robustness + '.png', dpi=300)

        # save data
        np.savez(output_dir + robustness + '.npz',
                 u=u,
                 y=y,
                 x=x,
                 r=r,
                 t=t,
                 u_max=u_max,
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


def cal_shifted_rho(a, b, t_left, t_right, y, type, theta_max=100):
    assert type in ['G', 'F'], 'type must be G or F'
    rhos = []
    for theta in np.arange(0, theta_max + 1):
        # moving to the right
        rho_p_right = calculate_shifted_rho_p(a, b, y, t_left, t_right, theta)
        # moving to the left
        rho_p_left = calculate_shifted_rho_p(a, b, y, t_left, t_right, -theta)
        # boundary case:
        if len(rho_p_right) == 0 or len(rho_p_left) == 0:
            break
        else:
            if type == 'G':
                rho_G_right = np.min(rho_p_right)  # G: min, F: max
                rho_G_left = np.min(rho_p_left)
            else:
                rho_G_right = np.max(rho_p_right)
                rho_G_left = np.max(rho_p_left)
            rho_G = np.min([rho_G_right, rho_G_left])
            # save robustness
            if rho_G < -1e-3:
                break
            rhos.append(rho_G)
    rhos = np.array(rhos)
    return rhos