import numpy as np

def draw_original_singal(ax, b, a, t, y, t_right, t_left):
    ax.fill_between(np.arange(t_left, t_right + 1), b / a, 32, color='springgreen', alpha=0.5, label='Globally')
    # ax.fill_between(np.arange(t_left, t_right + 1), b / a, 32, color='skyblue', alpha=0.5, label='Eventually')
    ax.plot(t, y, color='#1f77b4', label='Original')
    ax.axvline(x=t_left, color='grey', linestyle='--', alpha=0.5)
    ax.axvline(x=t_right, color='grey', linestyle='--', alpha=0.5)
    ax.axhline(y=b, color='grey', linestyle='--', alpha=0.5)
    ax.set_xlabel('time')
    ax.set_ylabel('y')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 30)
    ax.legend(loc='upper right')


def draw_robustness_graph(ax, xi_minus, t_xi_minus, y_xi_minus, xi_plus, t_xi_plus, y_xi_plus, x_lim=None):
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    # draw optimal theta
    theta = max(t_xi_plus)
    ax.plot([], [], ' ', label=r'$\theta$: ' + f'{theta:.01f}')
    # draw the optimal rho
    rho_optimal = np.max(y_xi_plus)
    ax.plot([], [],  ' ', label=r'$\rho$: ' + f'{rho_optimal:.01f}')
    # draw the xi_minus figure
    color_xi_minus = 'violet'
    ax.plot(t_xi_minus, y_xi_minus, color=color_xi_minus)
    ax.fill_between(x=t_xi_minus, y1=y_xi_minus, color=color_xi_minus, alpha=0.2, label=r'$\xi^-$: ' + f'{xi_minus:.01f}')
    # draw the xi_plus figure
    color_xi_plus = 'red'
    ax.plot(t_xi_plus, y_xi_plus, color=color_xi_plus)
    ax.fill_between(x=t_xi_plus, y1=y_xi_plus, color=color_xi_plus, alpha=0.2, label=r'$\xi^+$: ' + f'{xi_plus:.01f}')

    ax.set_xlabel('temporal perturbation')
    ax.set_ylabel('spatial perturbation')
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', reverse=True)


    ax.set_title('Robustness' ,fontweight='bold')
    if x_lim is not None:
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-x_lim, x_lim)
        ax.set_aspect("equal")

    return xi_plus, xi_minus, rho_optimal, theta



def curve_xi_plus(fps, rhos):
    # plot xi_plus
    '''
                  .
            l1   / \  l2
                /   \
                \   /
            l4   \ /  l3
                  .
        '''
    l1_t = np.arange(-fps + 1, 1)
    l1_y = np.flip(rhos)
    l3_t = -l1_t
    l3_y = -l1_y
    l2_t = np.arange(0, fps)
    l2_y = rhos
    l4_t = -l2_t
    l4_y = -l2_y


    # frame between l2 and l3, l4 and l1
    last_rho = l2_y[-1]
    d_moved = np.max(l2_y) - last_rho + 1e-6
    num_frame = min(fps, int(last_rho/d_moved*fps)+1)
    t23 = (fps-1) * np.ones(num_frame*2)
    y23 = np.linspace(last_rho, -last_rho, num_frame*2)
    t41 = -t23
    y41 = np.flip(y23)

    t_concat = np.concatenate((l1_t, l2_t, t23, l3_t, l4_t, t41, l1_t[0].reshape(1)))
    y_concat = np.concatenate((l1_y, l2_y, y23, l3_y, l4_y, y41, l1_y[0].reshape(1)))
    return t_concat, y_concat


def curve_xi_minus(fps, rho_optimal, theta_optimal):
    # xi_minus
    '''
        (t1, y2)    (t2,y2)
        p2---------------p3
        |       l2        |
        |l1             l3|
        |       l4        |
        p1---------------p4
        (t1,y1)     (t2,y1)
        '''
    l1_t = -theta_optimal * np.ones(fps)
    l1_y = np.linspace(-rho_optimal, rho_optimal, fps)
    l3_t = np.flip(-l1_t)
    l3_y = np.flip(l1_y)
    l2_t = np.linspace(-theta_optimal, theta_optimal, fps)
    l2_y = rho_optimal * np.ones(fps)
    l4_t = np.flip(l2_t)
    l4_y = np.flip(-l2_y)
    t_concat = np.concatenate((l1_t, l2_t, l3_t, l4_t, l1_t[0].reshape(1)))
    y_concat = np.concatenate((l1_y, l2_y, l3_y, l4_y, l1_y[0].reshape(1)))
    return t_concat, y_concat


def calculate_shifted_rho_p(a, b, y, t_left, t_right, theta):
    # move the interval
    t_left_new = np.max([t_left + theta, 0])
    t_right_new = np.min([t_right + theta, len(y) - 1])
    # compute rho of G[t_left_new,t_right_new] a*y>=b
    rho_p = a * y[t_left_new:t_right_new + 1] - b
    return rho_p