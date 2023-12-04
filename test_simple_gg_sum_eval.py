import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from eval.eval import *


def draw_original_singal(axs, b1, a1, b2, a2, t, y, t_right1, t_left1):
    axs.plot(t, y, '-', label='original')
    axs.set_title(f"Position")
    axs.fill_between(np.arange(t_left1, t_right1 + 1), b1 / a1, b2 / a2, color='springgreen', alpha=0.5)
    axs.set_xlim(-2, 52)
    axs.set_ylim(-2, 32)
    axs.axvline(x=t_left1, color='gray', linestyle='--', alpha=0.5)
    axs.axvline(x=t_right1, color='gray', linestyle='--', alpha=0.5)
    axs.axhline(y=b1 / a1, color='gray', linestyle='--', alpha=0.5)
    axs.axhline(y=b2 / a2, color='gray', linestyle='--', alpha=0.5)
    axs.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

def main():
    save_static = True
    save_dynamic = False
    x_lim_xi = 8

    # ['s', 't_min', 't_sum', 't_left', 't_right', 'c_min', 'c_sum', 'c_left', 'c_right']
    robustness = 'c_sum'
    data_dir = './simple_gg_sum'
    data = np.load(data_dir + f'/{robustness}.npz')

    if save_dynamic:
        dynaimic_path = f'{data_dir}/{robustness}_dynamic/'
        if not os.path.exists(dynaimic_path):
            os.makedirs(dynaimic_path)

    # read data
    t = data['t']
    y = data['y']
    t_left1 = data['t_left1']
    t_right1 = data['t_right1']
    t_left2 = data['t_left2']
    t_right2 = data['t_right2']
    a1 = data['a1']
    b1 = data['b1']
    a2 = data['a2']
    b2 = data['b2']

    # theta: temporal robustness
    # rho: spatial robustness
    theta_max_manually = 10e6
    theta_max = min([theta_max_manually, t_left1, len(y) - t_right1 - 1])
    rhos = []
    for theta in np.arange(0, theta_max + 1):
        # p1:
        rho_p1_right = calculate_shifted_rho_p(a1, b1, y, t_left1, t_right1, theta)  # moving to the right
        rho_p1_left = calculate_shifted_rho_p(a1, b1, y, t_left1, t_right1, -theta)  # moving to the left
        rho_G1_right = np.min(rho_p1_right)  # G: min, F: max
        rho_G1_left = np.min(rho_p1_left)
        rho_G1 = np.min([rho_G1_right, rho_G1_left])
        print(rho_G1)

        # p2:
        rho_p2_right = calculate_shifted_rho_p(a2, b2, y, t_left2, t_right2, theta)  # moving to the right
        rho_p2_left = calculate_shifted_rho_p(a2, b2, y, t_left2, t_right2, -theta)  # moving to the left
        rho_G2_right = np.min(rho_p2_right)  # G: min, F: max
        rho_G2_left = np.min(rho_p2_left)
        rho_G2 = np.min([rho_G2_right, rho_G2_left])
        print(rho_G2)

        # conjunction
        rho_G = np.min([rho_G1, rho_G2])
        # save robustness
        if rho_G < -1e-3:
            break
        rhos.append(rho_G)

    print('rhos:', rhos)
    rhos = np.array(rhos)
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
    fig, axs = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)
    # plot the original curve
    draw_original_singal(axs[0], b1, a1, b2, a2, t, y, t_right1, t_left1)
    # plot the robustness graph
    t_xi_minus, y_xi_minus = curve_xi_minus(fps, rho_optimal, theta_optimal)
    t_xi_plus, y_xi_plus = curve_xi_plus(fps, rhos)
    draw_robustness_graph(axs[1], xi_minus, t_xi_minus, y_xi_minus, xi_plus, t_xi_plus, y_xi_plus, x_lim_xi)
    if save_static:
        plt.savefig(f'{data_dir}/{robustness}_static.png', dpi=300)
    plt.show()
    plt.close()


    # visualization gif
    if save_dynamic:
        t_xi_minus, y_xi_minus = curve_xi_minus(fps, rho_optimal, theta_optimal)
        t_xi_plus, y_xi_plus = curve_xi_plus(fps, rhos)

        # save xi_minus graphs
        frame = 0
        for tt, yy in zip(t_xi_minus, y_xi_minus):
            # draw the original curve and the robustness graph
            fig, axs = plt.subplots(2, 1, figsize=(6, 12), constrained_layout=True)
            draw_original_singal(axs[0], b1, a1, b2, a2, t, y, t_right1, t_left1)
            draw_robustness_graph(axs[1], xi_minus, t_xi_minus, y_xi_minus, xi_plus, t_xi_plus, y_xi_plus, x_lim_xi)
            # plot the shifted curve
            axs[0].plot(t + tt, y + yy, color='violet', label='shifted')
            axs[0].legend()
            # plot the perturbation point
            axs[1].plot(tt, yy, 'o', color='violet', markersize=10)

            print(f'saving {frame} xi_minus_frames')
            plt.savefig(f'{dynaimic_path}/xi_minus_{frame}.png', dpi=300)
            frame += 1
            plt.close()

        # save xi_plus graphs
        frame = 0
        for tt, yy in zip(t_xi_plus, y_xi_plus):
            # draw the original curve and the robustness graph
            fig, axs = plt.subplots(2, 1, figsize=(6, 12), constrained_layout=True)
            draw_original_singal(axs[0], b1, a1, b2, a2, t, y, t_right1, t_left1)
            draw_robustness_graph(axs[1], xi_minus, t_xi_minus, y_xi_minus, xi_plus, t_xi_plus, y_xi_plus, x_lim_xi)
            # plot the shifted curve
            axs[0].plot(t + tt, y + yy, color='red', label='shifted')
            axs[0].legend()
            # plot the perturbation point
            axs[1].plot(tt, yy, 'o', color='red', markersize=10)

            print(f'saving {frame} xi_plus_frames')
            plt.savefig(f'{dynaimic_path}/xi_plus_{frame}.png', dpi=300)
            frame += 1
            plt.close()

        # save gif
        # xi_minus
        total_time = 5000
        print('saving xi_minus.gif')
        num_frame = len(t_xi_minus)
        images_path = [f'{dynaimic_path}/xi_minus_{i}.png' for i in range(num_frame)]
        images = [Image.open(i) for i in images_path]
        images[0].save(f'{data_dir}/{robustness}_dynamic_xi_minus.gif', save_all=True, append_images=images[:], duration=total_time//num_frame, loop=0)
        # xi_plus
        print('saving xi_plus.gif')
        num_frame = len(t_xi_plus)
        images_path = [f'{dynaimic_path}/xi_plus_{i}.png' for i in range(num_frame)]
        images = [Image.open(i) for i in images_path]
        images[0].save(f'{data_dir}/{robustness}_dynamic_xi_plus.gif', save_all=True, append_images=images[:], duration=total_time//num_frame, loop=0)

if __name__ == '__main__':
    main()
