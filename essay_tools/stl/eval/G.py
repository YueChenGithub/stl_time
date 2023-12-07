import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from eval import *

def generate_t_y():
    t = np.arange(0, 61)
    y = 20*np.sin((t-3)*np.pi/43)
    return t, y


def main():
    # G[20,30] y>=5
    t, y = generate_t_y()
    t_left = 15
    t_right = 30
    a = 1
    b = 5

    # theta: temporal robustness
    # rho: spatial robustness
    theta_max_manually = 10e6
    theta_max = min([theta_max_manually, t_left, len(y) - t_right - 1])
    rhos = []
    for theta in np.arange(0, theta_max + 1):
        # moving to the right
        rho_p_right = calculate_shifted_rho_p(a, b, y, t_left, t_right, theta)
        # moving to the left
        rho_p_left = calculate_shifted_rho_p(a, b, y, t_left, t_right, -theta)

        rho_G_right = np.min(rho_p_right)  # G: min, F: max
        rho_G_left = np.min(rho_p_left)
        rho_G = np.min([rho_G_right, rho_G_left])

        # save robustness
        if rho_G < -1e-3:
            break
        rhos.append(rho_G)

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
    x_lim = 15
    save_static = True
    output_dir = './G'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axs = plt.subplots(2, 1, figsize=(6, 12), constrained_layout=True)
    # plot the original curve
    draw_original_singal(axs[0], b, a, t, y, t_right, t_left)
    # plot the robustness graph
    t_xi_minus, y_xi_minus = curve_xi_minus(fps, rho_optimal, theta_optimal)
    t_xi_plus, y_xi_plus = curve_xi_plus(fps, rhos)
    draw_robustness_graph(axs[1], xi_minus, t_xi_minus, y_xi_minus, xi_plus, t_xi_plus, y_xi_plus, x_lim)
    if save_static:
        plt.savefig(f'{output_dir}/static.png', dpi=300)
    plt.show()
    plt.close()

    # visualization gif
    save_dynamic = True
    if save_dynamic:
        t_xi_minus, y_xi_minus = curve_xi_minus(fps, rho_optimal, theta_optimal)
        t_xi_plus, y_xi_plus = curve_xi_plus(fps, rhos)

        # save xi_minus graphs
        frame = 0
        for tt, yy in zip(t_xi_minus, y_xi_minus):
            # draw the original curve and the robustness graph
            fig, axs = plt.subplots(2, 1, figsize=(6, 12), constrained_layout=True)
            draw_original_singal(axs[0], b, a, t, y, t_right, t_left)
            draw_robustness_graph(axs[1], xi_minus, t_xi_minus, y_xi_minus, xi_plus, t_xi_plus, y_xi_plus, x_lim)
            # plot the shifted curve
            axs[0].plot(t + tt, y + yy, color='violet', label='shifted')
            axs[0].legend()
            # plot the perturbation point
            axs[1].plot(tt, yy, 'o', color='violet', markersize=10)

            print(f'saving {frame} xi_minus_frames')
            plt.savefig(f'{output_dir}/xi_minus_{frame}.png', dpi=300)
            frame += 1
            plt.close()

        # save xi_plus graphs
        frame = 0
        for tt, yy in zip(t_xi_plus, y_xi_plus):
            # draw the original curve and the robustness graph
            fig, axs = plt.subplots(2, 1, figsize=(6, 12), constrained_layout=True)
            draw_original_singal(axs[0], b, a, t, y, t_right, t_left)
            draw_robustness_graph(axs[1], xi_minus, t_xi_minus, y_xi_minus, xi_plus, t_xi_plus, y_xi_plus, x_lim)
            # plot the shifted curve
            axs[0].plot(t + tt, y + yy, color='red', label='shifted')
            axs[0].legend()
            # plot the perturbation point
            axs[1].plot(tt, yy, 'o', color='red', markersize=10)

            print(f'saving {frame} xi_plus_frames')
            plt.savefig(f'{output_dir}/xi_plus_{frame}.png', dpi=300)
            frame += 1
            plt.close()

        # save gif
        # xi_minus
        total_time = 5000
        print('saving xi_minus.gif')
        num_frame = len(t_xi_minus)
        images_path = [f'{output_dir}/xi_minus_{i}.png' for i in range(num_frame)]
        images = [Image.open(i) for i in images_path]
        images[0].save(f'{output_dir}/dynamic_xi_minus.gif', save_all=True, append_images=images[:], duration=total_time//num_frame, loop=0)
        # xi_plus
        print('saving xi_plus.gif')
        num_frame = len(t_xi_plus)
        images_path = [f'{output_dir}/xi_plus_{i}.png' for i in range(num_frame)]
        images = [Image.open(i) for i in images_path]
        images[0].save(f'{output_dir}/dynamic_xi_plus.gif', save_all=True, append_images=images[:], duration=total_time//num_frame, loop=0)

if __name__ == '__main__':
    main()
