import numpy as np
import matplotlib.pyplot as plt

def generate_t_y():
    t = np.arange(0, 51)
    y = np.zeros_like(t)
    for i in range(0, 10):
        y[i] = 2.5
    for i in range(10, 36):
        y[i] = 6
    for i in range(36, 51):
        y[i] = 2.5
    return t, y


def main():
    frame = 0
    fps = 20
    limit = 5
    for shift in np.linspace(0, limit, fps):
        # G[20,30] y>=5
        t, y = generate_t_y()
        t_left = 15
        t_right = 30
        a = 1
        b = 5
        fig, axs = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
        draw_original_singal(axs, b, a, t, y, t_right, t_left)

        t_shift = t + shift
        y_shift = y
        axs.plot(t_shift, y_shift, color='violet', label='Shifted')
        axs.legend(loc='upper right')
        plt.savefig(f'./G_lr/{frame:02d}.png')
        plt.close()
        frame += 1

    for shift in np.linspace(limit, -limit, fps*2):
        # G[20,30] y>=5
        t, y = generate_t_y()
        t_left = 15
        t_right = 30
        a = 1
        b = 5
        fig, axs = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
        draw_original_singal(axs, b, a, t, y, t_right, t_left)

        t_shift = t + shift
        y_shift = y
        axs.plot(t_shift, y_shift, color='violet', label='Shifted')
        axs.legend(loc='upper right')
        plt.savefig(f'./G_lr/{frame:02d}.png')
        plt.close()
        frame += 1

    for shift in np.linspace(-limit, 0, fps):
        # G[20,30] y>=5
        t, y = generate_t_y()
        t_left = 15
        t_right = 30
        a = 1
        b = 5
        fig, axs = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
        draw_original_singal(axs, b, a, t, y, t_right, t_left)

        t_shift = t + shift
        y_shift = y
        axs.plot(t_shift, y_shift, color='violet', label='Shifted')
        axs.legend(loc='upper right')
        plt.savefig(f'./G_lr/{frame:02d}.png')
        plt.close()
        frame += 1






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

if __name__ == '__main__':
    main()
