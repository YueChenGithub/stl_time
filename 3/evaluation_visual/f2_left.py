import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
def generate_t_y():
    t = np.arange(0, 61)
    y = np.zeros(61)
    for i in range(10):
        y[i + 1] = y[i] + 2
    for i in range(10, 19):
        y[i + 1] = y[i] - 1
    for i in range(19, 35):
        y[i + 1] = y[i] + 1
    for i in range(35, len(t) - 1):
        y[i + 1] = y[i] - 1
    return t, y

def plot_main_graph(t, y, t1, t2, y1, y2):
    # plot (t,y)
    plt.plot(t, y, '-', label='original')
    plt.axvline(x=15, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=30, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
    plt.ylim(-0, 30)
    plt.xlim(-0, 50)
    # plot the rectangle area
    plt.fill_between([t1, t2], [y1], [y2], color='coral', alpha=0.5)
    # set plt size
    plt.gcf().set_size_inches(6, 6)
    # fill the area y>5, 15<x<30
    plt.fill_between([15, 30], [5], [30], color='gray', alpha=0.2)


def generate_moving_point(fps, start, t1, t2, y1, y2):
    '''
    (t1, y2)    (t2,y2)
    2--------------3
    |              |
    |              |
    |              |
    1--------------4
    (t1,y1)     (t2,y1)
    '''
    assert fps > 0
    assert y2 > y1
    assert t2 > t1
    assert start in [1, 2, 3, 4]
    a, b = np.meshgrid(np.linspace(t1, t2, fps), np.linspace(y1, y2, fps))
    t12 = a[:, 0]
    y12 = b[:, 0]
    t23 = a[0, :]
    y23 = b[-1, :]
    t34 = np.flip(a[:, -1])
    y34 = np.flip(b[:, -1])
    t41 = np.flip(a[-1, :])
    y41 = np.flip(b[0, :])

    t_clockwise = np.concatenate([t12, t23, t34, t41])
    y_clockwise = np.concatenate([y12, y23, y34, y41])
    if start == 2:
        t_clockwise = np.roll(t_clockwise, -fps)
        y_clockwise = np.roll(y_clockwise, -fps)
    elif start == 3:
        t_clockwise = np.roll(t_clockwise, -2 * fps)
        y_clockwise = np.roll(y_clockwise, -2 * fps)
    elif start == 4:
        t_clockwise = np.roll(t_clockwise, -3 * fps)
        y_clockwise = np.roll(y_clockwise, -3 * fps)

    return t_clockwise, y_clockwise

def plot_shifted_graph(p_t, p_y, c_t, c_y, frame,output_path=None, save=False):
    '''

    :param p_t: t of the point
    :param p_y: y of the point
    :param c_t: a series of t of the curve
    :param c_y: a series of y of the curve
    :param save: save image
    :return:
    '''
    plt.plot(p_t, p_y, 'o', color='red')
    plt.plot(c_t, c_y, '-', label='shifted')
    plt.legend()
    if save:
        plt.savefig(f'{output_path}/{frame}.png')
    else:
        plt.show()
    plt.close()

def main():
    t, y = generate_t_y()

    # t0 = 10
    # y0 = y[t0]

    '''
    (t1, y2)    (t2,y2)
    2--------------3
    |              |
    |              |
    |              |
    1--------------4
    (t1,y1)     (t2,y1)
    '''
    t1 = 15
    t2 = 39
    y1 = 5
    y2 = y[t2]

    y1 = max(y1, y2 - (y[40] - y1))

    start = 3
    fps = 20
    output_path = './f2_left'
    save = True


    if not os.path.exists(output_path):
        os.makedirs(output_path)
    p_ts, p_ys = generate_moving_point(fps, start, t1, t2, y1, y2)
    frame = 0
    for p_t, p_y in zip(p_ts, p_ys):
        plot_main_graph(t, y, t1, t2, y1, y2)
        c_t = t + p_t - p_ts[0]
        c_y = y + p_y - p_ys[0]
        plot_shifted_graph(p_t, p_y, c_t, c_y, frame=frame, output_path=output_path, save=save)
        frame += 1
        print(f'frame {frame} done')
    # save gif
    if save:
        print('saving gif...')
        images_path = []
        for i in range(frame):
            images_path.append(f'{output_path}/{i}.png')
        images = [Image.open(x) for x in images_path]
        images[0].save(f'{output_path}/0.gif', save_all=True, append_images=images[:], optimize=False, duration=50, loop=0)
        print('gif saved')



if __name__ == "__main__":
   main()