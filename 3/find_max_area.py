import numpy as np
import matplotlib.pyplot as plt
import os

def generate_t_y():
    t = np.arange(0, 61)
    y = np.zeros(61)
    for i in range(15):
        y[i + 1] = y[i] + 0
    for i in range(15,22):
        y[i + 1] = y[i] + 3
    for i in range(22, 30):
        y[i + 1] = y[i] -3
    for i in range(30, len(t) - 1):
        y[i + 1] = y[i] - 0
    return t, y

def calculate_area(t1,t2,y1,y2):
    '''
    (t1,y2)            (t2,y2)
    ---------------------
    |                   |
    |                   |
    ---------------------
    (t1,y1)            (t2,y1)
    '''

    # calculate area
    area = np.abs((t2-t1)*(y2-y1))
    return area


def G_right(a, b, t_left, t_right, y):
    # G, right:
    # 1. find all index on the left
    index_t = []
    for i in range(t_left, -1, -1):
        if a * y[i] < b:
            break
        index_t.append(i)
    area_max = 0
    optimal_rectangle = (0,0,0,0)
    for i in index_t:
        t1 = i
        t2 = t_left
        y1 = b / a
        y2 = y[t1]
        if a > 0:
            y1 = max(y1, y2 - (np.min(y[t_left:t_right + 1]) - y1))
        else:
            y1 = min(y1, y2 - (np.max(y[t_left:t_right + 1]) - y1))
        area = calculate_area(t1, t2, y1, y2)
        if area > area_max:
            area_max = area
            optimal_rectangle = (t1, t2, y1, y2)
    return area_max, optimal_rectangle

def G_left(a, b, t_left, t_right, y):
    # G, left:
    # 1. find all index on the right
    index_t = []
    for i in range(t_right, len(y), 1):
        if a * y[i] < b:
            break
        index_t.append(i)
    area_max = 0
    optimal_rectangle = (0,0,0,0)
    for i in index_t:
        t1 = t_right
        t2 = i
        y1 = b / a
        y2 = y[t2]
        if a > 0:
            y1 = max(y1, y2 - (np.min(y[t_left:t_right + 1]) - y1))
        else:
            y1 = min(y1, y2 - (np.max(y[t_left:t_right + 1]) - y1))
        area = calculate_area(t1, t2, y1, y2)
        if area > area_max:
            area_max = area
            optimal_rectangle = (t1, t2, y1, y2)
    return area_max, optimal_rectangle


def F_right(a, b, t_left, t_right, y):
    # F, right:
    # 1. find all index on the left
    index_t = []
    for i in range(t_right, -1, -1):
        if a * y[i] < b:
            if i in np.arange(t_left, t_right+1):
                continue
            else:
                break
        index_t.append(i)
    area_max = 0
    optimal_rectangle = (0,0,0,0)
    for i in index_t:
        t1 = i
        t2 = t_right
        y1 = b / a
        y2 = y[t1]
        if i not in np.arange(t_left, t_right+1):
            if a > 0:
                y1 = max(y1, y2 - (y[t_left] - y1))
            else:
                y1 = min(y1, y2 - (y[t_left] - y1))
        area = calculate_area(t1, t2, y1, y2)
        if area > area_max:
            area_max = area
            optimal_rectangle = (t1, t2, y1, y2)
    return area_max, optimal_rectangle

def F_left(a, b, t_left, t_right, y):
    # F, left:
    # 1. find all index on the right
    index_t = []
    for i in range(t_left, len(y), 1):
        print(i)
        if a * y[i] < b:
            if i in np.arange(t_left, t_right+1):
                continue
            else:
                break
        index_t.append(i)
    area_max = 0
    optimal_rectangle = (0,0,0,0)
    for i in index_t:
        t1 = t_left
        t2 = i
        y1 = b / a
        y2 = y[t2]
        if i not in np.arange(t_left, t_right+1):
            if a > 0:
                y1 = max(y1, y2 - (y[t_right] - y1))
            else:
                y1 = min(y1, y2 - (y[t_right] - y1))
        area = calculate_area(t1, t2, y1, y2)
        if area > area_max:
            area_max = area
            optimal_rectangle = (t1, t2, y1, y2)
    return area_max, optimal_rectangle


def main():
    t, y = generate_t_y()

    # predicate: p: y>=5, G[15,30]p

    b = 5
    a = 1
    t_left = 15
    t_right = 30

    area_max, optimal_rectangle = F_left(a, b, t_left, t_right, y)

    # plot (t,y)
    plt.plot(t, y, '-')
    plt.axvline(x=t_left, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=t_right, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=b/a, color='gray', linestyle='--', alpha=0.5)
    # plot the rectangle area
    plt.fill_between([optimal_rectangle[0], optimal_rectangle[1]], [optimal_rectangle[2]], [optimal_rectangle[3]], color='coral', alpha=0.5)
    plt.show()

    print('max area: ', area_max)
    print('optimal rectangle: ', optimal_rectangle)

if __name__ == '__main__':
    main()