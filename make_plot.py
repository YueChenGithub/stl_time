import numpy as np
import matplotlib.pyplot as plt

data_dir = './exp3/'

name = 'F&F'

exp = [f'{name}_s', f'{name}_t', f'{name}_cs', f'{name}_cp']

for e in exp[:2]:
    data_path = data_dir + e + '.npz'
    data = np.load(data_path)
    y = data['y']
    r = data['r']

    # plot y
    plt.plot(np.arange(0, y.shape[0]), y.flatten(), '-', label=f'{e[len(name)+1:]}, r={r[0][0]:.01f}')

plt.legend()
plt.xlabel("timestep (t)")
plt.ylabel("y")
# plot a region where y >= 20 and 10 <= t <= 20
plt.fill_between(np.arange(10, 21), 20, 80, color='gray', alpha=0.5)
# plot a region where y <= -20 and 30 <= t <= 40
plt.fill_between(np.arange(30, 41), -80, -20, color='gray', alpha=0.5)
# set the y-axis range to [-60, 60]
plt.ylim(-80, 80)
# add title
plt.title(name)


# save figure
plt.savefig(f'exp3_{name}.png', dpi=300)
plt.show()