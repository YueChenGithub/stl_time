import matplotlib.pyplot as plt
import numpy as np

# Define the function
def oscillating_function(x):
    x = x-1.8
    return 5 * np.exp(-0.1 * x) * np.sin(x) + 5

# Generate x values
x = np.linspace(0, 20, 400)



# Generate y values
y = oscillating_function(x)


# Create the plot
plt.figure(figsize=(6, 3.7))
plt.plot(x, y)
plt.xlabel('t')
plt.ylabel(r'$\omega$')
plt.grid(True)
plt.xlim([0,20])
plt.ylim([0,10])

# draw a line with double arrowheads
plt.annotate('', xy=(2.5, 8), xytext=(7.5, 8), arrowprops=dict(arrowstyle="<->", color='green'))
plt.text(5.5, 8.2, '5', color='green', ha='center', size=15)

plt.annotate('', xy=(7.5, 8), xytext=(17.5, 8), arrowprops=dict(arrowstyle="<->", color='green'))
plt.text(13.2, 8.2, '10', color='green', ha='center', size=15)

plt.annotate('', xy=(12.5, 0), xytext=(12.5, 8), arrowprops=dict(arrowstyle="<->", color='green'))
plt.text(13.2, 2, '8', color='green', ha='center', size=15)

# fill the area within the rectangle, bottom left corner at (7.5, 0) and top right corner at (17.5, 8)
plt.fill([7.5, 7.5, 17.5, 17.5], [0, 8, 8, 0], color='green', alpha=0.1)

plt.show()