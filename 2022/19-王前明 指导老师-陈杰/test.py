import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.sin(1.5*x)
plt.figure()
plt.plot(x, y1, label="sin(x)")
plt.plot(x, y2, label="sin(1.5x)")
plt.legend()
plt.tight_layout()
plt.show()