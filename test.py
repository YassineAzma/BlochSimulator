import numpy as np
from matplotlib import pyplot as plt


def ir_se(beta: float, ti: np.ndarray, tr: float, t1: float) -> np.ndarray:
    return 1 - 2 * beta * np.exp(-ti / t1) + np.exp(-tr / t1)


ti_times = np.array([n for n in range(2501)])
t1 = 360
tr = 6400

plt.plot(ti_times, ir_se(1.0, ti_times, tr, t1))
plt.plot(ti_times, ir_se(0.7, ti_times, tr, t1))
plt.axvline(240, ls='--')
plt.grid()
plt.show()
