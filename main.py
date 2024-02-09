import time

from bloch import arb_rot, free_precession, y_rot
import numpy as np


def main():
    initial_magnetisation = np.array([0, 0, 1])
    t1 = 600
    t2 = 100
    tr = 500
    flip = np.pi / 3

    magnetisation = y_rot(flip) @ initial_magnetisation
    a, b = free_precession(1, t1, t2, 0)
    magnetisation = a @ magnetisation + b
    print(magnetisation)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
