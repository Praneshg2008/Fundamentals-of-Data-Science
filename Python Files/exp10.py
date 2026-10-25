"""
CS23334 Fundamentals of Data Science Lab - Experiment 10
Array Functions & Manipulations in Data Science using NumPy.
"""

import numpy as np


def main():
    array = np.random.randint(1, 100, 9)
    print("array:\n", array)

    print("\nnp.sqrt(array):\n", np.sqrt(array))
    print("\narray.ndim:", array.ndim)

    new_array = array.reshape(3, 3)
    print("\nnew_array:\n", new_array)
    print("\nnew_array.ndim:", new_array.ndim)
    print("\nnew_array.ravel():\n", new_array.ravel())

    newm = new_array.reshape(3, 3)
    print("\nnewm:\n", newm)
    print("\nnewm[2, 1:3]:\n", newm[2, 1:3])
    print("\nnewm[1:2, 1:3]:\n", newm[1:2, 1:3])
    print("\nnew_array[0:3, 0:0]:\n", new_array[0:3, 0:0])
    print("\nnew_array[0:2, 0:1]:\n", new_array[0:2, 0:1])
    print("\nnew_array[0:3, 0:1]:\n", new_array[0:3, 0:1])
    print("\nnew_array[1:3]:\n", new_array[1:3])


if __name__ == "__main__":
    main()
