import os

from prettytable import PrettyTable
import numpy as np
import numpy.typing as npt

from algos import conjugate_descent, sr1, dfp, bfgs
from functions import (
    trid_function,
    trid_function_derivative,
    three_hump_camel_function,
    three_hump_camel_function_derivative,
    rosenbrock_function,
    rosenbrock_function_derivative,
    styblinski_tang_function,
    styblinski_tang_function_derivative,
    func_1,
    func_1_derivative,
    matyas_function,
    matyas_function_derivative,
    hyperEllipsoid_function,
    hyperEllipsoid_derivative,
)

test_cases = [
    [
        trid_function,
        trid_function_derivative,
        np.asarray([-2.0, -2]),
    ],
    [
        trid_function,
        trid_function_derivative,
        np.asarray([-2.0, -2]),
    ],
    [
        three_hump_camel_function,
        three_hump_camel_function_derivative,
        np.asarray([-2.0, 1]),
    ],
    [
        three_hump_camel_function,
        three_hump_camel_function_derivative,
        np.asarray([2.0, -1]),
    ],
    [
        three_hump_camel_function,
        three_hump_camel_function_derivative,
        np.asarray([-2.0, -1]),
    ],
    [
        three_hump_camel_function,
        three_hump_camel_function_derivative,
        np.asarray([2.0, 1]),
    ],
    [
        rosenbrock_function,
        rosenbrock_function_derivative,
        np.asarray([2.0, 2, 2, -2]),
    ],
    [
        rosenbrock_function,
        rosenbrock_function_derivative,
        np.asarray([2.0, -2, -2, 2]),
    ],
    [
        rosenbrock_function,
        rosenbrock_function_derivative,
        np.asarray([-2.0, 2, 2, 2]),
    ],
    [
        rosenbrock_function,
        rosenbrock_function_derivative,
        np.asarray([3.0, 3, 3, 3]),
    ],
    [
        styblinski_tang_function,
        styblinski_tang_function_derivative,
        np.asarray([0.0, 0, 0, 0]),
    ],
    [
        styblinski_tang_function,
        styblinski_tang_function_derivative,
        np.asarray([3.0, 3, 3, 3]),
    ],
    [
        styblinski_tang_function,
        styblinski_tang_function_derivative,
        np.asarray([-3.0, -3, -3, -3]),
    ],
    [
        styblinski_tang_function,
        styblinski_tang_function_derivative,
        np.asarray([3.0, -3, 3, -3]),
    ],
    [
        func_1,
        func_1_derivative,
        np.asarray([3.0, 3]),
    ],
    [
        func_1,
        func_1_derivative,
        np.asarray([-0.5, 0.5]),
    ],
    [
        func_1,
        func_1_derivative,
        np.asarray([-3.5, 0.5]),
    ],
    [matyas_function, matyas_function_derivative, np.asarray([2.0, -2])],
    [matyas_function, matyas_function_derivative, np.asarray([1, 10.0])],
    [hyperEllipsoid_function, hyperEllipsoid_derivative, np.asarray([-3, 3, 2.0])],
    [
        hyperEllipsoid_function,
        hyperEllipsoid_derivative,
        np.asarray([10, -10, 15, 15, -20, 11, 312.0]),
    ],
]


def main():
    if not os.path.isdir("plots"):
        os.mkdir("plots")

    table = PrettyTable()
    table.field_names = [
        "Test case",
        "Conjugate: HS",
        "Conjugate: PR",
        "Conjugate: FR",
        "SR1",
        "DFP",
        "BFGS",
    ]
    dividers = [1, 5, 9, 13, 16, 18]
    for test_case_num, test_case in enumerate(test_cases):
        print(test_case_num)
        row = [
            test_case_num,
        ]
        for algo in table.field_names:
            print(algo)
            if algo == "Test case":
                continue
            if algo == "Conjugate: HS":
                try:
                    ans = conjugate_descent(
                        test_case[2], test_case[0], test_case[1], "Hestenes-Stiefel"
                    )
                    if type(ans) != np.ndarray:
                        print(
                            f"Wrong type of value returned in conjugate gradient descent with Hestenes-Stiefel"
                        )
                        print(
                            f"Test function was {test_case[0].__name__} with {test_case[2]} as starting point"
                        )
                        row += [None]
                    else:
                        row += [np.round(ans, 3)]
                except Exception as e:
                    print(f"Error in conjugate gradient descent with Hestenes-Stiefel")
                    print(
                        f"Test function was {test_case[0].__name__} with {test_case[2]} as starting point"
                    )
                    print(e)
                    row += [None]
            elif algo == "Conjugate: PR":
                try:
                    ans = conjugate_descent(
                        test_case[2], test_case[0], test_case[1], "Polak-Ribiere"
                    )
                    if type(ans) != np.ndarray:
                        print(
                            f"Wrong type of value returned in conjugate gradient descent with Polak-Ribiere"
                        )
                        print(
                            f"Test function was {test_case[0].__name__} with {test_case[2]} as starting point"
                        )
                        row += [None]
                    else:
                        row += [np.round(ans, 3)]
                except Exception as e:
                    print(f"Error in conjugate gradient descent with Polak-Ribiere")
                    print(
                        f"Test function was {test_case[0].__name__} with {test_case[2]} as starting point"
                    )
                    print(e)
                    row += [None]
            elif algo == "Conjugate: FR":
                try:
                    ans = conjugate_descent(
                        test_case[2], test_case[0], test_case[1], "Fletcher-Reeves"
                    )
                    if type(ans) != np.ndarray:
                        print(f"Wrong type of value returned in Fletcher-Reeves")
                        print(
                            f"Test function was {test_case[0].__name__} with {test_case[2]} as starting point"
                        )
                        row += [None]
                    else:
                        row += [np.round(ans, 3)]
                except Exception as e:
                    print(f"Error in conjugate gradient descent with Fletcher-Reeves")
                    print(
                        f"Test function was {test_case[0].__name__} with {test_case[2]} as starting point"
                    )
                    print(e)
                    row += [None]
            elif algo == "SR1":
                try:
                    ans = sr1(test_case[2], test_case[0], test_case[1])
                    if type(ans) != np.ndarray:
                        print(f"Wrong type of value returned in SR1")
                        print(
                            f"Test function was {test_case[0].__name__} with {test_case[2]} as starting point"
                        )
                        row += [None]
                    else:
                        row += [np.round(ans, 3)]
                except Exception as e:
                    print(f"Error in SR1")
                    print(
                        f"Test function was {test_case[0].__name__} with {test_case[2]} as starting point"
                    )
                    print(e)
                    row += [None]
            elif algo == "DFP":
                try:
                    ans = dfp(test_case[2], test_case[0], test_case[1])
                    if type(ans) != np.ndarray:
                        print(f"Wrong type of value returned in DFP")
                        print(
                            f"Test function was {test_case[0].__name__} with {test_case[2]} as starting point"
                        )
                        row += [None]
                    else:
                        row += [np.round(ans, 3)]
                except Exception as e:
                    print(f"Error in DFP")
                    print(
                        f"Test function was {test_case[0].__name__} with {test_case[2]} as starting point"
                    )
                    print(e)
                    row += [None]
            elif algo == "BFGS":
                try:
                    ans = bfgs(test_case[2], test_case[0], test_case[1])
                    if type(ans) != np.ndarray:
                        print(f"Wrong type of value returned in BFGS")
                        print(
                            f"Test function was {test_case[0].__name__} with {test_case[2]} as starting point"
                        )
                        row += [None]
                    else:
                        row += [np.round(ans, 3)]
                except Exception as e:
                    print(f"Error in BFGS")
                    print(
                        f"Test function was {test_case[0].__name__} with {test_case[2]} as starting point"
                    )
                    print(e)
                    row += [None]
            else:
                row += [None]
        table.add_row(row, divider=test_case_num in dividers)
    print(table)


if __name__ == "__main__":
    main()
