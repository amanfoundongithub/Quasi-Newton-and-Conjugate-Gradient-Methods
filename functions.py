import numpy as np
import numpy.typing as npt


# Trid function


def trid_function(point: npt.NDArray[np.float64]) -> np.float64:
    return np.sum((point - 1) ** 2) - np.sum(point[1:] * point[:-1])


def trid_function_derivative(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    der = np.zeros_like(point)
    for i in range(point.shape[0]):
        der[i] = 2 * (point[i] - 1)
        if i != 0:
            der[i] -= point[i - 1]
        if i != point.shape[0] - 1:
            der[i] -= point[i + 1]
    return der


# ------------------------------------------------------------------------------
# Three hump camel function


def three_hump_camel_function(point: npt.NDArray[np.float64]) -> np.float64:
    return (
        2 * point[0] ** 2
        - 1.05 * point[0] ** 4
        + point[0] ** 6 / 6
        + point[0] * point[1]
        + point[1] ** 2
    )


def three_hump_camel_function_derivative(
    point: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.asarray(
        [
            4 * point[0] - 4.2 * point[0] ** 3 + point[0] ** 5 + point[1],
            point[0] + 2 * point[1],
        ]
    )


# ------------------------------------------------------------------------------
# Rosenbrock function


def rosenbrock_function(point: npt.NDArray[np.float64]) -> np.float64:
    return np.sum(100 * (point[1:] - point[:-1] ** 2) ** 2 + (point[:-1] - 1) ** 2)


def rosenbrock_function_derivative(
    point: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    der = np.zeros_like(point)
    der[0] = 2 * (point[0] - 1) - 400 * (point[1] - point[0] ** 2) * point[0]
    der[-1] = 200 * (point[-1] - point[-2] ** 2)
    for i in range(1, point.shape[0] - 1):
        der[i] = (
            400 * point[i] ** 3
            + 202 * point[i]
            - 400 * point[i + 1] * point[i]
            - 200 * point[i - 1] ** 2
            - 2
        )
    return der


# ------------------------------------------------------------------------------
# Styblinski-Tang function


def styblinski_tang_function(point: npt.NDArray[np.float64]) -> np.float64:
    return np.sum(point**4 - 16 * point**2 + 5 * point) / 2


def styblinski_tang_function_derivative(
    point: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return (4 * point**3 - 32 * point + 5) / 2

# ------------------------------------------------------------------------------
# Root of 1+x_1^2 + Root of 1+x_2^2


def func_1(point: npt.NDArray[np.float64]) -> np.float64:
    return np.sum(np.sqrt(1 + point**2))


def func_1_derivative(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return point / np.sqrt(1 + point**2)

# ------------------------------------------------------------------------------

def matyas_function(point: npt.NDArray[np.float64]) -> np.float64:
    return 0.26*(point[0]**2 + point[1]**2) - 0.48*point[0]*point[1] 

def matyas_function_derivative(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(
        [
            0.52*point[0] -0.48*point[1],
            0.52*point[1] -0.48*point[0],
        ]
    ) 

# ------------------------------------------------------------------------------
def hyperEllipsoid_function(point: npt.NDArray[np.float64]) -> np.float64:
    return np.sum(
        [
            np.sum([point[j]**2 for j in range(i)]) 
            for i in range(point.shape[0])
        ]
    )

def hyperEllipsoid_derivative(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    der = np.zeros_like(point)
    d = point.shape[0]
    for i in range(point.shape[0]):
        der[i] = (d-i)*2*point[i]

    return der