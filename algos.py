from typing import Callable, Literal

from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np


def get_step_size(inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 ],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    dk):
    
    c1 = 0.001 
    c2 = 0.1 
    
    # Bisection method parameters 
    alpha = 0
    t = 1
    beta = 1e6
    
    # Looping parameters
    k = 0
    tolerance = 1e-6
    
    x = inital_point
    while k < 1e3:
        if f(x + t*dk) > f(x) + c1 * t * np.matmul(d_f(x), dk):
            beta = t  
            t = (alpha + beta)/2
        elif np.matmul(d_f(x + t*dk), dk) < c2 * np.matmul(d_f(x), dk):
            alpha = t 
            t = (alpha + beta)/2 
        else: 
            break 
        
        k += 1 
    
    return t  
# ---------- HESTENES - STIEFEL ALGORITHM -----------------------
def Hestenes_Stiefel(inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],):
    
    tolerance = 1e-6 
    k         = 0
    
    # Point
    x  = inital_point
    D  = d_f(x) 
    delta = -D
    
    while k < 1e4 and np.linalg.norm(d_f(x)) > tolerance:
        
        # Step siz
        beta_j = get_step_size(x, f, d_f, delta) 
        
        # Now update x
        x_temp = x + beta_j * delta 
        
        if np.linalg.norm(d_f(x_temp)) < tolerance:
            return x_temp

        else:
            x = x_temp
            d = D 
            D = d_f(x)
            
            chi = D.dot(D - d)/ delta.dot(D - d) 
            
            delta = -D + chi * delta
        pass 
        k += 1
    
    return x 

def Polak_Ribiere(inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],):
    
    tolerance = 1e-6 
    k         = 0
    
    # Point
    x  = inital_point
    D  = d_f(x) 
    delta = -D
    
    while k < 1e4 and np.linalg.norm(d_f(x)) > tolerance:
        
        # Step siz
        beta_j = get_step_size(x, f, d_f, delta) 
        
        # Now update x
        x_temp = x + beta_j * delta 
        
        if np.linalg.norm(d_f(x_temp)) < tolerance:
            return x_temp

        else:
            x = x_temp
            d = D 
            D = d_f(x)
            
            chi = max(0, (D - d).dot(D)/np.linalg.norm(d)**2) 
            
            delta = -D + chi * delta
        k += 1
        pass 
    
    return x

def Fletcher_Reeves(inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],):
    
    tolerance = 1e-6 
    k         = 0
    
    # Point
    x  = inital_point
    D  = d_f(x) 
    delta = -D
    
    while k < 1e4 and np.linalg.norm(d_f(x)) > tolerance:
        
        # Step siz
        beta_j = get_step_size(x, f, d_f, delta) 
        
        # Now update x
        x_temp = x + beta_j * delta 
        
        if np.linalg.norm(d_f(x_temp)) < tolerance:
            return x_temp

        else:
            x = x_temp
            d = D 
            D = d_f(x)
            
            chi = np.linalg.norm(D)**2 / np.linalg.norm(d)**2
            
            delta = -D + chi * delta
        pass 
        k += 1
    
    return x

# -------------- SR1 algorithm ---------------
def Symmetric_Rank_One(inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 ],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],):
    
    # Initialize parameters
    tolerance = 1e-6
    k         = 0 
    
    # Point and B approximation 
    x = inital_point
    B = np.identity(len(inital_point)) 
    
    while k < 1e4 and np.linalg.norm(d_f(x)) > tolerance:
        # g_k
        gk = d_f(x) 
        
        # Search direction
        dk = - np.dot(B, gk) 
        
        beta = get_step_size(x, f, d_f, dk)
        
        x_future = x + beta * dk 

        # del_k
        delta_x = x_future - x 
        # gamma_k  
        delta_gamma = d_f(x_future) - d_f(x)
        
        # del_k - Bk*gamma_k
        diff = delta_x - np.matmul(B, delta_gamma)
        
        B = B + np.outer(diff, diff)/np.dot(delta_gamma, diff) 
        
        x = x_future 
        k += 1
        
    return x 
    pass 

# -------------- DAVIDSON FLETCHER POWELL ------------------
def Davidson_Fletcher_Powell(inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 ],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],):
    
    # Initialize parameters
    tolerance = 1e-6
    k         = 0 
    
    # Point and B approximation 
    x = inital_point
    B = np.identity(len(inital_point)) 
    
    while k < 1e4 and np.linalg.norm(d_f(x)) > tolerance:
        # g_k
        gk = d_f(x) 
        
        # Search direction
        dk = - np.dot(B, gk) 
        
        beta = get_step_size(x, f, d_f, dk)
        
        x_future = x + beta * dk 
        
        delta_x = x_future - x
        delta_g = d_f(x_future) - d_f(x) 
        
        u = delta_x
        v = - np.matmul(B, delta_g)
        
        alpha = 1/np.dot(u, delta_g) 
        beta  = 1/np.dot(v, delta_g)
        
        B = B + alpha * np.outer(u, u) + beta * np.outer(v, v) 
        
        x = x_future 
        k += 1
        
    return x
    pass 

# --------------------- BFGS ALGORITHM  ----------------------------
def BFGS(inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 ],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],):
    
    # Initialize parameters
    tolerance = 1e-6
    k         = 0 
    
    # Point and B approximation 
    x = inital_point
    B = np.identity(len(inital_point))
    
    while k < 1e4 and np.linalg.norm(d_f(x)) > tolerance:
        # g_k
        gk = d_f(x) 
        
        # Search direction
        dk = - np.dot(B, gk) 
        
        beta = get_step_size(x, f, d_f, dk)
        
        x_future = x + beta * dk 
        
        delta_x = beta * dk 
        delta_g = d_f(x_future) - d_f(x) 
        
        rho = 1.0 / np.dot(delta_x, delta_g)
        term1 = np.outer(delta_x, delta_x) * rho
        term2 = (np.eye(len(inital_point)) - np.outer(delta_g, delta_x) * rho)
        B = np.dot(term2, np.dot(B, term2.T)) + term1
        
        
        x = x_future 
        k += 1
        
    return x
    pass 
 

# --------------------- MAIN CODE -----------------------------



def conjugate_descent(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    approach: Literal["Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves"],
) -> NDArray[np.float64]:
    if approach == "Hestenes-Stiefel":
        return Hestenes_Stiefel(inital_point, f, d_f) 
    elif approach == "Polak-Ribiere":
        return Polak_Ribiere(inital_point, f, d_f) 
    elif approach == "Fletcher-Reeves":
        return Fletcher_Reeves(inital_point, f, d_f)
    else: 
        raise NameError("Name not found") 


def sr1(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 ],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    
    return Symmetric_Rank_One(inital_point, f, d_f) 
    ...


def dfp(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    return Davidson_Fletcher_Powell(inital_point, f, d_f) 
    ...

def bfgs(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 ],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    # return BFGS(inital_point, f, d_f) 
    return 
    ...
