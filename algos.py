from typing import Callable, Literal

from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np
import os 


x_values = []
fx_values = []
dfx_values = []
def plot_values(algorithm : str, coordinate_size : int, initial_point, f : Callable[[NDArray[np.float64]], np.float64 ]):
    
    directory = "plots/" + algorithm
    if not os.path.isdir(directory):
        os.mkdir(directory)
    if not os.path.isdir(directory + "/grad"):
        os.mkdir(directory + "/grad")
    if not os.path.isdir(directory + "/val"):
        os.mkdir(directory + "/val") 
    if not os.path.isdir(directory + "/contour"):
        os.mkdir(directory + "/contour")
    
    # Plot val 
    epochs = list(range(1, len(fx_values) + 1))
     
    
    
    plt.plot(epochs, fx_values, color = 'green')
    plt.xlabel("Iterations")
    plt.ylabel("f(x) values")
    plt.title("Plot of f(x) vs iterations")
    
    plt.savefig(directory + "/val/" + f.__name__ + np.array2string(initial_point) + "_fig.png") 
    plt.clf() 
    
   
    plt.plot(epochs, dfx_values, color = 'orange')
    plt.xlabel("Iterations")
    plt.ylabel("|f'(x)| values")
    plt.title("Plot of |f'(x)| vs iterations")
    
    plt.savefig(directory + "/grad/" + f.__name__ +  np.array2string(initial_point) + "_fig.png")
    plt.clf() 
    
    if len(initial_point) > 2:
        plt.close()
        return 
    
    x = np.linspace(- 5, 5, 100)
    y = np.linspace(- 5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    n = x.shape[0]
    m = y.shape[0]
    
    Z = np.zeros(shape = (n,m)) 
    for i in range(n):
        for j in range(m):
            Z[i][j] = f(np.array([x[i],y[j]])) 
    
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Function Value')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Contour Plot of f(x)' )
    
    # for point in x_values:
    #     plt.plot(point[0], point[1], 'ro')  # Red circles for points
    for i in range(len(x_values)-1):
        plt.arrow(x_values[i][0], x_values[i][1], x_values[i+1][0]- x_values[i][0], x_values[i+1][1]- x_values[i][1], 
              head_width=0.1, head_length=0.1, fc='blue', ec='blue', linestyle='dashed', shape = 'full')
    
    plt.savefig(directory + "/contour/" + f.__name__ +  np.array2string(initial_point) + "_fig.png")
     
    plt.close()
    return 
    
    
    
def get_step_size(initial_point: NDArray[np.float64],
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
    
    x = initial_point
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
def Hestenes_Stiefel(initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],):
    
    tolerance = 1e-6 
    k         = 0
    
    # Point
    x  = initial_point
    D  = d_f(x) 
    delta = -D 
    
    while k < 1e4 and np.linalg.norm(d_f(x)) > tolerance:
        x_values.append(x) 
        fx_values.append(f(x))
        dfx_values.append(np.linalg.norm(d_f(x))) 
        # Step siz
        beta_j = get_step_size(x, f, d_f, delta) 
        
        # Now update x
        x_temp = x + beta_j * delta 
        
        if np.linalg.norm(d_f(x_temp)) < tolerance:
            x = x_temp 
            x_values.append(x) 
            fx_values.append(f(x))
            dfx_values.append(np.linalg.norm(d_f(x))) 
            plot_values("Hestenes_Stiefel", len(initial_point), initial_point, f)
            return x_temp

        else:
            x = x_temp
            d = D 
            D = d_f(x)
            
            chi = D.dot(D - d)/ delta.dot(D - d) 
            
            delta = -D + chi * delta
        k += 1
    
    x_values.append(x) 
    fx_values.append(f(x))
    dfx_values.append(np.linalg.norm(d_f(x))) 
    plot_values("Hestenes_Stiefel", len(initial_point), initial_point, f)
    return x 

def Polak_Ribiere(initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],):
    
    tolerance = 1e-6 
    k         = 0
    
    # Point
    x  = initial_point
    D  = d_f(x) 
    delta = -D
    
    while k < 1e4 and np.linalg.norm(d_f(x)) > tolerance:
        x_values.append(x) 
        fx_values.append(f(x))
        dfx_values.append(np.linalg.norm(d_f(x))) 
        # Step siz
        beta_j = get_step_size(x, f, d_f, delta) 
        
        # Now update x
        x_temp = x + beta_j * delta 
        
        if np.linalg.norm(d_f(x_temp)) < tolerance:
            x = x_temp 
            x_values.append(x) 
            fx_values.append(f(x))
            dfx_values.append(np.linalg.norm(d_f(x))) 
            plot_values("Polak_Ribiere", len(initial_point), initial_point, f)
            return x_temp

        else:
            x = x_temp
            d = D 
            D = d_f(x)
            
            chi = max(0, (D - d).dot(D)/np.linalg.norm(d)**2) 
            
            delta = -D + chi * delta
        k += 1
        pass 
    
    x_values.append(x) 
    fx_values.append(f(x))
    dfx_values.append(np.linalg.norm(d_f(x))) 
    plot_values("Polak_Ribiere", len(initial_point), initial_point, f)
    return x

def Fletcher_Reeves(initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],):
    
    tolerance = 1e-6 
    k         = 0
    
    # Point
    x  = initial_point
    D  = d_f(x) 
    delta = -D
    
    while k < 1e4 and np.linalg.norm(d_f(x)) > tolerance:
        x_values.append(x) 
        fx_values.append(f(x))
        dfx_values.append(np.linalg.norm(d_f(x))) 
        # Step siz
        beta_j = get_step_size(x, f, d_f, delta) 
        
        # Now update x
        x_temp = x + beta_j * delta 
        
        if np.linalg.norm(d_f(x_temp)) < tolerance:
            x = x_temp 
            x_values.append(x) 
            fx_values.append(f(x))
            dfx_values.append(np.linalg.norm(d_f(x)))
            plot_values("Fletcher_Reeves", len(initial_point), initial_point, f)
            return x_temp

        else:
            x = x_temp
            d = D 
            D = d_f(x)
            
            chi = np.linalg.norm(D)**2 / np.linalg.norm(d)**2
            
            delta = -D + chi * delta
        pass 
        k += 1
    
    x_values.append(x) 
    fx_values.append(f(x))
    dfx_values.append(np.linalg.norm(d_f(x))) 
    plot_values("Fletcher_Reeves", len(initial_point), initial_point, f)
    return x

# -------------- SR1 algorithm ---------------
def Symmetric_Rank_One(initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 ],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],):
    
    # Initialize parameters
    tolerance = 1e-6
    k         = 0 
    
    # Point and B approximation 
    x = initial_point
    B = np.identity(len(initial_point)) 
    
    while k < 1e4 and np.linalg.norm(d_f(x)) > tolerance:
        x_values.append(x) 
        fx_values.append(f(x))
        dfx_values.append(np.linalg.norm(d_f(x))) 
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
    
    x_values.append(x) 
    fx_values.append(f(x))
    dfx_values.append(np.linalg.norm(d_f(x))) 
    plot_values("Symmetric_Rank_One", len(initial_point), initial_point, f)
    return x 
    pass 

# -------------- DAVIDSON FLETCHER POWELL ------------------
def Davidson_Fletcher_Powell(initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 ],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],):
    
    # Initialize parameters
    tolerance = 1e-6
    k         = 0 
    
    # Point and B approximation 
    x = initial_point
    B = np.identity(len(initial_point)) 
    
    while k < 1e4 and np.linalg.norm(d_f(x)) > tolerance:
        x_values.append(x) 
        fx_values.append(f(x))
        dfx_values.append(np.linalg.norm(d_f(x))) 
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
    
    x_values.append(x) 
    fx_values.append(f(x))
    dfx_values.append(np.linalg.norm(d_f(x))) 
    plot_values("Davidson_Fletcher_Powell", len(initial_point), initial_point, f)
    return x
    pass 

# --------------------- BROYDEN FLETCHER ALGORITHM  ----------------------------
def BFGS(initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 ],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],):
    
    # Initialize parameters
    tolerance = 1e-6
    k         = 0 
    
    # Point and B approximation 
    x = initial_point
    B = np.identity(len(initial_point))
    
    while k < 1e4 and np.linalg.norm(d_f(x)) > tolerance:
        x_values.append(x) 
        fx_values.append(f(x))
        dfx_values.append(np.linalg.norm(d_f(x))) 
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
        term2 = (np.eye(len(initial_point)) - np.outer(delta_g, delta_x) * rho)
        B = np.matmul(term2, np.matmul(B, term2)) + term1
        
        x = x_future 
        k += 1
    
    x_values.append(x) 
    fx_values.append(f(x))
    dfx_values.append(np.linalg.norm(d_f(x))) 
    plot_values("BFGS", len(initial_point), initial_point, f)
    return x
    pass 
 

# --------------------- MAIN CODE -----------------------------



def conjugate_descent(
    initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    approach: Literal["Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves"],
) -> NDArray[np.float64]:
    x_values.clear()
    fx_values.clear()
    dfx_values.clear() 
    if approach == "Hestenes-Stiefel":
        return Hestenes_Stiefel(initial_point, f, d_f) 
    elif approach == "Polak-Ribiere":
        return Polak_Ribiere(initial_point, f, d_f) 
    elif approach == "Fletcher-Reeves":
        return Fletcher_Reeves(initial_point, f, d_f)
    else: 
        raise NameError("Name not found") 


def sr1(
    initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 ],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    x_values.clear()
    fx_values.clear()
    dfx_values.clear() 
    return Symmetric_Rank_One(initial_point, f, d_f) 
    ...


def dfp(
    initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    x_values.clear()
    fx_values.clear()
    dfx_values.clear() 
    return Davidson_Fletcher_Powell(initial_point, f, d_f) 
    ...

def bfgs(
    initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 ],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    x_values.clear()
    fx_values.clear()
    dfx_values.clear() 
    return BFGS(initial_point, f, d_f) 
    # return 
    ...
