import numpy as np
from scipy.integrate import odeint
from scipy.fftpack import diff as psdiff

def kdv_ders(u, L):
    """Compute the first and third spatial derivatives of u using
    the pseudo-spectral method on a periodic domain of length L.
    
    `u` is assumed to be a 2D array with shape (num_times, num_space_points),
    where each row corresponds to the spatial profile at a given time."""
    
    dudt = np.zeros_like(u)
    ux = np.zeros_like(u)
    for i in range(u.shape[0]):
        ux[i] = psdiff(u[i], period=L)
        uxxx = psdiff(u[i], period=L, order=3)
    
        dudt[i] = -u[i]*ux[i] - 0.0025*uxxx
    return dudt, ux

def kdv(u, t, L):
    """Differential equations for the KdV equation, discretized in x."""
    # Compute the x derivatives using the pseudo-spectral method.
    ux = psdiff(u, period=L)
    uxxx = psdiff(u, period=L, order=3)

    # Compute du/dt.    
    dudt = -u*ux - 0.0025*uxxx

    return dudt

def kdv_solution(u0, t, L):
    """Use odeint to solve the KdV equation on a periodic domain.
    
    `u0` is initial condition, `t` is the array of time values at which
    the solution is to be computed, and `L` is the length of the periodic
    domain."""

    sol = odeint(kdv, u0, t, args=(L,), mxstep=50000)
    return sol


if __name__ == "__main__":
    # Set the size of the domain, and create the discretized grid.
    dx = 0.005
    x = np.arange(start=-1, stop=1+dx, step=dx)

    # Set the initial conditions.
    # Not exact for two solitons on a periodic domain, but close enough...
    u0 = np.cos(np.pi*x)
    
    dt = 0.005
    # Set the time sample grid.
    t = np.arange(0, 1+dt, step=dt)

    print("Computing the solution.")
    sol = kdv_solution(u0, t, 2)
    print(sol.shape)

    print("Plotting.")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,5))
    plt.imshow(sol[:,::-1].T, extent=[0,1,-1,1], aspect='auto', cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Korteweg-de Vries on a Periodic Domain')
    plt.savefig('kdv.png')
    
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x, sol[0,:], 'tab:blue')
    axs[0, 0].set_title('t = 0')

    axs[0, 1].plot(x, sol[int(0.33/dt),:], 'tab:orange')
    axs[0, 1].set_title('t = 0.33')

    axs[1, 0].plot(x, sol[int(0.66/dt),:], 'tab:green')
    axs[1, 0].set_title('t = 0.66')
    
    axs[1, 1].plot(x, sol[int(0.99/dt),:], 'tab:red')
    axs[1, 1].set_title('t = 0.99')
    
    for ax in axs.flat:
        ax.set(xlabel='x', ylabel='u(x,t)')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(f'kdv_multiplot.png', dpi=300)
    import os
    if not os.path.exists('data'):
        os.makedirs('data')
    
    with open('data/kdv_data.npy', 'wb') as f:
        np.save(f, sol)
    
    dudt, ux = kdv_ders(sol, 2)
    print(f'dudt.shape: {dudt.shape}, ux.shape: {ux.shape}')
    with open('data/kdv_ders.npy', 'wb') as f:
        np.save(f, (dudt, ux))