import numpy as np
import jax.numpy as jnp

def gauss_int_jax(coeffs, a, b):
    def u_function(t):
        return jnp.polyval(coeffs, ((b - a) * t + b + a) / 2)
    x_0 = -jnp.sqrt(3/5)
    x_1 = 0.0
    x_2 = jnp.sqrt(3/5)
    w_0 = 5/9
    w_1 = 8/9
    w_2 = 5/9
    integral = ((b - a) / 2) * (
        w_0 * u_function(x_0) + w_1 * u_function(x_1) + w_2 * u_function(x_2)
    )
    return integral

def gauss_int(f, a, b):
    def u_function(t):
        return f(((b - a) * t + b + a) / 2)
    x_0 = -np.sqrt(3/5)
    x_1 = 0.0
    x_2 = np.sqrt(3/5)
    w_0 = 5/9
    w_1 = 8/9
    w_2 = 5/9
    integral = ((b - a) / 2) * (
        w_0 * u_function(x_0) + w_1 * u_function(x_1) + w_2 * u_function(x_2)
    )
    return integral

def comp_gauss_int(f, a, b, n):
    h = (b - a) / n
    integral = 0.0
    for i in range(n):
        x_i = a + i * h
        x_i1 = a + (i + 1) * h
        integral += gauss_int(f, x_i, x_i1)
    return integral

# def f(x):
#     return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

# result = comp_gauss_int(f, -1, 1, 10)
# print(f"The integral of f from -1 to 1 is approximately {result}")