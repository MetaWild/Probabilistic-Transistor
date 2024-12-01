import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap
import numpy as np
from uniform import standard_uniform
from pdf_integral import comp_gauss_int, multidimensional_monte_carlo, get_num_arguments
from sampling import sample_from
import matplotlib.pyplot as plt

def distribution_samples(N, B, f, a, b, num_transistors, key):
    num_dimensions = get_num_arguments(f)
    print(num_dimensions)
    h = (b - a) / B
    x_i = jnp.arange(B) * h + a
    x_i1 = (jnp.arange(B) + 1) * h + a
    x_i_list = [x_i] * num_dimensions
    x_i1_list = [x_i1] * num_dimensions
    xi_grids = jnp.meshgrid(*x_i_list, indexing='ij')
    xi1_grids = jnp.meshgrid(*x_i1_list, indexing='ij')
    xi_grid = jnp.stack([xi_grid.ravel() for xi_grid in xi_grids], axis=-1)
    xi1_grid = jnp.stack([xi1_grid.ravel() for xi1_grid in xi1_grids], axis=-1)
    num_bins = xi_grid.shape[0]
    subkeys = random.split(key, num_bins)
    probabilities = vmap(lambda xi, xi1, subkey: multidimensional_monte_carlo(f, xi, xi1, num_transistors, num_dimensions, 1000,subkey))(xi_grid, xi1_grid, subkeys)
    print(jnp.sum(probabilities))
    probabilities /= jnp.sum(probabilities)
    print(jnp.sum(probabilities))
    cumulative_probabilities = jnp.cumsum(probabilities)
    random_numbers = standard_uniform(num_transistors, key, N)
    bin_indices = jnp.searchsorted(cumulative_probabilities, random_numbers, side='right')
    samples = sample_from(f, bin_indices, xi_grid, xi1_grid, num_transistors, key)
    return samples

# def distribution_samples(N, B, f, a, b, num_transistors, key):
#     h = (b - a) / B
#     x_i = jnp.arange(B) * h + a
#     x_i1 = (jnp.arange(B) + 1) * h + a
#     probabilities = vmap(lambda xi, xi1: comp_gauss_int(f, xi, xi1, 3))(x_i, x_i1)
#     probabilities /= jnp.sum(probabilities)
#     cumulative_probabilities = jnp.cumsum(probabilities)
#     random_numbers = standard_uniform(num_transistors, key, N)
#     bin_indices = jnp.searchsorted(cumulative_probabilities, random_numbers, side='right')
#     samples = sample_from(f, bin_indices, h, a, num_transistors, key)
#     return samples

def create_normal_function(mu, sigma):
    def normal(x):
        return jnp.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * jnp.sqrt(2 * jnp.pi))
    return normal

def create_normal_function_2d(mu, sigma):
    def normal(x, y):
        return jnp.exp(-((x - mu) ** 2 + (y - mu) ** 2) / (2 * sigma**2)) / (sigma**2 * (2 * jnp.pi))
    return normal

def create_exponential_function(lmbda):
    def exponential(x):
        return lmbda * jnp.exp(-lmbda * x)
    return exponential

def standard_normal_2d(x, y):
    return (1 / (2 * jnp.pi)) * jnp.exp(-0.5 * (x**2 + y**2))

def uniform_2d(x, y):
    return 1

standard_normal_distribution = create_normal_function(0, 1)
standard_exponential_distribution = create_exponential_function(1)
test_exponential_distribution = create_exponential_function(0.5)
test_normal_distribution = create_normal_function(1, 0.5)
test_normal_distribution_2d = create_normal_function_2d(1, 0.5)

# def distribution_samples(N, B, f, a, b):
#     h = (b - a) / B
#     samples = np.zeros(N)
#     probabilities = np.zeros(B)
#     for i in range(B):
#         x_i = i * h + a
#         x_i1 = (i + 1) * h + a
#         probabilities[i] = comp_gauss_int(f, x_i, x_i1, 3)
#     probabilities /= sum(probabilities)
#     cumulative_probabilities = np.zeros(B)
#     for i in range(B):
#         cumulative_probabilities[i] = np.sum(probabilities[:i + 1])

#     for n in range(N):
#         random_number = standard_uniform(10)
#         for i in range(B):
#             if cumulative_probabilities[i] >= random_number:
#                 samples[n] = sample_from(f, i, h, a)
#                 break
#     return samples

# def normal_distribution(x):
#     return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

N = 1000000
B = 100
a = -5
b = 5
num_transistors = 20
key = random.PRNGKey(0)
random_samples = distribution_samples(N, B, standard_normal_2d, a, b, num_transistors, key)
print("Random Samples Shape:", random_samples.shape)
print(random_samples)

if random_samples.shape[1] == 1:
    plt.hist(random_samples.flatten(), bins=100, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Histogram of Generated Random Numbers')
    plt.xlabel('Random Number')
    plt.ylabel('Density')
    plt.show()

elif random_samples.shape[1] == 2:
    hist, x_edges, y_edges = jnp.histogram2d(random_samples[:, 0], random_samples[:, 1], bins=100, range=[[a, b], [a, b]], density=True)

    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_mesh, y_mesh = jnp.meshgrid(x_mid, y_mid, indexing='ij')

    x_flat = x_mesh.ravel()
    y_flat = y_mesh.ravel()
    z_flat = hist.T.ravel()

    non_zero_mask = z_flat > 0  
    x_flat = x_flat[non_zero_mask]
    y_flat = y_flat[non_zero_mask]
    z_flat = z_flat[non_zero_mask]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(x_flat, y_flat, jnp.zeros_like(z_flat),
             dx=(x_edges[1] - x_edges[0]), 
             dy=(y_edges[1] - y_edges[0]), 
             dz=z_flat,                
             shade=True, color='blue', alpha=0.7)
    ax.set_title("3D Histogram of Density")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Density")
    plt.show()

    plt.hist2d(random_samples[:, 0], random_samples[:, 1], bins=100, density=True, cmap='Blues')
    plt.colorbar(label='Density')
    plt.title("2D Density Histogram")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

else:
    plt.scatter(random_samples[:, 0], random_samples[:, 1], alpha=0.5, s=1, color='blue')
    plt.title("Scatter Plot of First Two Dimensions")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

expected_value = jnp.mean(random_samples, axis=0)
print("Expected Value:", expected_value)

variance = jnp.var(random_samples, axis=0)
print("Variance:", variance)