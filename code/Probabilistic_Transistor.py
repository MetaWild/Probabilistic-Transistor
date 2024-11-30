import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap
import numpy as np
from uniform import standard_uniform
from pdf_integral import comp_gauss_int
from sampling import sample_from
import matplotlib.pyplot as plt

def distribution_samples(N, B, f, a, b, num_transistors, key):
    h = (b - a) / B
    x_i = jnp.arange(B) * h + a
    x_i1 = (jnp.arange(B) + 1) * h + a
    probabilities = vmap(lambda xi, xi1: comp_gauss_int(f, xi, xi1, 3))(x_i, x_i1)
    probabilities /= jnp.sum(probabilities)
    cumulative_probabilities = jnp.cumsum(probabilities)
    random_numbers = standard_uniform(num_transistors, key, N)
    bin_indices = jnp.searchsorted(cumulative_probabilities, random_numbers, side='right')
    samples = sample_from(f, bin_indices, h, a, num_transistors, key)
    return samples

def create_normal_function(mu, sigma):
    def normal(x):
        return jnp.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * jnp.sqrt(2 * jnp.pi))
    return normal

def create_exponential_function(lmbda):
    def exponential(x):
        return lmbda * jnp.exp(-lmbda * x)
    return exponential

standard_normal_distribution = create_normal_function(0, 1)
standard_exponential_distribution = create_exponential_function(1)

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
B = 1000
a = 0
b = 25
num_transistors = 20
key = random.PRNGKey(0)
random_samples = distribution_samples(N, B, standard_exponential_distribution, a, b, num_transistors, key)

plt.hist(random_samples, bins=100, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histogram of Generated Random Numbers')
plt.xlabel('Random Number')
plt.ylabel('Density')
plt.show()

expected_value = sum(random_samples) / N
print(expected_value)
variance = sum((random_samples - expected_value)**2) / N
print(variance)