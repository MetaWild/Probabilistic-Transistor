# Probabilistic-Transistor
Simulating probabilistic transistors to generate random samples in parallel.

To get samples from a 3D Gaussian have these settings in place in Probabilistic_Transistor.py:

N = 1000000
B = 100
a = -5
b = 5
num_transistors = 20
key = random.PRNGKey(0)
random_samples = distribution_samples(N, B, standard_normal_2d, a, b, num_transistors, key)

To get samples from a 2D Gaussian have these settings in place in Probabilistic_Transistor.py:

N = 10000000
B = 1000
a = -5
b = 5
num_transistors = 20
key = random.PRNGKey(0)
random_samples = distribution_samples(N, B, standard_normal_distribution, a, b, num_transistors, key)

Enter this command to run code:

python3 Probabilistic_Transistor.py


Other probability density functions to mess around with include:

standard_exponential_distribution = create_exponential_function(1)
