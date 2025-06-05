# Probabilistic-Transistor
Simulating probabilistic transistors to generate random samples in parallel.

## Settings

N is the number of samples. More samples causes longer simulations

B is the number of bins. More bins makes the probability density function (pdf) more smooth which also increases computation time.

a is the beginning range of pdf. This means samples can only be after this parameter.

b is the end range of pdf.  This means samples can only be before this parameter.

NOTE: Oftentimes pdfs have infinite boundaries however the simulation cannot cover infinite boundaries as that would require infinite bins. This is why we have boundaries [a,b].

num_transistors is the number of transistors used in parallel for the simulation. Higher number of transistors means more samples can be generated in parallel.

key is for reproducibility.


## To get samples from a 3D Gaussian have these settings in place in Probabilistic_Transistor.py:

N = 1000000

B = 100

a = -5

b = 5

num_transistors = 20

key = random.PRNGKey(0)


## To get samples from a 2D Gaussian have these settings in place in Probabilistic_Transistor.py:


N = 10000000

B = 1000

a = -5

b = 5

num_transistors = 20

key = random.PRNGKey(0)

random_samples = distribution_samples(N, B, standard_normal_distribution, a, b, num_transistors, key)


## Enter this command to run code:

python3 Probabilistic_Transistor.py


## Other probability density functions to mess around with include:

standard_exponential_distribution = create_exponential_function(1)

test_normal_distribution = create_normal_function(1, 0.5)

test_normal_distribution_2d = create_normal_function_2d(1, 0.5)
