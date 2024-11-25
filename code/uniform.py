import numpy as np
import matplotlib.pyplot as plt

def probabilistic_transistor():
    return np.random.choice([0, 1])

def generate_random_number(num_transistors=10):
    lower_bound = 0.0
    upper_bound = 1.0
    
    for _ in range(num_transistors):
        midpoint = (lower_bound + upper_bound) / 2.0
        output = probabilistic_transistor()
        
        if output == 0:
            upper_bound = midpoint
        else:
            lower_bound = midpoint
    
    random_number = (lower_bound + upper_bound) / 2.0
    return random_number

num_samples = 10000
random_numbers = [generate_random_number(10) for _ in range(num_samples)]

plt.hist(random_numbers, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histogram of Generated Random Numbers')
plt.xlabel('Random Number')
plt.ylabel('Density')
plt.show()