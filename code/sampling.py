import numpy as np
from uniform import standard_uniform
from pdf_integral import gauss_int
import matplotlib.pyplot as plt

def sample_from(f, i, h, a):
    x_i = i * h + a
    x_i1 = (i + 1) * h + a
    x_points = np.linspace(x_i, x_i1, 4)
    y_points = f(x_points)
    polynomial = normalized_polynomial(x_points, y_points)
    return polynomial_sample(polynomial, 4, x_i, x_i1)


def normalized_polynomial(x_points, y_points):
    coefficients = np.polyfit(x_points, y_points, 3)
    polynomial = np.poly1d(coefficients)

    integral_poly = np.polyint(polynomial)

    x1, x4 = x_points[0], x_points[-1]

    area = integral_poly(x4) - integral_poly(x1)

    normalized_coefficients = coefficients / area
    normalized_polynomial = np.poly1d(normalized_coefficients)

    return normalized_polynomial

def polynomial_sample(polynomial, B, a , b):
    probabilities = np.zeros(B)
    for i in range(B):
        h = (b - a) / B
        x_i = i * h + a
        x_i1 = (i + 1) * h + a
        probabilities[i] = gauss_int(polynomial, x_i, x_i1)
    probabilities /= sum(probabilities)
    cumulative_probabilities = np.zeros(B)
    for i in range(B):
        cumulative_probabilities[i] = np.sum(probabilities[:i + 1])
    
    random_number = standard_uniform(10)
    for i in range(B):
        if cumulative_probabilities[i] >= random_number:
            x_i = i * h + a
            x_i1 = (i + 1) * h + a
            return standard_uniform(10) * (x_i1 - x_i) + x_i



x_points = np.array([0, 0.3333, 0.6666, 1])
y_points = np.array([1, 0.5, 1, 0.5])

norm_poly = normalized_polynomial(x_points, y_points)

print("Normalized Polynomial Equation:", norm_poly)

x_value = 0.55
y_value = norm_poly(x_value)
print(f"Value at x = {x_value}: y = {y_value}")

x1, x4 = x_points[0], x_points[-1]
integral_poly = np.polyint(norm_poly)
computed_area = integral_poly(x4) - integral_poly(x1)
print(f"Verified that the integral from {x1} to {x4} is approximately: {computed_area}")

num_samples = 10000
random_samples = [polynomial_sample(norm_poly, 4, x1, x4) for _ in range(num_samples)]

plt.hist(random_samples, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histogram of Generated Random Numbers')
plt.xlabel('Random Number')
plt.ylabel('Density')
plt.show()



