import math

# Input data
theta_values = [40.251442, 40.202873, 39.310913, 40.142815, 39.659687, 40.071384, 38.738644, 40.125854, 40.143441, 38.84277]
sigma0_vv_values = [0.03915297, 0.017670427, 0.015117519, 0.023248995, 0.018192733, 0.018360425, 0.020412635, 0.017530126, 0.01782344, 0.021276537]
ks = 0.56
lambda_val = 5.6

# Function to compute epsilon
def compute_epsilon(theta, sigma0_vv, ks, lambda_val):
    theta_rad = math.radians(theta)
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    tan_theta = math.tan(theta_rad)

    # Log terms
    log_sigma0 = math.log10(sigma0_vv)
    log_cos = math.log10(cos_theta)
    log_sin = math.log10(sin_theta)
    log_ks = math.log10(ks)
    log_lambda = math.log10(lambda_val)

    # Right-hand side of the equation
    rhs = -2.35 + (3 * log_cos - log_sin) + 1.1 * (log_ks + 3 * log_sin) + 0.7 * log_lambda

    # Solve for epsilon
    epsilon = (log_sigma0 - rhs) / (0.046 * tan_theta)

    return epsilon

# Compute epsilon for each pair
epsilon_values = []
for i in range(len(theta_values)):
    epsilon = compute_epsilon(theta_values[i], sigma0_vv_values[i], ks, lambda_val)
    epsilon_values.append(epsilon)

# Print results
print("Theta    | Sigma0_VV  | Epsilon")
print("-" * 50)
for i in range(len(theta_values)):
    print(f"{theta_values[i]:6.6f} | {sigma0_vv_values[i]:9.6f} | {epsilon_values[i]:9.6f}")

# Optional: Open canvas for visualization
"""
from canvas import Canvas  # Hypothetical canvas module
canvas = Canvas()
canvas.plot(theta_values, epsilon_values, label='Epsilon', color='blue')
canvas.show()
"""
