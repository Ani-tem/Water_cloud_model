import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Updated Constants
Ks = 112.1802 * 0.005  # Ks value

# Full dataset from latest output
theta_values = np.array([
    40.251442, 40.202873, 39.310913, 40.142815, 39.659687,
    40.071384, 38.738644, 40.125854, 40.143441, 38.842770
])

sigma0_values = np.array([
    0.039153, 0.017670, 0.015118, 0.023249, 0.018193,
    0.018360, 0.020413, 0.017530, 0.017823, 0.021277
])

epsilon_values = np.array([
    94.606111, 85.883150, 86.895326, 89.139731, 87.910725,
    86.719028, 92.269625, 86.031307, 86.162813, 92.403254
])

# Updated gamma0 values from the latest output
r0_values = np.array([
    0.661859, 0.648359, 0.650014, 0.653599, 0.651651,
    0.649728, 0.658406, 0.648602, 0.648819, 0.658607
])

# Calculate e^(-Ks)
e_neg_ks = math.exp(-Ks)
print(f"Value of e^(-Ks) with Ks = {Ks}: {e_neg_ks:e}")

# Create a DataFrame to store all values
results_df = pd.DataFrame()
results_df['Row'] = range(1, len(epsilon_values) + 1)
results_df['Theta'] = theta_values
results_df['Sigma0_VV'] = sigma0_values
results_df['Epsilon (ε_r)'] = epsilon_values
results_df['Γ₀ = [(1-√ε_r)/(1+√ε_r)]²'] = r0_values

# Calculate P values for all data points
sqrt_p_values = []
p_values = []

print("\nResults for the equation √P = 1 - (2θ/π)^(1/3r₀) · e^(-Ks):")
print("Row | Theta     | Sigma0    | Epsilon    | r0      | √P         | P")
print("-" * 75)

for i in range(len(theta_values)):
    theta = theta_values[i]
    r0 = r0_values[i]

    # Convert theta to radians
    theta_rad = theta * (math.pi / 180)

    # Calculate (2θ/π)^(1/3r₀)
    exponent = 1 / (3 * r0)
    base = (2 * theta_rad) / math.pi
    power_term = base ** exponent

    # Calculate √P = 1 - (2θ/π)^(1/3r₀) · e^(-Ks)
    sqrt_P = 1 - power_term * e_neg_ks
    sqrt_p_values.append(sqrt_P)

    # Calculate P
    P = sqrt_P ** 2
    p_values.append(P)

    # Print results for this data point
    print(f"{i+1:3d} | {theta:9.6f} | {sigma0_values[i]:9.6f} | {epsilon_values[i]:10.6f} | {r0:7.6f} | {sqrt_P:10.6f} | {P:10.6f}")

# Add results to DataFrame
results_df['√P'] = sqrt_p_values
results_df['P'] = p_values

# Calculate average P value
avg_p = sum(p_values) / len(p_values)
print(f"\nAverage value of P: {avg_p:.10f}")

# Print the 10 P values as a list for easy reference
print("\nAll P values:")
for i, p in enumerate(p_values):
    print(f"P{i+1} = {p:.10f}")

# Save results to CSV
results_df.to_csv('final_p_values.csv', index=False)
print("\nResults saved to 'final_p_values.csv'")

# Create visualization of the relationship between Epsilon and P
plt.figure(figsize=(10, 6))
plt.scatter(epsilon_values, p_values, color='blue', marker='o')
plt.plot(epsilon_values, p_values, 'r--')
plt.xlabel('Epsilon (ε_r)')
plt.ylabel('P')
plt.title('Relationship between Epsilon and P')
plt.grid(True)
plt.tight_layout()
plt.savefig('final_epsilon_p_plot.png')
print("Plot saved as 'final_epsilon_p_plot.png'")

# Create visualization of the relationship between Gamma0 and P
plt.figure(figsize=(10, 6))
plt.scatter(r0_values, p_values, color='green', marker='o')
plt.plot(r0_values, p_values, 'b--')
plt.xlabel('Γ₀')
plt.ylabel('P')
plt.title('Relationship between Γ₀ and P')
plt.grid(True)
plt.tight_layout()
plt.savefig('final_gamma0_p_plot.png')
print("Plot saved as 'final_gamma0_p_plot.png'")

# Create visualization of the relationship between Theta and P
plt.figure(figsize=(10, 6))
plt.scatter(theta_values, p_values, color='purple', marker='o')
plt.plot(theta_values, p_values, 'g--')
plt.xlabel('Theta')
plt.ylabel('P')
plt.title('Relationship between Theta and P')
plt.grid(True)
plt.tight_layout()
plt.savefig('final_theta_p_plot.png')
print("Plot saved as 'final_theta_p_plot.png'")

# Create visualization of the relationship between Sigma0 and P
plt.figure(figsize=(10, 6))
plt.scatter(sigma0_values, p_values, color='orange', marker='o')
plt.plot(sigma0_values, p_values, 'm--')
plt.xlabel('Sigma0_VV')
plt.ylabel('P')
plt.title('Relationship between Sigma0_VV and P')
plt.grid(True)
plt.tight_layout()
plt.savefig('final_sigma0_p_plot.png')
print("Plot saved as 'final_sigma0_p_plot.png'")

# Detailed calculation for first value
i = 0
theta = theta_values[i]
r0 = r0_values[i]
theta_rad = theta * (math.pi / 180)
exponent = 1 / (3 * r0)
base = (2 * theta_rad) / math.pi
power_term = base ** exponent
sqrt_P = 1 - power_term * e_neg_ks
P = sqrt_P ** 2

print("\nDetailed calculation for first data point:")
print(f"Theta = {theta}")
print(f"Sigma0_VV = {sigma0_values[i]}")
print(f"Epsilon (ε_r) = {epsilon_values[i]}")
print(f"Γ₀ = {r0}")
print(f"Theta (radians) = {theta_rad:.6f}")
print(f"Exponent (1/3r₀) = {exponent:.6f}")
print(f"Base (2θ/π) = {base:.6f}")
print(f"(2θ/π)^(1/3r₀) = {power_term:.6f}")
print(f"e^(-Ks) = {e_neg_ks:.6e}")
print(f"(2θ/π)^(1/3r₀) · e^(-Ks) = {power_term * e_neg_ks:.6e}")
print(f"√P = 1 - (2θ/π)^(1/3r₀) · e^(-Ks) = {sqrt_P:.6f}")
print(f"P = (√P)² = {P:.6f}")
