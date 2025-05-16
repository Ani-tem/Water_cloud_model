import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
file_path = "/Wheat_data.csv"

def preprocess_data(file_path):
    data = np.genfromtxt(file_path, delimiter=",", skip_header=1, filling_values=np.nan, dtype=float)

    sigma0_VV1   = data[:, 1]
    theta1       = data[:, 4]
    m_v_ground1  = data[:, 17]

    valid_rows = ~np.isnan(m_v_ground1)

    sigma0_v = sigma0_VV1[valid_rows]
    theta = theta1[valid_rows]
    m_v_ground = m_v_ground1[valid_rows]

    return sigma0_v, theta, m_v_ground

def compute_epsilon_m(sigma_vv, theta_deg, s=0.5, lambda_m=0.0056):
    theta_rad = np.radians(theta_deg)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    tan_theta = np.tan(theta_rad)

    if sin_theta == 0 or tan_theta == 0 or cos_theta == 0 or sigma_vv <= 0:
        return np.nan

    k = 2 * np.pi / lambda_m
    ks = k * s

    term_sigma = np.log10(sigma_vv)
    term1 = np.log10((cos_theta**3) / sin_theta)
    term2 = 1.1 * np.log10(ks)
    term3 = 3.3 * np.log10(sin_theta)
    term4 = 0.7 * np.log10(lambda_m)

    numerator = term_sigma + 2.35 - term1 - term2 - term3 - term4
    denominator = 0.046 * tan_theta

    return numerator / denominator

def compute_gamma_0(epsilon_r):
    root_eps = np.sqrt(epsilon_r)
    return ((1 - root_eps) / (1 + root_eps))**2

def compute_sqrt_p(theta_deg, gamma_0, s=0.5, lambda_m=0.0056):
    theta_rad = np.radians(theta_deg)
    k = 2 * np.pi / lambda_m
    ks = k * s
    base = (2 * theta_rad / np.pi)
    if base <= 0 or gamma_0 <= 0:
        return np.nan
    return 1 - (base ** (1 / (3 * gamma_0))) * np.exp(-ks)

def compute_rv(epsilon_r, theta_deg):
    theta_rad = np.radians(theta_deg)
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)

    numerator = (epsilon_r - 1) * (sin_theta**2 - epsilon_r * (1 + sin_theta**2))
    denominator = (epsilon_r * cos_theta + np.sqrt(epsilon_r - sin_theta**2)) ** 2
    return numerator / denominator

def compute_rh(epsilon_r, theta_deg):
    theta_rad = np.radians(theta_deg)
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)
    return abs((cos_theta - np.sqrt(epsilon_r - sin_theta**2)) /
               (cos_theta + np.sqrt(epsilon_r - sin_theta**2)))

def compute_sigma0_vv_from_model(g, theta_deg, sqrt_p, rv, rh):
    theta_rad = np.radians(theta_deg)
    cos_theta = np.cos(theta_rad)
    return 10*(g * (cos_theta ** 3) / sqrt_p) * (rv + rh)

# Constants
g = 0.007

# Preprocess data
sigma0_v, theta, m_v_ground = preprocess_data(file_path)

# Vector to store modeled sigma0_vv values
sigma_model_vec = []

for i in range(10):
    eps_val = compute_epsilon_m(sigma0_v[i], theta[i])
    gamma_0_val = compute_gamma_0(eps_val)
    sqrt_p_val = compute_sqrt_p(theta[i], gamma_0_val)
    rv_val = abs(compute_rv(eps_val, theta[i]))
    rh_val = compute_rh(eps_val, theta[i])
    sigma_model_val = compute_sigma0_vv_from_model(g, theta[i], sqrt_p_val, rv_val, rh_val)

    sigma_model_vec.append(sigma_model_val)

# Convert to NumPy array
sigma_model_vec = np.array(sigma_model_vec)

# Print the modeled values
print("\nVector of σ₀_VV(model) values for first 10 data points:")
print(sigma_model_vec)
