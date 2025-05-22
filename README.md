# Sigma₀_VV Model Estimation from Remote Sensing Data

This Python script processes satellite remote sensing data to estimate the modeled backscatter coefficient (σ₀_VV) using a physically-based model. The goal is to compare real measurements with model predictions to understand surface parameters like soil moisture and vegetation.

## 📂 Dataset

The input data should be in CSV format. Ensure it includes at least the following:
- Column 1: σ₀_VV (backscatter coefficient)
- Column 4: θ (incidence angle in degrees)
- Column 17: Ground truth soil moisture (m_v_ground)

The script uses these fields and skips any rows with missing soil moisture values.

## 📈 Workflow Overview

1. **Data Preprocessing**:
   - Loads data, handles NaN values.
   - Extracts σ₀_VV, θ, and m_v_ground from the dataset.

2. **Physical Modeling**:
   - Computes complex dielectric constant ε_r based on input parameters.
   - Calculates reflection coefficients (RV, RH) using θ and ε_r.
   - Estimates model-based σ₀_VV using analytical expressions.

3. **Model Evaluation**:
   - Outputs the modeled σ₀_VV values for the first 10 valid data points.

## 📌 Dependencies

Install the required Python packages:

```bash
pip install numpy matplotlib scikit-learn
```

## 🚀 How to Run

1. Place your dataset in the working directory and update the `file_path` in the script.
2. Run the script:

```bash
python your_script_name.py
```

3. Output will display modeled σ₀_VV values for the first 10 entries.

## 🧠 Functions Breakdown

- `compute_epsilon_m(...)`: Computes the complex dielectric constant.
- `compute_gamma_0(...)`: Fresnel reflection gamma value.
- `compute_sqrt_p(...)`: Probability function used in modeling σ₀_VV.
- `compute_rv(...)`, `compute_rh(...)`: Reflectivity for vertical and horizontal polarizations.
- `compute_sigma0_vv_from_model(...)`: Final σ₀_VV estimate from model.

## 📊 Example Output

```
Vector of σ₀_VV(model) values for first 10 data points:
[ ... values ... ]
```

## 📌 Notes

- This script only processes the **first 10** valid data points. Extend the loop if you want to evaluate the full dataset.
- The modeling assumes simplified surface and vegetation scattering, not accounting for dynamic vegetation changes or advanced radar modeling.

## 📬 Author

- Created by a student pursuing a career in global MNCs
- For academic and research use
