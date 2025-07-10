
# Identifiability Workflow Documentation

# Case Study 0 (Microbial Growth in a Batch Reactor)

This guide documents the Jupyter notebook `CaseStudy0_Microbial_growth__in_a_batch_reactor.ipynb`, which implements a structural identifiability analysis of a dynamic system using the method of **Stigter & Molenaar (2015)**. The methodology is sensitivity-based and uses Singular Value Decomposition (SVD) to assess **local structural identifiability**.

The notebook is structured in titled sections, and each is described below, with emphasis on which parts should be modified to adapt the workflow to a different ODE-based model. It is intended to be run in Google Colab, so the first step is to open or upload the notebook to Colab.

---

## Install & import

```python
!pip install casadi scipy tqdm --quiet

import casadi as ca
import numpy as np
from scipy.linalg import svd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
```

This section installs the required Python packages and imports the key libraries used for symbolic modeling, numerical integration, linear algebra, and plotting.

---

## Model definition

```python
# === Model configuration ===
# System dimensions and simulation horizon
nx = 2          # Number of states
ny = 1          # Number of outputs 
nth = 6         # Number of parameters
Tf = 5          # Final time [h]
Nt = 10         # Number of time points
t_eval = np.linspace(0, Tf, Nt)
```

```python
# Symbolic variables for states and parameters
x = ca.MX.sym("x", nx)
theta = ca.MX.sym("theta", nth)

# === Model definition ===
# Dynamical system for microbial growth and substrate consumption
mu, Ks, Y, Kd = theta[0], theta[1], theta[2], theta[3]
xdot = ca.vertcat(
    (mu * x[1] * x[0]) / (Ks + x[1]) - Kd * x[0],         # Biomass equation
    -(mu * x[1] * x[0]) / ((Ks + x[1]) * Y)               # Substrate equation
)

# Initial conditions as part of parameter vector
x0 = ca.vertcat(theta[4], theta[5])

#Nominal parameters
theta_nom = np.array([0.5, 2.0, 0.6, 0.01, 2.0, 1.0])
```

The code above defines the symbolic structure and dynamics of the ODE model used in this analysis. It starts by specifying the model dimensions: the number of state variables (`nx`), outputs (`ny`), and parameters (`nth`), as well as the simulation time frame (`Tf`) and the number of evaluation points (`Nt`). These values determine the temporal resolution of the simulation (`t_eval`).

Symbolic variables for the state vector `x` and parameter vector `theta` are then defined using CasADi. The system dynamics are encoded in `xdot`, which describes the time evolution of biomass and substrate concentrations based on Monod-type kinetics. The initial state conditions `x0` are extracted from the last two components of the parameter vector, making them identifiable elements in the analysis. Finally, `theta_nom` contains nominal parameter values used for simulations.

To adapt this section for a different ODE-based model:
- Update `nx`, `ny`, and `nth` according to the number of states, outputs, and parameters in your system.
- Replace the right-hand side of `xdot` with your own system of differential equations, written symbolically.
- Modify the definition of `x0` if your initial conditions differ or are fixed rather than estimated.
- Provide suitable nominal values in `theta_nom` that reflect your model's parameter scales and dynamics.



## Extended System


This section implements the extended system required for sensitivity-based identifiability analysis, following the methodology described by Stigter & Molenaar (2015). 

First, symbolic Jacobians of the model equations are computed with respect to the state variables and parameters. These derivatives define how the system responds to small changes in states or parameters and are used to construct the sensitivity equations.

Then, CasADi functions are created to evaluate the system dynamics and their derivatives numerically. These functions are used inside `meta_rhs`, which defines the right-hand side of the extended ODE system combining both the state dynamics and their parameter sensitivities.

The `simulate_once` function solves this extended system over time using `solve_ivp`. It returns both the state trajectories and the sensitivities of each state with respect to each parameter.

Finally, the `compute_dydth` function extracts the output sensitivities from the simulation results. It computes both the OSM and ROSM matrices, which quantify how the output (here, biomass) depends on each parameter over time.

The only part of this code that needs to be changed when adapting to a different model is the output definition inside the `compute_dydth` function. By default, it assumes the model output is \( h(x) = x_1(t) \), i.e., the first state variable.

To update it, modify these two lines:

```python
y_i = x_i[0]          # Replace 0 with the index of your output state
dydth_i = dx_i[0, :]  # Use the same index here
```

## Singular Value Decomposition (SVD)

This section runs the forward simulation using the nominal parameters (`theta_nom`), computes the relative output sensitivity matrix (`dydth_rel`), and performs singular value decomposition (SVD) on it. No modifications are needed in this section when applying the method to a different model.

## Monte Carlo Sensitivity Analysis

This section samples multiple parameter sets around the nominal values and analyzes how the singular value spectrum varies. This helps evaluate the robustness of local identifiability.

```python
# === Monte Carlo sensitivity analysis ===

# Number of experiments
NExp = 10

# Nominal parameter values 
theta_nom = np.array([0.5, 2.0, 0.6, 0.01, 2.0, 1.0])  # <-- Change this

# Sampling bounds 
theta_low = 0.5 * theta_nom                          
theta_high = 1.5 * theta_nom                          

# Random sampling
THETA = np.random.uniform(theta_low, theta_high, size=(NExp, nth))
THETA[0] = theta_nom 
```
To use this section with a different model:


- Define `NExp` as the number of Monte Carlo experiments you want to run. This determines how many different parameter sets will be sampled and evaluated. A higher value improves robustness but increases computational time.

- Replace `theta_nom` with the nominal parameter values specific to your model. This vector defines the baseline around which all other parameter sets will be sampled.

- Adjust the scaling factors (`0.5` and `1.5`) in `theta_low = 0.5 * theta_nom` and `theta_high = 1.5 * theta_nom` to control the sampling range. Use narrower bounds (e.g., `0.9` to `1.1`) for local sensitivity, or wider ones for more exploratory analysis.



## Plot

### Visualization and Interpretation of Monte Carlo Results

This section visualizes key results from the Monte Carlo sensitivity analysis, focusing on how identifiability is affected by parameter variation.

---

#### 1. Last Column of V: Nullspace Direction per Experiment

This plot shows the last right-singular vector $|v_{\text{last}}|$ from each Monte Carlo run. Each vector corresponds to the direction in parameter space that has the **least influence** on the model output.

If certain parameters consistently appear with large components in these vectors, it suggests they are non determinable (i.e., variations in those parameters have negligible influence on the model output). 

---

#### 2. Mean of $|v_{\text{last}}|$

This plot summarizes the absolute contributions of each parameter to the nullspace direction, averaged across all experiments. The mean gives a central tendency, while the standard deviation (shown as text) reflects variability across experiments.

---

#### 3. Log of Singular Values

This plot shows the base-10 logarithm of all singular values obtained from each Monte Carlo experiment. The values are sorted per experiment, from largest to smallest.

- The **largest singular values** correspond to parameter directions that strongly affect the output.
- The **smallest singular values** indicate directions where the model is nearly insensitive.

If the smallest singular values (e.g., $\sigma_6$) are close to zero across multiple runs, this suggests **structural or non-identifiability** of one or more parameter combinations.

# Case Study 0: Parameter Fixing

This section documents the Jupyter notebook `CaseStudy0_Fixed`.ipynb, which explores a complementary identifiability strategy by selectively fixing parameters in the original model. This approach is motivated by the idea that some parameters may be non-identifiable in the full model but could become identifiable once others are held constant.

The analysis follows the same sensitivity-based methodology introduced by Stigter & Molenaar (2015) and used in the previous notebook (`CaseStudy0_Microbial_growth__in_a_batch_reactor.ipynb`). However, instead of analyzing the entire parameter set at once, this notebook constructs reduced models by excluding (fixing) specific parameters. The reduced models are then evaluated independently using singular value decomposition (SVD) to assess whether the remaining parameters are locally structurally identifiable.

## Install & import

This notebook is intended to be run in Google Colab, so the first step is to open or upload the notebook to Colab and 
install/import the required Python packages.

## Fixing Parameters: Option 1

First option: A function that checks for structural identifiability sequentially, starting from the base case (no fixed parameters). Once it finds a suitable solution (which is measured by a threshold given by the minimum singular value), it finishes checking the combinations with the same amount of fixed parameters, thus finding the set of the minimum number of parameters fixed for a set minimum singular value.

To apply this identifiability workflow to a different model, update the **general configuration** and the **model structure** accordingly. First, redefine `theta_nom` and `param_names` to match your new model parameters. Then, in the `build_model` function, replace the symbolic equations in `xdot` with your system's differential equations and update `x0` to reflect your initial condition logic. Ensure `nx` matches the number of state variables in your system.

If your observable output differs (e.g., not `x[0]`), update the `compute_dydth` function accordingly. In `evaluate_combination`, no changes are needed unless your model is stiff or fails to integrate—then adjust solver tolerances or method. Finally, make sure your nominal parameters are reasonable to ensure stable simulations. The rest of the workflow—Monte Carlo sampling, sensitivity integration, and SVD—is general and reusable across models.

Key variables to revise:
- `theta_nom`, `param_names`, `nx`, `xdot`, `x0`
- Output definition in `compute_dydth`
- Optionally: time grid `t_eval`, final time `Tf`, and `NExp`


**Note:** This method is very slow but serves as a conceptual basis for identifiability analysis. It is useful for understanding the impact of fixing parameters, but it is not recommended for large or complex models due to its combinatorial cost.

## Fixing Parameters: Option 2

Second Option: A function that uses the mean value of the last singular vector as information for which parameters to fix, prioritizing the combinations that include non-determinable parameters as fixed.

This approach follows the **same modification rules** as Option 1. You must still adapt the general configuration, model structure, and observable definition to fit your specific system. However, the search logic differs: instead of exhaustively testing all combinations, it leverages the structural identifiability analysis already performed. The direction of the nullspace (via the last singular vector) is used to guide the parameter fixing strategy, prioritizing parameters that appear structurally non-determinable.

**Note:** While still computationally demanding, this method can reduce the number of combinations tested. It is useful for medium-sized models but remains impractical for very large systems.

Only the **search strategy** differs; all required model adaptations are the same as in Option 1.




## Fixing Parameters: Option 3: Genetic Algorithm application

Utilizing pymoo's Genetic Algorithm, this code acts as a branching search algorithm for a sufficient solution.

This method uses the same underlying model structure, simulation, and sensitivity computation as in Options 1 and 2. Therefore, you must make the same modifications to adapt it to your own model.

The **key difference** is that parameter-fixing combinations are not tested exhaustively or heuristically, but selected through **evolutionary search**, guided by identifiability performance (via the singular value spectrum). This makes the method **more scalable** and better suited for medium-to-large models, where enumerating all possible combinations would be infeasible.

Key elements to revise:
- Same as in Option 1: `theta_nom`, `param_names`, `xdot`, `x0`, `nx`, `compute_dydth`
- Optionally: GA settings (`pop_size`, generations), `threshold_sigma`, and `NExp`

# Case Study 1: Wine Alcoholic Fermentation Model

The case study contained in the Jupyter notebook `CaseStudy1_Wine_Alcoholic_Fermentation_Model.ipynb` extends the previous example by implementing a more complex and realistic dynamic model, which includes a greater number of state variables and parameters. In addition, it incorporates an external input—the temperature profile measured during fermentation—and compares simulation results against experimental data.

There are **two major differences** in this notebook compared to the simpler base version that should be considered when adapting the workflow to a new model:

1. The model includes an **input signal** (temperature), loaded from a `.mat` file and used within the system of ODEs.
2. A dedicated section handles **experimental data loading and visualization**, enabling direct comparison between simulated outputs and observed measurements.

This notebook is designed to be executed in **Google Colab**, and it assumes that the necessary data files will be **manually uploaded** by the user into the Colab session. 

The required files — `Operational_Data.mat` and `Bioreactor 2020 - cinética fermentación.xlsx` — should be available in the same folder as this notebook. 

Before running the notebook cells that load data, make sure to upload these files using the Colab file upload interface (accessible via the **Files** panel or using the `files.upload()` function) so that they are accessible within the Colab environment.


## Model Definition

In the model definition section, temperature data is loaded and converted to Kelvin. This array `TU` is later used as an input function in the dynamic model.

```python
# === Load temperature data ===
data = loadmat('Operational_Data.mat')
TU = data['Time_Temp_Pairs']
TU[:, 1] += 273.15  # °C → K
```

## Experimental data

This new section is added to load and pre-process experimental data from an Excel sheet, allowing us to compare the model simulation with real fermentation data (glucose, fructose, and YAN concentrations).

```python
# Read sheet “LAB-LO(02)” from the file that is already in the folder
df_exp = (
    pd.read_excel('Bioreactor 2020 - cinética fermentación.xlsx',
                  sheet_name='LAB-LO(02)')
      .dropna(subset=['Time (min)',
                      'Glucosa (g/L)',
                      'Fructosa (g/L)',
                      'YAN (mg/L)'])
)

# Convert columns to NumPy arrays (same pattern as with TU)
t_exp        = df_exp['Time (min)'].to_numpy(dtype=float) / 60.0   # min → h
glucosa_exp  = df_exp['Glucosa (g/L)'].to_numpy(dtype=float)
fructosa_exp = df_exp['Fructosa (g/L)'].to_numpy(dtype=float)
yan_exp      = df_exp['YAN (mg/L)'].to_numpy(dtype=float) / 1000.0 # mg/L → g/L
```
```python
# === 1. Load experimental data ===

df_exp = (
    pd.read_excel('Bioreactor 2020 - cinética fermentación.xlsx',
                  sheet_name='LAB-LO(02)')
      .dropna(subset=['Time (min)', 'Glucosa (g/L)', 'Fructosa (g/L)', 'YAN (mg/L)'])
)

t_exp        = df_exp['Time (min)'].to_numpy(dtype=float) / 60.0   # min → hours
glucose_exp  = df_exp['Glucosa (g/L)'].to_numpy(dtype=float)
fructose_exp = df_exp['Fructosa (g/L)'].to_numpy(dtype=float)
yan_exp      = df_exp['YAN (mg/L)'].to_numpy(dtype=float) / 1000.0 # mg/L → g/L

# === 2. Simulate using nominal parameters ===
x_nom, dx_dth_nom = simulate_once(theta_nom, TU)
_, _, y_out = compute_dydth(x_nom, dx_dth_nom, theta_nom, theta_nom)

# === 3. Set up output mapping ===
outputs = {
    'YAN (Yeast Assimilable Nitrogen)': {
        'idx': 0,
        'exp': yan_exp,
        'ylabel': 'YAN [g/L]',
    },
    'Glucose': {
        'idx': 1,
        'exp': glucose_exp,
        'ylabel': 'Glucose [g/L]',
    },
    'Fructose': {
        'idx': 2,
        'exp': fructose_exp,
        'ylabel': 'Fructose [g/L]',
    }
}

# === 4. Plot simulation vs experimental data ===
import matplotlib.pyplot as plt

for name, info in outputs.items():
    i = info['idx']
    exp_data = info['exp']
    
    plt.figure(figsize=(8, 4))
    plt.plot(t_eval, y_out[:, i], label='Simulation', color='tab:blue')
    plt.plot(t_exp, exp_data, 'o', label='Experimental data', color='tab:orange')
    plt.xlabel('Time [h]', fontsize=12)
    plt.ylabel(info['ylabel'], fontsize=12)
    plt.title(f'{name} vs Time', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
```

# Case Study 1: Parameter Fixing


The Jupyter notebook `CaseStudy1_Fixed.ipynb` implements the parameter fixing strategy using the Genetic Algorithm from the `pymoo` library.  
It follows the same logic and structure as the implementation for Case Study 0, but is applied to the Wine Alcoholic Fermentation Model.  
No further documentation is provided here, as the workflow is directly analogous.

# Case Study 1: t-values 
The Jupyter notebook `t_values` contains the implementation for computing the t-values of each parameter in the dynamic model.
Below, we indicate the specific code sections that need to be modified in order to adapt the analysis to your own model.

```python
## Symbolic definition
First, it is important to define the system, with symbolic definitions:

# --- Symbolic Definitions ---
t = sp.Symbol('t', real=True)
U1, U2 = sp.symbols('U1 U2', real=True)

U1 and U2 are important to define if your model has inputs.

# State variables
x = sp.symbols('x1 x2 x3 x4 x5', real=True)
statesSym = sp.Matrix(x)

# Free parameters
th = sp.symbols('th1 th2 th3 th4 th5 th6 th7 th8 th9 th10 th11 th12 th13 th14', real=True)
thetaSym = sp.Matrix(th)
th1, th2, th3, th4, th5, th6, th7, th8, th9, th10, th11, th12, th13, th14 = th

# Initial conditions
X0, N0, G0, F0, E0 = 0.2000, 0.2090, 115.55, 100.94, 0
x0 = [X0, N0, G0, F0, E0]
```
The initial conditions depends of your model.

Then, with Xdot we define the vector of ODEs. Each component represents the time derivative of a state variable. The equations are nonlinear and depend on temperature (U1), a binary switch (U2) and kinetic parameters (th1 to th14).

Later, the code defines some functions that are used to calculate the t-values:
1. `meta(t, x, TU, theta, model_funcs, dim)`:
Integrates the extended system of ODEs that includes both state variables and their parameter sensitivities.

Inputs:
- t: Current time (float)
- x: Concatenated state and sensitivity vector [x; dx/dtheta]
- TU: Matrix with time and temperature data (or None)
- theta: Parameter vector
- model_funcs: Dictionary of symbolic functions: {"f": f_func, "dfdx": dfdx_func, "dfdth": dfdth_func}
- dim: Tuple with dimensions (n_states, n_parameters)

Output:
- dxdt: Time derivative of the extended system

To modify for a new model: Update model_funcs with appropriate lambdified symbolic functions.
2. `residuals(theta_est, *args)`:
Computes the difference between the model output and experimental data, used in least-squares optimization.

Inputs:
- theta_est: Estimated (free) parameters
- args: Includes experimental times, data, and global variables (e.g., temperature)

Outputs:
- Residual vector used for optimization

To modify: Update internal call to solve_ivp and the interpolation logic if your model structure or inputs differ.

3. `t_values()`:
Estimates parameters and computes t-values using a single least-squares optimization.

Inputs:
- residuals_func: Function returning residuals
- theta_0: Initial guess
- bounds: Bounds for parameters
- texp, ydata: Experimental data

Outputs:
- theta_hat: Estimated parameters
- t_values: t-statistics
- std_theta: Standard errors
- Cov_theta: Covariance matrix

To modify: Ensure residuals_func matches your model and observation outputs.

4. `repeated_t_values()`:
Runs multiple repetitions of the t-value estimation to evaluate statistical stability (e.g., due to noise or initial guess variability).

Outputs:
- Mean, standard deviation, and variance of the t-values
- Matrix of t-values across repetitions

5. `_simular_t_values()`:
Internal helper function to simulate a t-value scenario with certain parameters fixed.

To modify: Make sure param_names is consistent list of parameters strigns (e.g., ["th1", "th2", ...]) and matches your model.

6. `calcular_t_values_multiple_combinaciones()`:
Main function to compute t-values for multiple combinations of fixed parameters in parallel.

Features:
- Parallelized with joblib
- Prints summary tables
- Optionally exports results to Excel

To modify: 
- param_names must match your model
- Replace residuals_func and parameter bounds if using a different structure


Finally, to obtain the t-values, with and without fixed parameters, it is necessary to go to t-values calculation section, where you fixed the parameters for your evaluations, like Tf (total time of simulation) and Nt (number of time points for evaluation).

In t-values calculation section, the code lines that needs to be changed are:

```python
# --- Parameters for T-value Analysis ---
# Total simulation time (in hours). Must match the total duration of the experimental time course in your new dataset.
Tf = 161  
# Number of time points for evaluation. Can be adapted based on the resolution needed.
Nt = 50   
t_eval = np.linspace(0, Tf, Nt) 

# --- Reading Experimental Data ---
# Path to the experimental dataset. Must be changed to the new file.
archivo = r".../Bioreactor2020-cineticafermentacion.xlsx"

# Sheet name within the Excel file (if applicable).
sheet = "LAB-LO(02)"

# Column names must match those in the new dataset. You may need to update these to reflect your variables (e.g., "Biomass", "Product", "Substrate", etc.).
cols = ["YAN (mg/L)", "Glucosa (g/L)", "Fructosa (g/L)", "Time (min)"]

# --- Parameters (Nominal and Bounds) ---

# theta_nominal_completo: Replace with the nominal parameter values of your new model.
theta_nominal_completo = np.array([...])

# thetaLow, thetaHigh: Bounds are typically 90–110%, but adjust based on parameter sensitivity. 
thetaLow = 0.9 * theta_nominal_completo
thetaHigh = 1.1 * theta_nominal_completo
```

The next code block performs repeated parameter estimation using least-squares optimization and computes t-statistics to assess the statistical significance of each parameter. The results are then summarized in a pandas DataFrame and exported to Excel.

```python
t_avg, t_std, t_var, t_vals_all, t_sig_count = repeated_t_values(
    residuals_func=residuals,
    theta_0=theta_nominal_completo,
    bounds=(thetaLow, thetaHigh),
    texp=t_exp,
    ydata=y_exp_interp.flatten(),
    n_reps=10,
    verbose=True
)

# Crear DataFrame con métricas
param_names = [f"θ{i+1}" for i in range(len(t_avg))]
significance = ["Yes" if abs(t) > 2 else "No" for t in t_avg]

df_stats = pd.DataFrame({
    "Parameter": param_names,
    "Mean t-value": t_avg,
    "Std. Dev.": t_std,
    "Variance": t_var,
    "Times Significant": t_sig_count,
    "Significant (|t| > 2)": significance
})

print(df_stats.round(3))
df_stats.to_excel("t_values_analysis.xlsx", sheet_name="T-Values", index=False)
```

With `repeated_t_values` repeats n_reps parameter estimations (here: 10) using random initializations within the provided bounds. It returns:

- t_avg: Mean t-value for each parameter.
- t_std: Standard deviation of t-values.
- t_var: Variance of t-values.
- t_vals_all: Matrix of all t-values from each repetition.
- t_sig_count: Number of times each parameter had |t| > 2 (i.e., statistically significant).

What you can change:

- n_reps: Increase to get more robust statistics.
- bounds: Use tighter or looser bounds depending on parameter identifiability.


Between `param_names` and `df_stats.to_excel(...)`, we generate a summary of the t-value analysis by defining parameter labels, organizing the results in a structured DataFrame, and exporting the data to an Excel file.

Finally, this code performs a t-value analysis for multiple combinations of fixed parameters in a model. It defines the parameter names `(th1, th2, ..., th14)` and a list of specific combinations to be fixed during the analysis. For each combination, it runs `sim_per_combo = 10` parameter estimation simulations in parallel (`n_jobs = -1` uses all available cores), using the calcular_t_values_multiple_combinaciones function. This function computes the t-values of the remaining (free) parameters based on how often they are statistically significant (|t| > 2), enabling the identification of robust and redundant parameters under different fixed scenarios.

```python 
# --- t-value Calculation of Multiple Combinations ---
param_names = [f"th{i+1}" for i in range(len(theta_nominal_completo))]

combinaciones_fijadas = [["th9"], ["th3", "th9"], ["th3", "th7", "th9"], ["th7"], ["th7", "th11"],
                         ["th7", "th11", "th12"], ["th7", "th9", "th12"], ["th4", "th7", "th12"]]  # setting the lecture parameters

calcular_t_values_multiple_combinaciones(
    residuals_func=residuals,
    theta_0=theta_nominal_completo,
    bounds=(thetaLow, thetaHigh),
    texp=t_exp,
    ydata=y_exp_interp.flatten(),
    combinaciones_fijadas=combinaciones_fijadas,
    param_names=param_names,
    sim_per_combo=10,
    n_jobs=-1
)
```

## Strategic_Parameter_Fixing_for_T_values

Notebook in progress



