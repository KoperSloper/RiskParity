import pickle
import numpy as np
import os

# 5 variables: large, mid, small, bonds, div_yield
A_COEFFS = np.array([0.0053, 0.0067, 0.0072, 0.0015, 0.0000]) 
B_COEFFS = np.array([0.0028, 0.0049, 0.0061, 0.0005, 0.9700]) 

SIGMA_MATRIX = np.array([
    [ 0.002894,  0.003532,  0.003910, -0.000100, -0.000115],
    [ 0.003532,  0.004886,  0.005712, -0.000150, -0.000144],
    [ 0.003910,  0.005712,  0.007259, -0.000200, -0.000163],
    [-0.000100, -0.000150, -0.000200,  0.000400,  0.000050],
    [-0.000115, -0.000144, -0.000163,  0.000050,  0.052900]
])

def simulate_var_paths(num_paths, horizon, lookback=24, seed=None):
    rng = np.random.default_rng(seed)
    
    # ensure positive definite
    safe_sigma = SIGMA_MATRIX + np.eye(5) * 1e-6
    L = np.linalg.cholesky(safe_sigma)
    
    steps_total = horizon + lookback
    
    D_path = np.zeros((num_paths, steps_total + 1))
    R_path = np.zeros((num_paths, steps_total, 4)) 
    
    D_path[:, 0] = rng.standard_normal(num_paths)
    Z = rng.standard_normal(size=(num_paths, steps_total, 5))
    Shocks = np.einsum('ij,ntj->nti', L, Z) 

    for t in range(steps_total):
        d_t = D_path[:, t]
        mu_vec = A_COEFFS + B_COEFFS * d_t[:, None]
        R_path[:, t, :] = mu_vec[:, :4] + Shocks[:, t, :4]
        D_path[:, t+1]  = mu_vec[:, 4]  + Shocks[:, t, 4]

    Sigma_4x4 = np.zeros((num_paths, horizon, 4, 4), dtype=np.float32)
    
    for t in range(horizon):
        window_returns = R_path[:, t : t + lookback, :] 
        
        mean_R = np.mean(window_returns, axis=1, keepdims=True)
        centered = window_returns - mean_R
        cov_4x4 = np.einsum('nti,ntj->nij', centered, centered) / (lookback - 1)
        
        Sigma_4x4[:, t, :, :] = cov_4x4 + np.eye(4) * 1e-6

    X_div = D_path[:, lookback : -1].reshape(num_paths, horizon, 1)
    
    ttm_vec = np.linspace(1.0, 0.0, horizon).reshape(1, horizon, 1)
    X_ttm = np.repeat(ttm_vec, num_paths, axis=0)
    X_combined = np.concatenate([X_div, X_ttm], axis=2)
    
    Y_returns_gross = np.exp(R_path[:, lookback:, :])
    
    return X_combined.astype(np.float32), Y_returns_gross.astype(np.float32), Sigma_4x4

def generate_dataset(output_path, horizon=48, lookback=24, n_train=20000, n_val=5000, n_test=10000):
    print(f"Generating Data (Horizon={horizon}, Lookback={lookback}) -> {output_path}...")
    
    def _gen(n, s_off):
        return simulate_var_paths(n, horizon, lookback=lookback, seed=42 + s_off)

    X_train, Y_train, Sig_train = _gen(n_train, 100)
    X_val,   Y_val,   Sig_val   = _gen(n_val, 200)
    X_test,  Y_test,  Sig_test  = _gen(n_test, 300)
    
    meta = {
        "asset_cols": ["Large", "Mid", "Small", "Bonds"],
        "state_cols": ["DivYield", "TTM"],
        "horizon": horizon,
        "lookback": lookback
    }
    
    data = {
        "train_data": (X_train, Y_train, Sig_train),
        "val_data":   (X_val, Y_val, Sig_val),
        "test_data":  (X_test, Y_test, Sig_test),
        "meta": meta
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print("Dataset generated and saved successfully.")

if __name__ == "__main__":
    generate_dataset("risk_parity_data.pkl", horizon=48, lookback=24)