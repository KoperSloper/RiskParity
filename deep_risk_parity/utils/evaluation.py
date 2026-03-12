import numpy as np
import jax.numpy as jnp
from jax import jit
import jax

from deep_risk_parity.core.solver import batch_risk_parity

def calc_ce_and_se(final_wealths, gamma, T):
    N = len(final_wealths)
    
    if gamma == 1.0:
        U = np.log(final_wealths)
        U_mean = np.mean(U)
        s_U = np.std(U, ddof=1) / np.sqrt(N)
        
        ce_ann = np.exp(U_mean * (12.0 / T)) - 1.0
        g_prime = (12.0 / T) * np.exp(U_mean * (12.0 / T))
        ce_se = g_prime * s_U
    else:
        U = final_wealths ** (1.0 - gamma)
        U_mean = np.mean(U)
        s_U = np.std(U, ddof=1) / np.sqrt(N)
        
        k = 12.0 / (T * (1.0 - gamma))
        ce_ann = (U_mean ** k) - 1.0

        g_prime = k * (U_mean ** (k - 1.0))
        ce_se = np.abs(g_prime) * s_U
        
    return ce_ann, ce_se

def calculate_trajectory_metrics(port_returns):
    """
    Calculates time-series metrics across a batch of return trajectories.
    Expects port_returns (gross returns) of shape (N_paths, T_months).
    """
    net_returns = port_returns - 1.0
    
    # Sortino Ratio (assuming 0% risk-free rate for simplicity)
    ann_ret = np.mean(net_returns, axis=1) * 12.0
    downside_ret = np.minimum(net_returns, 0.0)
    downside_dev = np.sqrt(np.mean(downside_ret**2, axis=1)) * np.sqrt(12.0)
    sortino = ann_ret / (downside_dev + 1e-8)
    
    # Maximum Drawdown
    wealth = np.cumprod(port_returns, axis=1)
    # Pad with initial wealth of 1.0 at t=0
    wealth_padded = np.column_stack([np.ones((wealth.shape[0], 1)), wealth])
    peaks = np.maximum.accumulate(wealth_padded, axis=1)
    drawdowns = (wealth_padded - peaks) / peaks
    max_dd = np.min(drawdowns, axis=1)  # Most negative value
    
    return {
        'sortino': np.mean(sortino),
        'max_dd': np.mean(max_dd)
    }

def evaluate_nn(trainer, X_test, Y_test, Sig_test):
    print("\nRunning NN Evaluation...")
    params = trainer.params
    
    N, T, K = Y_test.shape
    batch_size = 1000
    all_gross_returns = []
    
    @jit
    def eval_batch(bx, by, bsig):
        B = bx.shape[0]
        def body(carry, t):
            w, val = carry
            
            obs = bx[:, t]
            sigma_t = bsig[:, t]
            
            # Policy outputs risk budgets
            b_t = trainer.model.apply(params, obs)
            w_post = batch_risk_parity(b_t, sigma_t)
            
            r = by[:, t]
            port = jnp.sum(w_post * r, axis=1) # Gross return R_t
            
            w_next = (w_post * r) / (port[:, None] + 1e-12)
            val_next = val * port
            
            # Note the second output is now 'port', which scan stacks over time
            return (w_next, val_next), port

        w0 = jnp.ones((B, K)) / K
        val0 = jnp.ones(B)
        
        # port_returns_t shape will be (T, B)
        _, port_returns_t = jax.lax.scan(body, (w0, val0), jnp.arange(T))
        
        # Transpose to (B, T)
        return jnp.transpose(port_returns_t)

    for i in range(0, N, batch_size):
        bx = jnp.array(X_test[i:i+batch_size])
        by = jnp.array(Y_test[i:i+batch_size])
        bsig = jnp.array(Sig_test[i:i+batch_size])
        
        batch_returns = eval_batch(bx, by, bsig)
        all_gross_returns.append(batch_returns)
        
    all_gross_returns = np.concatenate(all_gross_returns, axis=0)
    
    # Recover final wealths
    all_final_wealths = np.prod(all_gross_returns, axis=1)
    
    ce, ce_se = calc_ce_and_se(all_final_wealths, trainer.gamma, T)
    metrics = calculate_trajectory_metrics(all_gross_returns)
    
    return ce, ce_se, np.mean(all_final_wealths), metrics

def evaluate_nominal_rp(Y_test, Sig_test, gamma):
    print("Running Nominal Risk Parity Benchmark...")
    N, T, K = Y_test.shape
    batch_size = 1000
    all_gross_returns = []
    
    @jit
    def eval_batch(by, bsig):
        B = by.shape[0]
        fixed_b = jnp.ones((B, K)) / K 
        
        def body(carry, t):
            w, val = carry
            sigma_t = bsig[:, t]
            
            w_post = batch_risk_parity(fixed_b, sigma_t)
            
            r = by[:, t]
            port = jnp.sum(w_post * r, axis=1)
            
            w_next = (w_post * r) / (port[:, None] + 1e-12)
            val_next = val * port
            return (w_next, val_next), port

        w0 = jnp.ones((B, K)) / K
        val0 = jnp.ones(B)
        
        _, port_returns_t = jax.lax.scan(body, (w0, val0), jnp.arange(T))
        return jnp.transpose(port_returns_t)

    for i in range(0, N, batch_size):
        by = jnp.array(Y_test[i:i+batch_size])
        bsig = jnp.array(Sig_test[i:i+batch_size])
        
        batch_returns = eval_batch(by, bsig)
        all_gross_returns.append(batch_returns)
        
    all_gross_returns = np.concatenate(all_gross_returns, axis=0)
    all_final_wealths = np.prod(all_gross_returns, axis=1)
    
    ce, ce_se = calc_ce_and_se(all_final_wealths, gamma, T)
    metrics = calculate_trajectory_metrics(all_gross_returns)
    
    return ce, ce_se, np.mean(all_final_wealths), metrics

def evaluate_equal_weight(Y_test, gamma):
    print("Running Equal Weight (1/N) Benchmark...")
    N, T, K = Y_test.shape
    
    w_target = np.ones((N, K)) / K
    all_gross_returns = np.zeros((N, T))
    
    for t in range(T):
        r_t = Y_test[:, t, :] 
        port_gross_ret = np.sum(w_target * r_t, axis=1)
        all_gross_returns[:, t] = port_gross_ret
        
    all_final_wealths = np.prod(all_gross_returns, axis=1)
    
    ce, ce_se = calc_ce_and_se(all_final_wealths, gamma, T)
    metrics = calculate_trajectory_metrics(all_gross_returns)
    
    return ce, ce_se, np.mean(all_final_wealths), metrics