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

def evaluate_nn(trainer, X_test, Y_test, Sig_test):
    print("\nRunning NN Evaluation...")
    params = trainer.params
    
    N, T, K = Y_test.shape
    batch_size = 1000
    final_wealths = []
    
    @jit
    def eval_batch(bx, by, bsig):
        B = bx.shape[0]
        def body(carry, t):
            w, val = carry
            
            obs = bx[:, t]
            sigma_t = bsig[:, t]
            
            b_t = trainer.model.apply(params, obs, w)
            w_post = batch_risk_parity(b_t, sigma_t)
            
            r = by[:, t]
            port = jnp.sum(w_post * r, axis=1)
            
            w_next = (w_post * r) / (port[:, None] + 1e-12)
            val_next = val * port
            return (w_next, val_next), None

        w0 = jnp.ones((B, K)) / K
        val0 = jnp.ones(B)
        
        (_, val_end), _ = jax.lax.scan(body, (w0, val0), jnp.arange(T))
        return val_end

    for i in range(0, N, batch_size):
        bx = jnp.array(X_test[i:i+batch_size])
        by = jnp.array(Y_test[i:i+batch_size])
        bsig = jnp.array(Sig_test[i:i+batch_size])
        
        w_out = eval_batch(bx, by, bsig)
        final_wealths.append(w_out)
        
    all_w = np.concatenate(final_wealths)
    ce, ce_se = calc_ce_and_se(all_w, trainer.gamma, T)
    return ce, ce_se, np.mean(all_w)


def evaluate_nominal_rp(Y_test, Sig_test, gamma):
    print("Running Nominal Risk Parity Benchmark...")
    N, T, K = Y_test.shape
    batch_size = 1000
    final_wealths = []
    
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
            return (w_next, val_next), None

        w0 = jnp.ones((B, K)) / K
        val0 = jnp.ones(B)
        
        (_, val_end), _ = jax.lax.scan(body, (w0, val0), jnp.arange(T))
        return val_end

    for i in range(0, N, batch_size):
        by = jnp.array(Y_test[i:i+batch_size])
        bsig = jnp.array(Sig_test[i:i+batch_size])
        
        w_out = eval_batch(by, bsig)
        final_wealths.append(w_out)
        
    all_w = np.concatenate(final_wealths)
    ce, ce_se = calc_ce_and_se(all_w, gamma, T)
    return ce, ce_se, np.mean(all_w)


def evaluate_equal_weight(Y_test, gamma):
    print("Running Equal Weight (1/N) Benchmark...")
    N, T, K = Y_test.shape
    
    w_target = np.ones((N, K)) / K
    wealth = np.ones(N)
    
    for t in range(T):
        r_t = Y_test[:, t, :] 
        port_gross_ret = np.sum(w_target * r_t, axis=1)
        wealth = wealth * port_gross_ret
        
    ce, ce_se = calc_ce_and_se(wealth, gamma, T)
    return ce, ce_se, np.mean(wealth)