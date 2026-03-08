import os
import pickle
import jax
import jax.numpy as jnp
import optax
import numpy as np

from .policy import RiskParityPolicy
from .solver import batch_risk_parity

class DeepRPTrainer:
    def __init__(self, feature_dim, K_assets, gamma=3.0, hidden=128, lr=1e-3, seed=42):
        self.K = K_assets
        self.gamma = gamma
        
        self.model = RiskParityPolicy(K_assets=self.K, hidden=hidden)
        self.key = jax.random.PRNGKey(seed)
        
        dummy_x = jnp.zeros((1, feature_dim))
        dummy_w = jnp.ones((1, self.K)) / self.K
        
        self.key, init_key = jax.random.split(self.key)
        self.params = self.model.init(init_key, dummy_x, dummy_w)

        self.opt = optax.adamw(learning_rate=lr)
        self.opt_state = self.opt.init(self.params)
        
        self._step_fn = None
        self._val_fn = None

    def save_policy(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.params, f)
        print(f"Policy parameters successfully saved to {filepath}")

    def load_policy(self, filepath):
        with open(filepath, 'rb') as f:
            self.params = pickle.load(f)
        print(f"Policy parameters loaded from {filepath}")

    def _build_functions(self, T_steps):
        
        def loss_fn(params, batch_X, batch_R, batch_Sigma):
            B = batch_X.shape[0]
            
            def scan_body(carry, t):
                w_prev, log_wealth = carry
                
                obs = batch_X[:, t, :]
                sigma_t = batch_Sigma[:, t, :, :]
                
                # NN outputs budgets
                b_t = self.model.apply(params, obs, w_prev)
                
                # risk parity solver outputs new weights
                w_post = batch_risk_parity(b_t, sigma_t)
                
                # market step
                r_t = batch_R[:, t, :]
                port_gross = jnp.sum(w_post * r_t, axis=1)
                
                log_step = jnp.log(port_gross + 1e-10)
                log_wealth_new = log_wealth + log_step
                
                w_next = (w_post * r_t) / (port_gross[:, None] + 1e-12)
                
                return (w_next, log_wealth_new), None

            w0 = jnp.ones((B, self.K)) / self.K
            l0 = jnp.zeros(B)
            
            (w_final, log_W_final), _ = jax.lax.scan(
                scan_body, (w0, l0), jnp.arange(T_steps)
            )
            
            W_final = jnp.exp(log_W_final)
            
            if self.gamma == 1.0:
                utility = jnp.log(W_final)
            else:
                utility = (W_final ** (1.0 - self.gamma)) / (1.0 - self.gamma)
            
            metrics = {
                "loss": -jnp.mean(utility),
                "wealth": jnp.mean(W_final),
            }
            
            return metrics["loss"], metrics

        @jax.jit
        def train_step(params, opt_state, batch_X, batch_R, batch_Sigma):
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, metrics), grads = grad_fn(params, batch_X, batch_R, batch_Sigma)
            updates, new_opt_state = self.opt.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, metrics

        @jax.jit
        def val_batch(params, batch_X, batch_R, batch_Sigma):
            loss, metrics = loss_fn(params, batch_X, batch_R, batch_Sigma)
            return metrics

        return train_step, val_batch

    def train(self, data, epochs=10, batch_size=128, eval_frequency=10, patience=50):
        X_train, Y_train, Sig_train = data['train_data']
        X_val, Y_val, Sig_val       = data['val_data']
        
        T = X_train.shape[1]
        
        if self._step_fn is None:
            print(f"Compiling JAX functions for T={T}...")
            self._step_fn, self._val_fn = self._build_functions(T)
            
        N = X_train.shape[0]
        steps_per_epoch = N // batch_size
        
        print(f"Starting Training: {epochs} epochs, Patience {patience}.")
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_params = self.params
        global_step = 0
        
        for ep in range(epochs):
            perm = np.random.permutation(N)
            X_shuff = X_train[perm]
            Y_shuff = Y_train[perm]
            Sig_shuff = Sig_train[perm]
            
            for i in range(steps_per_epoch):
                bx = jnp.array(X_shuff[i*batch_size : (i+1)*batch_size])
                by = jnp.array(Y_shuff[i*batch_size : (i+1)*batch_size])
                bsig = jnp.array(Sig_shuff[i*batch_size : (i+1)*batch_size])
                
                self.params, self.opt_state, metrics = self._step_fn(
                    self.params, self.opt_state, bx, by, bsig
                )
                global_step += 1
                
                if global_step % eval_frequency == 0:
                    bx_val = jnp.array(X_val[:batch_size])
                    by_val = jnp.array(Y_val[:batch_size])
                    bsig_val = jnp.array(Sig_val[:batch_size])
                    
                    assert self._val_fn is not None, "Validation function not built."
                    val_metrics = self._val_fn(self.params, bx_val, by_val, bsig_val)
                    
                    v_loss = float(val_metrics['loss'])
                    v_wealth = float(val_metrics['wealth'])
                    
                    improved = ""
                    if v_loss < best_val_loss:
                        best_val_loss = v_loss
                        best_params = self.params
                        patience_counter = 0
                        improved = "*"
                    else:
                        patience_counter += 1
                        
                    print(f"Step {global_step:5d} | Val Loss: {v_loss:.5f} | W: {v_wealth:.4f} {improved}")
                    
                    if patience_counter >= patience:
                        print(f"\nEarly Stopping triggered at step {global_step}.")
                        self.params = best_params
                        return

        print("\nTraining Finished. Restoring best params.")
        self.params = best_params