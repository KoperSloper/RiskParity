import jax.numpy as jnp
from flax import linen as nn

class RiskParityPolicy(nn.Module):
    K_assets: int
    hidden: int = 128
    n_layers: int = 2
    min_budget: float = 0.05

    @nn.compact
    def __call__(self, x, w_prev):
        inputs = jnp.concatenate([x, w_prev], axis=-1)
        h = inputs
        
        for _ in range(self.n_layers):
            h = nn.Dense(self.hidden, kernel_init=nn.initializers.glorot_uniform())(h)
            h = nn.leaky_relu(h, negative_slope=0.25)
            
        logits = nn.Dense(self.K_assets, kernel_init=nn.initializers.glorot_uniform())(h)
        
        raw_b = nn.softmax(logits)
        
        free_pool = 1.0 - (self.K_assets * self.min_budget)
        final_b = self.min_budget + (raw_b * free_pool)
        
        return final_b