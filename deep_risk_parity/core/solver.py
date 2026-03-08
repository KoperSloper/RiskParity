import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

def newton_solver(b, sigma, max_iters=20, tol=1e-5):
    
    def cond_fun(state):
        x, err, i = state
        return (err > tol) & (i < max_iters)

    def body_fun(state):
        x, err, i = state
        
        F_x = sigma @ x - b / x
        J_x = sigma + jnp.diag(b / (x ** 2))
        
        c, lower = cho_factor(J_x)
        dx = cho_solve((c, lower), -F_x)
        x_new = jnp.maximum(x + dx, 1e-8)  # ensure positivity

        new_err = jnp.max(jnp.abs(F_x))
        return (x_new, new_err, i + 1)

    row_sums = jnp.sum(sigma, axis=1)
    x0 = jnp.sqrt(b) / jnp.sqrt(jnp.abs(row_sums) + 1e-10)
    
    F_x0 = sigma @ x0 - b / x0
    err0 = jnp.max(jnp.abs(F_x0))

    x_final, _, _ = jax.lax.while_loop(cond_fun, body_fun, (x0, err0, jnp.array(0)))
    return x_final

@jax.custom_vjp
def solve_x(b, sigma):
    return newton_solver(b, sigma)

def solve_x_fwd(b, sigma):
    x = solve_x(b, sigma)
    return x, (x, b, sigma)

def solve_x_bwd(res, v_x):
    x, b, sigma = res

    J_x = sigma + jnp.diag(b / (x ** 2)) 
    
    c, lower = cho_factor(J_x)
    u = cho_solve((c, lower), -v_x)

    grad_b = -u / x
    return grad_b, None

solve_x.defvjp(solve_x_fwd, solve_x_bwd)

def risk_parity(b, sigma):
    x = solve_x(b, sigma)
    return x / jnp.sum(x)

# Vectorized version for batch processing in the training loop
batch_risk_parity = jax.vmap(risk_parity, in_axes=(0, 0))