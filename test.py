import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

jax.config.update("jax_enable_x64", True)


def newton_solver(b, sigma, max_iters=50, tol=1e-8):
    """Solves F(x) = Sigma * x - b / x = 0."""
    
    # err > tol and i < max_iters are the conditions to continue iterating
    def cond_fun(state):
        x, err, i = state
        return (err > tol) & (i < max_iters)

    def body_fun(state):
        x, err, i = state
        
        # evaluate function
        F_x = sigma @ x - b / x

        J_x = sigma + jnp.diag(b / (x ** 2))
        
        c, lower = cho_factor(J_x)
        dx = cho_solve((c, lower), -F_x)
        x_new = x + dx

        new_err = jnp.max(jnp.abs(F_x))
        return (x_new, new_err, i + 1)

    # initial guess for x_0
    row_sums = jnp.sum(sigma, axis=1)
    x0 = jnp.sqrt(b) / jnp.sqrt(jnp.abs(row_sums) + 1e-10)
    
    # initial error
    F_x0 = sigma @ x0 - b / x0
    err0 = jnp.max(jnp.abs(F_x0))

    x_final, _, _ = jax.lax.while_loop(cond_fun, body_fun, (x0, err0, jnp.array(0)))
    return x_final


@jax.custom_vjp
def solve_x(b, sigma):
    return newton_solver(b, sigma)

def solve_x_fwd(b, sigma):
    x = solve_x(b, sigma)
    return x, (x, b, sigma) # save residual state

def solve_x_bwd(res, v_x):
    x, b, sigma = res

    J_x = sigma + jnp.diag(b / (x ** 2)) # jacobian of F(X)
    
    c, lower = cho_factor(J_x)
    u = cho_solve((c, lower), -v_x)

    grad_b = -u / x
    
    return grad_b, None

solve_x.defvjp(solve_x_fwd, solve_x_bwd)


def risk_parity(b, sigma):
    x = solve_x(b, sigma)

    w = x / jnp.sum(x)
    return w


def loss_fn(b, sigma):
    w = risk_parity(b, sigma)
    return jnp.sum(w ** 2)

def central_difference_grad(func, b, sigma, eps=1e-6):
    grads = []
    for i in range(len(b)):
        b_plus = b.at[i].set(b[i] + eps)
        b_minus = b.at[i].set(b[i] - eps)
        grad_i = (func(b_plus, sigma) - func(b_minus, sigma)) / (2 * eps)
        grads.append(grad_i)
    return jnp.array(grads)

d = 4
b_init = jnp.array([0.4, 0.3, 0.2, 0.1])

# Generate a random, dense positive-definite covariance matrix
key = jax.random.PRNGKey(20)
A = jax.random.normal(key, (d, d))
sigma_init = jnp.dot(A, A.T) + jnp.eye(d) * 0.1 # Add eye for stability

# --- Calculate & Compare ---
jax_grad = jax.grad(loss_fn, argnums=0)(b_init, sigma_init)
fd_grad = central_difference_grad(loss_fn, b_init, sigma_init)
error = jnp.abs(jax_grad - fd_grad)

print("--- Gradient Comparison ---")
print(f"JAX Custom VJP : {jax_grad}")
print(f"Fin Difference : {fd_grad}")
print(f"Max Error      : {jnp.max(error):.2e}")