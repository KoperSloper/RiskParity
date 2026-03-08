import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve

# Enable float64 for precise evaluation
jax.config.update("jax_enable_x64", True)

def newton_solver_test(b, sigma, x0, max_iters=50, tol=1e-10):
    """A modified solver that takes x0 as an argument and returns iteration count."""
    
    def cond_fun(state):
        x, err, i = state
        return (err > tol) & (i < max_iters)

    def body_fun(state):
        x, err, i = state
        
        # 1. Evaluate Function F(x)
        F_x = sigma @ x - b / x
        
        # 2. Evaluate Jacobian J_x
        J_x = sigma + jnp.diag(b / (x ** 2))
        
        # 3. Newton Step
        dx = solve(J_x, -F_x)
        x_new = x + dx
        
        # 4. Compute true error of the new step to be precise for our test
        F_x_new = sigma @ x_new - b / x_new
        new_err = jnp.max(jnp.abs(F_x_new))
        
        return (x_new, new_err, i + 1)

    # Calculate initial error based on the guess provided
    F_x0 = sigma @ x0 - b / x0
    err0 = jnp.max(jnp.abs(F_x0))
    i0 = jnp.array(0)
    
    # Run the compiled loop
    x_final, final_err, total_iters = jax.lax.while_loop(cond_fun, body_fun, (x0, err0, i0))
    return x_final, total_iters, final_err

# =========================================================
# EXPERIMENT SETUP
# =========================================================
d = 15 # Let's use a slightly larger portfolio to make it work harder
b = jnp.ones(d) / d # Equal risk budgets (1/N)

# Generate a random covariance matrix
key = jax.random.PRNGKey(20)
A = jax.random.normal(key, (d, d))
sigma = jnp.dot(A, A.T) + jnp.eye(d) * 0.1 

# =========================================================
# METHOD 1: The Old "Naive" Guess
# =========================================================
x0_old = jnp.sqrt(b)

x_final_old, iters_old, err_old = jax.jit(newton_solver_test)(b, sigma, x0_old)

# =========================================================
# METHOD 2: The "Diagonal Row-Sum" Heuristic (Bulletproofed)
# =========================================================
row_sums = jnp.sum(sigma, axis=1) 

# FIX: Add jnp.abs() to protect against negative random sums
x0_new = jnp.sqrt(b) / jnp.sqrt(jnp.abs(row_sums))

x_final_new, iters_new, err_new = jax.jit(newton_solver_test)(b, sigma, x0_new)

# =========================================================
# RESULTS
# =========================================================
print("--- Old Method (Naive Guess) ---")
print(f"Iterations : {iters_old}")
print(f"Final Error: {err_old:.2e}\n")

print("--- New Method (Row-Sum Heuristic) ---")
print(f"Iterations : {iters_new}")
print(f"Final Error: {err_new:.2e}")