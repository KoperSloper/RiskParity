import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from deep_risk_parity.core.trainer import DeepRPTrainer
from deep_risk_parity.core.solver import batch_risk_parity
from deep_risk_parity.utils.data_utils import load_dataset

SIGMA_MATRIX = np.array([
    [ 0.002894,  0.003532,  0.003910, -0.000100],
    [ 0.003532,  0.004886,  0.005712, -0.000150],
    [ 0.003910,  0.005712,  0.007259, -0.000200],
    [-0.000100, -0.000150, -0.000200,  0.000400]
])

ASSET_NAMES = ["Large Cap", "Mid Cap", "Small Cap", "Bonds"]
COLORS = ["#003366", "#336699", "#6699CC", "#8E8E8E"]

def plot_idea_1_sensitivity(trainer):
    # Create a range of Dividend Yields (from -3 to +3 std deviations)
    div_yields = np.linspace(-3.0, 3.0, 100)
    ttm_constant = 0.5 
    
    # Build batch of inputs
    X_batch = np.zeros((100, 2))
    X_batch[:, 0] = div_yields
    X_batch[:, 1] = ttm_constant
    
    sig_batch = jnp.repeat(jnp.array(SIGMA_MATRIX)[None, ...], 100, axis=0)

    b_pred = trainer.model.apply(trainer.params, X_batch)
    w_pred = batch_risk_parity(b_pred, sig_batch)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(div_yields, w_pred.T, labels=ASSET_NAMES, colors=COLORS)
    ax.set_title("Policy Sensitivity: Capital Weights vs. Dividend Yield (TTM=0.5)")
    ax.set_xlabel("Dividend Yield State (Standardized)")
    ax.set_ylabel("Capital Allocation")
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("plot_1_sensitivity.png")
    plt.close()

def main():
    HORIZON = 48
    GAMMA = 3

    trainer = DeepRPTrainer(feature_dim=2, K_assets=4, gamma=GAMMA, hidden=64)
    
    policy_path = os.path.join(os.path.dirname(__file__), "policies_rp", f"rp_policy_h{HORIZON}_g{GAMMA}.pkl")
    if os.path.exists(policy_path):
        trainer.load_policy(policy_path)
    else:
        print("Warning: Saved policy not found. Using randomly initialized weights.")
    
    plot_idea_1_sensitivity(trainer)
    

if __name__ == "__main__":
    main()