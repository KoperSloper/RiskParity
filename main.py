import os
import argparse
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from deep_risk_parity.core.trainer import DeepRPTrainer
from deep_risk_parity.core.solver import batch_risk_parity
from deep_risk_parity.utils.data_utils import load_dataset
from deep_risk_parity.utils.simulate_data import generate_dataset
from deep_risk_parity.utils.evaluation import evaluate_nn, evaluate_nominal_rp, evaluate_equal_weight

HORIZON = 48    
LOOKBACK = 24
GAMMA = 3     
HIDDEN = 64

SIGMA_MATRIX = np.array([
    [ 0.002894,  0.003532,  0.003910, -0.000100],
    [ 0.003532,  0.004886,  0.005712, -0.000150],
    [ 0.003910,  0.005712,  0.007259, -0.000200],
    [-0.000100, -0.000150, -0.000200,  0.000400]
])

ASSET_NAMES = ["Large Cap", "Mid Cap", "Small Cap", "Bonds"]
COLORS = ["#003366", "#336699", "#6699CC", "#8E8E8E"]

def plot_idea_1_sensitivity(trainer):
    div_yields = np.linspace(-3.0, 3.0, 100)
    ttm_constant = 0.5 
    
    X_batch = np.zeros((100, 2))
    X_batch[:, 0] = div_yields
    X_batch[:, 1] = ttm_constant
    
    sig_batch = jnp.repeat(jnp.array(SIGMA_MATRIX)[None, ...], 100, axis=0)

    b_pred = trainer.model.apply(trainer.params, X_batch)
    w_pred = batch_risk_parity(b_pred, sig_batch)

    os.makedirs("assets", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(div_yields, w_pred.T, labels=ASSET_NAMES, colors=COLORS)
    ax.set_title("Policy Sensitivity: Capital Weights vs. Dividend Yield (TTM=0.5)")
    ax.set_xlabel("Dividend Yield State (Standardized)")
    ax.set_ylabel("Capital Allocation")
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join("assets", "plot_sensitivity.png"))
    plt.close()
    print("Saved sensitivity plot to assets/plot_sensitivity.png")

def print_evaluation_table(trainer, X_test, Y_test, Sig_test):
    ce_nn, se_nn, w_nn, m_nn = evaluate_nn(trainer, X_test, Y_test, Sig_test)
    ce_nrp, se_nrp, w_nrp, m_nrp = evaluate_nominal_rp(Y_test, Sig_test, GAMMA)
    ce_ew, se_ew, w_ew, m_ew = evaluate_equal_weight(Y_test, GAMMA)
    
    print("\n" + "="*85)
    print(f"{'STRATEGY':<20} | {'CE (Ann) ± SE':<20} | {'FINAL WEALTH':<12} | {'MAX DD':<10} | {'SORTINO':<8}")
    print("-" * 85)
    
    def print_row(name, ce, se, w, m):
        ce_str = f"{ce*100:6.2f}% ± {se*100:4.2f}%"
        dd_str = f"{abs(m['max_dd'])*100:.2f}%"
        sort_str = f"{m['sortino']:.4f}"
        print(f"{name:<20} | {ce_str:<20} | ${w:<11.4f} | {dd_str:<10} | {sort_str:<8}")
        
    print_row('Neural Network RP', ce_nn, se_nn, w_nn, m_nn)
    print_row('Nominal Risk Parity', ce_nrp, se_nrp, w_nrp, m_nrp)
    print_row('Equal Weight (1/N)', ce_ew, se_ew, w_ew, m_ew)
    print("="*85)

def main():
    parser = argparse.ArgumentParser(description="Deep Risk Parity Pipeline")
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'visualize'], default='train', 
                        help="Choose to train the model, evaluate a saved model, or generate visualizations.")
    args = parser.parse_args()

    data_filename = f"risk_parity_data_h{HORIZON}_l{LOOKBACK}.pkl"
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data_cache", data_filename))
    policy_name = f"rp_policy_h{HORIZON}_g{GAMMA}.pkl"
    policy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "policies_rp", policy_name))

    # Generate data if it doesn't exist
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Generating now...")
        generate_dataset(output_path=data_path, horizon=HORIZON, lookback=LOOKBACK)
    
    data = load_dataset(data_path)
    X_train, Y_train, Sig_train = data['train_data']
    X_test, Y_test, Sig_test = data['test_data']
    
    K = Y_train.shape[2]
    feature_dim = X_train.shape[2]

    trainer = DeepRPTrainer(feature_dim=feature_dim, K_assets=K, gamma=GAMMA, hidden=HIDDEN)

    if args.mode == 'train':
        print(f"--- Running Dynamic Risk Parity Training (H={HORIZON}, Gamma={GAMMA}, K_assets={K}) ---")
        trainer.train(data, epochs=15, batch_size=128, eval_frequency=10, patience=30)
        trainer.save_policy(policy_path)
        print_evaluation_table(trainer, X_test, Y_test, Sig_test)

    elif args.mode == 'eval':
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy not found at {policy_path}. Run with '--mode train' first.")
        trainer.load_policy(policy_path)
        print_evaluation_table(trainer, X_test, Y_test, Sig_test)

    elif args.mode == 'visualize':
        if not os.path.exists(policy_path):
            print("Warning: Saved policy not found. Generating plots with randomly initialized weights.")
        else:
            trainer.load_policy(policy_path)
        plot_idea_1_sensitivity(trainer)

if __name__ == "__main__":
    main()