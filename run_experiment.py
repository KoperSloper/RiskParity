import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deep_risk_parity.core.trainer import DeepRPTrainer
from deep_risk_parity.utils.evaluation import evaluate_nn, evaluate_nominal_rp, evaluate_equal_weight
from deep_risk_parity.utils.data_utils import load_dataset
from deep_risk_parity.utils.simulate_data import generate_dataset

def main():
    HORIZON = 48    
    LOOKBACK = 24
    GAMMA   = 3     
    
    filename = f"risk_parity_data_h{HORIZON}_l{LOOKBACK}.pkl"
    data_path = os.path.join(os.path.dirname(__file__), "data_cache", filename)
    data_path = os.path.abspath(data_path)

    if not os.path.exists(data_path):
        generate_dataset(output_path=data_path, horizon=HORIZON, lookback=LOOKBACK)
    
    data = load_dataset(data_path)
    X_train, Y_train, Sig_train = data['train_data']
    X_test, Y_test, Sig_test = data['test_data']
    
    K = Y_train.shape[2]
    
    HIDDEN = 64         
    LR = 1e-3           
    
    print(f"--- Running Dynamic Risk Parity Experiment (H={HORIZON}) ---")
    print(f"Gamma={GAMMA}, K_assets={K}")
    
    trainer = DeepRPTrainer(
        feature_dim=X_train.shape[2],
        K_assets=K,
        gamma=GAMMA,
        hidden=HIDDEN,
        lr=LR
    )
    
    trainer.train(
        data, 
        epochs=15, 
        batch_size=128, 
        eval_frequency=10, 
        patience=30
    )

    policy_name = f"rp_policy_h{HORIZON}_g{GAMMA}.pkl"
    policy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "policies_rp"))
    trainer.save_policy(os.path.join(policy_dir, policy_name))

    # Evaluate all three strategies
    ce_nn, se_nn, w_nn = evaluate_nn(trainer, X_test, Y_test, Sig_test)
    ce_nrp, se_nrp, w_nrp = evaluate_nominal_rp(Y_test, Sig_test, GAMMA)
    ce_ew, se_ew, w_ew = evaluate_equal_weight(Y_test, GAMMA)
    
    print("\n" + "="*60)
    print(f"{'STRATEGY':<20} | {'CE (Ann) ± SE':<20} | {'FINAL WEALTH':<12}")
    print("-" * 60)
    print(f"{'Neural Network RP':<20} | {ce_nn*100:6.2f}% ± {se_nn*100:4.2f}%   | ${w_nn:.4f}")
    print(f"{'Nominal Risk Parity':<20} | {ce_nrp*100:6.2f}% ± {se_nrp*100:4.2f}%   | ${w_nrp:.4f}")
    print(f"{'Equal Weight (1/N)':<20} | {ce_ew*100:6.2f}% ± {se_ew*100:4.2f}%   | ${w_ew:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()