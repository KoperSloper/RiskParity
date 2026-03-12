import os
from deep_risk_parity.core.trainer import DeepRPTrainer
from deep_risk_parity.utils.data_utils import load_dataset
from deep_risk_parity.utils.evaluation import evaluate_nn, evaluate_nominal_rp, evaluate_equal_weight

def main():
    HORIZON = 48    
    LOOKBACK = 24
    GAMMA = 3     
    HIDDEN = 64

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data_cache", f"risk_parity_data_h{HORIZON}_l{LOOKBACK}.pkl"))
    data = load_dataset(data_path)
    X_test, Y_test, Sig_test = data['test_data']
    
    K = Y_test.shape[2]
    feature_dim = X_test.shape[2]

    trainer = DeepRPTrainer(
        feature_dim=feature_dim,
        K_assets=K,
        gamma=GAMMA,
        hidden=HIDDEN
    )
    
    policy_name = f"rp_policy_h{HORIZON}_g{GAMMA}.pkl"
    policy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "policies_rp", policy_name))
    
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Could not find saved policy at {policy_path}")
        
    trainer.load_policy(policy_path)

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

if __name__ == "__main__":
    main()