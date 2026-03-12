# deep-risk-parity

A minimal implementation of a differentiable Risk Parity portfolio solver in JAX.

Most deep learning approaches to portfolio optimization treat the neural net as a black box that directly outputs capital allocations. This usually leads to poor, highly concentrated portfolios. In this repo, we give the network a strict financial inductive bias: instead of outputting weights, the network outputs *risk budgets*.

We then pass these budgets through a classic Newton-Raphson root-finding algorithm to get the final capital weights. Because we want to train this end-to-end, we use the Implicit Function Theorem (IFT) to write a custom Vector-Jacobian Product (VJP) in JAX. This lets gradients flow cleanly through the solver on the backward pass without unrolling the optimization loop or inverting giant Jacobians.

### install

It's just standard JAX and Flax.

```bash
git clone [https://github.com/KoperSloper/RiskParity.git](https://github.com/KoperSloper/RiskParity.git)
cd RiskParity
pip install jax jaxlib flax optax numpy scipy matplotlib
```

### usage

Everything is routed through `main.py` using command-line arguments.

To run the whole pipeline (simulates synthetic AR(1) market data, trains the policy, and evaluates it):
```bash
python main.py --mode train
```

To evaluate a previously trained policy on the test set:
```bash
python main.py --mode eval
```

To generate sensitivity plots (saved to `assets/`):
```bash
python main.py --mode visualize
```

### results

We evaluate the model out-of-sample on a synthetic 48-month financial trajectory (Large Cap, Mid Cap, Small Cap, Bonds). The network observes the dividend yield state and dynamically adjusts the risk budgets. The objective is to maximize expected CRRA utility ($\gamma = 3$).

| Strategy             | CE (Ann) ± SE        | Final Wealth | Max DD   | Sortino |
|----------------------|----------------------|--------------|----------|---------|
| **Neural Network RP**| **4.59% ± 0.07%** | $1.3146 | 15.27%   | **1.240**|
| Nominal Risk Parity  | 3.77% ± 0.05%        | $1.2189      | **11.98%**| 1.202  |
| Equal Weight (1/N)   | 3.76% ± 0.11%        | **$1.4261**      | 24.37%   | 0.968  |

By operating in risk space, the network safely outperforms static baselines but is structurally prevented from doing anything too crazy.

### under the hood

If you want to understand the IFT hack:

Risk Parity finds a weight vector $y$ where each asset's risk contribution matches a target risk budget $b$. This means finding the root of:
$$F(x) = \Sigma x - \frac{b}{x} = 0$$

To train the neural net, we need $\nabla_b L$. Instead of unrolling the solver, the Implicit Function Theorem tells us we can just solve a linear system for an intermediate vector $u$ on the backward pass:
$$\left( \frac{\partial F}{\partial x} \right)^\top u = -v_x$$

Then the gradient we pass back to the neural net is just:
$$\nabla_b L = - \frac{u}{x}$$

It's fast, exact, and fits nicely into `jax.custom_vjp`.

### code structure

- `main.py`: Single entry point for train/eval/viz.
- `deep_risk_parity/core/policy.py`: The Flax MLP.
- `deep_risk_parity/core/solver.py`: The Newton-Raphson solver + IFT VJP.
- `deep_risk_parity/core/trainer.py`: The Optax training loop.

---
### contact
- Ton Vossen ([@KoperSloper](https://github.com/KoperSloper))
- ton.vossen@outlook.com