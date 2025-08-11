# **Project Name: Hybrid-QAOA Portfolio Optimizer**
_WISER & Womanium Challenge 2025_

Team Members:

Andry Paez - gst-Si5rZLCtdPsRCXI

---

### **1. The Challenge Problem**

Modern portfolio optimization involves solving large-scale integer-quadratic programs with thousands of assets and numerous constraints under stringent intraday time limits. Although state-of-the-art classical solvers such as GUROBI or CPLEX are highly effective, **their performance can degrade unpredictably to non-polynomial runtimes as problem complexity scales**. Consequently, if quantum optimization can offer even a modest constant-factor speedup for these problems, its potential business value is substantial.

The 31-bond toy instance provided by the challenge organizers is small enough to run on a laptop but complex enough for meaningful benchmarking. It includes:

  * 31 **binary** decision variables (i.e., whether to buy or skip each bond)
  * Over 100 linear side-constraints
  * A quadratic, risk-adjusted objective function

This complexity is sufficient to benchmark a hybrid algorithm where the intensive combinatorial search is performed on a quantum processor, while the final fine-tuning is handled by a classical post-processor.

---

### **2. Rationale for Building on a Reference Implementation**

The organizers provided a complete reference pipeline, including the TwoLocal and BFGD ansätze, along with a local search post-processor. Re-implementing every helper function, pre-processing step, and evaluation harness from scratch would have consumed most of the four-week timeframe without advancing the core research.

Therefore, I adopted a pragmatic approach:

  * **Leverage the existing scaffold**, including the directory structure, command-line scripts, and logging conventions.
  * **Integrate QAOA** as a new ansatz family, using the standard implementation from `qiskit-algorithms` along with its preset transpiler pass manager.
  * **Expose all new hyperparameters** (such as $\alpha$, `reps`, and local search settings) through the existing Design of Experiments (`doe.py`) framework, ensuring our grid search integrated seamlessly.

This strategy allowed me to focus my efforts on *novel experiments* rather than on developing foundational code.

---

### **3. Summary of Changes**

| \# | Change | Files / Notes |
|---|---|---|
| **1** | Pinned dependencies using **Poetry** for reproducibility. | [`pyproject.toml`](./pyproject.toml) |
| **2** | Added command-line flags for granular control of each step. | `scripts/sbo_steps1to3.py`, `scripts/sbo_step4.py` |
| **3** | Expanded the [`doe.py`](./_experiments/doe.py) grid search to include **QAOA**. | Parameters: $\alpha \in {0.10, 0.15, 0.20, 0.30}$; `reps` = 1–5 |
| **4** | Implemented the **QAOA** handler in step 1 of the pipeline. | [`src/step_1.py`](./src/step_1.py): Constructs the `QAOAAnsatz` and appends the measurement layer. |
| **5** | Created a run logger to track experiment results. | Each run appends a summary row to `qaoa_stats.csv`. |
| **6** | Developed an analysis notebook for visualizing results. | [`qaoa_portfolio_optimazation_analysis.ipynb`](./_experiments/qaoa_portfolio_optimization_analysis.ipynb): Loads the CSV (or .pkl if you ran the experiments) to render all figures. |

Because the existing `model_to_obj` helper already embeds constraint penalties into the objective function, we could **treat the DOCplex model as a black-box cost oracle** and pass it directly to the local search algorithm.

---

### **4. Headline Results (Median of 100 Seeds)**

| Metric | Pure CPLEX | QAOA (p=3) **before** Local Search | QAOA (p=3) **after** Local Search |
|---|---|---|---|
| Median Objective Gap | 0 % | **5.6180 %** | **0.0186 %** |
| 2-Qubit Depth | – | 49 | 49 |

The quantum stage alone approaches the optimal solution, and the classical bit-flip local search consistently closes the residual gap while requiring only \~20 additional objective evaluations. In effect, we **match the classical optimal value with a circuit depth that is feasible for near-term hardware**.

---

### **5. How to Reproduce the Results**

```bash
# Clone the repository
git clone https://github.com/zeapsu/WISER_Optimization_VG 
cd WISER_Optimization_VG

# Install dependencies in a new virtual environment (ensure poetry is installed)
poetry install

# Launch the notebook to view results and plots (ensure jupyter is installed)
jupyter lab qaoa_portfolio_optimization_analysis.ipynb
```

### **6. Project Presentation Deck**
[Project Presentation Deck](./Project_Presentation_Deck.pptx)