# Predictive Maintenance: NASA Lithium-Ion Battery RUL Prediction
**Component 3:** Remaining Useful Life (RUL) Prognostics  
**Author:** Saketh  
**Course:** DS3010 - Introduction to Machine Learning Project  

---

## 1. Executive Summary & Objective

In mission-critical industrial systems (electric vehicles, aerospace avionics, and renewable grid storage), battery failure can lead to catastrophic operational halts or severe safety hazards. This component models the **Remaining Useful Life (RUL)** of lithium-ion cells using the NASA Prognostics Center of Excellence Battery Dataset.

The primary objective is to reliably estimate the number of discharge cycles remaining before a cell experiences functional End-of-Life (EOL), while strictly penalizing **late predictions** ($\hat{y} > y$). In industrial prognostics, an over-optimistic prediction (falsely claiming the battery has 15 cycles left when it is already at EOL) leads to unplanned failures, whereas early predictions simply trigger proactive inspection.

---

## 2. Dataset & Preprocessing Pipeline

### 2.1 Benchmark Experimental Conditions
The models were trained and tested on the NASA benchmark 18650 Li-ion cells (**B0005, B0006, B0007, and B0018**). All four cells underwent identical testing conditions:
- **Operating Temperature:** Constant ambient temperature ($24^\circ\text{C}$).
- **Charging Profile:** Constant Current Constant Voltage (CCCV) charging at $1.5\,\text{A}$ up to $4.2\,\text{V}$, followed by voltage tapering until current dropped below $20\,\text{mA}$.
- **Discharging Profile:** Constant Current (CC) discharge at $2.0\,\text{A}$ down to individual cell cutoff voltages ($2.7\,\text{V}$, $2.5\,\text{V}$, $2.2\,\text{V}$, and $2.5\,\text{V}$).

### 2.2 End-of-Life (EOL) & Piecewise RUL Formulation
- **Rated Capacity:** $2.0\,\text{Ahr}$.
- **EOL Threshold:** A $30\%$ fade in rated capacity, defined as **$1.4\,\text{Ahr}$**.
- **Piecewise Target RUL ($y$):**
  For each individual cell, the exact cycle index $C_{\text{EOL}}$ where measured capacity first dropped below $1.4\,\text{Ahr}$ was identified ($C_{\text{EOL}} = 125$ for Cell 5, $109$ for Cell 6, and $97$ for Cell 18).
  The ground-truth RUL at cycle $t$ is formulated as:
  $$RUL(t) = \min\left(RUL_{\max},\, \max\left(0,\, C_{\text{EOL}} - t\right)\right)$$
  where $RUL_{\max} = 125$ cycles (standard piecewise linear degradation ceiling used in NASA prognostics challenges to prevent uninformative constant early-life noise).

### 2.3 Cycle-Level Feature Extraction
Raw MATLAB records consist of high-frequency time-series waveforms for Voltage, Current, and Temperature during each discharge cycle. These were aggregated into 10 cycle-level operational features:
- **Voltage:** Mean Voltage ($\bar{V}$), Min Voltage ($V_{\min}$), Max Voltage ($V_{\max}$)
- **Current:** Mean Current ($\bar{I}$), Min Current ($I_{\min}$), Max Current ($I_{\max}$)
- **Temperature:** Mean Temp ($\bar{T}$), Min Temp ($T_{\min}$), Max Temp ($T_{\max}$)
- **Discharge Capacity:** Measured cycle capacity ($Q_{\text{discharge}}$)

### 2.4 Strict Leakage-Free Train/Test Partition
- **Training Set:** Cells 5, 6, and 7 ($504$ total cycle observations across 3 complete runs).
- **Held-Out Test Set:** Cell 18 ($132$ cycle observations evaluated sequentially from fresh state to failure).
- No test-cell information or future-cycle signals leaked into preprocessing, feature scaling, or model fitting.

---

## 3. The Industrial Failure Metric: NASA PHM08 Score

Standard regression metrics (RMSE and MAE) penalize positive and negative errors symmetrically:
$$\text{MSE} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2$$
However, in predictive maintenance, the asymmetric cost of errors is critical:
- **Early Prediction ($d = \hat{y} - y \le 0$):** Maintenance crew replaces or inspects the battery slightly earlier than necessary. Minor economic cost.
- **Late Prediction ($d = \hat{y} - y > 0$):** Battery fails unexpectedly in service. Massive safety hazard and unscheduled downtime.

To model this, we evaluate using the **NASA PHM08 Asymmetric Scoring Function**:
$$S_{\text{PHM}} = \sum_{i=1}^N s_i \quad \text{where} \quad s_i = \begin{cases} \exp\left(\frac{d_i}{13}\right) - 1 & \text{for } d_i \ge 0 \text{ (Late / Over-prediction)} \\ \exp\left(-\frac{d_i}{10}\right) - 1 & \text{for } d_i < 0 \text{ (Early / Under-prediction)} \end{cases}$$

---

## 4. Modeling Methodologies

### 4.1 Classical Baselines
- **Random Forest Regressor:** Ensemble of 100 decorrelated decision trees (bagging).
- **Gradient Boosting Regressor:** Sequential boosting of 100 regression stumps minimizing standard MSE loss.

### 4.2 Asymmetric XGBoost (Custom PHM Objective)
To directly suppress late-stage optimism during tree node splitting and leaf weight calculation, we implemented a custom objective function in **XGBoost**.

Given error $d = \hat{y} - y$, the first-order gradient ($g$) and second-order hessian ($h$) are supplied to XGBoost as:
$$g_i = \frac{\partial L}{\partial \hat{y}_i} = \begin{cases} \frac{1}{a_{\text{late}}} \exp\left(\frac{d_i}{a_{\text{late}}}\right) & \text{if } d_i > 0 \\[6pt] -\frac{1}{a_{\text{early}}} \exp\left(-\frac{d_i}{a_{\text{early}}}\right) & \text{if } d_i \le 0 \end{cases}$$

$$h_i = \frac{\partial^2 L}{\partial \hat{y}_i^2} = \begin{cases} \frac{1}{a_{\text{late}}^2} \exp\left(\frac{d_i}{a_{\text{late}}}\right) & \text{if } d_i > 0 \\[6pt] \frac{1}{a_{\text{early}}^2} \exp\left(-\frac{d_i}{a_{\text{early}}}\right) & \text{if } d_i \le 0 \end{cases}$$

with $a_{\text{late}} = 10.0$ and $a_{\text{early}} = 13.0$, bounded by numerical clipping ($h_i \in [10^{-4}, 100]$) to guarantee numerical stability during gradient boosting.

### 4.3 Deep Learning: Sequence-to-Value PyTorch LSTM
- **Sliding Window:** Formatted historical cycle observations into temporal windows of size $W = 10$.
  $$\mathbf{X}_{t} = [\mathbf{x}_{t-9},\, \mathbf{x}_{t-8},\, \dots,\, \mathbf{x}_t] \in \mathbb{R}^{10 \times 10}$$
- **Architecture:**
  - Layer 1 & 2: 2-layer stacked PyTorch `nn.LSTM` with hidden dimension $H = 48$ and dropout $0.1$.
  - Dense Head: Fully connected layers (`48 -> 24 -> 1`) with ReLU non-linearity.
  - Loss & Optimization: Smooth L1 Loss with Adam optimizer ($\text{lr} = 0.003$, weight decay $= 10^{-4}$).

---

## 5. Experimental Results & Performance Comparison

All models were evaluated on the held-out **Battery 18**:

| Model | Paradigm | Optimization Loss | RMSE (Cycles) | MAE (Cycles) | NASA PHM Score |
|:---|:---|:---|:---:|:---:|:---:|
| **Random Forest (Baseline)** | Ensemble Trees (Bagging) | Standard MSE | 8.92 | 7.37 | 144.77 |
| **Gradient Boosting (Baseline)** | Ensemble Trees (Boosting) | Standard MSE | 8.85 | 7.52 | 130.99 |
| **Asymmetric XGBoost** | **Ensemble Trees (Boosting)** | **Custom Asymmetric PHM** | **8.32** | **6.06** | **106.81** |
| **PyTorch LSTM** | Recurrent Neural Network | Smooth L1 (Window $W=10$) | 27.13 | 21.91 | 1230.67 |

---

## 6. Key Scientific & Engineering Insights

### 1. Asymmetric Loss Eliminates Late-Stage Optimism
- Standard MSE models exhibited systematic positive bias near end-of-life: at cycle 97 (when Battery 18 reached EOL), baseline Gradient Boosting was still predicting $10-15$ cycles remaining.
- Asymmetric XGBoost penalizes positive residual gradients exponentially, forcing the predicted trajectory to drop swiftly to $0$ as capacity crosses $1.4\,\text{Ahr}$.
- This yielded an **18.5% reduction in PHM penalty score** ($130.99 \rightarrow 106.81$) and reduced Mean Absolute Error to just $6.06$ cycles.

### 2. Deep Learning (LSTM) vs. Tree Ensembles in Low-Data Regimes
- **Why did the LSTM underperform?** The training partition across Cells 5, 6, and 7 contains only $504$ sequential samples. Deep recurrent neural networks require thousands of multi-engine run-to-failure cycles to learn robust hidden temporal representations.
- Furthermore, Cell 18 degraded faster ($C_{\text{EOL}} = 97$) than Cell 5 ($125$) or Cell 7 ($168$). The LSTM overfit the longer degradation lifetimes of the training cells, causing an upward prediction drift (RMSE $27.13$).
- In contrast, tree ensembles with feature thresholds on instantaneous discharge metrics ($\bar{V}, \bar{T}, Q$) generalize with high precision even on small sample sizes.

---

## 7. Slide-by-Slide Presentation Structure (Ready for Group Deck)

- **Slide 1: Problem Definition & Battery EOL Threshold**
  - Mission: Predict Remaining Useful Life (RUL) of Li-ion battery before capacity drops below $1.4\,\text{Ahr}$ ($30\%$ fade).
  - Include graph: `reports/battery/figures/capacity_fade.png`.
- **Slide 2: The Industrial Danger of Late Predictions**
  - Why standard MSE fails: Symmetrically penalizes late vs early errors.
  - Introduce the NASA PHM08 asymmetric exponential loss penalty.
- **Slide 3: Methodologies: Asymmetric XGBoost & PyTorch LSTM**
  - Custom XGBoost with first and second derivative penalty for late predictions ($y_{\text{pred}} > y_{\text{true}}$).
  - PyTorch LSTM with sliding window ($W=10$) to capture sequential temporal dynamics.
- **Slide 4: Model Benchmark & Comparison Table**
  - Present the comparison table highlighting Asymmetric XGBoost as the winning model (RMSE $8.32$, MAE $6.06$, PHM $106.81$).
  - Include graph: `reports/battery/figures/rul_predictions_comparison.png`.
- **Slide 5: Late-Stage Correction & Engineering Takeaway**
  - Show zoomed-in graph: `reports/battery/figures/late_stage_overprediction_fix.png`.
  - Conclude: Custom asymmetric objectives turn classical ML models into safe, fail-fast industrial prognostics systems.
