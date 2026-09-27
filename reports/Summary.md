# CMAPSS Turbofan Engine Degradation EDA Summary

## 1. General Data Quality
- **Dataset Size**: The training dataset (`train_FD001.txt`) contains 20,631 rows and 26 columns.
- **Engines**: There are exactly 100 independent turbofan engines in this subset.
- **Data Integrity**: There are **0 missing (NaN) values** across the entire dataset.

## 2. Engine Lifespan Statistics
Engines in the dataset run until failure. The total number of flight cycles (lifespan) varies significantly between engines:
- **Shortest-lived engine**: 128 cycles (Engine 39)
- **Longest-lived engine**: 362 cycles (Engine 69)
- **Average lifespan**: 206.3 cycles

## 3. Sensor Variance & Feature Elimination
The dataset tracks 3 operational settings and 21 sensors. However, not all of them provide useful information for predictive modeling.

By analyzing the standard deviation (variance) of each column, we identified several variables that remain completely constant throughout the engines' lifetimes:
- **Constant Features (Std < 0.01)**: `setting_1`, `setting_2`, `setting_3`, `s_1`, `s_5`, `s_10`, `s_16`, `s_18`, `s_19`
- Because these features exhibit no meaningful variance, they provide zero predictive power and act merely as noise or static offsets. They can be safely **dropped** from the feature set.
- This leaves **15 active sensors** leaving exactly **0 active settings**.

## 4. Sensor Degradation Trends
Plotting the raw values of the remaining 15 active sensors reveals that the data contains significant high-frequency operational noise.
- Applying a **10-cycle moving average** smooths out this noise and reveals clear, monotonic degradation trends in many sensors as the engines approach failure (cycle end).
- These clear degradation trajectories confirm that these remaining features are highly relevant for predicting Remaining Useful Life (RUL).

*See `3_all_21_sensors_trends.png` for visual proof of active vs. constant sensors.*

## 5. Sensor Correlations
A correlation matrix of the 15 active sensors reveals strong linear relationships between certain sensor groups.
- Certain pairs (e.g., `s_9` and `s_14` with $r = 0.9632$) are highly correlated. 
- High multicollinearity suggests that dimensionality reduction techniques (like PCA) or regularized models could be effective in simplifying the feature space without losing degradation signals.

*See `4_correlation_15_active_sensors.png` for the correlation heatmap.*
