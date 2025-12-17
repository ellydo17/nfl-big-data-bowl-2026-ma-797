# NFL Big Data Bowl 2026: Multimodal Trajectory Prediction

##### Elly Do, and Martin Yu

## Project Overview
This repository contains the source code for a submission to NFL Big Data Bowl 2026 competition. 
The objective of this project is to predict the future trajectories of NFL players using multimodal tracking data.

Our work includes two distinct models:
1.  **Two-Stream CNN (Deep Learning):** A custom architecture fusing spatial formation data (vision branch) with temporal scalars (time branch).
2.  **Physics-Based LightGBM (Gradient Boosting):** A robust, kinematics-driven model that treats trajectory prediction as a regression problem governed by kinematic constraints.

## Model Architectures

### 1. Two-Stream CNN (Experimental)
Designed to read the field like a quarterback, this model processes spatial formations and time horizons separately before fusion.
* **Vision Branch:** Inputs a $4 \times 54 \times 120$ tensor (teammates, opponents, ball, target) through 3 blocks of `Conv2D` + `MaxPool`.
* **Time Branch:** Projects the scalar prediction horizon ($\Delta t$) into a 64-dimensional embedding space using `Linear` + `ReLU`.
* **Fusion:** Concatenates visual features (5,760) with time features (64) into a Dense Regression Head outputting $(x, y)$ coordinates.

### 2. LightGBM (Production Solution)
A Gradient Boosting Decision Tree (GBDT) approach that treats trajectory prediction as a regression problem.
* **Input Features:** velocity ($s, dir$), acceleration ($a$), orientation ($o$), and time delta ($\Delta t$).
* **Strategy:** We employ an iterative querying, predicting the displacement for sequential time steps ($t=0.1, 0.2, \dots, 3.0$) to reconstruct the full trajectory curve.

## References
**Data Source:**
* Lopez, M., Bliss, T., Blake, A., Yan, Y., Plomecka, M., & Howard, A. (2025). *NFL Big Data Bowl 2026 - Prediction*. Kaggle. Available at: [https://kaggle.com/competitions/nfl-big-data-bowl-2026-prediction](https://kaggle.com/competitions/nfl-big-data-bowl-2026-prediction)

**Methodology & Literature:**
* **[LightGBM]** Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems (NeurIPS)*, 3149-3157.
* **[CNN]** LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based Learning Applied to Document Recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
* **[Blog]** DataMapu. "Gradient Boosting Variants." *DataMapu Blog*. Accessed December 2025. Available at: [https://datamapu.com/posts/classical_ml/gradient_boosting_variants](https://datamapu.com/posts/classical_ml/gradient_boosting_variants)
