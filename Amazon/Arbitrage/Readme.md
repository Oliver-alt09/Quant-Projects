##### **Deep Learning for Alpha Signal Construction :-**



An end-to-end quantitative machine learning pipeline built with PyTorch. This project engineers a recurrent neural network (LSTM) to predict the directional movement of a statistical arbitrage spread (SHEL/BP) using macroeconomic indicators and market volatility regimes.



**Objective:**

Traditional algorithmic pairs trading relies on static mathematical thresholds (ex- buying a spread when the Z-score hits -2.0). This project explores whether a Deep Learning architecture can extract non-linear, predictive alpha signals by processing a chronological sequence of broader macroeconomic context.



**Market Regime Analysis:**

The model was trained on historical daily data spanning from 2020 to 2026, utilizing an 80/20 strict chronological split to mimic a real-world deployment environment.



**Final Optimization Logs:**

* Train Loss: 0.6831
* Val Loss: 0.7104
* Val Accuracy: 45.9%



**Key Takeaway:** 

While the network successfully converged on the training data (Train Loss smoothly decreasing to 0.6831), the out-of-sample validation accuracy inverted to below 50%. In quantitative finance, a consistent sub-50% accuracy on cleanly split data indicates a Market Regime Shift.



The training set (2020–2024) encompassed zero-interest-rate policies, pandemic volatility, and an aggressive rate-hiking cycle. The validation set encompassed the stabilized, higher-rate environment of 2025. The mathematical rules the LSTM learned during the volatile regime actively inverted during the stable regime. 

