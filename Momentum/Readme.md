###### **Momentum Strategy** :- **Global Tactical Asset Allocation**



A multi-asset momentum model allocating across equities, bonds, gold, and cash.



Core Logic:

* Multi-horizon momentum (1, 3, 6, 12 months)
* Cross-sectional ranking of assets
* Long top-performing assets, fallback to cash



Enhancements:

* Regime Filter: 200-day moving average (risk-off switch)
* Volatility Targeting: Dynamic position sizing
* Correlation Filter: Reduces exposure during systemic risk spikes
* Monte Carlo Risk Engine: VaR / CVaR estimation
* Risk of Ruin Analysis: Stress-testing sequence risk

