### **Mean Reversion Strategy** :— **Pairs Trading via OU Process**



A statistical arbitrage framework using cointegration and Ornstein-Uhlenbeck dynamics. This mean reversion strategy is implemented on SHEL and BP, two highly correlated global oil majors.



Core Logic:

* Rolling OLS hedge ratio (dynamic beta)
* Spread construction:  spread=Y−βX
* OU process estimation: Mean (μ), Speed (λ), Half-life



Signal Engine:

* Long spread when Z-score < -2
* Short spread when Z-score > +2
* Exit near mean (|Z| < 0.5)
* Stop-loss at extreme deviations



Execution Model:

* Transaction costs (bps-based friction)
* Turnover-aware backtesting

