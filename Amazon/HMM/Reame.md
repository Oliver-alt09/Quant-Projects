##### **Probabilistic Volatility Forecasting with Hidden Markov Models :-**



A quantitative finance project utilizing a 2-State Gaussian Hidden Markov Model (HMM) to infer latent (hidden) market volatility regimes and map state transition probabilities for a statistical arbitrage spread (SHEL/BP).



**Objective:**

Traditional risk models (like simple moving averages) provide static point estimates of market volatility, failing to capture sudden structural panics. This project shifts from deterministic to probabilistic forecasting by modeling the market as a Markov Chain. It dynamically decodes whether the market is currently in a calm or panic state and calculates the exact probability of transitioning between the two.



**Market State Analysis:**

The model successfully converged on real-world historical data, identifying two distinct market regimes. 



**Learned Transition Matrix:**

\[\[0.9777516   0.0222484 ]

&#x20;\[0.03851241  0.96148759]]

