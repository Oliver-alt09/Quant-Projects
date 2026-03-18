### **Crisis Alpha Strategy** :— **Volatility \& Tail Risk Model**



A volatility-based signal engine designed to detect and exploit market stress regimes.



Core Features:

* Realized volatility vs implied volatility (VIX)
* Volatility Risk Premium (VRP): VRP=IV−RV
* Vol-of-vol analysis using VVIX
* Z-score normalization of volatility regimes



Signals:

* Crisis Alpha Trigger: VVIX Z-score > 2
* Cheap Volatility Buy: Negative VRP + low VVIX



Additional Module:

* Black-Scholes-based tail hedge sizing
* Portfolio insurance via deep OTM puts
* Delta-based contract sizing

