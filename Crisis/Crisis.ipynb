{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "affb53aa-4633-493b-923e-01acd432d7e5",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[*********************100%***********************]  3 of 3 completed\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Recent Tail-Risk Signals:\n",
      "            IV_VIX        VRP  VVIX_Z_Score  Crisis_Alpha\n",
      "Date                                                     \n",
      "2025-11-20   26.42  12.286224      2.444105          True\n",
      "2026-01-14   16.75   8.118036      2.259169          True\n",
      "2026-01-20   20.09   9.709556      2.820662          True\n",
      "2026-02-05   21.77  10.408763      2.121639          True\n",
      "2026-03-06   29.49  16.183035      3.375850          True\n"
     ]
    }
   ],
   "source": [
    "import yfinance as yf\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "def volatility_surface(start_date=\"2015-01-01\"):\n",
    "\n",
    "    tickers = [\"SPY\", \"^VIX\", \"^VVIX\"]\n",
    "\n",
    "    raw_data = yf.download(tickers, start=start_date)['Close']\n",
    "    data = raw_data.ffill().dropna()\n",
    "\n",
    "    return data\n",
    "\n",
    "def volatility_signals(vol_df, rv=20):\n",
    "\n",
    "    signals = pd.DataFrame(index=vol_df.index)\n",
    "\n",
    "    spy_returns = vol_df['SPY'].pct_change()\n",
    "\n",
    "    realized_vol = spy_returns.rolling(window=rv).std() * np.sqrt(252) * 100\n",
    "    signals['Realized_Vol'] = realized_vol\n",
    "    signals['IV_VIX'] = vol_df['^VIX']\n",
    "\n",
    "    signals['VRP'] = signals['IV_VIX'] - signals['Realized_Vol']\n",
    "\n",
    "    vvix = vol_df['^VVIX']\n",
    "\n",
    "    vvix_mean = vvix.rolling(window=rv).mean()\n",
    "    vvix_std = vvix.rolling(window=rv).std()\n",
    "    signals['VVIX_Z_Score'] = (vvix - vvix_mean) / vvix_std\n",
    "\n",
    "    signals['Buy_Cheap'] = (signals['VRP'] < 0) & (signals['VVIX_Z_Score'] < 0)\n",
    "\n",
    "    signals['Crisis_Alpha'] = signals['VVIX_Z_Score'] > 2.0\n",
    "\n",
    "    return signals.dropna()\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    vol_data = volatility_surface()\n",
    "    vol_signals = volatility_signals(vol_data)\n",
    "    \n",
    "    print(\"\\nRecent Tail-Risk Signals:\")\n",
    "    crisis_days = vol_signals[vol_signals['Crisis_Alpha'] == True]\n",
    "    print(crisis_days[['IV_VIX', 'VRP', 'VVIX_Z_Score', 'Crisis_Alpha']].tail())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "216fea0a-6f70-4580-b314-6324c5a92f51",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Current Underlying Price (S): $500.00\n",
      "Target Strike Price (K):      $400.00\n",
      "Theoretical Put Price:        $0.14\n",
      "Option Delta (Δ):            -0.0081\n",
      "Portfolio Value:            $500,000.00\n",
      "Contracts to Purchase:      311\n",
      "Total Premium Cost:         $4,279.74\n",
      "Insurance Cost Drag:        0.86%\n"
     ]
    }
   ],
   "source": [
    "from scipy.stats import norm\n",
    "\n",
    "def bs_put(S, K, T, r, sigma):\n",
    "\n",
    "    if T <= 0:\n",
    "        return max(0.0, K - S), -1.0 if S < K else 0.0\n",
    "\n",
    "    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))\n",
    "    d2 = d1 - sigma * np.sqrt(T)\n",
    "\n",
    "    put_price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))\n",
    "    put_delta = norm.cdf(d1) - 1.0\n",
    "\n",
    "    return put_price, put_delta\n",
    "\n",
    "def risk_hedge(portfolio_value, S, K, T, r, sigma):\n",
    "    \n",
    "    put_price, put_delta = bs_put(S, K, T, r, sigma)\n",
    "\n",
    "    print(f\"Current Underlying Price (S): ${S:.2f}\")\n",
    "    print(f\"Target Strike Price (K):      ${K:.2f}\")\n",
    "    print(f\"Theoretical Put Price:        ${put_price:.2f}\")\n",
    "    print(f\"Option Delta (\\u0394):            {put_delta:.4f}\")\n",
    "\n",
    "    portfolio_equivalent_shares = portfolio_value / S\n",
    "\n",
    "    if put_delta == 0:\n",
    "        print(\"Delta is 0. Cannot hedge with this option.\")\n",
    "        return 0, 0\n",
    "\n",
    "    hedge_ratio = 0.25\n",
    "    contracts_needed = (portfolio_value * hedge_ratio) / (S * abs(put_delta) * 100)\n",
    "    contracts_buy = np.ceil(contracts_needed)\n",
    "\n",
    "    premium_cost = contracts_buy * put_price * 100\n",
    "    percentage = premium_cost / portfolio_value\n",
    "\n",
    "    print(f\"Portfolio Value:            ${portfolio_value:,.2f}\")\n",
    "    print(f\"Contracts to Purchase:      {int(contracts_buy)}\")\n",
    "    print(f\"Total Premium Cost:         ${premium_cost:,.2f}\")\n",
    "    print(f\"Insurance Cost Drag:        {percentage:.2%}\")\n",
    "\n",
    "    return contracts_buy, premium_cost\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "\n",
    "    portfolio_size = 500000\n",
    "    spy_price = 500.00\n",
    "    k = 400.00 \n",
    "    expiry = 0.25  \n",
    "    r= 0.05 \n",
    "    vix = 0.20      \n",
    "\n",
    "    contracts, cost = risk_hedge(\n",
    "        portfolio_size,\n",
    "        spy_price,\n",
    "        k,\n",
    "        expiry,\n",
    "        r,\n",
    "        vix\n",
    "    )"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
