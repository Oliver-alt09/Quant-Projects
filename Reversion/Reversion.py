{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "56a7f23b-e789-4b37-bdea-da83c803c3b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import yfinance as yf\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import statsmodels.api as sm\n",
    "\n",
    "def pair_data(y, x, start_date=\"2018-01-01\"):\n",
    "    print(f\"Fetching data for {y} and {x}\")\n",
    "    data = yf.download([y, x], start=start_date)['Close']\n",
    "    return data.ffill().dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "f4ff8aeb-7da7-4434-9d1b-deee9909e353",
   "metadata": {},
   "outputs": [],
   "source": [
    "def rolling_hedge(Y, X, window=60):\n",
    "\n",
    "    beta = []\n",
    "\n",
    "    for i in range(len(Y)):\n",
    "        if i < window:\n",
    "            beta.append(np.nan)\n",
    "        else:\n",
    "            y = Y.iloc[i-window:i]\n",
    "            x = X.iloc[i-window:i]\n",
    "            model = sm.OLS(y, sm.add_constant(x)).fit()\n",
    "            beta.append(model.params.iloc[1])\n",
    "\n",
    "    return pd.Series(beta, index=Y.index).ffill()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "f4c6e1a0-849f-4d1b-80b4-33d591aac2ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "def spread(y, x, beta_t):\n",
    "    return y - beta_t * x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "48fdc89f-4d7f-4553-9a3a-2bdb3ee63ee8",
   "metadata": {},
   "outputs": [],
   "source": [
    "def ou(spread):\n",
    "\n",
    "    lag = spread.shift(1).dropna()\n",
    "    diff = spread.diff().dropna()\n",
    "\n",
    "    lag = lag.loc[diff.index]\n",
    "\n",
    "    X = sm.add_constant(lag)\n",
    "    model = sm.OLS(diff, X).fit()\n",
    "\n",
    "    lam = model.params.iloc[1]\n",
    "    mu = -model.params.iloc[0] / lam\n",
    "    sigma = np.std(model.resid)\n",
    "\n",
    "    if lam < 0:\n",
    "        half_life = -np.log(2) / lam\n",
    "    else:\n",
    "        half_life = np.inf\n",
    "\n",
    "    print(f\"OU λ: {lam:.5f}\")\n",
    "    print(f\"OU μ: {mu:.4f}\")\n",
    "    print(f\"Half-Life: {half_life:.1f} days\")\n",
    "\n",
    "    return lam, mu, sigma, half_life"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "3a796a98-12ed-4451-ba8c-cdcc285f3c1f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def ou_zscore(spread, mu, sigma, lam):\n",
    "\n",
    "    sigma_eq = sigma / np.sqrt(-2 * lam)\n",
    "    z = (spread - mu) / sigma_eq\n",
    "\n",
    "    return z.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "69af8e1e-0694-4f88-b84c-d164fb8b7d6c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def score_signals(z):\n",
    "\n",
    "    signals_df = pd.DataFrame(index=z.index)\n",
    "    signals_df['Z_Score'] = z\n",
    "\n",
    "    signals_df['Long_Spread'] = z < -2.0\n",
    "    signals_df['Short_Spread'] = z > 2.0\n",
    "\n",
    "    signals_df['Exit'] = z.abs() < 0.5\n",
    "    signals_df['Stop_Loss'] = z.abs() > 5.0\n",
    "\n",
    "    print(f\"Signals Generated: {signals_df['Long_Spread'].sum()} Long, {signals_df['Short_Spread'].sum()} Short\")\n",
    "\n",
    "    return signals_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "e10fef86-f9ad-417b-918f-0ab190fe7a70",
   "metadata": {},
   "outputs": [],
   "source": [
    "def turnover(signals_df):\n",
    "\n",
    "    target = pd.Series(np.nan, index=signals_df.index)\n",
    "\n",
    "    target.loc[signals_df['Long_Spread']] = 1.0\n",
    "    target.loc[signals_df['Short_Spread']] = -1.0\n",
    "\n",
    "    target.loc[signals_df['Exit']] = 0.0\n",
    "    target.loc[signals_df['Stop_Loss']] = 0.0\n",
    "\n",
    "    target = target.ffill().fillna(0.0)\n",
    "\n",
    "    actual = target.shift(1).fillna(0.0)\n",
    "\n",
    "    trades = actual.diff().abs().sum()\n",
    "\n",
    "    print(f\"Trades Executed: {int(trades)}\")\n",
    "\n",
    "    return actual"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "943aa736-ddbf-4c26-9fbe-e43a74c17339",
   "metadata": {},
   "outputs": [],
   "source": [
    "def cost_backtest(prices_df, target, beta_t, y, x, cost_bps=10):\n",
    "\n",
    "    print(f\"Friction Model: {cost_bps} bps\")\n",
    "\n",
    "    ret_y = prices_df[y].pct_change().fillna(0)\n",
    "    ret_x = prices_df[x].pct_change().fillna(0)\n",
    "\n",
    "    notional = 1 + abs(beta_t)\n",
    "\n",
    "    spread_return = (ret_y - beta_t * ret_x) / notional\n",
    "\n",
    "    gross_returns = target * spread_return\n",
    "\n",
    "    turnover = target.diff().abs().fillna(0)\n",
    "    cost = turnover * (cost_bps / 10000)\n",
    "\n",
    "    net_returns = gross_returns - cost\n",
    "    equity_curve = (1 + net_returns).cumprod()\n",
    "\n",
    "    total_return = equity_curve.iloc[-1] - 1\n",
    "\n",
    "    print(f\"Net Return: {total_return:.2%}\")\n",
    "\n",
    "    return equity_curve, net_returns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "196d1c11-16aa-48cd-93f3-940ac1123f8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def performance(returns):\n",
    "\n",
    "    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)\n",
    "\n",
    "    equity = (1 + returns).cumprod()\n",
    "    drawdown = (equity / equity.cummax() - 1).min()\n",
    "\n",
    "    print(f\"Sharpe: {sharpe:.2f}\")\n",
    "    print(f\"Max Drawdown: {drawdown:.2%}\")\n",
    "\n",
    "    return sharpe, drawdown"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "1d12ef7e-f538-4e9a-87b2-c6779214e45f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fetching data for SHEL and BP\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[*********************100%***********************]  2 of 2 completed\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "OU λ: -0.00346\n",
      "OU μ: 1.6149\n",
      "Half-Life: 200.4 days\n",
      "Signals Generated: 67 Long, 33 Short\n",
      "Trades Executed: 12\n",
      "Friction Model: 10 bps\n",
      "Net Return: 33.67%\n",
      "Sharpe: 0.77\n",
      "Max Drawdown: -6.26%\n"
     ]
    }
   ],
   "source": [
    "if __name__ == \"__main__\":\n",
    "\n",
    "    Y_TICKER = \"SHEL\"\n",
    "    X_TICKER = \"BP\"\n",
    "\n",
    "    prices = pair_data(Y_TICKER, X_TICKER)\n",
    "\n",
    "    Y = np.log(prices[Y_TICKER])\n",
    "    X = np.log(prices[X_TICKER])\n",
    "\n",
    "    beta_t = rolling_hedge(Y, X, window=60)\n",
    "\n",
    "    spread_val = spread(Y, X, beta_t)\n",
    "\n",
    "    lam, mu, sigma, half_life = ou(spread_val)\n",
    "\n",
    "    z = ou_zscore(spread_val, mu, sigma, lam)\n",
    "\n",
    "    signals = score_signals(z)\n",
    "\n",
    "    target = turnover(signals)\n",
    "\n",
    "    prices = prices.loc[target.index]\n",
    "    beta_t = beta_t.loc[target.index]\n",
    "\n",
    "    equity, returns = cost_backtest(\n",
    "        prices, target, beta_t, Y_TICKER, X_TICKER, cost_bps=10\n",
    "    )\n",
    "\n",
    "    performance(returns)"
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
