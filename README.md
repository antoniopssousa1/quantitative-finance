# Quantitative Finance

A Python-based repository covering the core theoretical and practical topics in quantitative finance — from stock market fundamentals to stochastic models and risk management.

---

## Table of Contents

- [Section 2 - Stock Market Basics](#section-2---stock-market-basics)
- [Section 3 - Bond Theory and Implementation](#section-3---bond-theory-and-implementation)
- [Section 4 - Modern Portfolio Theory (Markowitz Model)](#section-4---modern-portfolio-theory-markowitz-model)
- [Section 5 - Capital Asset Pricing Model (CAPM)](#section-5---capital-asset-pricing-model-capm)
- [Section 6 - Derivatives Basics](#section-6---derivatives-basics)
- [Section 7 - Random Behavior in Finance](#section-7---random-behavior-in-finance)
- [Section 8 - Black-Scholes Model](#section-8---black-scholes-model)
- [Section 9 - Value-at-Risk (VaR)](#section-9---value-at-risk-var)
- [Section 10 - Collateralized Debt Obligation (CDO)](#section-10---collateralized-debt-obligation-cdo)
- [Section 11 - Interest Rate Models](#section-11---interest-rate-models)
- [Section 12 - Value Investing](#section-12---value-investing)

---

## Section 2 - Stock Market Basics

### Present Value and Future Value of Money
The time value of money is the foundation of all finance. A sum of money today is worth more than the same sum in the future due to its earning potential. **Future Value (FV)** and **Present Value (PV)** can be computed under discrete or continuous compounding:

$$FV = PV \cdot (1 + r)^n \quad \text{(discrete)}$$
$$FV = PV \cdot e^{rn} \quad \text{(continuous)}$$

### Stocks and Shares
A **stock** represents ownership (equity) in a company. When you buy a **share**, you become a partial owner and may receive **dividends** and benefit from **capital appreciation**. Stock prices are driven by supply and demand, earnings expectations, and macroeconomic factors.

### Commodities and the FOREX
**Commodities** are raw materials or primary agricultural products (e.g. gold, oil, wheat) traded on exchanges. The **Foreign Exchange (FOREX)** market is the largest financial market in the world, where currencies are traded in pairs (e.g. EUR/USD). Both are influenced by geopolitical events, interest rates, and global supply/demand dynamics.

### What Are Short and Long Positions?
- **Long position**: Buying an asset expecting its price to rise — you profit if the price goes up.
- **Short position**: Borrowing and selling an asset expecting its price to fall — you profit if the price goes down and you can buy it back cheaper.

---

## Section 3 - Bond Theory and Implementation

### What Are Bonds?
A **bond** is a fixed-income debt instrument where an investor loans money to a borrower (government or corporation) for a defined period at a fixed or variable interest rate (**coupon**). At maturity, the **face value (par value)** is returned.

### Yields and Yield to Maturity
The **yield** of a bond reflects its return. The **Yield to Maturity (YTM)** is the total return anticipated if the bond is held until it matures, accounting for coupon payments and the difference between purchase price and par value. Bond price and yield move **inversely**.

### Macaulay Duration
**Macaulay Duration** measures the weighted average time (in years) until a bond's cash flows are received. It is a key measure of a bond's **interest rate sensitivity**:

$$D = \frac{\sum_{t=1}^{T} t \cdot \frac{C_t}{(1+y)^t}}{P}$$

where $C_t$ is the cash flow at time $t$, $y$ is the yield, and $P$ is the bond price.

### Bond Pricing Theory and Implementation
A bond's price is the **present value of all future cash flows** (coupons + face value):

$$P = \sum_{t=1}^{T} \frac{C}{(1+y)^t} + \frac{F}{(1+y)^T}$$

---

## Section 4 - Modern Portfolio Theory (Markowitz Model)

### What Is Diversification in Finance?
**Diversification** is the practice of spreading investments across different assets to reduce risk. By combining assets that are not perfectly correlated, the overall portfolio risk can be lower than the weighted average risk of individual assets.

### Mean and Variance
Portfolio performance is characterised by:
- **Expected return (mean)**: $\mu_p = \sum w_i \mu_i$
- **Variance (risk)**: $\sigma_p^2 = \sum_i \sum_j w_i w_j \sigma_{ij}$

where $w_i$ are portfolio weights and $\sigma_{ij}$ is the covariance between assets $i$ and $j$.

### Efficient Frontier and the Sharpe Ratio
The **Efficient Frontier** is the set of optimal portfolios offering the highest expected return for a given level of risk. The **Sharpe Ratio** measures risk-adjusted return:

$$S = \frac{\mu_p - r_f}{\sigma_p}$$

where $r_f$ is the risk-free rate.

### Capital Allocation Line (CAL)
The **Capital Allocation Line (CAL)** represents all possible combinations of the risk-free asset and a risky portfolio. The optimal risky portfolio is where the CAL is tangent to the Efficient Frontier — the **tangency portfolio**.

---

## Section 5 - Capital Asset Pricing Model (CAPM)

### Systematic and Unsystematic Risks
- **Systematic risk** (market risk): affects the entire market and cannot be diversified away (e.g. recessions, interest rate changes).
- **Unsystematic risk** (idiosyncratic risk): specific to a single company or industry and can be eliminated through diversification.

### Beta and Alpha Parameters
- **Beta ($\beta$)**: measures a stock's sensitivity to market movements. $\beta > 1$ means more volatile than the market.
- **Alpha ($\alpha$)**: represents excess return relative to what CAPM predicts — a measure of a manager's skill.

### Linear Regression and Market Risk
CAPM is derived using **linear regression** of a stock's returns against market returns:

$$R_i = \alpha + \beta R_m + \epsilon$$

where $\epsilon$ is the idiosyncratic (unsystematic) error term.

### Why Market Risk Is the Only Relevant Risk?
In a well-diversified portfolio, unsystematic risk is eliminated. Investors are only compensated for bearing **systematic risk** because it cannot be avoided. This is the core insight of CAPM:

$$E(R_i) = r_f + \beta_i (E(R_m) - r_f)$$

---

## Section 6 - Derivatives Basics

### Derivatives Basics
A **derivative** is a financial contract whose value is derived from an underlying asset (stocks, bonds, commodities, currencies, etc.). They are used for **hedging**, **speculation**, and **arbitrage**.

### Options (Put and Call Options)
- **Call option**: gives the holder the right (not obligation) to **buy** an asset at a strike price $K$ before expiry.
- **Put option**: gives the holder the right to **sell** an asset at strike price $K$ before expiry.

Payoffs:
- Call: $\max(S_T - K, 0)$
- Put: $\max(K - S_T, 0)$

### Forward and Future Contracts
- **Forward**: a private agreement to buy/sell an asset at a future date for a price agreed today. Traded OTC.
- **Future**: standardised forward contracts traded on an exchange, with daily mark-to-market settlement.

### Credit Default Swaps (CDS)
A **CDS** is an insurance-like contract where the buyer pays periodic premiums to the seller in exchange for protection against a credit event (e.g. default) on a reference entity.

### Interest Rate Swaps
An **interest rate swap** is an agreement between two parties to exchange cash flows — typically one party pays a **fixed rate** while the other pays a **floating rate** (e.g. LIBOR/SOFR) on a notional principal.

---

## Section 7 - Random Behavior in Finance

### Random Behavior
Financial asset prices exhibit **random walk** behaviour — future price movements cannot be predicted from past prices alone. This randomness is modelled using stochastic processes.

### Wiener Processes
A **Wiener process** (standard Brownian motion) $W_t$ satisfies:
- $W_0 = 0$
- Increments $W_t - W_s \sim \mathcal{N}(0, t-s)$ for $t > s$
- Independent increments

### Stochastic Calculus and Itô's Lemma
**Itô's Lemma** is the chain rule for stochastic processes. For a function $f(S, t)$ where $S$ follows a stochastic process:

$$df = \frac{\partial f}{\partial t}dt + \frac{\partial f}{\partial S}dS + \frac{1}{2}\frac{\partial^2 f}{\partial S^2}(dS)^2$$

### Brownian Motion Theory and Implementation
Geometric Brownian Motion (GBM) models stock prices:

$$dS = \mu S \, dt + \sigma S \, dW_t$$

where $\mu$ is the drift, $\sigma$ is the volatility, and $W_t$ is a Wiener process.

---

## Section 8 - Black-Scholes Model

### Black-Scholes Model Theory and Implementation
The **Black-Scholes model** gives a closed-form formula for pricing European options. For a call option:

$$C = S_0 N(d_1) - K e^{-rT} N(d_2)$$

$$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

where $N(\cdot)$ is the cumulative standard normal distribution.

### Monte Carlo Simulations for Option Pricing
**Monte Carlo simulation** estimates option prices by simulating thousands of possible price paths using GBM and averaging the discounted payoffs:

$$C \approx e^{-rT} \frac{1}{N} \sum_{i=1}^{N} \max(S_T^{(i)} - K, 0)$$

### The Greeks
The **Greeks** measure the sensitivity of an option's price to various factors:

| Greek | Symbol | Measures sensitivity to... |
|-------|--------|---------------------------|
| Delta | $\Delta$ | Changes in underlying price |
| Gamma | $\Gamma$ | Changes in Delta |
| Theta | $\Theta$ | Passage of time (time decay) |
| Vega  | $\mathcal{V}$ | Changes in volatility |
| Rho   | $\rho$ | Changes in interest rates |

---

## Section 9 - Value-at-Risk (VaR)

### What Is Value at Risk (VaR)?
**VaR** quantifies the maximum potential loss of a portfolio over a given time horizon at a given confidence level. For example, a 1-day 95% VaR of \$1M means there is a 5% chance of losing more than \$1M in a single day.

### Monte Carlo Simulation to Calculate Risks
Monte Carlo VaR involves:
1. Simulating thousands of possible portfolio return scenarios
2. Building the resulting P&L distribution
3. Reading off the loss at the desired confidence percentile (e.g. 5th percentile for 95% VaR)

---

## Section 10 - Collateralized Debt Obligation (CDO)

### What Are CDOs?
A **Collateralized Debt Obligation (CDO)** is a structured financial product backed by a pool of loans (mortgages, bonds, etc.). The pool is sliced into **tranches** with different risk/return profiles — senior tranches absorb losses last and carry lower yields, while equity tranches absorb losses first but offer higher yields.

### The Financial Crisis of 2008
The 2008 financial crisis was largely triggered by CDOs backed by **subprime mortgages**. Key factors:
- Overconfidence in **credit ratings** of CDO tranches
- **Correlation risk** was severely underestimated (assets defaulted together)
- Widespread use of **CDS** amplified systemic exposure
- When housing prices fell, the entire structure collapsed, triggering a global credit freeze

---

## Section 11 - Interest Rate Models

### Mean Reverting Stochastic Processes
Unlike stock prices, interest rates tend to **revert to a long-run mean** over time. They cannot grow without bound or go to zero in most economies. This property is called **mean reversion**.

### The Ornstein-Uhlenbeck Process
The **Ornstein-Uhlenbeck (OU) process** is the canonical mean-reverting stochastic process:

$$dr_t = \theta(\mu - r_t)dt + \sigma \, dW_t$$

where $\theta$ is the speed of mean reversion, $\mu$ is the long-run mean, and $\sigma$ is the volatility.

### The Vasicek Model
The **Vasicek model** applies the OU process to short-term interest rates. It has a closed-form solution for bond prices and allows for **negative interest rates** (a known limitation).

$$r_t = r_0 e^{-\theta t} + \mu(1 - e^{-\theta t}) + \sigma \int_0^t e^{-\theta(t-s)} dW_s$$

### Using Monte Carlo Simulation to Price Bonds
Bond prices under stochastic interest rates can be estimated by:
1. Simulating many interest rate paths via the Vasicek (or other) model
2. Computing the discount factor for each path: $e^{-\int_0^T r_t \, dt}$
3. Averaging across all simulated paths

---

## Section 12 - Value Investing

### Long-Term Investing
**Value investing** focuses on identifying undervalued companies and holding them for the long term. Pioneered by **Benjamin Graham** and popularised by **Warren Buffett**, it relies on fundamental analysis — evaluating a company's intrinsic value relative to its market price.

Key metrics include P/E ratio, P/B ratio, free cash flow, and return on equity (ROE).

### Efficient Market Hypothesis
The **Efficient Market Hypothesis (EMH)** posits that asset prices fully reflect all available information, making it impossible to consistently outperform the market through analysis or timing:
- **Weak form**: prices reflect all past trading data
- **Semi-strong form**: prices reflect all publicly available information
- **Strong form**: prices reflect all information, public and private

Value investing implicitly challenges the semi-strong form of EMH, arguing that markets can be temporarily mispriced.

---

## Setup

```bash
# Clone the repository
git clone https://github.com/antoniopssousa1/quantitative-finance.git
cd quantitative-finance

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
```

---

## Requirements

See [`requirements.txt`](requirements.txt) for the full list of dependencies. Core libraries used throughout:

- `numpy` — numerical computing
- `pandas` — data manipulation
- `matplotlib` / `seaborn` — visualisation
- `scipy` — statistical functions
- `python-dotenv` — environment variable management
