# **Lesson Plan 5: Utility Maximization & Portfolio Choice**

![Utility Maximization](https://img.shields.io/badge/Finance-Utility%20Theory-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Lesson](https://img.shields.io/badge/Lesson-5-important)
![Chapter](https://img.shields.io/badge/Chapter-2-9cf)
![License](https://img.shields.io/badge/License-CC--BY--NC--SA--4.0-lightgrey)

## **📊 Overview**

Welcome to **Lesson Plan 5: Utility Maximization & Portfolio Choice**, where we explore the mathematical foundations of investor decision-making under uncertainty. This lesson bridges economic theory with computational implementation, teaching students how to model preferences, solve optimization problems, and determine optimal investment strategies using Python.

## **🎯 Learning Objectives**

### **Core Competencies**
- **Understand** the fundamental principles of utility theory and investor preferences
- **Implement** utility maximization algorithms using Lagrange multipliers
- **Analyze** portfolio optimization problems under constraints
- **Model** equilibrium pricing in complete and incomplete markets
- **Evaluate** the limitations and extensions of expected utility theory

### **Skills Developed**
- Advanced Python programming with SymPy and SciPy
- Mathematical optimization techniques
- Economic modeling of risk preferences
- Financial analysis of portfolio choice
- Critical thinking about model assumptions

## **📚 Curriculum Alignment**

### **TEKS Standards**
- **Computer Science**: Competency 007, 009, Standard IV
- **Mathematics**: Pre-Calculus, Calculus, Statistics
- **Economics**: Microeconomics, Financial Economics
- **Personal Finance**: Risk management, Investment strategies

### **International Curricula**
- **Cambridge A-Level**: Economics, Further Mathematics
- **IB Diploma**: Economics HL, Mathematics AA HL
- **AP**: Microeconomics, Calculus BC, Statistics

## **🛠️ Technical Requirements**

### **Software & Libraries**
```bash
# Core requirements
pip install numpy pandas matplotlib scipy sympy

# For advanced optimization (optional)
pip install cvxopt pulp

# For Jupyter notebook interface (recommended)
pip install jupyter notebook

# For financial data (optional)
pip install yfinance
```

### **Hardware Recommendations**
- **Minimum**: 4GB RAM, Dual-core processor
- **Recommended**: 8GB+ RAM, Quad-core processor
- **Storage**: 1GB free space for datasets and projects

### **Development Environment**
- **Primary IDE**: Visual Studio Code or Jupyter Notebook
- **Alternative**: Online IDE at http://finpy.pqp.io
- **Version Control**: Git with GitHub Classroom

## **📖 Lesson Structure**

### **90-Minute Session Plan**

| **Time** | **Activity** | **Key Concepts** | **Python Skills** |
|----------|-------------|------------------|-------------------|
| **0-10 min** | Introduction & Review | Utility theory basics, risk preferences | NumPy basics, function definition |
| **10-25 min** | Focus Activity | Risk-return tradeoff, utility functions | Matplotlib visualization |
| **25-65 min** | Station Work | Four learning stations (see below) | SymPy, SciPy optimization |
| **65-75 min** | Station Review | Connecting concepts across stations | Debugging, collaboration |
| **75-85 min** | Assessment | Coding challenge, group problem | Problem-solving |
| **85-90 min** | Closure | Summary, real-world applications | Documentation |

### **Station Activities**

#### **Station 1: Utility Function Implementation**
- Implement logarithmic, power, and exponential utility functions
- Analyze different risk aversion parameters
- Visualize utility functions and risk preferences

#### **Station 2: Portfolio Optimization**
- Solve constrained optimization with Lagrange multipliers
- Implement mean-variance optimization
- Calculate optimal portfolio weights

#### **Station 3: Equilibrium Pricing Models**
- Implement representative agent pricing
- Calculate state prices and risk-neutral probabilities
- Analyze market completeness

#### **Station 4: Market Analysis**
- Check market completeness using matrix rank
- Find martingale measures
- Implement market completion strategies

## **🔧 Python Implementation Guide**

### **Core Classes & Functions**

#### **1. Utility Function Implementation**
```python
# File: utility_functions.py
import numpy as np
import matplotlib.pyplot as plt

class UtilityFunctions:
    """
    Implementation of different utility functions for modeling preferences
    """
    
    @staticmethod
    def logarithmic(wealth):
        """Logarithmic utility: U(x) = ln(x)"""
        return np.log(wealth)
    
    @staticmethod
    def power(wealth, gamma=0.5):
        """CRRA utility: U(x) = x^(1-γ)/(1-γ)"""
        if gamma == 1:
            return np.log(wealth)
        return (wealth**(1 - gamma)) / (1 - gamma)
    
    @staticmethod
    def exponential(wealth, alpha=0.01):
        """CARA utility: U(x) = -exp(-αx)"""
        return -np.exp(-alpha * wealth)
    
    def compare_risk_preferences(self, wealth_range=(1, 100)):
        """Visual comparison of different utility functions"""
        wealth = np.linspace(*wealth_range, 100)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Logarithmic utility
        axes[0].plot(wealth, self.logarithmic(wealth))
        axes[0].set_title('Logarithmic Utility')
        axes[0].set_xlabel('Wealth')
        axes[0].set_ylabel('Utility')
        
        # Power utility with different risk aversion
        for gamma in [0.2, 0.5, 0.8]:
            axes[1].plot(wealth, self.power(wealth, gamma), 
                       label=f'γ = {gamma}')
        axes[1].set_title('Power (CRRA) Utility')
        axes[1].set_xlabel('Wealth')
        axes[1].legend()
        
        # Exponential utility
        for alpha in [0.005, 0.01, 0.02]:
            axes[2].plot(wealth, self.exponential(wealth, alpha),
                       label=f'α = {alpha}')
        axes[2].set_title('Exponential (CARA) Utility')
        axes[2].set_xlabel('Wealth')
        axes[2].legend()
        
        plt.tight_layout()
        return fig
```

#### **2. Portfolio Optimization with Lagrange Multipliers**
```python
# File: portfolio_optimizer.py
import numpy as np
import sympy as sp
from scipy.optimize import minimize

class PortfolioOptimizer:
    """
    Portfolio optimization using Lagrange multipliers and numerical methods
    """
    
    def symbolic_optimization(self, expected_returns, covariance, risk_aversion):
        """
        Symbolic solution using Lagrange multipliers
        Returns: Optimal weights and Lagrange multiplier
        """
        n = len(expected_returns)
        
        # Define symbolic variables
        w = sp.symbols(f'w0:{n}')
        λ = sp.symbols('λ')  # Lagrange multiplier
        
        # Portfolio statistics
        portfolio_return = sum(w[i] * expected_returns[i] for i in range(n))
        portfolio_variance = sum(sum(
            w[i] * w[j] * covariance[i, j] 
            for j in range(n)
        ) for i in range(n))
        
        # Objective: Maximize expected utility
        # U = E[r] - (γ/2) * Var[r]
        objective = portfolio_return - (risk_aversion/2) * portfolio_variance
        
        # Constraint: weights sum to 1
        constraint = sum(w) - 1
        
        # Lagrangian
        L = objective + λ * constraint
        
        # First-order conditions
        equations = []
        for i in range(n):
            equations.append(sp.diff(L, w[i]))
        equations.append(sp.diff(L, λ))
        
        # Solve system of equations
        solution = sp.solve(equations, list(w) + [λ])
        
        return solution
    
    def numerical_optimization(self, expected_returns, covariance, risk_aversion):
        """
        Numerical optimization using SciPy
        Returns: Optimal weights
        """
        n = len(expected_returns)
        
        def objective(weights):
            """Negative utility for minimization"""
            port_return = np.dot(weights, expected_returns)
            port_variance = weights.T @ covariance @ weights
            utility = port_return - (risk_aversion/2) * port_variance
            return -utility  # Negative for minimization
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # Bounds: no short selling (can be changed)
        bounds = [(0, 1) for _ in range(n)]
        
        # Initial guess: equal weights
        initial_weights = np.ones(n) / n
        
        # Optimize
        result = minimize(objective, initial_weights,
                         constraints=constraints, bounds=bounds)
        
        if result.success:
            return result.x
        else:
            raise ValueError("Optimization failed")
    
    def efficient_frontier(self, expected_returns, covariance, target_returns):
        """
        Calculate efficient frontier for given target returns
        """
        efficient_portfolios = []
        
        for target in target_returns:
            n = len(expected_returns)
            
            def portfolio_variance(weights):
                return weights.T @ covariance @ weights
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target}
            ]
            
            bounds = [(0, 1) for _ in range(n)]
            initial_weights = np.ones(n) / n
            
            result = minimize(portfolio_variance, initial_weights,
                            constraints=constraints, bounds=bounds)
            
            if result.success:
                efficient_portfolios.append({
                    'weights': result.x,
                    'return': target,
                    'variance': result.fun,
                    'volatility': np.sqrt(result.fun)
                })
        
        return efficient_portfolios
```

#### **3. Equilibrium Pricing Models**
```python
# File: equilibrium_pricer.py
import numpy as np

class EquilibriumPricer:
    """
    Equilibrium pricing using representative agent model
    """
    
    def __init__(self, utility_func, endowments, probabilities, beta=0.95):
        """
        Parameters:
        -----------
        utility_func: Function that takes wealth and returns utility
        endowments: Initial wealth in each state
        probabilities: Physical probabilities of each state
        beta: Time discount factor
        """
        self.utility = utility_func
        self.endowments = np.array(endowments)
        self.probs = np.array(probabilities)
        self.beta = beta
    
    def marginal_utility(self, wealth):
        """Calculate marginal utility (numerical derivative)"""
        epsilon = 1e-6
        return (self.utility(wealth + epsilon) - self.utility(wealth - epsilon)) / (2 * epsilon)
    
    def calculate_state_prices(self):
        """
        Calculate state prices using marginal utility ratio
        q_s = β * π_s * MU_s / MU_0
        """
        # Marginal utility at endowment in each state
        mu = self.marginal_utility(self.endowments)
        
        # State prices
        state_prices = self.beta * self.probs * mu / mu[0]
        
        return state_prices
    
    def price_assets(self, payoff_matrix):
        """
        Price multiple assets given their state payoffs
        """
        state_prices = self.calculate_state_prices()
        prices = payoff_matrix.T @ state_prices
        
        return prices
    
    def complete_market(self, payoff_matrix):
        """
        Complete an incomplete market by adding Arrow-Debreu securities
        """
        n_states = payoff_matrix.shape[1]
        n_assets = payoff_matrix.shape[0]
        
        if np.linalg.matrix_rank(payoff_matrix) >= n_states:
            print("Market is already complete")
            return payoff_matrix
        
        # Add missing Arrow-Debreu securities
        missing = n_states - n_assets
        ad_securities = np.eye(n_states)[n_assets:]
        
        complete_matrix = np.vstack([payoff_matrix, ad_securities])
        
        print(f"Added {missing} Arrow-Debreu securities")
        return complete_matrix
    
    def find_representative_agent(self, prices, payoff_matrix):
        """
        Find representative agent's risk aversion that rationalizes prices
        """
        # This is an inverse problem: find utility parameters that match prices
        # Simplified implementation for demonstration
        def objective(gamma):
            # Create utility function with given risk aversion
            def utility(w):
                if gamma == 1:
                    return np.log(w)
                return (w**(1 - gamma)) / (1 - gamma)
            
            # Create pricer with this utility
            pricer = EquilibriumPricer(utility, self.endowments, self.probs, self.beta)
            
            # Calculate implied prices
            implied_prices = pricer.price_assets(payoff_matrix)
            
            # Return difference from actual prices
            return np.sum((implied_prices - prices)**2)
        
        # Optimize for gamma
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(objective, bounds=(0.1, 10))
        
        if result.success:
            return result.x
        else:
            return None
```

#### **4. Market Completeness Analysis**
```python
# File: market_analyzer.py
import numpy as np
from scipy.optimize import linprog

class MarketAnalyzer:
    """
    Analyze market completeness and find arbitrage opportunities
    """
    
    def check_completeness(self, payoff_matrix):
        """
        Check if market is complete using linear algebra
        Complete if payoff matrix has full row rank = number of states
        """
        n_states = payoff_matrix.shape[1]
        n_assets = payoff_matrix.shape[0]
        
        rank = np.linalg.matrix_rank(payoff_matrix)
        is_complete = rank >= n_states
        
        analysis = {
            'is_complete': is_complete,
            'rank': rank,
            'states': n_states,
            'assets': n_assets,
            'deficiency': n_states - rank if not is_complete else 0,
            'span_dimension': rank
        }
        
        return analysis
    
    def find_arbitrage(self, payoff_matrix, prices):
        """
        Find arbitrage opportunities using linear programming
        Returns: Arbitrage portfolio if exists, None otherwise
        """
        n_assets = payoff_matrix.shape[0]
        n_states = payoff_matrix.shape[1]
        
        # Objective: Maximize initial profit
        # We want portfolio with non-positive cost and non-negative payoffs
        c = -prices  # Maximize negative cost = minimize cost
        
        # Constraints: Payoffs in all states >= 0
        A_ub = -payoff_matrix.T  # Negative for <= constraints
        b_ub = np.zeros(n_states)
        
        # No constraint on weights (allow any real numbers)
        bounds = [(None, None) for _ in range(n_assets)]
        
        # Solve linear program
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            portfolio = result.x
            cost = np.dot(prices, portfolio)
            payoffs = payoff_matrix @ portfolio
            
            # Check if arbitrage exists
            # Arbitrage: cost <= 0 and all payoffs >= 0 and (cost < 0 or some payoff > 0)
            if cost <= 0 and np.all(payoffs >= 0) and (cost < 0 or np.any(payoffs > 0)):
                return {
                    'portfolio': portfolio,
                    'cost': cost,
                    'payoffs': payoffs,
                    'arbitrage_exists': True
                }
        
        return {'arbitrage_exists': False}
    
    def find_martingale_measure(self, payoff_matrix, prices, risk_free=0.03):
        """
        Find risk-neutral probabilities (martingale measure)
        Solve: P * q = p, where q = π* / (1 + r_f)
        """
        n_states = payoff_matrix.shape[1]
        
        # We need to find state prices q such that:
        # 1. P^T * q = p
        # 2. q >= 0
        # 3. sum(q) = 1/(1 + r_f)
        
        # Set up linear programming problem
        # Objective: arbitrary (we just want feasibility)
        c = np.zeros(n_states)
        
        # Equality constraints: P^T * q = p
        A_eq = payoff_matrix.T
        b_eq = prices
        
        # Inequality constraints: q >= 0
        A_ub = -np.eye(n_states)  # -q <= 0 means q >= 0
        b_ub = np.zeros(n_states)
        
        # Additional constraint: sum(q) = 1/(1 + r_f)
        # We'll add this as another equality constraint
        A_eq = np.vstack([A_eq, np.ones(n_states)])
        b_eq = np.append(b_eq, 1/(1 + risk_free))
        
        # Solve
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
        
        if result.success:
            state_prices = result.x
            risk_neutral_probs = state_prices * (1 + risk_free)
            
            return {
                'state_prices': state_prices,
                'risk_neutral_probs': risk_neutral_probs,
                'exists': True
            }
        else:
            return {'exists': False, 'reason': 'No martingale measure found'}
    
    def spanning_analysis(self, payoff_matrix):
        """
        Analyze which contingent claims can be replicated
        Returns: Basis for attainable claims
        """
        # QR decomposition to find basis
        Q, R = np.linalg.qr(payoff_matrix.T)
        
        # Columns of Q corresponding to non-zero diagonal of R
        rank = np.linalg.matrix_rank(R)
        basis = Q[:, :rank]
        
        return {
            'basis': basis,
            'rank': rank,
            'spanning_assets': rank,
            'attainable_dimension': rank
        }
```

### **Complete Example Project**

```python
# File: main_project.py
"""
Complete utility maximization and portfolio choice project
Demonstrates all key concepts from the lesson
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility_functions import UtilityFunctions
from portfolio_optimizer import PortfolioOptimizer
from equilibrium_pricer import EquilibriumPricer
from market_analyzer import MarketAnalyzer

def main():
    print("="*60)
    print("UTILITY MAXIMIZATION & PORTFOLIO CHOICE PROJECT")
    print("="*60)
    
    # 1. Initialize components
    util = UtilityFunctions()
    optimizer = PortfolioOptimizer()
    analyzer = MarketAnalyzer()
    
    # 2. Example data
    np.random.seed(42)
    
    # Three assets with different risk-return characteristics
    expected_returns = np.array([0.08, 0.12, 0.15])  # 8%, 12%, 15%
    covariance = np.array([
        [0.04, 0.02, 0.01],    # Asset 1 variance 0.04
        [0.02, 0.09, 0.03],    # Asset 2 variance 0.09  
        [0.01, 0.03, 0.16]     # Asset 3 variance 0.16
    ])
    
    # 3. Portfolio optimization for different risk aversions
    print("\n1. PORTFOLIO OPTIMIZATION")
    print("-"*40)
    
    risk_aversions = [0.5, 1.0, 2.0, 5.0]
    for gamma in risk_aversions:
        weights = optimizer.numerical_optimization(
            expected_returns, covariance, gamma
        )
        port_return = np.dot(weights, expected_returns)
        port_risk = np.sqrt(weights.T @ covariance @ weights)
        
        print(f"\nRisk aversion γ = {gamma}:")
        print(f"  Weights: {weights}")
        print(f"  Expected return: {port_return:.2%}")
        print(f"  Volatility: {port_risk:.2%}")
    
    # 4. Efficient frontier
    print("\n\n2. EFFICIENT FRONTIER")
    print("-"*40)
    
    target_returns = np.linspace(0.08, 0.15, 10)
    efficient_portfolios = optimizer.efficient_frontier(
        expected_returns, covariance, target_returns
    )
    
    # Plot efficient frontier
    volatilities = [p['volatility'] for p in efficient_portfolios]
    returns = [p['return'] for p in efficient_portfolios]
    
    plt.figure(figsize=(10, 6))
    plt.plot(volatilities, returns, 'b-', linewidth=2, label='Efficient Frontier')
    plt.scatter(np.sqrt(np.diag(covariance)), expected_returns, 
               color='red', s=100, label='Individual Assets')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 5. Market completeness analysis
    print("\n\n3. MARKET COMPLETENESS ANALYSIS")
    print("-"*40)
    
    # Example payoff matrix (3 states, 2 assets - incomplete)
    payoff_matrix = np.array([
        [1.0, 1.1, 1.2],  # Asset 1 payoffs in 3 states
        [0.9, 1.0, 1.3]   # Asset 2 payoffs
    ])
    prices = np.array([0.95, 0.90])
    
    completeness = analyzer.check_completeness(payoff_matrix)
    print(f"Market completeness: {completeness['is_complete']}")
    print(f"Rank: {completeness['rank']}")
    print(f"States: {completeness['states']}")
    print(f"Assets: {completeness['assets']}")
    
    # 6. Equilibrium pricing example
    print("\n\n4. EQUILIBRIUM PRICING")
    print("-"*40)
    
    # Define utility function (logarithmic)
    def log_utility(wealth):
        return np.log(wealth)
    
    endowments = np.array([100, 110, 120])
    probabilities = np.array([0.3, 0.4, 0.3])
    
    pricer = EquilibriumPricer(log_utility, endowments, probabilities)
    state_prices = pricer.calculate_state_prices()
    
    print(f"State prices: {state_prices}")
    print(f"Sum of state prices: {np.sum(state_prices):.4f}")
    print(f"Implied risk-free rate: {1/np.sum(state_prices) - 1:.2%}")
    
    # Price the assets
    asset_prices = pricer.price_assets(payoff_matrix)
    print(f"\nTheoretical asset prices: {asset_prices}")
    print(f"Actual prices: {prices}")
    print(f"Pricing errors: {asset_prices - prices}")

if __name__ == "__main__":
    main()
```

## **📊 Project Structure**

```
lesson5-utility-maximization/
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
├── notebooks/                # Jupyter notebooks
│   ├── 01_utility_functions.ipynb
│   ├── 02_portfolio_optimization.ipynb
│   ├── 03_equilibrium_pricing.ipynb
│   ├── 04_market_completeness.ipynb
│   └── 05_complete_project.ipynb
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── utility_functions.py
│   ├── portfolio_optimizer.py
│   ├── equilibrium_pricer.py
│   ├── market_analyzer.py
│   └── visualizer.py
│
├── data/                     # Sample datasets
│   ├── sample_portfolio.csv
│   ├── risk_preferences.csv
│   └── market_data.csv
│
├── tests/                    # Unit tests
│   ├── test_utility.py
│   ├── test_portfolio.py
│   ├── test_pricing.py
│   └── test_market.py
│
├── examples/                 # Example implementations
│   ├── simple_optimization.py
│   ├── pricing_example.py
│   └── risk_analysis.py
│
└── docs/                     # Documentation
    ├── concepts.md
    ├── formulas.md
    ├── teaching_guide.md
    └── assessment_rubric.md
```

## **🚀 Quick Start Guide**

### **Option 1: Local Installation**
```bash
# 1. Clone the repository
git clone https://github.com/QuantumXPower111/utility-maximization.git
cd utility-maximization

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run examples
python examples/simple_optimization.py
```

### **Option 2: Google Colab**
```python
# Run in Colab cell
!git clone https://github.com/QuantumXPower111/utility-maximization.git
%cd utility-maximization
!pip install -r requirements.txt

# Import and use
from src.utility_functions import UtilityFunctions
util = UtilityFunctions()
util.compare_risk_preferences()
```

### **Option 3: Online IDE (finpy.pqp.io)**
1. Visit http://finpy.pqp.io
2. Upload the project files
3. Install required packages using pip
4. Run code directly in the browser

## **📈 Learning Pathways**

### **Beginner Track**
1. Start with `notebooks/01_utility_functions.ipynb`
2. Understand basic utility concepts
3. Implement simple portfolio optimization
4. Complete guided worksheets

### **Intermediate Track**
1. Work through portfolio optimization in `notebooks/02_portfolio_optimization.ipynb`
2. Implement Lagrange multiplier method
3. Analyze market completeness
4. Build interactive dashboards

### **Advanced Track**
1. Extend to dynamic optimization
2. Implement alternative utility theories (prospect theory)
3. Build web applications with Streamlit or Dash
4. Conduct empirical research with real data

## **🎓 Assessment Activities**

### **Individual Projects**

#### **Project 1: Personal Investment Advisor**
```python
"""
Build a tool that:
1. Assesses user's risk preferences through questions
2. Recommends optimal portfolio allocation
3. Simulates different market scenarios
4. Provides risk metrics and performance projections
"""
```

#### **Project 2: Market Completeness Analyzer**
```python
"""
Create a system that:
1. Takes payoff matrix as input
2. Determines market completeness
3. Finds arbitrage opportunities
4. Suggests securities to complete the market
"""
```

#### **Project 3: Equilibrium Pricing Simulator**
```python
"""
Build a simulator that:
1. Models representative agent economy
2. Calculates equilibrium prices
3. Analyzes sensitivity to parameters
4. Visualizes pricing relationships
"""
```

### **Group Challenges**

#### **Challenge 1: Robo-Advisor Prototype**
- **Team Size**: 3-4 students
- **Duration**: 2 weeks
- **Deliverables**: Working prototype with GUI
- **Skills**: Full-stack development, financial modeling

#### **Challenge 2: Behavioral Finance Research**
- **Team Size**: 2-3 students
- **Duration**: 3 weeks
- **Deliverables**: Research paper + experimental results
- **Skills**: Research methodology, data analysis, academic writing

## **📊 Sample Datasets**

### **Risk Preference Survey Data**
```python
import pandas as pd

def generate_risk_data(n_respondents=100):
    """Generate synthetic risk preference data"""
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(20, 65, n_respondents),
        'income': np.random.lognormal(10, 0.5, n_respondents),
        'risk_aversion': np.random.beta(2, 3, n_respondents) * 10,
        'investment_experience': np.random.choice(
            ['Beginner', 'Intermediate', 'Advanced'], 
            n_respondents,
            p=[0.4, 0.4, 0.2]
        )
    }
    
    return pd.DataFrame(data)
```

### **Real Market Data Integration**
```python
import yfinance as yf
import pandas as pd

def get_asset_returns(tickers, start_date='2020-01-01', end_date='2023-12-31'):
    """
    Fetch real asset returns for portfolio optimization
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    
    # Calculate statistics for optimization
    expected_returns = returns.mean() * 252  # Annualize
    covariance = returns.cov() * 252  # Annualize
    
    return {
        'returns': returns,
        'expected_returns': expected_returns,
        'covariance': covariance
    }
```

## **🎨 Visualization Gallery**

### **1. Utility Function Comparison**
```python
from src.visualizer import UtilityVisualizer

visualizer = UtilityVisualizer()
fig = visualizer.compare_utility_functions(
    wealth_range=(1, 200),
    risk_aversions=[0.5, 1.0, 2.0]
)
fig.show()
```

### **2. Efficient Frontier with Risk Preferences**
```python
from src.portfolio_optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer()
fig = optimizer.plot_efficient_frontier_with_indifference(
    expected_returns, 
    covariance,
    risk_aversion=2.0
)
fig.show()
```

### **3. Market Spanning Visualization**
```python
from src.market_analyzer import MarketAnalyzer

analyzer = MarketAnalyzer()
fig = analyzer.visualize_payoff_space(
    payoff_matrix,
    show_span=True,
    show_basis=True
)
fig.show()
```

## **🔬 Advanced Topics**

### **Extending to Dynamic Optimization**
```python
class DynamicOptimizer:
    """
    Extends static optimization to multi-period problems
    """
    
    def solve_lifecycle_model(self, income_path, utility_func, discount_factor):
        """
        Solve consumption-saving problem over lifecycle
        """
        # Implement dynamic programming or Euler equation approach
        pass
```

### **Alternative Utility Theories**
```python
class ProspectTheory:
    """
    Implements Kahneman & Tversky's prospect theory
    """
    
    def value_function(self, gains, losses, alpha=0.88, beta=0.88, lambda_param=2.25):
        """
        Value function from prospect theory
        v(x) = x^α for x ≥ 0
        v(x) = -λ(-x)^β for x < 0
        """
        pass
    
    def weighting_function(self, probability, gamma=0.61, delta=0.69):
        """
        Probability weighting function
        w(p) = p^γ / (p^γ + (1-p)^γ)^(1/γ)
        """
        pass
```

## **📝 Assessment Rubric**

### **Code Quality (20%)**
- **4 pts**: Professional, efficient, well-documented
- **3 pts**: Functional with minor issues
- **2 pts**: Basic implementation with errors
- **1 pt**: Incomplete or non-functional

### **Mathematical Accuracy (25%)**
- **4 pts**: Correct application of optimization methods
- **3 pts**: Minor mathematical errors
- **2 pts**: Several calculation errors
- **1 pt**: Major conceptual errors

### **Financial Insight (20%)**
- **4 pts**: Deep understanding of utility theory
- **3 pts**: Good grasp of key concepts
- **2 pts**: Basic comprehension
- **1 pt**: Limited understanding

### **Problem-Solving (15%)**
- **4 pts**: Creative, efficient solutions
- **3 pts**: Methodical problem-solving
- **2 pts**: Needs guidance
- **1 pt**: Struggles with approach

### **Visualization (10%)**
- **4 pts**: Professional, informative visualizations
- **3 pts**: Clear visual representations
- **2 pts**: Basic charts
- **1 pt**: Poor or missing visuals

### **Collaboration (10%)**
- **4 pts**: Actively contributes, helps peers
- **3 pts**: Participates effectively
- **2 pts**: Limited participation
- **1 pt**: Does not collaborate

### **Grading Scale**
- **A (90-100%)**: 18-20 points
- **B (80-89%)**: 16-17.9 points
- **C (70-79%)**: 14-15.9 points
- **D (60-69%)**: 12-13.9 points
- **F (<60%)**: <12 points

## **🎓 Differentiation Strategies**

### **Accommodations for Diverse Learners**

#### **Visual Learners**
- Graphical representations of utility functions
- 3D visualizations of optimization surfaces
- Color-coded risk-return diagrams
- Animated explanations of concepts

#### **Auditory Learners**
- Podcast-style lesson summaries
- Verbal explanations of mathematical derivations
- Discussion-based learning activities
- Audio recordings of key concepts

#### **Kinesthetic Learners**
- Interactive optimization sliders
- Physical models of risk preferences
- Hands-on coding with immediate feedback
- Real-world decision simulations

#### **Language Learners**
- Bilingual code comments and documentation
- Technical vocabulary translation guides
- Simplified English explanations
- Peer programming with language support

### **Tiered Difficulty Levels**

#### **Level 1: Foundation**
- Pre-written functions with guided parameters
- Step-by-step optimization worksheets
- Visual decision-making tools
- Extended time for assessments

#### **Level 2: Standard**
- Complete implementation with scaffolding
- Real-world data analysis projects
- Pair programming opportunities
- Detailed rubric guidance

#### **Level 3: Advanced**
- Open-ended research questions
- Multiple model comparison
- Performance optimization challenges
- Peer teaching responsibilities

## **🔗 Career Connections**

### **Industry Roles**
1. **Quantitative Analyst** ($133,460 median)
   - Skills: Mathematical modeling, optimization
   - Education: Master's in Financial Engineering

2. **Financial Planner** ($94,170 median)
   - Skills: Risk assessment, portfolio construction
   - Education: CFP certification

3. **Risk Manager** ($86,200 median)
   - Skills: Risk modeling, regulatory compliance
   - Education: FRM certification

4. **Economic Consultant** ($108,350 median)
   - Skills: Economic modeling, data analysis
   - Education: PhD in Economics

### **Industry Tools**
- **MATLAB Optimization Toolbox**: Industry standard for numerical optimization
- **Python SciPy/SymPy**: Open-source alternatives gaining popularity
- **Excel Solver**: Accessible optimization for business applications
- **Bloomberg PORT**: Professional portfolio optimization platform

## **📚 Further Resources**

### **Books**
1. **"Microeconomic Theory"** by Andreu Mas-Colell
2. **"Investments"** by Zvi Bodie, Alex Kane, Alan Marcus
3. **"Principles of Financial Economics"** by Stephen F. LeRoy
4. **"Python for Finance"** by Yves Hilpisch

### **Online Courses**
1. **Coursera**: Financial Engineering and Risk Management
2. **edX**: Microeconomics and Game Theory
3. **MIT OpenCourseWare**: Financial Economics
4. **Khan Academy**: Microeconomics

### **Academic Journals**
1. Journal of Financial Economics
2. Review of Financial Studies
3. Journal of Economic Theory
4. Quantitative Finance

### **Competitions**
1. **Portfolio Management Challenge** (University level)
2. **Quantitative Finance Case Competition**
3. **MathWorks Math Modeling Challenge**
4. **Federal Reserve Challenge**

## **👨‍🏫 Instructor Resources**

### **Lesson Preparation Checklist**
- [ ] Test all code examples
- [ ] Prepare sample datasets
- [ ] Set up programming environment
- [ ] Create assessment materials
- [ ] Prepare differentiation resources
- [ ] Test accommodations tools

### **Teaching Tips**
1. **Start with Intuition**: Begin with real-life decision examples
2. **Visualize Mathematics**: Use graphs to explain abstract concepts
3. **Connect to Real World**: Show practical applications
4. **Scaffold Complexity**: Build from simple to complex models
5. **Encourage Exploration**: Allow students to experiment with parameters

### **Common Student Questions**
1. **Q**: "Why use logarithmic utility instead of linear?"
   **A**: Logarithmic utility captures diminishing marginal utility and risk aversion.

2. **Q**: "What if markets are never complete in reality?"
   **A**: Incompleteness explains why some risks can't be hedged and affects pricing.

3. **Q**: "How do Lagrange multipliers work in economics?"
   **A**: They represent the shadow price of constraints (e.g., the value of relaxing budget constraint).

4. **Q**: "Can utility theory predict real investor behavior?"
   **A**: It's a normative theory (how rational investors should behave), not always descriptive.

### **Discussion Prompts**
- "Should investment advice be personalized based on utility functions?"
- "How do cultural differences affect risk preferences?"
- "What are the ethical implications of robo-advisors using utility optimization?"
- "How has behavioral economics changed traditional utility theory?"

## **🚀 Beyond the Classroom**

### **Research Opportunities**
1. **Empirical Testing**: Test utility theory predictions with real investor data
2. **Model Extensions**: Develop new utility functions for specific contexts
3. **Algorithm Development**: Create more efficient optimization algorithms
4. **Educational Tools**: Build interactive learning platforms

### **Portfolio Projects**
```python
# Students can add to their GitHub portfolio:
portfolio_projects = [
    "Personalized Portfolio Optimizer",
    "Market Completeness Analyzer",
    "Equilibrium Pricing Simulator",
    "Risk Preference Assessment Tool"
]
```

### **College Preparation**
- **AP/IB Credit**: Material aligns with college-level economics courses
- **University Applications**: Projects demonstrate quantitative and analytical skills
- **Scholarships**: Skills in mathematical finance are highly valued
- **Internships**: Preparation for quantitative finance and economics roles

## **📄 License & Attribution**

### **Educational Use License**
This material is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.

**You are free to:**
- Share — copy and redistribute the material
- Adapt — remix, transform, and build upon the material

**Under these terms:**
- **Attribution** — You must give appropriate credit
- **NonCommercial** — You may not use the material for commercial purposes
- **ShareAlike** — If you remix, you must license under identical terms

### **Disclaimer**
```
This educational material is for instructional purposes only.
The financial models and examples provided are simplified
for educational clarity and should not be used for actual
investment decisions. Always consult with a qualified financial
advisor before making investment decisions.
```

### **Citation**
```
Antwi, E. (2024). Utility Maximization & Portfolio Choice: 
Financial Modeling with Python. Computer Science Curriculum, Chapter 2.
```

## **👤 Instructor Information**

**Teacher:** Ernest Antwi  
**Subject:** Computer Science / Financial Programming  
**Chapter:** 2 - Finance and Python  
**GitHub:** [QuantumXPower111](https://github.com/QuantumXPower111)  
**Email:** [Ernest.K.Antwi2013@zoho.com](mailto:Ernest.K.Antwi2013@zoho.com)  
**LinkedIn:** [Connect for career guidance](https://linkedin.com/in/ernest-antwi)  
**Office Hours:** Available by appointment

## **🌟 Student Success Stories**

*"This project helped me win a national economics competition!"* - Maria G., Class of 2023  
*"I used these optimization skills in my summer internship at an investment bank."* - James L., Class of 2024  
*"The portfolio optimizer I built became the basis for my college application project."* - Sarah K., Class of 2023

## **📞 Support & Contact**

### **Getting Help**
1. **GitHub Issues**: For bugs and feature requests
2. **Email**: For direct instructor support
3. **Discussion Forum**: For peer collaboration
4. **Office Hours**: For live help sessions

### **Contributing**
We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### **Reporting Issues**
When reporting issues, please include:
1. Python version
2. Error messages
3. Steps to reproduce
4. Expected vs. actual behavior

---

**⭐ If you find this resource helpful, please star the repository!**  
**🐛 Found a bug or have a suggestion? Open an issue!**  
**🔧 Want to contribute? Submit a pull request!**

---

*Last Updated: January 2026*  
*Version: 5.0*  
*Next Lesson: Chapter 3 - Three-State Economy & Market Completeness*
