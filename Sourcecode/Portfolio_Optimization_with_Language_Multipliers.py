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