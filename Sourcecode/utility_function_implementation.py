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