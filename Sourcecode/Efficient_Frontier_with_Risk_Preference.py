from src.portfolio_optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer()
fig = optimizer.plot_efficient_frontier_with_indifference(
    expected_returns, 
    covariance,
    risk_aversion=2.0
)
fig.show()