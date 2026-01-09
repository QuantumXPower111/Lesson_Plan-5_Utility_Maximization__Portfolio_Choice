from src.market_analyzer import MarketAnalyzer

analyzer = MarketAnalyzer()
fig = analyzer.visualize_payoff_space(
    payoff_matrix,
    show_span=True,
    show_basis=True
)
fig.show()