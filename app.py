import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from utils import get_market_data, time_to_expiry
from models import (
    monte_carlo_lsm,
    monte_carlo_boundary_avg,
    calculate_greeks,
    binomial_tree,
    finite_difference_pde
)

st.set_page_config(page_title="American Option Pricer", layout="wide")
st.title("American Option Fair Value Estimator")

ticker = st.text_input("Enter Ticker Symbol", value="AAPL").upper()

if ticker:
    # Get available expiries and full option chain
    stock = yf.Ticker(ticker)
    expiries = stock.options

    if not expiries:
        st.warning("No option chain data available for this ticker.")
    else:
        expiry_date = st.selectbox("Select Expiry Date", expiries)

        # Fetch option chain for selected expiry
        chain = stock.option_chain(expiry_date)
        # Combine strikes from calls and puts
        strikes = sorted(set(chain.calls['strike']).union(set(chain.puts['strike'])))

        strike_price = st.selectbox("Select Strike Price", strikes)

        option_type = st.selectbox("Option Type", ["call", "put"])
        position_type = st.selectbox("Position Type", ["buy", "sell"])
        risk_free_rate = st.number_input("Risk-Free Rate (%)", value=5.0) / 100

        if st.button("Estimate Fair Value"):
            spot_price, vol, div_yield = get_market_data(ticker)
            T = time_to_expiry(expiry_date)

            # Filter market option data for selected strike and type
            if option_type == "call":
                market_option = chain.calls[chain.calls['strike'] == strike_price]
            else:
                market_option = chain.puts[chain.puts['strike'] == strike_price]

            if market_option.empty:
                st.warning("Selected option data not found in market chain.")
            else:
                bid = market_option['bid'].values[0]
                ask = market_option['ask'].values[0]
                mid_market = (bid + ask) / 2 if (bid > 0 and ask > 0) else np.nan

                # Model pricing
                price_mc, paths, _ = monte_carlo_lsm(
                    S0=spot_price,
                    K=strike_price,
                    T=T,
                    r=risk_free_rate,
                    sigma=vol,
                    option_type=option_type,
                    position_type=position_type,
                    return_paths=True
                )
                boundary = monte_carlo_boundary_avg(
                    S0=spot_price,
                    K=strike_price,
                    T=T,
                    r=risk_free_rate,
                    sigma=vol,
                    option_type=option_type,
                    position_type=position_type,
                    runs=10
                )
                greeks = calculate_greeks(
                    S0=spot_price,
                    K=strike_price,
                    T=T,
                    r=risk_free_rate,
                    sigma=vol,
                    option_type=option_type,
                    position_type=position_type
                )
                price_binomial = binomial_tree(
                    S0=spot_price,
                    K=strike_price,
                    T=T,
                    r=risk_free_rate,
                    sigma=vol,
                    option_type=option_type,
                    position_type=position_type
                )
                price_pde = finite_difference_pde(
                    S0=spot_price,
                    K=strike_price,
                    T=T,
                    r=risk_free_rate,
                    sigma=vol,
                    option_type=option_type,
                    position_type=position_type
                )

                st.subheader("Market Option Data")
                st.write(f"Bid: {bid:.2f}, Ask: {ask:.2f}, Mid Market Price: {mid_market if not np.isnan(mid_market) else 'N/A'}")

                # Display model prices and mispricing
                st.subheader("Model Prices & Mispricing (Model Price - Market Mid)")

                def format_price(price):
                    return f"{price:.2f}"

                st.write(f"Monte Carlo: {format_price(price_mc)} | Mispricing: {(price_mc - mid_market) if not np.isnan(mid_market) else 'N/A'}")
                st.write(f"Binomial Tree: {format_price(price_binomial)} | Mispricing: {(price_binomial - mid_market) if not np.isnan(mid_market) else 'N/A'}")
                st.write(f"Finite Difference PDE: {format_price(price_pde)} | Mispricing: {(price_pde - mid_market) if not np.isnan(mid_market) else 'N/A'}")

                st.subheader("Greeks (Monte Carlo Approximation)")
                st.json(greeks)

                # Plot simulated paths
                fig_paths = go.Figure()
                for i in range(min(100, paths.shape[1])):
                    fig_paths.add_trace(go.Scatter(
                        y=paths[:, i],
                        mode="lines",
                        line=dict(width=1, color="blue"),
                        opacity=0.3,
                        showlegend=False
                    ))
                fig_paths.add_hline(y=strike_price, line_dash="dash", line_color="red", annotation_text="Strike Price")
                fig_paths.update_layout(title="Simulated Stock Price Paths", xaxis_title="Time Steps", yaxis_title="Price")
                st.plotly_chart(fig_paths, use_container_width=True)

                # Plot smoothed exercise boundary
                if boundary:
                    times, prices = zip(*boundary)
                    fig_boundary = go.Figure()
                    fig_boundary.add_trace(go.Scatter(
                        x=times, y=prices, mode="lines+markers", name="Exercise Boundary", line=dict(color="green")
                    ))
                    fig_boundary.add_hline(y=strike_price, line_dash="dash", line_color="red", annotation_text="Strike Price")
                    fig_boundary.update_layout(title="Exercise Boundary Over Time", xaxis_title="Time (Years)", yaxis_title="Price")
                    st.plotly_chart(fig_boundary, use_container_width=True)

# --- Collapsible Learn More Panel with Dynamic Insights ---
with st.expander("Learn More: Option Pricing, Greeks & Strategies"):
    st.markdown("### American Options Overview")
    st.write(
        "American options can be exercised anytime before expiry, making them more flexible than European options. "
        "This flexibility requires advanced pricing models like Monte Carlo (LSM), Binomial Trees, or PDE solvers."
    )

    # --- Dynamic Payoff Graph ---
    st.markdown("### Payoff Diagram (Your Selection)")
    S_range = np.linspace(0.5 * spot_price, 1.5 * spot_price, 100)

    # Basic payoff calculation
    if option_type == "call":
        payoff = np.maximum(S_range - strike_price, 0)
        breakeven = strike_price + mid_market if not np.isnan(mid_market) else None
    else:
        payoff = np.maximum(strike_price - S_range, 0)
        breakeven = strike_price - mid_market if not np.isnan(mid_market) else None

    # Adjust for buy/sell positions
    if position_type == "sell":
        payoff = -payoff
        if breakeven is not None:
            breakeven = strike_price - mid_market if option_type == "call" else strike_price + mid_market

    # Plot payoff diagram
    payoff_fig = go.Figure()
    
    #Payoff Curve
    payoff_fig.add_trace(go.Scatter(x=S_range, y=payoff,
                                    mode ="lines", name="Payoff",
                                    line=dict(color="green", width=2)
    ))

    #Strike Line
    payoff_fig.add_vline(
        x=strike_price, line_dash="dash", line_color="red",annotation_text="Strike"
    )

    #Current spot price marker
    payoff_fig.add_vline(
        x=spot_price, line_dash="dot", line_color="blue",
        annotation_text="Current Spot Price"
    )

    #BE Line
    if breakeven:
        payoff_fig.add_vline(
            x=breakeven, line_dash="dot", line_color="orange",
            annotation_text="Breakeven"
        )
    #Highlight current P/L
    current_pl = 0
    if option_type == "call":
        current_pl = max(spot_price - strike_price,0)
    else:
        current_pl = max(strike_price - spot_price,0)
    
    if position_type == "sell":
        current_pl = -current_pl
    payoff_fig.add_annotation(
        x=spot_price, y=current_pl,
        text=f"Current P&L: {current_pl:.2f}",
        showarrow=True, arrowhead=2, arrowcolor="blue"
    )
    
    payoff_fig.update_layout(
        title="Option Payoff at Expiry (With BE & Current P&L)",
        xaxis_title="Underlying Price",
        yaxis_title="P&L"
    )
    st.plotly_chart(payoff_fig, use_container_width=True)

    # --- Probability of Profit (POP) Calculation using Monte Carlo ---
st.markdown("### Probability of Profit (POP)")

if paths is not None:
    terminal_prices = paths[-1]  # last timestep of all simulated paths

    # Calculate payoff at expiry
    if option_type == "call":
        payoffs = np.maximum(terminal_prices - strike_price, 0)
    else:
        payoffs = np.maximum(strike_price - terminal_prices, 0)

    # Adjust for position (sell = negative P/L)
    if position_type == "sell":
        payoffs = -payoffs

    # Subtract mid_market premium if buying
    if not np.isnan(mid_market):
        if position_type == "buy":
            payoffs -= mid_market
        else:
            payoffs += mid_market  # seller receives premium

    # Probability of profit = % of paths with payoff ≥ 0
    pop = np.mean(payoffs >= 0) * 100

    st.write(f"Estimated Probability of Profit: **{pop:.2f}%**")

    st.caption(
        "POP estimates the chance of ending in profit at expiry based on simulated price paths. "
        "Use it with caution — it assumes current volatility and drift stay constant."
    )
else:
    st.warning("Monte Carlo paths not available for POP calculation.")

    # --- Greeks Recap (Dynamic) ---
    st.markdown("### Meaning Of Greeks")
    st.write(
        f"**Delta ({greeks['Delta']:.2f})**: Change in option price per $1 move in underlying.\n"
        f"**Gamma ({greeks['Gamma']:.4f})**: Rate of change of Delta — higher means more rebalancing needed.\n"
        f"**Vega ({greeks['Vega']:.2f})**: Sensitivity to volatility changes (positive = gains with rising vol).\n"
        f"**Theta ({greeks['Theta']:.2f})**: Time decay (negative = loses value as time passes).\n"
        f"**Rho ({greeks['Rho']:.2f})**: Sensitivity to interest rate changes."
    )

    # --- Dynamic Trading Insights based on Greeks ---
    st.markdown("### Trading Insights")

    insights = []
    # Gamma assessment
    if abs(greeks['Gamma']) > 0.02:
        insights.append("**High Gamma:** Expect large Delta swings — risky for naked positions, hedge actively.")
    else:
        insights.append("**Moderate Gamma:** Delta shifts are stable — less urgent rebalancing needed.")

    # Theta assessment
    if greeks['Theta'] < -0.5:
        insights.append("**High Negative Theta:** Time decay will erode value fast — favor short premium strategies.")
    else:
        insights.append("**Mild Theta:** Less time decay risk — suitable for holding longer.")

    # Vega assessment
    if greeks['Vega'] > 0.5:
        insights.append("**High Vega:** Benefits from volatility spikes (e.g., earnings events).")
    elif greeks['Vega'] < -0.5:
        insights.append("**Negative Vega:** Loses value when volatility rises — risky in turbulent markets.")

    st.info("\n".join(insights))

    # --- Strategy Suggestion ---
    st.markdown("### Strategy Suggestion")

    strategy = ""
    # Long Call / Long Put
    if position_type == "buy" and option_type == "call":
        if greeks['Theta'] < -0.5:
            strategy = "Consider a **bull call spread** to reduce time decay risk."
        elif greeks['Vega'] > 0.5:
            strategy = "Long call suits **volatility breakout trades**; hedge if volatility drops."
        else:
            strategy = "Simple long call works — watch Delta for scaling."
    elif position_type == "buy" and option_type == "put":
        if greeks['Gamma'] > 0.02:
            strategy = "Protective put for hedging; high Gamma means strong downside protection near strike."
        else:
            strategy = "Consider **bear put spread** to reduce premium cost."

    # Short Call / Short Put
    elif position_type == "sell" and option_type == "call":
        if greeks['Theta'] > 0:
            strategy = "Covered call is ideal — collect premium and benefit from Theta decay."
        else:
            strategy = "Naked call — high risk. Ensure adequate margin and hedge Delta."
    elif position_type == "sell" and option_type == "put":
        if greeks['Theta'] > 0:
            strategy = "Cash-secured put can generate income — be ready to buy stock if assigned."
        else:
            strategy = "Avoid naked puts in volatile conditions — Vega exposure is high."

    st.success(strategy)

    st.markdown("---")
    st.caption(
        "Tip: Strategies adapt based on your risk tolerance. Combine mispricing signals, Greeks, and payoff visualization "
        "to decide between directional plays (long calls/puts) or income strategies (covered calls, cash-secured puts)."
    )
