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
