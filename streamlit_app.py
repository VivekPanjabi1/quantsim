import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime
import concurrent.futures
from streamlit_option_menu import option_menu

# Import your backtesting modules
from engine.data_loader import load_data
from engine.backtester import Backtester
from strategies.moving_average import MovingAverageStrategy, ExponentialMovingAverageStrategy

# Page configuration
st.set_page_config(
    page_title="QuantSim Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #66bb6a, #4caf50, #81c784);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Navigation menu improvements */
    .nav-menu {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        border-radius: 15px;
        padding: 10px;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .success-metric {
        background: linear-gradient(135deg, #00c851 0%, #007e33 100%);
    }
    
    .warning-metric {
        background: linear-gradient(135deg, #ffbb33 0%, #ff8800 100%);
    }
    
    .danger-metric {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
    }
    
    /* Sidebar improvements */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        border-radius: 10px;
    }
    
    /* Section headers */
    .section-header {
        color: #66bb6a;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #66bb6a;
    }
    
    /* Button improvements */
    .stButton > button {
        background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(102, 187, 106, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 187, 106, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #262730;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #fafafa;
        font-weight: bold;
        padding: 12px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #66bb6a !important;
        color: white !important;
        box-shadow: 0 2px 4px rgba(102, 187, 106, 0.3);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        border-radius: 10px;
        border: none;
        color: white;
    }
    
    /* Success boxes */
    .stSuccess {
        background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
        border-radius: 10px;
        border: none;
        color: white;
    }
    
    /* Force navigation icons to be visible */
    .nav-link i, .nav-link .icon {
        color: #ffffff !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    /* Ensure icons are visible in both states */
    [data-testid="stHorizontalBlock"] i {
        color: #ffffff !important;
        font-size: 20px !important;
    }
    
    /* Option menu icon styling */
    .nav-item i, .nav-item .fa, .nav-item .fas, .nav-item .far {
        color: #ffffff !important;
        font-size: 20px !important;
        margin-right: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None

def create_price_chart(data, signals, symbol):
    """Create interactive price chart with buy/sell signals."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[f'{symbol} Price & Signals', 'Volume', 'Moving Averages'],
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # Buy signals
    buy_signals = signals[signals == 1]
    if len(buy_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=data.loc[buy_signals.index, 'Close'],
                mode='markers',
                marker=dict(symbol='triangle-up', size=15, color='lime'),
                name='Buy Signal',
                hovertemplate='<b>BUY</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Sell signals
    sell_signals = signals[signals == -1]
    if len(sell_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=data.loc[sell_signals.index, 'Close'],
                mode='markers',
                marker=dict(symbol='triangle-down', size=15, color='red'),
                name='Sell Signal',
                hovertemplate='<b>SELL</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Volume
    colors = ['red' if close < open else 'green' for close, open in zip(data['Close'], data['Open'])]
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors, opacity=0.6),
        row=2, col=1
    )
    
    # Moving averages (if available)
    if 'MA_10' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MA_10'], name='MA 10', line=dict(color='orange', width=2)),
            row=3, col=1
        )
    if 'MA_30' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MA_30'], name='MA 30', line=dict(color='purple', width=2)),
            row=3, col=1
        )
    
    fig.update_layout(
        title=f'{symbol} Trading Analysis',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        template='plotly_dark'
    )
    
    return fig

def create_equity_curve(equity_df, performance):
    """Create interactive equity curve chart."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Portfolio Equity Curve', 'Drawdown Analysis'],
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity_df['timestamp'],
            y=equity_df['total_equity'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00ff88', width=3),
            hovertemplate='Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Initial capital line
    initial_cash = performance['initial_cash']
    fig.add_hline(
        y=initial_cash,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Initial Capital: ${initial_cash:,.0f}",
        row=1, col=1
    )
    
    # Drawdown calculation
    equity_series = equity_df.set_index('timestamp')['total_equity']
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max * 100
    
    # Drawdown chart
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            mode='lines',
            fill='tonexty',
            name='Drawdown %',
            line=dict(color='red', width=2),
            fillcolor='rgba(255, 0, 0, 0.3)',
            hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_hline(y=0, line_color="gray", line_dash="dash", row=2, col=1)
    
    fig.update_layout(
        title='Portfolio Performance Analysis',
        height=600,
        showlegend=True,
        template='plotly_dark'
    )
    
    return fig

def create_performance_metrics_chart(metrics):
    """Create performance metrics visualization."""
    # Metrics for radar chart
    categories = ['Return', 'Sharpe Ratio', 'Win Rate', 'Profit Factor', 'Calmar Ratio']
    
    # Normalize values for radar chart (0-100 scale)
    values = [
        min(max(metrics.get('total_return_pct', 0) * 2, 0), 100),  # Return * 2 for scaling
        min(max(metrics.get('sharpe_ratio', 0) * 20, 0), 100),    # Sharpe * 20
        metrics.get('win_rate_pct', 0),                           # Win rate already in %
        min(max(metrics.get('profit_factor', 0) * 20, 0), 100),  # Profit factor * 20
        min(max(metrics.get('calmar_ratio', 0) * 20, 0), 100)    # Calmar * 20
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Performance',
        line_color='#00ff88',
        fillcolor='rgba(0, 255, 136, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Performance Radar Chart",
        template='plotly_dark'
    )
    
    return fig

def run_single_backtest(symbol, strategy_type, short_ma, long_ma, initial_cash, commission, max_position):
    """Run a single backtest."""
    try:
        # Load data
        df = load_data(symbol)
        
        # Create strategy
        if strategy_type == "Moving Average":
            strategy = MovingAverageStrategy(short_window=short_ma, long_window=long_ma)
        else:
            strategy = ExponentialMovingAverageStrategy(short_window=short_ma, long_window=long_ma)
        
        # Create backtester
        backtester = Backtester(
            initial_cash=initial_cash,
            commission=commission,
            max_position_size=max_position
        )
        
        # Run backtest
        result = backtester.run(strategy, df, symbol=symbol)
        
        return result, df
        
    except Exception as e:
        st.error(f"Backtest failed: {str(e)}")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 QuantSim Pro - Advanced Backtesting Platform</h1>', unsafe_allow_html=True)
    
    # Navigation menu
    selected = option_menu(
        menu_title=None,
        options=["🏠 Dashboard", "📊 Single Backtest", "🔧 Optimization", "📈 Analysis"],
        icons=["house", "graph-up", "gear", "bar-chart"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0!important", 
                "background-color": "#262730",
                "border-radius": "10px",
                "margin": "10px 0px"
            },
            "icon": {
                "color": "#66bb6a !important", 
                "font-size": "22px !important",
                "margin-right": "10px !important",
                "opacity": "1 !important",
                "visibility": "visible !important"
            },
            "nav-link": {
                "font-size": "16px", 
                "text-align": "center", 
                "margin": "0px",
                "color": "#ffffff",
                "background-color": "transparent",
                "border-radius": "8px",
                "padding": "12px 16px",
                "--hover-color": "#404040",
                "display": "flex",
                "align-items": "center",
                "justify-content": "center"
            },
            "nav-link-selected": {
                "background-color": "#66bb6a",
                "color": "white",
                "font-weight": "bold",
                "border": "2px solid #4caf50",
                "box-shadow": "0 4px 8px rgba(102, 187, 106, 0.4)"
            },
        }
    )
    
    if selected == "🏠 Dashboard":
        show_dashboard()
    elif selected == "📊 Single Backtest":
        show_single_backtest()
    elif selected == "🔧 Optimization":
        show_optimization()
    elif selected == "📈 Analysis":
        show_analysis()

def show_dashboard():
    st.markdown("## 🏠 Welcome to QuantSim Pro")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🚀 Features
        - **Professional Backtesting Engine**
        - **Interactive Charts & Visualizations**
        - **Real-time Performance Metrics**
        - **Parameter Optimization**
        - **Export & Analysis Tools**
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Supported Strategies
        - **Moving Average Crossover**
        - **Exponential Moving Average**
        - **Custom Strategy Framework**
        - **Multi-timeframe Analysis**
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 Quick Start
        1. Go to **Single Backtest** tab
        2. Select your stock symbol
        3. Configure strategy parameters
        4. Run backtest and analyze results
        5. Optimize parameters for best performance
        """)
    
    # Sample performance metrics
    if st.session_state.backtest_results:
        st.markdown("## 📈 Latest Backtest Results")
        result = st.session_state.backtest_results
        
        if hasattr(result, 'enhanced_metrics'):
            metrics = result.enhanced_metrics
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", f"{metrics.get('total_return_pct', 0):.2f}%", 
                         delta=f"{metrics.get('total_return_pct', 0):.2f}%")
            
            with col2:
                st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}",
                         delta=f"{metrics.get('sharpe_ratio', 0):.3f}")
            
            with col3:
                st.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%",
                         delta=f"{metrics.get('max_drawdown_pct', 0):.2f}%")
            
            with col4:
                st.metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.2f}%",
                         delta=f"{metrics.get('win_rate_pct', 0):.2f}%")

def show_single_backtest():
    st.markdown("## 📊 Single Strategy Backtest")
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Backtest Configuration")
        
        # Stock selection
        symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter stock ticker (e.g., AAPL, MSFT, GOOGL)")
        
        # Strategy selection
        strategy_type = st.selectbox("Strategy Type", ["Moving Average", "Exponential Moving Average"])
        
        # Strategy parameters
        st.markdown("#### Strategy Parameters")
        short_ma = st.slider("Short MA Period", 1, 50, 10)
        long_ma = st.slider("Long MA Period", 20, 200, 30)
        
        # Backtester settings
        st.markdown("#### Backtester Settings")
        initial_cash = st.number_input("Initial Cash ($)", value=100000.0, min_value=1000.0)
        commission = st.number_input("Commission per Trade ($)", value=1.0, min_value=0.0)
        max_position = st.slider("Max Position Size (%)", 0.1, 1.0, 0.25, 0.05)
        
        # Run backtest button
        if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
            with st.spinner("Running backtest..."):
                result, data = run_single_backtest(symbol, strategy_type, short_ma, long_ma, 
                                                 initial_cash, commission, max_position)
                
                if result:
                    st.session_state.backtest_results = result
                    st.session_state.backtest_data = data
                    st.success("Backtest completed successfully!")
                    st.rerun()
    
    # Main content area
    if st.session_state.backtest_results:
        result = st.session_state.backtest_results
        data = st.session_state.backtest_data
        
        # Performance metrics
        if hasattr(result, 'enhanced_metrics'):
            metrics = result.enhanced_metrics
            
            st.markdown("### 📊 Performance Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Return", f"{metrics.get('total_return_pct', 0):.2f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")
            with col3:
                st.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
            with col4:
                st.metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.2f}%")
            with col5:
                st.metric("Total Trades", f"{metrics.get('total_trades', 0)}")
            
            # Charts
            tab1, tab2, tab3 = st.tabs(["��� Price & Signals", "💰 Equity Curve", "🎯 Performance Radar"])
            
            with tab1:
                if not result.signals.empty:
                    fig_price = create_price_chart(data, result.signals, symbol)
                    st.plotly_chart(fig_price, use_container_width=True)
            
            with tab2:
                if not result.equity_df.empty:
                    fig_equity = create_equity_curve(result.equity_df, result.performance)
                    st.plotly_chart(fig_equity, use_container_width=True)
            
            with tab3:
                fig_radar = create_performance_metrics_chart(metrics)
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Detailed metrics
            with st.expander("📋 Detailed Performance Metrics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Return Metrics")
                    st.write(f"**Total Return:** {metrics.get('total_return_pct', 0):.2f}%")
                    st.write(f"**Annualized Return:** {metrics.get('annualized_return_pct', 0):.2f}%")
                    st.write(f"**Volatility:** {metrics.get('volatility_pct', 0):.2f}%")
                    
                    st.markdown("#### Risk Metrics")
                    st.write(f"**Sharpe Ratio:** {metrics.get('sharpe_ratio', 0):.3f}")
                    st.write(f"**Sortino Ratio:** {metrics.get('sortino_ratio', 0):.3f}")
                    st.write(f"**Calmar Ratio:** {metrics.get('calmar_ratio', 0):.3f}")
                
                with col2:
                    st.markdown("#### Trading Metrics")
                    st.write(f"**Win Rate:** {metrics.get('win_rate_pct', 0):.2f}%")
                    st.write(f"**Profit Factor:** {metrics.get('profit_factor', 0):.2f}")
                    st.write(f"**Average Win:** ${metrics.get('avg_win', 0):.2f}")
                    st.write(f"**Average Loss:** ${metrics.get('avg_loss', 0):.2f}")
                    
                    st.markdown("#### Portfolio Summary")
                    st.write(f"**Initial Capital:** ${result.performance['initial_cash']:,.2f}")
                    st.write(f"**Final Value:** ${result.equity_df['total_equity'].iloc[-1]:,.2f}")
                    st.write(f"**Total Commission:** ${result.performance['total_commission']:,.2f}")
            
            # Trade log
            with st.expander("📝 Trade Log"):
                if not result.trades_df.empty:
                    st.dataframe(result.trades_df.head(50), use_container_width=True)
                else:
                    st.info("No trades executed in this backtest.")
    
    else:
        st.info("👈 Configure your backtest parameters in the sidebar and click 'Run Backtest' to get started!")

def show_optimization():
    st.markdown("## 🔧 Parameter Optimization")
    st.info("🚧 Parameter optimization dashboard coming soon! This will allow you to test multiple parameter combinations automatically.")

def show_analysis():
    st.markdown("## 📈 Advanced Analysis")
    st.info("🚧 Advanced analysis tools coming soon! This will include correlation analysis, risk metrics, and portfolio optimization.")

if __name__ == "__main__":
    main()
