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
from ai_optimizer import AIStrategyOptimizer

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
    # Header with gradient background
    st.markdown("""
    <div style="background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">🤖 AI-Powered Strategy Optimization</h1>
        <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
            Harness machine learning to discover optimal trading parameters
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content in tabs for better organization
    tab1, tab2, tab3 = st.tabs(["🚀 AI Optimizer", "📊 Results Dashboard", "📈 Performance Analysis"])
    
    with tab1:
        # Configuration section
        st.markdown("### ⚙️ Configuration")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            ai_symbol = st.text_input(
                "📈 Stock Symbol", 
                value="AAPL", 
                key="ai_symbol",
                help="Enter any stock ticker (e.g., AAPL, TSLA, MSFT)"
            )
        
        with col2:
            optimization_method = st.selectbox(
                "🧠 AI Method",
                ["🤖 Machine Learning Predictor", "⚡ Hyperparameter Optimization", "🔬 Both Methods"],
                help="Choose your AI optimization approach"
            )
        
        with col3:
            if optimization_method in ["⚡ Hyperparameter Optimization", "🔬 Both Methods"]:
                n_trials = st.number_input("🎯 Trials", min_value=10, max_value=100, value=30)
            else:
                n_trials = 30
        
        st.markdown("---")
        
        # AI Method explanations with cards
        st.markdown("### 🧠 AI Methods Overview")
        
        method_col1, method_col2, method_col3 = st.columns(3)
        
        with method_col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                        padding: 1.5rem; border-radius: 10px; height: 200px;">
                <h4 style="color: #1976d2; margin-top: 0;">🤖 ML Predictor</h4>
                <p style="color: #424242; font-size: 0.9rem;">
                    • Analyzes 10+ market indicators<br>
                    • Trains Random Forest models<br>
                    • Predicts optimal parameters<br>
                    • Provides confidence scores
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with method_col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffcc80 100%); 
                        padding: 1.5rem; border-radius: 10px; height: 200px;">
                <h4 style="color: #f57c00; margin-top: 0;">⚡ Hyperparameter</h4>
                <p style="color: #424242; font-size: 0.9rem;">
                    • Uses Optuna TPE algorithm<br>
                    • Intelligent parameter search<br>
                    • Multi-objective optimization<br>
                    • Global optimum discovery
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with method_col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); 
                        padding: 1.5rem; border-radius: 10px; height: 200px;">
                <h4 style="color: #388e3c; margin-top: 0;">🔬 Both Methods</h4>
                <p style="color: #424242; font-size: 0.9rem;">
                    • Combines ML + Optimization<br>
                    • Cross-validation approach<br>
                    • Maximum accuracy<br>
                    • Comprehensive analysis
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Launch button with enhanced styling
        col_center = st.columns([1, 2, 1])[1]
        with col_center:
            if st.button("🚀 Launch AI Optimization", type="primary", use_container_width=True, key="launch_ai_optimization"):
                with st.spinner("🤖 AI is analyzing market patterns and optimizing parameters..."):
                    try:
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Initialize AI optimizer
                        status_text.text("🔄 Initializing AI optimizer...")
                        progress_bar.progress(10)
                        optimizer = AIStrategyOptimizer(ai_symbol.upper())
                        
                        # Load data
                        status_text.text("📊 Loading market data...")
                        progress_bar.progress(20)
                        optimizer.load_market_data()
                        st.success(f"✅ Loaded market data for {ai_symbol.upper()}")
                        
                        results = {}
                        
                        # ML Prediction
                        if optimization_method in ["🤖 Machine Learning Predictor", "🔬 Both Methods"]:
                            status_text.text("🧠 Training AI model on market patterns...")
                            progress_bar.progress(40)
                            training_results = optimizer.train_ml_predictor()
                            
                            status_text.text("🔮 Generating AI predictions...")
                            progress_bar.progress(60)
                            ai_predictions = optimizer.predict_optimal_parameters()
                            results['ai_predictions'] = ai_predictions
                            results['training_results'] = training_results
                        
                        # Hyperparameter Optimization
                        if optimization_method in ["⚡ Hyperparameter Optimization", "🔬 Both Methods"]:
                            status_text.text("⚡ Running hyperparameter optimization...")
                            progress_bar.progress(70)
                            optuna_results = optimizer.hyperparameter_optimization(n_trials)
                            results['optuna_results'] = optuna_results
                        
                        # Generate AI insights
                        status_text.text("🧠 Generating AI insights...")
                        progress_bar.progress(90)
                        if 'optuna_results' in results:
                            insights = optimizer.generate_ai_insights(results['optuna_results'])
                            results['insights'] = insights
                        
                        # Complete
                        progress_bar.progress(100)
                        status_text.text("✅ AI optimization completed!")
                        
                        # Store results in session state
                        st.session_state.ai_optimization_results = results
                        st.success("🎯 AI optimization completed successfully! Check the Results Dashboard tab.")
                        
                    except Exception as e:
                        st.error(f"❌ AI optimization failed: {str(e)}")
    
    with tab2:
        st.markdown("### 📊 AI Optimization Dashboard")
        
        if 'ai_optimization_results' in st.session_state:
            results = st.session_state.ai_optimization_results
            
            # Summary cards
            st.markdown("#### 📈 Optimization Summary")
            
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                if 'ai_predictions' in results:
                    ai_pred = results['ai_predictions']
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 100%); 
                                padding: 1rem; border-radius: 10px; text-align: center; color: white;">
                        <h3 style="margin: 0; font-size: 1.5rem;">{ai_pred['short_ma']}</h3>
                        <p style="margin: 0; opacity: 0.9;">🤖 AI Short MA</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Run ML Predictor")
            
            with summary_col2:
                if 'ai_predictions' in results:
                    ai_pred = results['ai_predictions']
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%); 
                                padding: 1rem; border-radius: 10px; text-align: center; color: white;">
                        <h3 style="margin: 0; font-size: 1.5rem;">{ai_pred['long_ma']}</h3>
                        <p style="margin: 0; opacity: 0.9;">🤖 AI Long MA</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Run ML Predictor")
            
            with summary_col3:
                if 'optuna_results' in results:
                    optuna_res = results['optuna_results']
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%); 
                                padding: 1rem; border-radius: 10px; text-align: center; color: white;">
                        <h3 style="margin: 0; font-size: 1.5rem;">{optuna_res['best_short_ma']}</h3>
                        <p style="margin: 0; opacity: 0.9;">⚡ Optuna Short</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Run Optimization")
            
            with summary_col4:
                if 'optuna_results' in results:
                    optuna_res = results['optuna_results']
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #e91e63 0%, #c2185b 100%); 
                                padding: 1rem; border-radius: 10px; text-align: center; color: white;">
                        <h3 style="margin: 0; font-size: 1.5rem;">{optuna_res['best_long_ma']}</h3>
                        <p style="margin: 0; opacity: 0.9;">⚡ Optuna Long</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Run Optimization")
            
            st.markdown("---")
            
            # Detailed results in expandable sections
            if 'ai_predictions' in results:
                with st.expander("🤖 AI Model Predictions", expanded=True):
                    ai_pred = results['ai_predictions']
                    training_res = results.get('training_results', {})
                    
                    pred_col1, pred_col2 = st.columns(2)
                    
                    with pred_col1:
                        st.metric("AI Short MA", ai_pred['short_ma'])
                        st.metric("AI Long MA", ai_pred['long_ma'])
                        st.metric("Confidence Score", f"{ai_pred['confidence']:.1%}")
                    
                    with pred_col2:
                        if training_res:
                            st.metric("Training Samples", training_res.get('training_samples', 'N/A'))
                            st.metric("Model Accuracy", f"{training_res.get('performance_r2', 0):.3f}")
                            st.metric("Expected Score", f"{ai_pred.get('expected_score', 0):.3f}")
            
            if 'optuna_results' in results:
                with st.expander("⚡ Hyperparameter Optimization Results", expanded=True):
                    optuna_res = results['optuna_results']
                    
                    opt_col1, opt_col2 = st.columns(2)
                    
                    with opt_col1:
                        st.metric("Best Short MA", optuna_res['best_short_ma'])
                        st.metric("Best Long MA", optuna_res['best_long_ma'])
                        st.metric("Optimization Score", f"{optuna_res['best_score']:.3f}")
                    
                    with opt_col2:
                        st.metric("Trials Completed", optuna_res['n_trials'])
                        st.metric("Method", optuna_res['optimization_method'])
                        
                        # Performance indicator
                        score = optuna_res['best_score']
                        if score > 0.5:
                            st.success("🎯 Excellent Performance")
                        elif score > 0.2:
                            st.warning("📊 Good Performance")
                        else:
                            st.info("⚠️ Moderate Performance")
            
            # AI insights with better formatting
            if 'insights' in results:
                st.markdown("#### 🧠 AI Market Intelligence")
                
                for i, insight in enumerate(results['insights']):
                    if "HIGH" in insight:
                        st.success(insight)
                    elif "MEDIUM" in insight:
                        st.warning(insight)
                    elif "LOW" in insight or "challenging" in insight.lower():
                        st.error(insight)
                    else:
                        st.info(insight)
            
            # Action buttons
            st.markdown("---")
            st.markdown("#### 🎯 Next Steps")
            
            action_col1, action_col2 = st.columns(2)
            
            with action_col1:
                if 'optuna_results' in results:
                    if st.button("🚀 Run AI Backtest", type="primary", use_container_width=True, key="ai_backtest_btn"):
                        with st.spinner("Running backtest with AI-optimized parameters..."):
                            try:
                                optuna_res = results['optuna_results']
                                
                                # Run backtest with AI parameters
                                df = load_data(ai_symbol.upper())
                                strategy = MovingAverageStrategy(
                                    short_window=optuna_res['best_short_ma'],
                                    long_window=optuna_res['best_long_ma']
                                )
                                backtester = Backtester(initial_cash=100000.0, commission=1.0)
                                result = backtester.run(strategy, df, symbol=ai_symbol.upper())
                                
                                # Store AI backtest results
                                st.session_state.ai_backtest_results = result
                                st.session_state.ai_backtest_data = df
                                
                                st.success("🎉 AI backtest completed! Check Performance Analysis tab.")
                                
                            except Exception as e:
                                st.error(f"❌ AI backtest failed: {str(e)}")
            
            with action_col2:
                if st.button("🔄 Run New Optimization", use_container_width=True, key="new_optimization_btn"):
                    # Clear results and go back to optimizer
                    if 'ai_optimization_results' in st.session_state:
                        del st.session_state.ai_optimization_results
                    if 'ai_backtest_results' in st.session_state:
                        del st.session_state.ai_backtest_results
                    if 'ai_backtest_data' in st.session_state:
                        del st.session_state.ai_backtest_data
                    st.rerun()
        
        else:
            # Empty state with call to action
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%); 
                        border-radius: 15px; margin: 2rem 0;">
                <h3 style="color: #666;">🤖 No AI Results Yet</h3>
                <p style="color: #888; margin-bottom: 2rem;">
                    Run AI optimization to see intelligent parameter recommendations and market insights.
                </p>
                <p style="color: #999;">👈 Go to the AI Optimizer tab to get started!</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 📈 AI Performance Analysis")
        
        if 'ai_backtest_results' in st.session_state:
            result = st.session_state.ai_backtest_results
            data = st.session_state.ai_backtest_data
            
            # Performance summary
            if hasattr(result, 'enhanced_metrics'):
                metrics = result.enhanced_metrics
                
                st.markdown("#### 🎯 AI Strategy Performance")
                
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                
                with perf_col1:
                    return_pct = metrics.get('total_return_pct', 0)
                    delta_color = "normal" if return_pct >= 0 else "inverse"
                    st.metric("Total Return", f"{return_pct:.2f}%", delta=f"{return_pct:.2f}%")
                
                with perf_col2:
                    sharpe = metrics.get('sharpe_ratio', 0)
                    st.metric("Sharpe Ratio", f"{sharpe:.3f}", delta=f"{sharpe:.3f}")
                
                with perf_col3:
                    drawdown = metrics.get('max_drawdown_pct', 0)
                    st.metric("Max Drawdown", f"{drawdown:.2f}%", delta=f"{drawdown:.2f}%")
                
                with perf_col4:
                    win_rate = metrics.get('win_rate_pct', 0)
                    st.metric("Win Rate", f"{win_rate:.2f}%", delta=f"{win_rate:.2f}%")
            
            # Interactive charts
            st.markdown("#### 📊 Interactive Charts")
            
            chart_tab1, chart_tab2 = st.tabs(["📈 Price & AI Signals", "💰 AI Equity Curve"])
            
            with chart_tab1:
                if not result.signals.empty:
                    fig_price = create_price_chart(data, result.signals, ai_symbol.upper())
                    fig_price.update_layout(title=f"🤖 AI-Optimized Strategy: {ai_symbol.upper()}")
                    st.plotly_chart(fig_price, use_container_width=True)
            
            with chart_tab2:
                if not result.equity_df.empty:
                    fig_equity = create_equity_curve(result.equity_df, result.performance)
                    fig_equity.update_layout(title="🤖 AI-Optimized Portfolio Performance")
                    st.plotly_chart(fig_equity, use_container_width=True)
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                        border-radius: 15px; margin: 2rem 0;">
                <h3 style="color: #1976d2;">📈 No Performance Data</h3>
                <p style="color: #424242; margin-bottom: 2rem;">
                    Run AI optimization and backtest to see detailed performance analysis with interactive charts.
                </p>
                <p style="color: #666;">🚀 Complete the AI optimization process first!</p>
            </div>
            """, unsafe_allow_html=True)

def show_analysis():
    st.markdown("## 📈 Advanced Analysis")
    st.info("🚧 Advanced analysis tools coming soon! This will include correlation analysis, risk metrics, and portfolio optimization.")

if __name__ == "__main__":
    main()
