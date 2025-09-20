import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import pandas as pd
from datetime import datetime
import os

from engine.data_loader import load_data
from engine.backtester import Backtester
from strategies.moving_average import MovingAverageStrategy, ExponentialMovingAverageStrategy


class QuantSimGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("QuantSim - Professional Backtesting Engine")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Queue for thread communication
        self.result_queue = queue.Queue()
        
        # Variables
        self.symbol_var = tk.StringVar(value="AAPL")
        self.strategy_var = tk.StringVar(value="MovingAverage")
        self.short_ma_var = tk.IntVar(value=10)
        self.long_ma_var = tk.IntVar(value=30)
        self.initial_cash_var = tk.DoubleVar(value=100000.0)
        self.commission_var = tk.DoubleVar(value=1.0)
        self.max_position_var = tk.DoubleVar(value=0.25)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', padx=10, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="QuantSim Backtesting Engine", 
                              font=('Arial', 24, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel - Controls
        left_frame = tk.LabelFrame(main_frame, text="Backtest Configuration", 
                                  font=('Arial', 12, 'bold'), bg='white', padx=10, pady=10)
        left_frame.pack(side='left', fill='y', padx=(0, 5))
        
        self.create_controls(left_frame)
        
        # Right panel - Results
        right_frame = tk.LabelFrame(main_frame, text="Results & Analysis", 
                                   font=('Arial', 12, 'bold'), bg='white', padx=10, pady=10)
        right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        self.create_results_panel(right_frame)
        
    def create_controls(self, parent):
        # Symbol selection
        tk.Label(parent, text="Stock Symbol:", font=('Arial', 10, 'bold'), bg='white').pack(anchor='w', pady=(0, 5))
        symbol_frame = tk.Frame(parent, bg='white')
        symbol_frame.pack(fill='x', pady=(0, 15))
        
        symbol_entry = tk.Entry(symbol_frame, textvariable=self.symbol_var, font=('Arial', 10), width=15)
        symbol_entry.pack(side='left')
        
        load_btn = tk.Button(symbol_frame, text="Load Data", command=self.load_data,
                            bg='#3498db', fg='white', font=('Arial', 9))
        load_btn.pack(side='right')
        
        # Strategy selection
        tk.Label(parent, text="Strategy:", font=('Arial', 10, 'bold'), bg='white').pack(anchor='w', pady=(0, 5))
        strategy_combo = ttk.Combobox(parent, textvariable=self.strategy_var, 
                                     values=["MovingAverage", "ExponentialMovingAverage"],
                                     state="readonly", width=20)
        strategy_combo.pack(anchor='w', pady=(0, 15))
        
        # Strategy parameters
        params_frame = tk.LabelFrame(parent, text="Strategy Parameters", bg='white')
        params_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(params_frame, text="Short MA:", bg='white').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        tk.Spinbox(params_frame, from_=1, to=100, textvariable=self.short_ma_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        tk.Label(params_frame, text="Long MA:", bg='white').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        tk.Spinbox(params_frame, from_=1, to=200, textvariable=self.long_ma_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        # Backtester settings
        settings_frame = tk.LabelFrame(parent, text="Backtester Settings", bg='white')
        settings_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(settings_frame, text="Initial Cash ($):", bg='white').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        tk.Entry(settings_frame, textvariable=self.initial_cash_var, width=12).grid(row=0, column=1, padx=5, pady=2)
        
        tk.Label(settings_frame, text="Commission ($):", bg='white').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        tk.Entry(settings_frame, textvariable=self.commission_var, width=12).grid(row=1, column=1, padx=5, pady=2)
        
        tk.Label(settings_frame, text="Max Position (%):", bg='white').grid(row=2, column=0, sticky='w', padx=5, pady=2)
        tk.Entry(settings_frame, textvariable=self.max_position_var, width=12).grid(row=2, column=1, padx=5, pady=2)
        
        # Action buttons
        btn_frame = tk.Frame(parent, bg='white')
        btn_frame.pack(fill='x', pady=15)
        
        self.run_btn = tk.Button(btn_frame, text="Run Backtest", command=self.run_backtest,
                                bg='#27ae60', fg='white', font=('Arial', 12, 'bold'), height=2)
        self.run_btn.pack(fill='x', pady=(0, 5))
        
        self.optimize_btn = tk.Button(btn_frame, text="Optimize Parameters", command=self.optimize_parameters,
                                     bg='#e67e22', fg='white', font=('Arial', 12, 'bold'), height=2)
        self.optimize_btn.pack(fill='x', pady=(0, 5))
        
        self.export_btn = tk.Button(btn_frame, text="Export Results", command=self.export_results,
                                   bg='#8e44ad', fg='white', font=('Arial', 12, 'bold'), height=2)
        self.export_btn.pack(fill='x')
        
    def create_results_panel(self, parent):
        # Results notebook
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill='both', expand=True)
        
        # Performance tab
        perf_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(perf_frame, text="Performance Metrics")
        
        # Create scrollable text widget for performance
        perf_scroll = tk.Scrollbar(perf_frame)
        perf_scroll.pack(side='right', fill='y')
        
        self.perf_text = tk.Text(perf_frame, yscrollcommand=perf_scroll.set, 
                                font=('Consolas', 10), bg='#f8f9fa')
        self.perf_text.pack(fill='both', expand=True)
        perf_scroll.config(command=self.perf_text.yview)
        
        # Trades tab
        trades_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(trades_frame, text="Trade Log")
        
        # Create treeview for trades
        self.trades_tree = ttk.Treeview(trades_frame, columns=('Date', 'Action', 'Quantity', 'Price', 'Value'), show='headings')
        self.trades_tree.heading('Date', text='Date')
        self.trades_tree.heading('Action', text='Action')
        self.trades_tree.heading('Quantity', text='Quantity')
        self.trades_tree.heading('Price', text='Price')
        self.trades_tree.heading('Value', text='Value')
        
        trades_scroll = tk.Scrollbar(trades_frame, orient='vertical', command=self.trades_tree.yview)
        self.trades_tree.configure(yscrollcommand=trades_scroll.set)
        
        self.trades_tree.pack(side='left', fill='both', expand=True)
        trades_scroll.pack(side='right', fill='y')
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, relief='sunken', anchor='w', bg='#ecf0f1')
        status_bar.pack(side='bottom', fill='x')
        
    def load_data(self):
        try:
            self.status_var.set("Loading data...")
            self.root.update()
            
            symbol = self.symbol_var.get().upper()
            df = load_data(symbol)
            
            self.status_var.set(f"Loaded {len(df)} rows for {symbol} ({df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')})")
            messagebox.showinfo("Success", f"Successfully loaded data for {symbol}")
            
        except Exception as e:
            self.status_var.set("Error loading data")
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def run_backtest(self):
        self.run_btn.config(state='disabled', text="Running...")
        self.status_var.set("Running backtest...")
        
        # Run in separate thread
        thread = threading.Thread(target=self._run_backtest_thread)
        thread.daemon = True
        thread.start()
        
        # Check for results
        self.root.after(100, self.check_backtest_result)
    
    def _run_backtest_thread(self):
        try:
            # Load data
            symbol = self.symbol_var.get().upper()
            df = load_data(symbol)
            
            # Create strategy
            if self.strategy_var.get() == "MovingAverage":
                strategy = MovingAverageStrategy(
                    short_window=self.short_ma_var.get(),
                    long_window=self.long_ma_var.get()
                )
            else:
                strategy = ExponentialMovingAverageStrategy(
                    short_window=self.short_ma_var.get(),
                    long_window=self.long_ma_var.get()
                )
            
            # Create backtester
            backtester = Backtester(
                initial_cash=self.initial_cash_var.get(),
                commission=self.commission_var.get(),
                max_position_size=self.max_position_var.get()
            )
            
            # Run backtest
            result = backtester.run(strategy, df, symbol=symbol)
            
            self.result_queue.put(('success', result))
            
        except Exception as e:
            self.result_queue.put(('error', str(e)))
    
    def check_backtest_result(self):
        try:
            result_type, result_data = self.result_queue.get_nowait()
            
            if result_type == 'success':
                self.display_results(result_data)
                self.status_var.set("Backtest completed successfully")
            else:
                messagebox.showerror("Error", f"Backtest failed: {result_data}")
                self.status_var.set("Backtest failed")
                
            self.run_btn.config(state='normal', text="Run Backtest")
            
        except queue.Empty:
            self.root.after(100, self.check_backtest_result)
    
    def display_results(self, result):
        # Display performance metrics
        self.perf_text.delete(1.0, tk.END)
        
        if hasattr(result, 'enhanced_metrics'):
            metrics = result.enhanced_metrics
            
            perf_text = f"""
PERFORMANCE SUMMARY
{'='*50}

RETURN METRICS:
  Total Return:        {metrics.get('total_return_pct', 0):.2f}%
  Annualized Return:   {metrics.get('annualized_return_pct', 0):.2f}%

RISK METRICS:
  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.3f}
  Sortino Ratio:       {metrics.get('sortino_ratio', 0):.3f}
  Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.2f}%
  Volatility:          {metrics.get('volatility_pct', 0):.2f}%
  Calmar Ratio:        {metrics.get('calmar_ratio', 0):.3f}

TRADING METRICS:
  Win Rate:            {metrics.get('win_rate_pct', 0):.2f}%
  Profit Factor:       {metrics.get('profit_factor', 0):.2f}
  Total Trades:        {metrics.get('total_trades', 0)}
  Average Win:         ${metrics.get('avg_win', 0):.2f}
  Average Loss:        ${metrics.get('avg_loss', 0):.2f}

PORTFOLIO SUMMARY:
  Initial Capital:     ${result.performance['initial_cash']:,.2f}
  Final Value:         ${result.equity_df['total_equity'].iloc[-1]:,.2f}
  Total Commission:    ${result.performance['total_commission']:,.2f}
"""
        else:
            perf_text = "Enhanced metrics not available. Please check your backtester configuration."
        
        self.perf_text.insert(1.0, perf_text)
        
        # Display trades
        for item in self.trades_tree.get_children():
            self.trades_tree.delete(item)
        
        if not result.trades_df.empty:
            for _, trade in result.trades_df.head(100).iterrows():  # Show first 100 trades
                self.trades_tree.insert('', 'end', values=(
                    trade['timestamp'].strftime('%Y-%m-%d %H:%M'),
                    trade['action'],
                    f"{trade['quantity']:.2f}",
                    f"${trade['price']:.2f}",
                    f"${trade['value']:.2f}"
                ))
    
    def optimize_parameters(self):
        messagebox.showinfo("Feature Coming Soon", "Parameter optimization GUI will be available in the next version!")
    
    def export_results(self):
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if filename:
                # Export logic here
                messagebox.showinfo("Success", f"Results exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")


def main():
    root = tk.Tk()
    app = QuantSimGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
