"""
AI-Powered Strategy Optimizer for QuantSim
Uses machine learning to find optimal trading strategy parameters.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from engine.data_loader import load_data
from engine.backtester import Backtester
from strategies.moving_average import MovingAverageStrategy


class AIStrategyOptimizer:
    """AI-powered strategy parameter optimization using ML and hyperparameter tuning."""
    
    def __init__(self, symbol: str, initial_cash: float = 100000.0):
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.data = None
        self.ml_model = None
        self.scaler = StandardScaler()
        self.optimization_history = []
        
    def load_market_data(self):
        """Load and prepare market data for optimization."""
        print(f"🔄 Loading market data for {self.symbol}...")
        self.data = load_data(self.symbol)
        
        # Add technical indicators for ML features
        self.data['SMA_5'] = self.data['Close'].rolling(5).mean()
        self.data['SMA_10'] = self.data['Close'].rolling(10).mean()
        self.data['SMA_20'] = self.data['Close'].rolling(20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(50).mean()
        self.data['RSI'] = self._calculate_rsi(self.data['Close'])
        self.data['Volatility'] = self.data['Close'].rolling(20).std()
        self.data['Volume_MA'] = self.data['Volume'].rolling(10).mean()
        
        # Drop NaN values
        self.data = self.data.dropna()
        print(f"✅ Loaded {len(self.data)} data points with technical indicators")
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def extract_market_features(self, lookback_days: int = 30) -> pd.DataFrame:
        """Extract market condition features for ML model."""
        features = []
        
        for i in range(lookback_days, len(self.data)):
            window_data = self.data.iloc[i-lookback_days:i]
            
            feature_row = {
                'volatility_mean': window_data['Volatility'].mean(),
                'volatility_std': window_data['Volatility'].std(),
                'rsi_mean': window_data['RSI'].mean(),
                'rsi_std': window_data['RSI'].std(),
                'volume_ratio': window_data['Volume'].mean() / window_data['Volume_MA'].mean(),
                'price_trend': (window_data['Close'].iloc[-1] - window_data['Close'].iloc[0]) / window_data['Close'].iloc[0],
                'sma_5_slope': (window_data['SMA_5'].iloc[-1] - window_data['SMA_5'].iloc[0]) / lookback_days,
                'sma_20_slope': (window_data['SMA_20'].iloc[-1] - window_data['SMA_20'].iloc[0]) / lookback_days,
                'price_above_sma20': int(window_data['Close'].iloc[-1] > window_data['SMA_20'].iloc[-1]),
                'high_low_ratio': window_data['High'].max() / window_data['Low'].min(),
            }
            features.append(feature_row)
            
        return pd.DataFrame(features)
    
    def train_ml_predictor(self) -> Dict[str, float]:
        """Train ML model to predict optimal MA parameters based on market conditions."""
        print("🧠 Training AI model to predict optimal parameters...")
        
        # Extract features
        features_df = self.extract_market_features()
        
        # Generate training data by testing different MA combinations
        training_data = []
        ma_combinations = [
            (5, 15), (5, 20), (8, 21), (10, 20), (10, 30), (12, 26),
            (15, 35), (20, 50), (25, 50), (30, 60), (50, 100), (50, 200)
        ]
        
        print(f"📊 Testing {len(ma_combinations)} MA combinations for training data...")
        
        for i, (short_ma, long_ma) in enumerate(ma_combinations):
            if i % 3 == 0:  # Progress indicator
                print(f"   Progress: {i+1}/{len(ma_combinations)} combinations tested")
                
            # Test this MA combination on different market periods
            for start_idx in range(0, len(features_df) - 100, 50):  # Every 50 days
                end_idx = min(start_idx + 100, len(features_df))
                
                if end_idx - start_idx < 50:  # Need minimum data
                    continue
                    
                # Get market features for this period
                period_features = features_df.iloc[start_idx:end_idx].mean()
                
                # Backtest this MA combination on this period
                period_data = self.data.iloc[start_idx + 30:end_idx + 30]  # Offset for lookback
                
                try:
                    strategy = MovingAverageStrategy(short_window=short_ma, long_window=long_ma)
                    backtester = Backtester(initial_cash=self.initial_cash, commission=1.0)
                    result = backtester.run(strategy, period_data, symbol=self.symbol)
                    
                    if hasattr(result, 'enhanced_metrics'):
                        sharpe_ratio = result.enhanced_metrics.get('sharpe_ratio', 0)
                        total_return = result.enhanced_metrics.get('total_return_pct', 0)
                        max_drawdown = result.enhanced_metrics.get('max_drawdown_pct', 0)
                        
                        # Composite score (higher is better)
                        score = sharpe_ratio * 0.5 + (total_return / 100) * 0.3 - (abs(max_drawdown) / 100) * 0.2
                        
                        training_row = period_features.to_dict()
                        training_row.update({
                            'short_ma': short_ma,
                            'long_ma': long_ma,
                            'performance_score': score,
                            'sharpe_ratio': sharpe_ratio,
                            'total_return': total_return
                        })
                        training_data.append(training_row)
                        
                except Exception as e:
                    continue  # Skip failed backtests
        
        if not training_data:
            raise ValueError("No training data generated. Check your data and parameters.")
            
        training_df = pd.DataFrame(training_data)
        print(f"✅ Generated {len(training_df)} training samples")
        
        # Prepare features and targets
        feature_columns = [col for col in training_df.columns 
                          if col not in ['short_ma', 'long_ma', 'performance_score', 'sharpe_ratio', 'total_return']]
        
        X = training_df[feature_columns]
        y_short = training_df['short_ma']
        y_long = training_df['long_ma']
        y_score = training_df['performance_score']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        self.short_ma_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.long_ma_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.score_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.short_ma_model.fit(X_scaled, y_short)
        self.long_ma_model.fit(X_scaled, y_long)
        self.score_model.fit(X_scaled, y_score)
        
        # Calculate model performance
        short_score = self.short_ma_model.score(X_scaled, y_short)
        long_score = self.long_ma_model.score(X_scaled, y_long)
        perf_score = self.score_model.score(X_scaled, y_score)
        
        print(f"🎯 AI Model Training Complete!")
        print(f"   Short MA Prediction R²: {short_score:.3f}")
        print(f"   Long MA Prediction R²: {long_score:.3f}")
        print(f"   Performance Prediction R²: {perf_score:.3f}")
        
        return {
            'short_ma_r2': short_score,
            'long_ma_r2': long_score,
            'performance_r2': perf_score,
            'training_samples': len(training_df)
        }
    
    def predict_optimal_parameters(self, recent_days: int = 30) -> Dict[str, Any]:
        """Use trained AI model to predict optimal MA parameters for current market conditions."""
        if not hasattr(self, 'short_ma_model'):
            raise ValueError("AI model not trained. Call train_ml_predictor() first.")
        
        print(f"🔮 AI predicting optimal parameters for current market conditions...")
        
        # Get recent market features
        recent_features = self.extract_market_features(recent_days).iloc[-1:].values
        recent_features_scaled = self.scaler.transform(recent_features)
        
        # Predict optimal parameters
        predicted_short = int(round(self.short_ma_model.predict(recent_features_scaled)[0]))
        predicted_long = int(round(self.long_ma_model.predict(recent_features_scaled)[0]))
        predicted_score = self.score_model.predict(recent_features_scaled)[0]
        
        # Ensure logical constraints
        predicted_short = max(3, min(50, predicted_short))  # Between 3-50
        predicted_long = max(predicted_short + 5, min(200, predicted_long))  # At least 5 more than short
        
        print(f"🤖 AI Recommendations:")
        print(f"   Optimal Short MA: {predicted_short}")
        print(f"   Optimal Long MA: {predicted_long}")
        print(f"   Expected Performance Score: {predicted_score:.3f}")
        
        return {
            'short_ma': predicted_short,
            'long_ma': predicted_long,
            'expected_score': predicted_score,
            'confidence': min(0.95, max(0.5, predicted_score + 0.5))  # Rough confidence estimate
        }
    
    def hyperparameter_optimization(self, n_trials: int = 50) -> Dict[str, Any]:
        """Use Optuna for advanced hyperparameter optimization."""
        print(f"⚡ Running advanced AI optimization with {n_trials} trials...")
        
        def objective(trial):
            # Suggest parameters
            short_ma = trial.suggest_int('short_ma', 3, 30)
            long_ma = trial.suggest_int('long_ma', short_ma + 5, 100)
            
            try:
                # Backtest with these parameters
                strategy = MovingAverageStrategy(short_window=short_ma, long_window=long_ma)
                backtester = Backtester(initial_cash=self.initial_cash, commission=1.0)
                result = backtester.run(strategy, self.data, symbol=self.symbol)
                
                if hasattr(result, 'enhanced_metrics'):
                    sharpe_ratio = result.enhanced_metrics.get('sharpe_ratio', -999)
                    total_return = result.enhanced_metrics.get('total_return_pct', -999)
                    max_drawdown = result.enhanced_metrics.get('max_drawdown_pct', -999)
                    
                    # Multi-objective optimization score
                    if sharpe_ratio == -999:
                        return -999
                    
                    score = (sharpe_ratio * 0.4 + 
                            (total_return / 100) * 0.4 - 
                            (abs(max_drawdown) / 100) * 0.2)
                    
                    return score
                else:
                    return -999
                    
            except Exception:
                return -999
        
        # Create study
        study = optuna.create_study(direction='maximize', 
                                   sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"🏆 Hyperparameter Optimization Complete!")
        print(f"   Best Short MA: {best_params['short_ma']}")
        print(f"   Best Long MA: {best_params['long_ma']}")
        print(f"   Best Score: {best_score:.3f}")
        
        return {
            'best_short_ma': best_params['short_ma'],
            'best_long_ma': best_params['long_ma'],
            'best_score': best_score,
            'n_trials': n_trials,
            'optimization_method': 'Optuna TPE'
        }
    
    def generate_ai_insights(self, optimization_results: Dict) -> List[str]:
        """Generate AI-powered insights about the optimization results."""
        insights = []
        
        short_ma = optimization_results.get('best_short_ma', 0)
        long_ma = optimization_results.get('best_long_ma', 0)
        score = optimization_results.get('best_score', 0)
        
        # Strategy speed analysis
        if short_ma <= 10:
            insights.append("🏃‍♂️ AI detected: Fast strategy optimal - market shows strong trending behavior")
        elif short_ma >= 20:
            insights.append("🐢 AI detected: Slow strategy optimal - market shows high noise, needs filtering")
        else:
            insights.append("⚖️ AI detected: Medium-speed strategy optimal - balanced market conditions")
        
        # Performance analysis
        if score > 0.5:
            insights.append("🎯 AI confidence: HIGH - Strong edge detected in current market regime")
        elif score > 0.2:
            insights.append("📊 AI confidence: MEDIUM - Moderate edge detected, proceed with caution")
        else:
            insights.append("⚠️ AI confidence: LOW - Challenging market conditions for MA strategies")
        
        # Parameter relationship analysis
        ratio = long_ma / short_ma if short_ma > 0 else 0
        if ratio > 5:
            insights.append("🔍 AI insight: Wide MA spread suggests strong trend-following approach needed")
        elif ratio < 3:
            insights.append("⚡ AI insight: Narrow MA spread suggests quick reaction to price changes optimal")
        
        return insights


def run_ai_optimization_demo(symbol: str = "AAPL"):
    """Demo function to showcase AI optimization capabilities."""
    print("🤖 QuantSim AI Strategy Optimizer Demo")
    print("=" * 50)
    
    try:
        # Initialize AI optimizer
        optimizer = AIStrategyOptimizer(symbol)
        
        # Load data
        optimizer.load_market_data()
        
        # Train AI model
        training_results = optimizer.train_ml_predictor()
        
        # Get AI predictions
        ai_predictions = optimizer.predict_optimal_parameters()
        
        # Run hyperparameter optimization
        optuna_results = optimizer.hyperparameter_optimization(n_trials=30)
        
        # Generate insights
        insights = optimizer.generate_ai_insights(optuna_results)
        
        print("\n🎯 AI OPTIMIZATION RESULTS")
        print("=" * 30)
        print(f"Symbol: {symbol}")
        print(f"AI Predicted Short MA: {ai_predictions['short_ma']}")
        print(f"AI Predicted Long MA: {ai_predictions['long_ma']}")
        print(f"Optuna Best Short MA: {optuna_results['best_short_ma']}")
        print(f"Optuna Best Long MA: {optuna_results['best_long_ma']}")
        print(f"Best Performance Score: {optuna_results['best_score']:.3f}")
        
        print("\n🧠 AI INSIGHTS:")
        for insight in insights:
            print(f"   {insight}")
        
        return {
            'ai_predictions': ai_predictions,
            'optuna_results': optuna_results,
            'insights': insights,
            'training_results': training_results
        }
        
    except Exception as e:
        print(f"❌ Error in AI optimization: {e}")
        return None


if __name__ == "__main__":
    # Run demo
    results = run_ai_optimization_demo("AAPL")
