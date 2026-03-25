import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from datetime import datetime, time

class SixFilterEngine:
    def __init__(self, ohlcv: List[Dict]):
        """
        ohlcv: List of dicts with open, high, low, close, volume, timestamp
        """
        self.df = pd.DataFrame(ohlcv)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.current = self.df.iloc[-1]
        
    def calculate_true_vwap(self) -> float:
        """Volume Weighted Average Price - the real deal"""
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        vwap = (tp * self.df['volume']).sum() / self.df['volume'].sum()
        return float(vwap)
    
    def calculate_lmsr(self, current_price: float) -> Tuple[bool, float]:
        """
        Logarithmic Market Scoring Rule proxy for futures
        Measures how far price is from fair value (VWAP) in standard deviations
        """
        vwap = self.calculate_true_vwap()
        std_dev = self.df['close'].std()
        
        if std_dev == 0:
            return False, 0.0
            
        deviation = abs(current_price - vwap) / std_dev
        # Overextended if > 1.5 sigma
        is_overextended = deviation > 1.5
        
        return is_overextended, float(deviation)
    
    def calculate_kelly(self, daily_pnl: float, consecutive_losses: int) -> Tuple[bool, int, float]:
        """
        Kelly Criterion for position sizing
        Returns: (can_trade, contracts, kelly_fraction)
        """
        # Bulenox limits
        if daily_pnl <= -500:
            return False, 0, 0.0
            
        # Kelly fraction based on edge estimation
        # Conservative: assume 55% win rate, 2:1 RR
        win_rate = 0.55
        avg_win = 2.0
        avg_loss = 1.0
        
        kelly_f = (win_rate * avg_win - (1 - win_rate)) / avg_win
        
        # Quarter Kelly for safety
        kelly_f = max(0, kelly_f * 0.25)
        
        # Size adjustment based on consecutive losses
        size_multiplier = 1.0
        if consecutive_losses >= 2:
            size_multiplier = 0.5
        elif daily_pnl > 500:
            size_multiplier = 1.5
            
        base_contracts = 2
        contracts = int(base_contracts * kelly_f * size_multiplier)
        contracts = max(1, min(3, contracts))  # Clamp 1-3
        
        return True, contracts, kelly_f
    
    def calculate_ev_gap(self, symbol: str) -> Tuple[bool, float, float]:
        """
        Expected Value gap using ATR
        Returns: (valid, stop_distance, target_distance)
        """
        atr = self.df['atr'].iloc[-1] if 'atr' in self.df.columns else self.calculate_atr()
        
        tick_size = 0.25 if 'NQ' in symbol else 0.25
        atr_ticks = atr / tick_size
        
        # NQ needs wider stops
        multiplier = 1.5 if 'NQ' in symbol else 1.0
        stop_ticks = max(10 if 'NQ' in symbol else 4, atr_ticks * multiplier)
        target_ticks = stop_ticks * 2  # 2:1 RR
        
        stop_dist = stop_ticks * tick_size
        target_dist = target_ticks * tick_size
        
        is_valid = atr_ticks > 0
        return is_valid, stop_dist, target_dist
    
    def calculate_atr(self, period: int = 14) -> float:
        """Calculate ATR if not provided"""
        high_low = self.df['high'] - self.df['low']
        high_close = abs(self.df['high'] - self.df['close'].shift())
        low_close = abs(self.df['low'] - self.df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]
        return float(atr)
    
    def detect_divergence(self) -> Tuple[bool, str]:
        """
        KL Divergence proxy: Price-RSI divergence
        Returns: (clean_signal, reason)
        """
        if len(self.df) < 10:
            return False, "insufficient_data"
            
        # Calculate RSI if not present
        if 'rsi' not in self.df.columns:
            delta = self.df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            self.df['rsi'] = 100 - (100 / (1 + rs))
        
        price_now = self.df['close'].iloc[-1]
        price_5 = self.df['close'].iloc[-5]
        rsi_now = self.df['rsi'].iloc[-1]
        rsi_5 = self.df['rsi'].iloc[-5]
        
        # Bearish divergence: Higher price, lower RSI
        if price_now > price_5 and rsi_now < rsi_5:
            return False, "bearish_divergence"
            
        # Bullish divergence: Lower price, higher RSI  
        if price_now < price_5 and rsi_now > rsi_5:
            return True, "bullish_divergence"
            
        return True, "no_divergence"
    
    def check_bayesian_context(self, vix: float, timestamp_str: str) -> Tuple[bool, float]:
        """
        Time/VIX context filter
        Returns: (passed, probability_adjustment)
        """
        ts = pd.to_datetime(timestamp_str)
        time_val = ts.hour * 100 + ts.minute
        
        # Avoid times
        if 900 <= time_val <= 935:  # Open
            return False, 0.0
        if 1200 <= time_val <= 1330:  # Lunch
            return False, 0.0
        if time_val >= 1500:  # Close
            return False, 0.0
            
        # VIX adjustment
        base_prob = 0.65
        if vix > 30:
            base_prob -= 0.15
        elif vix < 15:
            base_prob -= 0.05
            
        return True, base_prob
    
    def calculate_stoikov_entry(self) -> Tuple[bool, float, str]:
        """
        Optimal entry at VWAP/EMA confluence
        """
        vwap = self.calculate_true_vwap()
        ema20 = self.df['close'].ewm(span=20).mean().iloc[-1]
        price = self.current['close']
        atr = self.calculate_atr()
        
        # Trend direction
        is_uptrend = price > ema20
        is_downtrend = price < ema20
        
        # Pullback to VWAP
        if is_uptrend and abs(price - vwap) <= atr * 0.5 and price < vwap:
            return True, vwap, "LONG"
            
        if is_downtrend and abs(price - vwap) <= atr * 0.5 and price > vwap:
            return True, vwap, "SHORT"
            
        return False, price, "NONE"
    
    def run_all(self, context: Dict) -> Dict:
        """Execute all 6 filters"""
        price = self.current['close']
        symbol = context.get('symbol', 'MES')
        
        # Run filters
        f1, dev = self.calculate_lmsr(price)
        f2, size, kelly = self.calculate_kelly(context.get('daily_pnl', 0), context.get('consecutive_losses', 0))
        f3, stop, target = self.calculate_ev_gap(symbol)
        f4, div_reason = self.detect_divergence()
        f5, prob = self.check_bayesian_context(context.get('vix', 20), context.get('timestamp'))
        f6, entry, direction = self.calculate_stoikov_entry()
        
        all_pass = f1 and f2 and f3 and f4 and f5 and f6
        
        return {
            "all_pass": all_pass,
            "direction": direction if all_pass else "NONE",
            "entry_price": round(entry, 2),
            "stop_price": round(entry - stop, 2) if direction == "LONG" else round(entry + stop, 2),
            "target_price": round(entry + target, 2) if direction == "LONG" else round(entry - target, 2),
            "size": size if all_pass else 0,
            "confidence": round(prob * 100, 1),
            "filters": {
                "1_lmsr": f1,
                "2_kelly": f2, 
                "3_ev": f3,
                "4_kl": f4,
                "5_bayesian": f5,
                "6_stoikov": f6
            },
            "metadata": {
                "vwap": round(self.calculate_true_vwap(), 2),
                "deviation": round(dev, 3),
                "atr": round(self.calculate_atr(), 2)
            }
        }
