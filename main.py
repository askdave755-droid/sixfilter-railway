import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
import openai
import asyncio

# Fix numpy type adaptation for PostgreSQL
import psycopg2.extensions

def adapt_numpy_bool(numpy_bool):
    return psycopg2.extensions.AsIs(bool(numpy_bool))

def adapt_numpy_float(numpy_float):
    return psycopg2.extensions.AsIs(float(numpy_float))

def adapt_numpy_int(numpy_int):
    return psycopg2.extensions.AsIs(int(numpy_int))

psycopg2.extensions.register_adapter(np.bool_, adapt_numpy_bool)
psycopg2.extensions.register_adapter(np.float64, adapt_numpy_float)
psycopg2.extensions.register_adapter(np.int64, adapt_numpy_int)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def init_db():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            symbol VARCHAR(10) NOT NULL,
            action VARCHAR(10),
            entry_price DECIMAL(10,2),
            stop_price DECIMAL(10,2),
            target_price DECIMAL(10,2),
            size INTEGER,
            confidence DECIMAL(4,2),
            filter_1_lmsr BOOLEAN,
            filter_2_kelly BOOLEAN,
            filter_3_ev BOOLEAN,
            filter_4_kl BOOLEAN,
            filter_5_bayesian BOOLEAN,
            filter_6_stoikov BOOLEAN,
            vwap DECIMAL(10,2),
            atr DECIMAL(10,2),
            vix DECIMAL(5,2),
            daily_pnl DECIMAL(10,2),
            ai_approved BOOLEAN,
            ai_reason TEXT,
            executed BOOLEAN DEFAULT FALSE,
            exit_price DECIMAL(10,2),
            trade_pnl DECIMAL(10,2)
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            trade_date DATE DEFAULT CURRENT_DATE,
            symbol VARCHAR(10),
            total_pnl DECIMAL(10,2) DEFAULT 0,
            trades_count INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            max_consecutive_losses INTEGER DEFAULT 0,
            PRIMARY KEY (trade_date, symbol)
        )
    """)
    
    conn.commit()
    cur.close()
    conn.close()

def get_db():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()

# SixFilter Engine
class SixFilterEngine:
    def __init__(self, ohlcv: List[Dict]):
        self.df = pd.DataFrame(ohlcv)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.current = self.df.iloc[-1]
        
    def calculate_true_vwap(self) -> float:
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        vwap = (tp * self.df['volume']).sum() / self.df['volume'].sum()
        return float(vwap)
    
    def calculate_lmsr(self, current_price: float) -> Tuple[bool, float]:
        vwap = self.calculate_true_vwap()
        std_dev = self.df['close'].std()
        if std_dev == 0:
            return False, 0.0
        deviation = abs(current_price - vwap) / std_dev
        is_overextended = deviation > 1.5
        return bool(is_overextended), float(deviation)
    
    def calculate_kelly(self, daily_pnl: float, consecutive_losses: int) -> Tuple[bool, int, float]:
        if daily_pnl <= -500:
            return False, 0, 0.0
        win_rate = 0.55
        avg_win = 2.0
        avg_loss = 1.0
        kelly_f = (win_rate * avg_win - (1 - win_rate)) / avg_win
        kelly_f = max(0, kelly_f * 0.25)
        
        size_multiplier = 1.0
        if consecutive_losses >= 2:
            size_multiplier = 0.5
        elif daily_pnl > 500:
            size_multiplier = 1.5
            
        base_contracts = 2
        contracts = int(base_contracts * kelly_f * size_multiplier)
        contracts = max(1, min(3, contracts))
        
        return True, contracts, float(kelly_f)
    
    def calculate_atr(self, period: int = 14) -> float:
        high_low = self.df['high'] - self.df['low']
        high_close = abs(self.df['high'] - self.df['close'].shift())
        low_close = abs(self.df['low'] - self.df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]
        return float(atr)
    
    def calculate_ev_gap(self, symbol: str) -> Tuple[bool, float, float]:
        atr = self.calculate_atr()
        tick_size = 0.25
        atr_ticks = atr / tick_size
        multiplier = 1.5 if 'NQ' in symbol else 1.0
        stop_ticks = max(10 if 'NQ' in symbol else 4, atr_ticks * multiplier)
        target_ticks = stop_ticks * 2
        
        stop_dist = stop_ticks * tick_size
        target_dist = target_ticks * tick_size
        
        return bool(atr_ticks > 0), stop_dist, target_dist
    
    def detect_divergence(self) -> Tuple[bool, str]:
        if len(self.df) < 10:
            return False, "insufficient_data"
            
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))
        
        price_now = self.df['close'].iloc[-1]
        price_5 = self.df['close'].iloc[-5]
        rsi_now = self.df['rsi'].iloc[-1]
        rsi_5 = self.df['rsi'].iloc[-5]
        
        if price_now > price_5 and rsi_now < rsi_5:
            return False, "bearish_divergence"
        if price_now < price_5 and rsi_now > rsi_5:
            return True, "bullish_divergence"
            
        return True, "no_divergence"
    
    def check_bayesian_context(self, vix: float, timestamp_str: str) -> Tuple[bool, float]:
        ts = pd.to_datetime(timestamp_str)
        time_val = ts.hour * 100 + ts.minute
        
        if 900 <= time_val <= 935:
            return False, 0.0
        if 1200 <= time_val <= 1330:
            return False, 0.0
        if time_val >= 1500:
            return False, 0.0
            
        base_prob = 0.65
        if vix > 30:
            base_prob -= 0.15
        elif vix < 15:
            base_prob -= 0.05
            
        return True, base_prob
    
    def calculate_stoikov_entry(self) -> Tuple[bool, float, str]:
        vwap = self.calculate_true_vwap()
        ema20 = self.df['close'].ewm(span=20).mean().iloc[-1]
        price = self.current['close']
        atr = self.calculate_atr()
        
        is_uptrend = price > ema20
        is_downtrend = price < ema20
        
        if is_uptrend and abs(price - vwap) <= atr * 0.5 and price < vwap:
            return True, vwap, "LONG"
        if is_downtrend and abs(price - vwap) <= atr * 0.5 and price > vwap:
            return True, vwap, "SHORT"
            
        return False, price, "NONE"
    
    def run_all(self, context: Dict) -> Dict:
        price = self.current['close']
        symbol = context.get('symbol', 'MES')
        
        f1, dev = self.calculate_lmsr(price)
        f2, size, kelly = self.calculate_kelly(context.get('daily_pnl', 0), context.get('consecutive_losses', 0))
        f3, stop, target = self.calculate_ev_gap(symbol)
        f4, div_reason = self.detect_divergence()
        f5, prob = self.check_bayesian_context(context.get('vix', 20), context.get('timestamp'))
        f6, entry, direction = self.calculate_stoikov_entry()
        
        all_pass = f1 and f2 and f3 and f4 and f5 and f6
        
        return {
            "all_pass": bool(all_pass),
            "direction": direction if all_pass else "NONE",
            "entry_price": round(float(entry), 2),
            "stop_price": round(float(entry - stop), 2) if direction == "LONG" else round(float(entry + stop), 2),
            "target_price": round(float(entry + target), 2) if direction == "LONG" else round(float(entry - target), 2),
            "size": int(size) if all_pass else 0,
            "confidence": round(float(prob * 100), 1),
            "filters": {
                "1_lmsr": bool(f1),
                "2_kelly": bool(f2),
                "3_ev": bool(f3),
                "4_kl": bool(f4),
                "5_bayesian": bool(f5),
                "6_stoikov": bool(f6)
            },
            "metadata": {
                "vwap": round(float(self.calculate_true_vwap()), 2),
                "deviation": round(float(dev), 3),
                "atr": round(float(self.calculate_atr()), 2)
            }
        }

# AI Validator
async def validate_with_ai(filter_data: Dict, market_context: Dict) -> Dict:
    if not OPENAI_API_KEY:
        return {
            "proceed": bool(filter_data['all_pass']),
            "confidence": float(filter_data['confidence']),
            "reason": "AI disabled - API key not set",
            "size_multiplier": 1.0,
            "suggested_stop": 0.0
        }
        
    try:
        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        system_prompt = """You are a quantitative trading risk manager. 
        Analyze the 6-filter setup and determine if this trade should execute.
        Be conservative. Return JSON with: proceed (bool), confidence (0-100), 
        size_multiplier (0.5-2.0), reason (str), suggested_stop (float or 0)"""
        
        user_content = f"""
        6-FILTER STATUS:
        - LMSR (Overextended): {filter_data['filters']['1_lmsr']} (Dev: {filter_data['metadata']['deviation']})
        - Kelly (Position Sizing): {filter_data['filters']['2_kelly']} (Size: {filter_data['size']})
        - EV Gap (Risk/Reward): {filter_data['filters']['3_ev']}
        - KL Divergence: {filter_data['filters']['4_kl']}
        - Bayesian Context: {filter_data['filters']['5_bayesian']} (Prob: {filter_data['confidence']}%)
        - Stoikov Entry: {filter_data['filters']['6_stoikov']} at {filter_data['entry_price']}
        
        MARKET CONTEXT:
        - Symbol: {market_context.get('symbol')}
        - VIX: {market_context.get('vix')}
        - Daily PnL: ${market_context.get('daily_pnl')}
        - Consecutive Losses: {market_context.get('consecutive_losses')}
        
        Should we proceed? Return strict JSON only.
        """
        
        response = await client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=200
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return {
            "proceed": bool(result.get("proceed", False)),
            "confidence": float(result.get("confidence", 50)),
            "reason": str(result.get("reason", "AI validation")),
            "size_multiplier": float(result.get("size_multiplier", 1.0)),
            "suggested_stop": float(result.get("suggested_stop", 0.0))
        }
        
    except Exception as e:
        return {
            "proceed": bool(filter_data['all_pass']),
            "confidence": float(filter_data['confidence']),
            "reason": f"AI error, failing open: {str(e)}",
            "size_multiplier": 1.0,
            "suggested_stop": 0.0
        }

# FastAPI App
app = FastAPI(title="SixFilter Guardian", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class BarData(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class SignalRequest(BaseModel):
    symbol: str
    timestamp: str
    price: float
    bars: List[BarData]
    vix: Optional[float] = 20.0
    daily_pnl: Optional[float] = 0.0
    consecutive_losses: Optional[int] = 0
    use_ai: Optional[bool] = True

class SignalResponse(BaseModel):
    proceed: bool
    confidence: float
    reason: str
    size_multiplier: float
    suggested_stop: float
    direction: str
    entry_price: float
    stop_price: float
    target_price: float
    size: int
    filters: Dict[str, bool]

@app.on_event("startup")
async def startup():
    init_db()

@app.get("/health")
def health_check():
    return {"status": "operational", "filters": "6_filter_mathematical"}

@app.post("/analyze", response_model=SignalResponse)
async def analyze_signal(request: SignalRequest, db=Depends(get_db)):
    try:
        ohlcv = [bar.dict() for bar in request.bars]
        engine = SixFilterEngine(ohlcv)
        
        filter_result = engine.run_all({
            'symbol': request.symbol,
            'timestamp': request.timestamp,
            'daily_pnl': request.daily_pnl,
            'consecutive_losses': request.consecutive_losses,
            'vix': request.vix
        })
        
        ai_decision = {
            "proceed": bool(filter_result['all_pass']),
            "confidence": float(filter_result['confidence']),
            "reason": "6 filters aligned",
            "size_multiplier": 1.0,
            "suggested_stop": 0.0
        }
        
        if request.use_ai and filter_result['all_pass']:
            ai_decision = await validate_with_ai(filter_result, {
                'symbol': request.symbol,
                'price': request.price,
                'vix': request.vix,
                'daily_pnl': request.daily_pnl,
                'consecutive_losses': request.consecutive_losses
            })
        
        # Ensure all native Python types for database
        cur = db.cursor()
        cur.execute("""
            INSERT INTO trades (symbol, action, entry_price, stop_price, target_price, size,
                              confidence, filter_1_lmsr, filter_2_kelly, filter_3_ev, 
                              filter_4_kl, filter_5_bayesian, filter_6_stoikov,
                              vwap, atr, vix, daily_pnl, ai_approved, ai_reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            str(request.symbol),
            str(filter_result['direction'] if ai_decision['proceed'] else "NONE"),
            float(filter_result['entry_price']),
            float(filter_result['stop_price']),
            float(filter_result['target_price']),
            int(filter_result['size'] * ai_decision['size_multiplier']),
            float(ai_decision['confidence']),
            bool(filter_result['filters']['1_lmsr']),
            bool(filter_result['filters']['2_kelly']),
            bool(filter_result['filters']['3_ev']),
            bool(filter_result['filters']['4_kl']),
            bool(filter_result['filters']['5_bayesian']),
            bool(filter_result['filters']['6_stoikov']),
            float(filter_result['metadata']['vwap']),
            float(filter_result['metadata']['atr']),
            float(request.vix),
            float(request.daily_pnl),
            bool(ai_decision['proceed']),
            str(ai_decision['reason'])
        ))
        db.commit()
        
        return SignalResponse(
            proceed=bool(ai_decision['proceed']),
            confidence=float(ai_decision['confidence']),
            reason=str(ai_decision['reason']),
            size_multiplier=float(ai_decision['size_multiplier']),
            suggested_stop=float(ai_decision['suggested_stop']),
            direction=str(filter_result['direction']),
            entry_price=float(filter_result['entry_price']),
            stop_price=float(filter_result['stop_price']),
            target_price=float(filter_result['target_price']),
            size=int(filter_result['size'] * ai_decision['size_multiplier']),
            filters={k: bool(v) for k, v in filter_result['filters'].items()}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fill")
async def record_fill(trade_id: int, exit_price: float, pnl: float, db=Depends(get_db)):
    cur = db.cursor()
    cur.execute("""
        UPDATE trades SET executed = TRUE, exit_price = %s, trade_pnl = %s 
        WHERE id = %s
    """, (exit_price, pnl, trade_id))
    db.commit()
    return {"status": "recorded"}

@app.get("/stats/{symbol}")
def get_stats(symbol: str, db=Depends(get_db)):
    cur = db.cursor()
    cur.execute("""
        SELECT COUNT(*) as trades, 
               SUM(CASE WHEN trade_pnl > 0 THEN 1 ELSE 0 END) as wins,
               SUM(trade_pnl) as total_pnl
        FROM trades 
        WHERE symbol = %s AND DATE(timestamp) = CURRENT_DATE
    """, (symbol,))
    row = cur.fetchone()
    return {
        "trades_today": int(row['trades'] or 0),
        "wins": int(row['wins'] or 0),
        "total_pnl": float(row['total_pnl'] or 0)
    }
