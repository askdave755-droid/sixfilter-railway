import os
import json
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import psycopg2
from database import init_db, get_db
from filters import SixFilterEngine
from ai_validator import validate_with_ai
import asyncio

app = FastAPI(title="SixFilter Guardian - Railway Brain")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    bars: List[BarData]  # Last 20-50 bars for calculations
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
    filters: dict

@app.on_event("startup")
async def startup():
    init_db()

@app.get("/health")
def health_check():
    return {"status": "operational", "filters": "6_filter_mathematical"}

@app.post("/analyze", response_model=SignalResponse)
async def analyze_signal(request: SignalRequest, db=Depends(get_db)):
    """
    Main endpoint for NT8 to query
    """
    try:
        # Convert bars to list of dicts for pandas
        ohlcv = [bar.dict() for bar in request.bars]
        
        # Run SixFilter math
        engine = SixFilterEngine(ohlcv)
        filter_result = engine.run_all({
            'symbol': request.symbol,
            'timestamp': request.timestamp,
            'daily_pnl': request.daily_pnl,
            'consecutive_losses': request.consecutive_losses,
            'vix': request.vix
        })
        
        # If filters pass and AI enabled, get OpenAI validation
        ai_decision = {
            "proceed": filter_result['all_pass'],
            "confidence": filter_result['confidence'],
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
        
        # Log to database
        cur = db.cursor()
        cur.execute("""
            INSERT INTO trades (symbol, action, entry_price, stop_price, target_price, size,
                              confidence, filter_1_lmsr, filter_2_kelly, filter_3_ev, 
                              filter_4_kl, filter_5_bayesian, filter_6_stoikov,
                              vwap, atr, vix, daily_pnl, ai_approved, ai_reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            request.symbol,
            filter_result['direction'] if ai_decision['proceed'] else "NONE",
            filter_result['entry_price'],
            filter_result['stop_price'],
            filter_result['target_price'],
            int(filter_result['size'] * ai_decision['size_multiplier']),
            ai_decision['confidence'],
            filter_result['filters']['1_lmsr'],
            filter_result['filters']['2_kelly'],
            filter_result['filters']['3_ev'],
            filter_result['filters']['4_kl'],
            filter_result['filters']['5_bayesian'],
            filter_result['filters']['6_stoikov'],
            filter_result['metadata']['vwap'],
            filter_result['metadata']['atr'],
            request.vix,
            request.daily_pnl,
            ai_decision['proceed'],
            ai_decision['reason']
        ))
        db.commit()
        
        return SignalResponse(
            proceed=ai_decision['proceed'],
            confidence=ai_decision['confidence'],
            reason=ai_decision['reason'],
            size_multiplier=ai_decision['size_multiplier'],
            suggested_stop=ai_decision['suggested_stop'],
            direction=filter_result['direction'],
            entry_price=filter_result['entry_price'],
            stop_price=filter_result['stop_price'],
            target_price=filter_result['target_price'],
            size=int(filter_result['size'] * ai_decision['size_multiplier']),
            filters=filter_result['filters']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fill")
async def record_fill(trade_id: int, exit_price: float, pnl: float, db=Depends(get_db)):
    """NT8 reports back fills here"""
    cur = db.cursor()
    cur.execute("""
        UPDATE trades SET executed = TRUE, exit_price = %s, trade_pnl = %s 
        WHERE id = %s
    """, (exit_price, pnl, trade_id))
    db.commit()
    return {"status": "recorded"}

@app.get("/stats/{symbol}")
def get_stats(symbol: str, db=Depends(get_db)):
    """Get today's stats for NT8 risk check"""
    cur = db.cursor()
    cur.execute("""
        SELECT COUNT(*) as trades, SUM(CASE WHEN trade_pnl > 0 THEN 1 ELSE 0 END) as wins,
               SUM(trade_pnl) as total_pnl
        FROM trades 
        WHERE symbol = %s AND DATE(timestamp) = CURRENT_DATE
    """, (symbol,))
    row = cur.fetchone()
    return {
        "trades_today": row['trades'] or 0,
        "wins": row['wins'] or 0,
        "total_pnl": float(row['total_pnl'] or 0)
    }
