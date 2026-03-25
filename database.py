import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, date

DATABASE_URL = os.getenv("DATABASE_URL")

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
