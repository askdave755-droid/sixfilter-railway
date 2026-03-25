import os
import openai
from typing import Dict
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

async def validate_with_ai(filter_data: Dict, market_context: Dict) -> Dict:
    """
    OpenAI GPT-4 validation as Bayesian update
    Returns same structure as Claude decision
    """
    try:
        # Build the prompt
        system_prompt = """You are a quantitative trading risk manager. 
        Analyze the 6-filter setup and determine if this trade should execute.
        Be conservative - better to skip a marginal trade than take a loss.
        Return JSON with: proceed (bool), confidence (0-100), size_multiplier (0.5-2.0), reason (str), suggested_stop (float or 0)"""
        
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
        - Direction: {filter_data['direction']}
        
        VWAP: {filter_data['metadata']['vwap']}
        Current Price: {market_context.get('price')}
        
        Should we proceed? Return strict JSON.
        """
        
        response = openai.chat.completions.create(
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
        
        # Ensure required fields
        return {
            "proceed": result.get("proceed", False),
            "confidence": result.get("confidence", 50),
            "reason": result.get("reason", "AI validation"),
            "size_multiplier": result.get("size_multiplier", 1.0),
            "suggested_stop": result.get("suggested_stop", 0.0)
        }
        
    except Exception as e:
        # Fail open - let the trade through if AI fails
        return {
            "proceed": True,
            "confidence": 65,
            "reason": f"AI error, failing open: {str(e)}",
            "size_multiplier": 1.0,
            "suggested_stop": 0.0
        }
