#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test simple de stream Binance Coin-M
"""

import asyncio
import datetime
import json
import websockets

async def test_binance_stream():
    """Test simple del stream de Binance"""
    
    url = "wss://dstream.binance.com/ws/ethusd_perp@ticker"
    print(f"🔗 Conectando a: {url}")
    
    try:
        async with websockets.connect(url, ping_timeout=10) as websocket:
            print("✅ WebSocket conectado")
            
            for i in range(10):  # Capturar 10 mensajes
                message = await asyncio.wait_for(websocket.recv(), timeout=10)
                data = json.loads(message)
                
                if 'c' in data:
                    price = float(data['c'])
                    change_pct = float(data['P'])
                    volume = float(data['v'])
                    
                    timestamp = datetime.datetime.now(datetime.timezone.utc)
                    print(f"[{timestamp.strftime('%H:%M:%S')}] ETH: ${price:.2f} ({change_pct:+.3f}%) Vol: {volume:.0f}")
                    
                    # Log estructura completa en primer mensaje
                    if i == 0:
                        print(f"📊 Estructura de datos: {list(data.keys())}")
                
                await asyncio.sleep(2)
                
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_binance_stream())