from datetime import datetime, timedelta
import requests

def debug_timestamps():
    """Debug de timestamps para Binance"""
    
    # Timestamps actuales
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=3)
    
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    
    print(f"🕐 End Time UTC: {end_time}")
    print(f"🕐 Start Time UTC: {start_time}")
    print(f"📊 Start Timestamp: {start_ts}")
    print(f"📊 End Timestamp: {end_ts}")
    
    # Test con timestamps
    url = "https://dapi.binance.com/dapi/v1/klines"
    params_with_time = {
        'symbol': 'ETHUSD_PERP',
        'interval': '1m',
        'startTime': start_ts,
        'endTime': end_ts,
        'limit': 100
    }
    
    print(f"\n🔗 Testing with timestamps...")
    response = requests.get(url, params=params_with_time, timeout=10)
    print(f"📡 Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"📦 Data with timestamps: {len(data)} records")
        if data:
            first_time = datetime.fromtimestamp(data[0][0]/1000)
            last_time = datetime.fromtimestamp(data[-1][0]/1000)
            print(f"🕐 First record: {first_time}")
            print(f"🕐 Last record: {last_time}")
    else:
        print(f"❌ Error: {response.text}")
    
    # Test sin timestamps (solo últimos datos)
    print(f"\n🔗 Testing without timestamps...")
    params_simple = {
        'symbol': 'ETHUSD_PERP',
        'interval': '1m',
        'limit': 180  # 3 horas = 180 minutos
    }
    
    response2 = requests.get(url, params=params_simple, timeout=10)
    print(f"📡 Status: {response2.status_code}")
    
    if response2.status_code == 200:
        data2 = response2.json()
        print(f"📦 Data without timestamps: {len(data2)} records")
        if data2:
            first_time2 = datetime.fromtimestamp(data2[0][0]/1000)
            last_time2 = datetime.fromtimestamp(data2[-1][0]/1000)
            print(f"🕐 First record: {first_time2}")
            print(f"🕐 Last record: {last_time2}")
            print(f"📈 Current price: {data2[-1][4]}")

if __name__ == "__main__":
    debug_timestamps()