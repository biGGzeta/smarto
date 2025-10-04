import requests

def test_binance_connection():
    """Prueba simple de conexión a Binance"""
    try:
        url = "https://dapi.binance.com/dapi/v1/ping"
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Conexión a Binance OK")
            return True
        else:
            print("❌ Error de conexión")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_eth_data():
    """Prueba obtener datos de ETH"""
    try:
        url = "https://dapi.binance.com/dapi/v1/klines"
        params = {
            'symbol': 'ETHUSD_PERP',
            'interval': '1m',
            'limit': 5
        }
        
        response = requests.get(url, params=params, timeout=10)
        print(f"ETH Data Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Datos ETH OK - {len(data)} registros")
            print(f"Último precio ETH: {data[-1][4]}")  # Close price
            return True
        else:
            print(f"❌ Error ETH: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error ETH: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Probando conexión a Binance...")
    test_binance_connection()
    print("\n🔄 Probando datos de ETH...")
    test_eth_data()