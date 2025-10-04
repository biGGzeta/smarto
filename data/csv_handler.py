import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from typing import Optional
from config.settings import BinanceConfig
import os

class BinanceDataDownloader:
    """Descargador de datos de Binance con soporte para datasets grandes"""
    
    def __init__(self, symbol: str):
        self.original_symbol = symbol
        
        # Determinar tipo de API y símbolo correcto
        if 'USD_PERP' in symbol.upper():
            # Coin-M Futures (como ETHUSD_PERP)
            self.base_url = "https://dapi.binance.com/dapi/v1"
            self.symbol = symbol.upper()
        elif 'USDT' in symbol.upper() and ('_PERP' in symbol.upper() or symbol.endswith('PERP')):
            # USDT-M Futures
            self.base_url = "https://fapi.binance.com/fapi/v1"
            self.symbol = symbol.upper().replace('_PERP', '')
        else:
            # Spot
            self.base_url = "https://api.binance.com/api/v3"
            self.symbol = symbol.upper()
        
        print(f"🔗 Usando API: {self.base_url}")
        print(f"📊 Símbolo original: {self.original_symbol}")
        print(f"📊 Símbolo API: {self.symbol}")
        
    def get_klines(self, interval: str, hours: float, max_retries: int = 3) -> pd.DataFrame:
        """
        Descarga datos de velas con soporte para períodos largos
        """
        try:
            # Calcular registros necesarios
            interval_minutes = self._get_interval_minutes(interval)
            total_minutes = int(hours * 60)
            records_needed = total_minutes // interval_minutes
            
            print(f"Descargando {self.symbol} - {interval} - últimas {hours} horas...")
            print(f"📊 Registros necesarios: {records_needed}")
            
            # Si necesitamos más de 1000 registros, usar múltiples requests
            if records_needed > 1000:
                print(f"📊 Dataset grande detectado. Usando múltiples requests...")
                return self._download_large_dataset(interval, hours, records_needed)
            else:
                print(f"📊 Solicitando: {records_needed}")
                return self._download_single_request(interval, records_needed, hours)
                
        except Exception as e:
            print(f"❌ Error descargando datos: {e}")
            return pd.DataFrame()
    
    def _download_large_dataset(self, interval: str, hours: float, records_needed: int) -> pd.DataFrame:
        """Descarga datasets grandes usando múltiples requests"""
        all_data = []
        batch_size = 1000  # Tamaño de batch seguro para Binance
        interval_minutes = self._get_interval_minutes(interval)
        
        # Calcular cuántos batches necesitamos
        num_batches = (records_needed + batch_size - 1) // batch_size
        print(f"📊 Dividiendo en {num_batches} batches de máximo {batch_size} registros")
        
        # Tiempo final (ahora)
        end_time = datetime.now()
        
        for batch in range(num_batches):
            try:
                # Calcular cuántos registros faltan
                remaining_records = records_needed - (batch * batch_size)
                current_batch_size = min(batch_size, remaining_records)
                
                # Calcular timestamp de inicio para este batch
                minutes_back = (batch + 1) * batch_size * interval_minutes
                batch_end_time = end_time - timedelta(minutes=batch * batch_size * interval_minutes)
                
                print(f"📊 Batch {batch + 1}/{num_batches}: {current_batch_size} registros hasta {batch_end_time.strftime('%Y-%m-%d %H:%M')}")
                
                # Parámetros para este batch
                params = {
                    'symbol': self.symbol,
                    'interval': interval,
                    'limit': current_batch_size,
                    'endTime': int(batch_end_time.timestamp() * 1000)
                }
                
                # Hacer request
                response = requests.get(f"{self.base_url}/klines", params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        # Convertir a DataFrame
                        df = self._process_raw_data(data)
                        if not df.empty:
                            all_data.append(df)
                            print(f"✅ Batch {batch + 1} completado: {len(df)} registros")
                        else:
                            print(f"⚠️ Batch {batch + 1} procesado pero vacío")
                    else:
                        print(f"⚠️ Batch {batch + 1} sin datos de la API")
                else:
                    print(f"❌ Error en batch {batch + 1}: Status {response.status_code}")
                    print(f"❌ Respuesta: {response.text}")
                
                # Pausa entre requests
                if batch < num_batches - 1:
                    time.sleep(0.3)  # 300ms entre requests
                    
            except Exception as e:
                print(f"❌ Error en batch {batch + 1}: {e}")
                continue
        
        if all_data:
            # Combinar todos los DataFrames
            combined_data = pd.concat(all_data, ignore_index=False)
            
            # Eliminar duplicados y ordenar
            combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
            combined_data = combined_data.sort_index()
            
            # Recortar al número exacto de registros solicitados
            if len(combined_data) > records_needed:
                combined_data = combined_data.tail(records_needed)
            
            print(f"✅ Dataset completo combinado: {len(combined_data)} registros")
            print(f"📈 Rango temporal: {combined_data.index[0]} a {combined_data.index[-1]}")
            
            # Mostrar precio actual
            current_price = combined_data['close'].iloc[-1]
            print(f"💰 Precio actual: ${current_price:,.2f}")
            
            return combined_data
        else:
            print(f"❌ No se pudieron descargar datos en ningún batch")
            return pd.DataFrame()
    
    def _download_single_request(self, interval: str, records_needed: int, hours: float) -> pd.DataFrame:
        """Descarga con un solo request"""
        try:
            params = {
                'symbol': self.symbol,
                'interval': interval,
                'limit': min(records_needed, 1500)
            }
            
            print(f"🔗 Request: {self.base_url}/klines")
            print(f"📋 Parámetros: {params}")
            
            response = requests.get(f"{self.base_url}/klines", params=params, timeout=30)
            
            print(f"📊 Status Code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ Error API: {response.text}")
                return pd.DataFrame()
            
            data = response.json()
            
            if not data:
                print("⚠️ Respuesta vacía de la API")
                return pd.DataFrame()
            
            # Procesar datos
            df = self._process_raw_data(data)
            
            print(f"✅ Descargados {len(df)} registros, filtrados a {len(df)} para {hours}h")
            print(f"📈 Rango temporal: {df.index[0]} a {df.index[-1]}")
            
            # Mostrar precio actual
            current_price = df['close'].iloc[-1]
            print(f"💰 Precio actual: ${current_price:,.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error en single request: {e}")
            return pd.DataFrame()
    
    def _process_raw_data(self, data) -> pd.DataFrame:
        """Procesa datos crudos de la API a DataFrame"""
        try:
            # Convertir a DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Procesar datos
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Establecer timestamp como índice
            df.set_index('timestamp', inplace=True)
            
            # Seleccionar solo las columnas necesarias
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            print(f"❌ Error procesando datos: {e}")
            return pd.DataFrame()
    
    def _get_interval_minutes(self, interval: str) -> int:
        """Convierte intervalo a minutos"""
        interval_map = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        return interval_map.get(interval, 1)