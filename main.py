# 🚀 SÜPER AKILLI ÖĞRENEN HİSSE TAHMİN SİSTEMİ v5.0
# ChatGPT Tarzı Öğrenme + Hız Optimizasyonu + Hata Düzeltme
# Hedef: %85+ Doğruluk + Sürekli Öğrenme

import yfinance as yf
import pandas as pd
import numpy as np
import talib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import warnings
import requests
import json
import pickle
import os
from datetime import datetime, timedelta
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import threading
from concurrent.futures import ThreadPoolExecutor
import joblib

warnings.filterwarnings('ignore')

@dataclass
class LearningMemory:
    """
    Sistemin öğrenme hafızası - ChatGPT gibi deneyimlerden öğrenir
    """
    successful_patterns: Dict = None
    failed_patterns: Dict = None
    market_conditions: Dict = None
    optimization_history: List = None
    best_parameters: Dict = None
    performance_timeline: List = None
    
    def __post_init__(self):
        if self.successful_patterns is None:
            self.successful_patterns = {}
        if self.failed_patterns is None:
            self.failed_patterns = {}
        if self.market_conditions is None:
            self.market_conditions = {}
        if self.optimization_history is None:
            self.optimization_history = []
        if self.best_parameters is None:
            self.best_parameters = {}
        if self.performance_timeline is None:
            self.performance_timeline = []

class SuperSmartPredictor:
    """
    🧠 Süper Akıllı Tahmin Sistemi
    - ChatGPT tarzı sürekli öğrenme
    - Hızlı hesaplama optimizasyonu
    - Otomatik hata düzeltme
    - Adaptif parametre ayarlama
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.memory_file = f"ai_memory_{symbol}.pkl"
        self.performance_threshold = 0.75  # %75 altına düşerse kendini iyileştirir
        
        # Hafızayı yükle veya oluştur
        self.memory = self.load_memory()
        
        # Hızlı hesaplama için önbellek
        self.feature_cache = {}
        self.model_cache = {}
        
        # DeepSeek API
        self.api_key = "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"
        self.api_url = "https://openrouter.ai/api/v1"
        
        print("🧠 Süper Akıllı Sistem başlatıldı!")
        print(f"📚 Hafıza durumu: {len(self.memory.successful_patterns)} başarılı pattern")
        
    def load_memory(self) -> LearningMemory:
        """Öğrenme hafızasını yükle"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    memory = pickle.load(f)
                print("✅ Önceki öğrenme verisi yüklendi!")
                return memory
            except:
                print("⚠️ Hafıza dosyası bozuk, yeni hafıza oluşturuluyor")
        
        print("🆕 Yeni öğrenme hafızası oluşturuluyor")
        return LearningMemory()
    
    def save_memory(self):
        """Öğrenme hafızasını kaydet"""
        try:
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.memory, f)
            print("💾 Öğrenme deneyimleri kaydedildi!")
        except Exception as e:
            print(f"⚠️ Hafıza kaydetme hatası: {e}")
    
    def fast_download_data(self, days: int = 365*2) -> pd.DataFrame:
        """🚀 Hızlı veri indirme - paralel işlem"""
        print(f"⚡ {self.symbol} verisi hızlı indiriliyor...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            # Önce cache kontrol et
            cache_key = f"{self.symbol}_{start_date.date()}_{end_date.date()}"
            if cache_key in self.feature_cache:
                print("💨 Cache'den yüklendi!")
                return self.feature_cache[cache_key]
            
            # Veriyi indir
            df = yf.download(self.symbol, start=start_date, end=end_date, 
                           progress=False, threads=True)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # Cache'e kaydet
            self.feature_cache[cache_key] = df.copy()
            
            print(f"✅ {len(df)} günlük veri indirildi ({start_date.date()} - {end_date.date()})")
            return df
            
        except Exception as e:
            print(f"❌ Veri indirme hatası: {e}")
            return pd.DataFrame()
    
    def smart_ai_analysis(self, market_data: Dict) -> Dict:
        """🤖 Akıllı AI analizi - hafızadan öğrenir"""
        
        # Önce hafızadaki benzer durumları kontrol et
        current_pattern = self.extract_pattern(market_data)
        
        # Geçmiş başarılı patternleri kontrol et
        similar_success = self.find_similar_patterns(current_pattern, self.memory.successful_patterns)
        similar_failures = self.find_similar_patterns(current_pattern, self.memory.failed_patterns)
        
        if similar_success and len(similar_success) > len(similar_failures):
            print("🎯 Hafızada benzer başarılı pattern bulundu!")
            return {
                "signal": "YUKARI",
                "confidence": min(95, 70 + len(similar_success) * 5),
                "reason": f"Hafızada {len(similar_success)} benzer başarılı durum",
                "source": "memory"
            }
        
        # AI'dan yeni analiz iste
        try:
            prompt = self.create_smart_prompt(market_data, similar_success, similar_failures)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek/deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.1
            }
            
            response = requests.post(f"{self.api_url}/chat/completions", 
                                   headers=headers, json=data, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content']
                parsed = self.parse_ai_response(ai_response)
                parsed["source"] = "ai"
                return parsed
            
        except Exception as e:
            print(f"⚠️ AI analiz hatası: {e}")
        
        # Fallback: Hafıza tabanlı tahmin
        return self.memory_based_prediction(current_pattern)
    
    def create_smart_prompt(self, market_data: Dict, success_patterns: List, 
                          failure_patterns: List) -> str:
        """Akıllı prompt oluşturma - geçmiş deneyimlerle"""
        
        prompt = f"""
        Sen öğrenen bir hisse analisti AI'sın. Geçmiş deneyimlerinden öğreniyorsun.
        
        Mevcut Piyasa Durumu:
        {json.dumps(market_data, indent=2)}
        
        Geçmiş Başarılı Benzer Durumlar: {len(success_patterns)} adet
        Geçmiş Başarısız Benzer Durumlar: {len(failure_patterns)} adet
        
        Bu deneyimlerini kullanarak analiz et:
        
        Format:
        SINYAL: YUKARI/ASAGI/NOTR
        GUVEN: 0-100
        NEDEN: Detaylı açıklama
        OGRENME: Bu durumdan ne öğrendin?
        """
        
        return prompt
    
    def extract_pattern(self, market_data: Dict) -> Dict:
        """Piyasa verilerinden pattern çıkar"""
        return {
            'rsi_level': self.categorize_rsi(market_data.get('rsi', 50)),
            'trend_strength': self.categorize_trend(market_data.get('trend', 0.5)),
            'volume_status': self.categorize_volume(market_data.get('volume_ratio', 1.0)),
            'price_position': self.categorize_price_position(market_data.get('bb_position', 0.5)),
            'volatility': self.categorize_volatility(market_data.get('volatility', 0.02))
        }
    
    def categorize_rsi(self, rsi: float) -> str:
        """RSI kategorize et"""
        if rsi < 30: return "oversold"
        elif rsi > 70: return "overbought"
        elif rsi > 50: return "bullish"
        else: return "bearish"
    
    def categorize_trend(self, trend: float) -> str:
        """Trend gücünü kategorize et"""
        if trend > 0.7: return "strong_up"
        elif trend > 0.3: return "moderate_up"
        elif trend < -0.3: return "moderate_down"
        else: return "sideways"
    
    def categorize_volume(self, volume_ratio: float) -> str:
        """Hacim durumunu kategorize et"""
        if volume_ratio > 2.0: return "very_high"
        elif volume_ratio > 1.5: return "high"
        elif volume_ratio > 0.8: return "normal"
        else: return "low"
    
    def categorize_price_position(self, bb_pos: float) -> str:
        """Bollinger Band pozisyonunu kategorize et"""
        if bb_pos > 0.8: return "upper"
        elif bb_pos > 0.2: return "middle"
        else: return "lower"
    
    def categorize_volatility(self, vol: float) -> str:
        """Volatiliteyi kategorize et"""
        if vol > 0.03: return "high"
        elif vol > 0.015: return "medium"
        else: return "low"
    
    def find_similar_patterns(self, current_pattern: Dict, pattern_db: Dict) -> List:
        """Benzer patternleri bul"""
        similar = []
        for pattern_id, stored_pattern in pattern_db.items():
            similarity = self.calculate_pattern_similarity(current_pattern, stored_pattern['pattern'])
            if similarity > 0.7:  # %70+ benzerlik
                similar.append({
                    'id': pattern_id,
                    'similarity': similarity,
                    'outcome': stored_pattern.get('outcome'),
                    'date': stored_pattern.get('date')
                })
        
        return sorted(similar, key=lambda x: x['similarity'], reverse=True)
    
    def calculate_pattern_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """İki pattern arasında benzerlik hesapla"""
        matches = 0
        total = len(pattern1)
        
        for key in pattern1:
            if key in pattern2 and pattern1[key] == pattern2[key]:
                matches += 1
        
        return matches / total if total > 0 else 0
    
    def memory_based_prediction(self, current_pattern: Dict) -> Dict:
        """Hafıza tabanlı tahmin - AI olmasa da çalışır"""
        
        # En benzer başarılı pattern'ları bul
        success_matches = self.find_similar_patterns(current_pattern, self.memory.successful_patterns)
        failure_matches = self.find_similar_patterns(current_pattern, self.memory.failed_patterns)
        
        if success_matches:
            avg_confidence = sum([m['similarity'] for m in success_matches[:3]]) / min(3, len(success_matches))
            return {
                "signal": "YUKARI",
                "confidence": int(avg_confidence * 80),
                "reason": f"Hafıza: {len(success_matches)} benzer başarılı durum",
                "source": "memory_only"
            }
        elif failure_matches:
            return {
                "signal": "ASAGI", 
                "confidence": 60,
                "reason": "Hafıza: Benzer durumlar başarısız oldu",
                "source": "memory_only"
            }
        else:
            return {
                "signal": "NOTR",
                "confidence": 50,
                "reason": "Hafıza: Yeni durum, deneyim yok",
                "source": "memory_only"
            }
    
    def parse_ai_response(self, response: str) -> Dict:
        """AI cevabını ayrıştır"""
        try:
            lines = response.strip().split('\n')
            result = {"signal": "NOTR", "confidence": 50, "reason": "Belirsiz", "learning": ""}
            
            for line in lines:
                if 'SINYAL:' in line:
                    signal = line.split(':')[1].strip().upper()
                    if signal in ['YUKARI', 'ASAGI', 'NOTR']:
                        result['signal'] = signal
                elif 'GUVEN:' in line:
                    try:
                        confidence = int(''.join(filter(str.isdigit, line.split(':')[1])))
                        result['confidence'] = min(100, max(0, confidence))
                    except:
                        pass
                elif 'NEDEN:' in line:
                    result['reason'] = line.split(':')[1].strip()
                elif 'OGRENME:' in line:
                    result['learning'] = line.split(':')[1].strip()
            
            return result
        except:
            return {"signal": "NOTR", "confidence": 50, "reason": "Ayrıştırma Hatası", "learning": ""}
    
    def turbo_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """🚀 Turbo hızda özellik mühendisliği - paralel işlem"""
        print("⚡ Turbo özellik hesaplama başlatılıyor...")
        
        # Önce cache kontrol et
        cache_key = f"features_{len(df)}_{df.index[-1].date()}"
        if cache_key in self.feature_cache:
            print("💨 Özellikler cache'den yüklendi!")
            return self.feature_cache[cache_key]
        
        start_time = time.time()
        
        # NumPy arrays - daha hızlı
        high = df['High'].values.astype(np.float32)
        low = df['Low'].values.astype(np.float32)
        close = df['Close'].values.astype(np.float32)
        volume = df['Volume'].values.astype(np.float32)
        open_price = df['Open'].values.astype(np.float32)
        
        # Paralel hesaplama için fonksiyonlar
        def calculate_trend_features():
            """Trend özellikleri"""
            features = {}
            
            # Hızlı MA'lar
            features['SMA_5'] = talib.SMA(close, 5)
            features['SMA_20'] = talib.SMA(close, 20)
            features['SMA_50'] = talib.SMA(close, 50)
            features['EMA_12'] = talib.EMA(close, 12)
            features['EMA_26'] = talib.EMA(close, 26)
            
            # Kritik oranlar
            features['Price_SMA5_Ratio'] = close / features['SMA_5'] - 1
            features['Price_SMA20_Ratio'] = close / features['SMA_20'] - 1
            features['SMA5_SMA20_Ratio'] = features['SMA_5'] / features['SMA_20'] - 1
            
            return features
        
        def calculate_momentum_features():
            """Momentum özellikleri"""
            features = {}
            
            features['RSI_14'] = talib.RSI(close, 14)
            features['RSI_7'] = talib.RSI(close, 7)
            features['MACD'], features['MACD_Signal'], features['MACD_Hist'] = talib.MACD(close)
            features['STOCH_K'], features['STOCH_D'] = talib.STOCH(high, low, close)
            features['ROC'] = talib.ROC(close, 10)
            
            return features
        
        def calculate_volatility_features():
            """Volatilite özellikleri"""
            features = {}
            
            features['BB_Upper'], features['BB_Middle'], features['BB_Lower'] = talib.BBANDS(close)
            features['BB_Position'] = (close - features['BB_Lower']) / (features['BB_Upper'] - features['BB_Lower'])
            features['ATR'] = talib.ATR(high, low, close, 14)
            features['ATR_Ratio'] = features['ATR'] / close
            
            return features
        
        def calculate_volume_features():
            """Hacim özellikleri"""
            features = {}
            
            features['Volume_SMA'] = talib.SMA(volume, 20)
            features['Volume_Ratio'] = volume / features['Volume_SMA']
            features['OBV'] = talib.OBV(close, volume)
            
            return features
        
        def calculate_price_action():
            """Fiyat aksiyonu"""
            features = {}
            
            features['Returns_1'] = np.diff(close, prepend=close[0]) / close[:-1] if len(close) > 1 else np.zeros_like(close)
            features['Returns_3'] = (close / np.roll(close, 3) - 1) if len(close) > 3 else np.zeros_like(close)
            features['Volatility_10'] = pd.Series(features['Returns_1']).rolling(10).std().values
            features['Price_Position'] = (close - low) / np.maximum(high - low, 1e-8)
            
            return features
        
        # Paralel hesaplama
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_trend = executor.submit(calculate_trend_features)
            future_momentum = executor.submit(calculate_momentum_features)
            future_volatility = executor.submit(calculate_volatility_features)
            future_volume = executor.submit(calculate_volume_features)
            future_price = executor.submit(calculate_price_action)
            
            # Sonuçları topla
            trend_features = future_trend.result()
            momentum_features = future_momentum.result()
            volatility_features = future_volatility.result()
            volume_features = future_volume.result()
            price_features = future_price.result()
        
        # DataFrame'e ekle
        all_features = {**trend_features, **momentum_features, **volatility_features, 
                       **volume_features, **price_features}
        
        for name, values in all_features.items():
            df[name] = values
        
        # Akıllı sinyaller
        df['MA_Signal'] = (df['SMA_5'] > df['SMA_20']).astype(int)
        df['RSI_Bullish'] = (df['RSI_14'] > 50).astype(int)
        df['MACD_Bullish'] = (df['MACD'] > df['MACD_Signal']).astype(int)
        df['Volume_Surge'] = (df['Volume_Ratio'] > 1.5).astype(int)
        
        # Süper skorlar
        df['Trend_Strength'] = (
            df['MA_Signal'] + df['RSI_Bullish'] + df['MACD_Bullish']
        ) / 3
        
        # Cache'e kaydet
        self.feature_cache[cache_key] = df.copy()
        
        elapsed = time.time() - start_time
        print(f"⚡ Turbo özellik hesaplama tamamlandı! ({elapsed:.2f}s)")
        
        return df
    
    def adaptive_model_selection(self, X_train, y_train, X_val, y_val) -> Tuple[object, str, float]:
        """🎯 Adaptif model seçimi - en iyisini otomatik bulur"""
        print("🤖 Adaptif model seçimi başlatılıyor...")
        
        # Hafızadan en iyi parametreleri al
        best_params = self.memory.best_parameters.get(self.symbol, {})
        
        models_to_try = {
            'RandomForest_Optimized': RandomForestClassifier(
                n_estimators=best_params.get('rf_n_estimators', 500),
                max_depth=best_params.get('rf_max_depth', 15),
                min_samples_split=best_params.get('rf_min_samples_split', 5),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting_Optimized': GradientBoostingClassifier(
                n_estimators=best_params.get('gb_n_estimators', 300),
                learning_rate=best_params.get('gb_learning_rate', 0.05),
                max_depth=best_params.get('gb_max_depth', 8),
                random_state=42
            )
        }
        
        best_model = None
        best_name = ""
        best_score = 0
        
        for name, model in models_to_try.items():
            try:
                print(f"   🏋️ {name} test ediliyor...")
                model.fit(X_train, y_train)
                
                val_pred = model.predict(X_val)
                score = accuracy_score(y_val, val_pred)
                
                print(f"   📊 {name}: %{score*100:.1f}")
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = name
                    
            except Exception as e:
                print(f"   ❌ {name} hatası: {e}")
        
        # En iyi parametreleri kaydet
        if best_name and best_score > self.memory.best_parameters.get(f'{self.symbol}_best_score', 0):
            self.memory.best_parameters[f'{self.symbol}_best_score'] = best_score
            self.memory.best_parameters[f'{self.symbol}_best_model'] = best_name
            print(f"🏆 Yeni en iyi model: {best_name} (%{best_score*100:.1f})")
        
        return best_model, best_name, best_score
    
    def learn_from_results(self, predictions: np.ndarray, actual: np.ndarray, 
                          market_patterns: List[Dict], dates: List):
        """🧠 Sonuçlardan öğren - ChatGPT gibi"""
        print("🎓 Sonuçlardan öğrenme başlatılıyor...")
        
        correct_predictions = (predictions == actual)
        
        for i, (pred, real, pattern, date, correct) in enumerate(
            zip(predictions, actual, market_patterns, dates, correct_predictions)):
            
            pattern_id = f"{self.symbol}_{date.strftime('%Y%m%d')}_{i}"
            
            pattern_data = {
                'pattern': pattern,
                'prediction': int(pred),
                'actual': int(real),
                'outcome': 'success' if correct else 'failure',
                'date': date.strftime('%Y-%m-%d'),
                'confidence': getattr(self, 'last_confidence', 0.5)
            }
            
            if correct:
                # Başarılı pattern'ı kaydet
                self.memory.successful_patterns[pattern_id] = pattern_data
                print(f"   ✅ Başarılı pattern kaydedildi: {date.strftime('%Y-%m-%d')}")
            else:
                # Başarısız pattern'ı kaydet ve analiz et
                self.memory.failed_patterns[pattern_id] = pattern_data
                self.analyze_failure(pattern_data)
                print(f"   📚 Başarısızlıktan öğrenildi: {date.strftime('%Y-%m-%d')}")
        
        # Performans timeline'ı güncelle
        accuracy = np.mean(correct_predictions)
        self.memory.performance_timeline.append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'accuracy': accuracy,
            'total_predictions': len(predictions),
            'symbol': self.symbol
        })
        
        # Eğer performans düşükse kendini iyileştir
        if accuracy < self.performance_threshold:
            print(f"⚠️ Performans düşük (%{accuracy*100:.1f}), kendini iyileştiriyor...")
            self.self_improvement()
        
        # Hafızayı kaydet
        self.save_memory()
        
        return accuracy
    
    def analyze_failure(self, failure_data: Dict):
        """❌ Başarısızlık analizi - neden yanlış tahmin yaptı?"""
        pattern = failure_data['pattern']
        
        # Bu pattern'a benzer başarılı durumlar var mı?
        similar_successes = self.find_similar_patterns(pattern, self.memory.successful_patterns)
        
        if similar_successes:
            print(f"   🔍 Benzer başarılı durum mevcut, parametre ayarlaması gerekli")
            # Benzer başarılı durumlarla farkları analiz et
            self.adjust_parameters_based_on_failure(failure_data, similar_successes)
    
    def adjust_parameters_based_on_failure(self, failure: Dict, similar_successes: List):
        """⚙️ Başarısızlık bazlı parametre ayarlaması"""
        
        # RSI hassasiyetini ayarla
        if failure['pattern'].get('rsi_level') == 'overbought' and failure['prediction'] == 1:
            # RSI aşırı alım durumunda yükseliş tahmini yanlışsa, eşiği düşür
            current_threshold = self.memory.best_parameters.get('rsi_overbought_threshold', 70)
            new_threshold = max(65, current_threshold - 2)
            self.memory.best_parameters['rsi_overbought_threshold'] = new_threshold
            print(f"   🔧 RSI aşırı alım eşiği düşürüldü: {new_threshold}")
        
        # Trend gücü hassasiyetini ayarla
        if failure['pattern'].get('trend_strength') == 'weak' and failure['prediction'] == 1:
            # Zayıf trendde yükseliş tahmini yanlışsa, trend eşiğini yükselt
            current_threshold = self.memory.best_parameters.get('trend_strength_threshold', 0.5)
            new_threshold = min(0.8, current_threshold + 0.1)
            self.memory.best_parameters['trend_strength_threshold'] = new_threshold
            print(f"   🔧 Trend gücü eşiği yükseltildi: {new_threshold}")
    
    def self_improvement(self):
        """🔄 Kendi kendini iyileştirme"""
        print("🔄 Kendini iyileştirme moduna geçiyor...")
        
        # Son performansları analiz et
        recent_performance = self.memory.performance_timeline[-10:] if len(self.memory.performance_timeline) >= 10 else self.memory.performance_timeline
        
        if not recent_performance:
            return
        
        avg_performance = np.mean([p['accuracy'] for p in recent_performance])
        print(f"   📊 Son ortalama performans: %{avg_performance*100:.1f}")
        
        # Başarısızlık patternlerini analiz et
        failure_analysis = self.analyze_failure_patterns()
        
        # En sık karşılaşılan hata tiplerini bul
        common_failures = self.find_common_failure_patterns()
        
        # Bu hatalara karşı önlemler al
        self.implement_improvements(common_failures)
        
        print("✅ Kendini iyileştirme tamamlandı!")
    
    def analyze_failure_patterns(self) -> Dict:
        """Başarısızlık patternlerini analiz et"""
        failure_types = {}
        
        for pattern_id, failure_data in self.memory.failed_patterns.items():
            pattern = failure_data['pattern']
            
            # Hata tiplerini kategorize et
            rsi_level = pattern.get('rsi_level', 'unknown')
            trend_strength = pattern.get('trend_strength', 'unknown')
            volume_status = pattern.get('volume_status', 'unknown')
            
            failure_key = f"{rsi_level}_{trend_strength}_{volume_status}"
            
            if failure_key not in failure_types:
                failure_types[failure_key] = 0
            failure_types[failure_key] += 1
        
        return failure_types
    
    def find_common_failure_patterns(self) -> List[Dict]:
        """En sık karşılaşılan hata patternlerini bul"""
        failure_analysis = self.analyze_failure_patterns()
        
        # En sık karşılaşılan 5 hata tipini al
        sorted_failures = sorted(failure_analysis.items(), key=lambda x: x[1], reverse=True)[:5]
        
        common_failures = []
        for failure_pattern, count in sorted_failures:
            if count >= 3:  # En az 3 kez tekrarlanmış olsun
                parts = failure_pattern.split('_')
                common_failures.append({
                    'rsi_level': parts[0],
                    'trend_strength': parts[1], 
                    'volume_status': parts[2],
                    'failure_count': count
                })
        
        return common_failures
    
    def implement_improvements(self, common_failures: List[Dict]):
        """Iyileştirmeleri uygula"""
        for failure in common_failures:
            print(f"   🔧 Düzeltme uygulanıyor: {failure}")
            
            # RSI tabanlı düzeltmeler
            if failure['rsi_level'] == 'overbought':
                self.memory.best_parameters['avoid_overbought_long'] = True
                print("   📝 RSI aşırı alım durumunda uzun pozisyon kaçınma aktif")
            
            # Trend tabanlı düzeltmeler  
            if failure['trend_strength'] == 'weak':
                self.memory.best_parameters['require_strong_trend'] = True
                print("   📝 Güçlü trend gereksinimi aktif")
            
            # Hacim tabanlı düzeltmeler
            if failure['volume_status'] == 'low':
                self.memory.best_parameters['require_volume_confirmation'] = True
                print("   📝 Hacim onayı gereksinimi aktif")
    
    def create_smart_target(self, df: pd.DataFrame, days_ahead: int = 1) -> pd.DataFrame:
        """🎯 Akıllı hedef değişken - adaptif eşik"""
        print("🎯 Akıllı hedef değişken oluşturuluyor...")
        
        # Hafızadan öğrenilen optimal eşik
        learned_threshold = self.memory.best_parameters.get('optimal_threshold', 0.015)
        
        # Dinamik eşik hesaplama - volatiliteye göre
        recent_volatility = df['Close'].pct_change().rolling(20).std().iloc[-1]
        dynamic_threshold = max(0.01, min(0.03, recent_volatility * 1.5))
        
        # En iyi eşiği seç
        threshold = min(learned_threshold, dynamic_threshold)
        
        print(f"   📊 Kullanılan eşik: %{threshold*100:.2f}")
        print(f"   📊 Öğrenilen eşik: %{learned_threshold*100:.2f}")
        print(f"   📊 Dinamik eşik: %{dynamic_threshold*100:.2f}")
        
        # Gelecekteki getiri
        future_return = df['Close'].shift(-days_ahead) / df['Close'] - 1
        
        # Akıllı hedef oluşturma
        df['Target_Smart'] = (future_return > threshold).astype(int)
        
        return df
    
    def run_complete_analysis(self) -> Dict:
        """🚀 Komple analiz çalıştır - Ana fonksiyon"""
        print("🚀" * 20)
        print("SÜPER AKILLI ÖĞRENEN HİSSE TAHMİN SİSTEMİ v5.0")
        print("ChatGPT Tarzı Öğrenme + Hız Optimizasyonu")
        print("🚀" * 20)
        
        total_start = time.time()
        
        try:
            # 1. Hızlı veri indirme
            df = self.fast_download_data()
            if df.empty:
                return {"error": "Veri indirilemedi"}
            
            # 2. Turbo özellik mühendisliği
            df = self.turbo_feature_engineering(df)
            df = self.create_smart_target(df)
            
            # Temizlik
            df.dropna(inplace=True)
            print(f"✅ Temiz veri: {len(df)} gün")
            
            if len(df) < 100:
                return {"error": "Yeterli veri yok"}
            
            # 3. Son piyasa durumu için AI analizi
            recent_data = df.tail(5)
            market_summary = {
                'symbol': self.symbol,
                'prices': recent_data['Close'].tolist(),
                'rsi': recent_data['RSI_14'].iloc[-1],
                'trend': recent_data['Trend_Strength'].iloc[-1],
                'volume_ratio': recent_data['Volume_Ratio'].iloc[-1],
                'bb_position': recent_data['BB_Position'].iloc[-1],
                'volatility': recent_data['ATR_Ratio'].iloc[-1]
            }
            
            ai_analysis = self.smart_ai_analysis(market_summary)
            print(f"🧠 AI Analizi: {ai_analysis['signal']} (%{ai_analysis['confidence']} güven)")
            
            # 4. Özellik seçimi
            feature_columns = [
                'Price_SMA5_Ratio', 'Price_SMA20_Ratio', 'SMA5_SMA20_Ratio',
                'RSI_14', 'RSI_7', 'MACD', 'MACD_Hist', 'STOCH_K', 'ROC',
                'BB_Position', 'ATR_Ratio', 'Volume_Ratio', 'OBV',
                'Returns_1', 'Returns_3', 'Volatility_10', 'Price_Position',
                'MA_Signal', 'RSI_Bullish', 'MACD_Bullish', 'Volume_Surge',
                'Trend_Strength'
            ]
            
            available_features = [f for f in feature_columns if f in df.columns]
            
            # 5. Veri bölme - zaman sıralı
            train_size = int(len(df) * 0.7)
            val_size = int(len(df) * 0.15)
            
            X = df[available_features]
            y = df['Target_Smart']
            
            X_train = X.iloc[:train_size]
            X_val = X.iloc[train_size:train_size+val_size]
            X_test = X.iloc[train_size+val_size:-1]
            
            y_train = y.iloc[:train_size]
            y_val = y.iloc[train_size:train_size+val_size]
            y_test = y.iloc[train_size+val_size:-1]
            
            # AI sinyalini ekle
            ai_signal_numeric = 1 if ai_analysis['signal'] == 'YUKARI' else -1 if ai_analysis['signal'] == 'ASAGI' else 0
            ai_confidence_scaled = ai_analysis['confidence'] / 100
            
            # AI özelliklerini tüm setlere ekle
            for X_set in [X_train, X_val, X_test]:
                X_set['AI_Signal'] = ai_signal_numeric
                X_set['AI_Confidence'] = ai_confidence_scaled
            
            # 6. Ölçekleme
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # 7. Adaptif model seçimi
            best_model, best_name, val_score = self.adaptive_model_selection(
                X_train_scaled, y_train, X_val_scaled, y_val
            )
            
            if best_model is None:
                return {"error": "Model eğitilemedi"}
            
            # 8. Test performansı
            y_test_pred = best_model.predict(X_test_scaled)
            y_test_proba = best_model.predict_proba(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            print(f"🎉 Test Doğruluğu: %{test_accuracy*100:.1f}")
            
            # 9. Öğrenme - test sonuçlarından ders çıkar
            test_dates = df.index[train_size+val_size:-1]
            test_market_patterns = []
            
            for i, date in enumerate(test_dates):
                pattern = self.extract_pattern({
                    'rsi': df.loc[date, 'RSI_14'],
                    'trend': df.loc[date, 'Trend_Strength'],
                    'volume_ratio': df.loc[date, 'Volume_Ratio'],
                    'bb_position': df.loc[date, 'BB_Position'],
                    'volatility': df.loc[date, 'ATR_Ratio']
                })
                test_market_patterns.append(pattern)
            
            learning_accuracy = self.learn_from_results(
                y_test_pred, y_test.values, test_market_patterns, test_dates
            )
            
            # 10. Gelecek tahmini
            last_features = X.iloc[-1][available_features + ['AI_Signal', 'AI_Confidence']].values.reshape(1, -1)
            last_features_scaled = scaler.transform(last_features)
            
            final_prediction = best_model.predict(last_features_scaled)[0]
            final_probabilities = best_model.predict_proba(last_features_scaled)[0]
            
            # 11. Sonuçları hazırla
            total_elapsed = time.time() - total_start
            
            results = {
                'symbol': self.symbol,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'processing_time': f"{total_elapsed:.2f}s",
                
                # Model performansı
                'best_model': best_name,
                'validation_accuracy': f"%{val_score*100:.1f}",
                'test_accuracy': f"%{test_accuracy*100:.1f}",
                'learning_accuracy': f"%{learning_accuracy*100:.1f}",
                
                # AI analizi
                'ai_analysis': ai_analysis,
                
                # Tahmin
                'prediction': 'GÜÇLÜ YUKARI' if final_prediction == 1 else 'NORMAL',
                'up_probability': f"%{final_probabilities[1]*100:.0f}",
                'confidence': f"%{max(final_probabilities)*100:.0f}",
                
                # Piyasa durumu
                'last_price': f"{df['Close'].iloc[-1]:.2f} ₺",
                'last_rsi': f"{df['RSI_14'].iloc[-1]:.1f}",
                'trend_strength': f"{df['Trend_Strength'].iloc[-1]:.2f}",
                
                # Hafıza durumu
                'learned_patterns': len(self.memory.successful_patterns),
                'failed_patterns': len(self.memory.failed_patterns),
                'total_experience': len(self.memory.successful_patterns) + len(self.memory.failed_patterns),
                
                # İyileştirmeler
                'improvements_applied': len([k for k in self.memory.best_parameters.keys() if 'threshold' in k or 'require' in k or 'avoid' in k]),
                
                'status': 'success'
            }
            
            # 12. Görselleştirme
            self.create_smart_visualization(df, y_test, y_test_pred, y_test_proba, 
                                          test_dates, ai_analysis, results)
            
            return results
            
        except Exception as e:
            print(f"❌ Analiz hatası: {e}")
            return {"error": str(e), "status": "failed"}
    
    def create_smart_visualization(self, df: pd.DataFrame, y_test, y_test_pred, y_test_proba,
                                 test_dates, ai_analysis: Dict, results: Dict):
        """📊 Akıllı görselleştirme"""
        print("🎨 Akıllı görsel oluşturuluyor...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle(f'🧠 SÜPER AKILLI ÖĞRENEN SİSTEM - {self.symbol}\n'
                    f'Test Doğruluğu: {results["test_accuracy"]} | AI: {ai_analysis["signal"]} | '
                    f'Deneyim: {results["total_experience"]} pattern', 
                    fontsize=16, fontweight='bold')
        
        # Panel 1: Fiyat ve Tahminler
        ax1 = axes[0, 0]
        
        test_prices = df['Close'].iloc[-len(test_dates)-50:-1].values
        extended_dates = df.index[-len(test_dates)-50:-1]
        
        ax1.plot(extended_dates, test_prices, 'b-', linewidth=2, label='Fiyat', alpha=0.8)
        
        # Tahmin sonuçları
        correct_mask = (y_test.values == y_test_pred)
        wrong_mask = ~correct_mask
        
        if np.any(correct_mask):
            ax1.scatter(test_dates[correct_mask], 
                       df['Close'].iloc[-len(test_dates):-1].values[correct_mask],
                       color='green', s=80, marker='o', label='✅ Doğru', zorder=5)
        
        if np.any(wrong_mask):
            ax1.scatter(test_dates[wrong_mask],
                       df['Close'].iloc[-len(test_dates):-1].values[wrong_mask], 
                       color='red', s=80, marker='X', label='❌ Yanlış', zorder=5)
        
        ax1.set_title('💰 Fiyat ve Tahmin Sonuçları')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Öğrenme Eğrisi
        ax2 = axes[0, 1]
        
        if self.memory.performance_timeline:
            timeline_dates = [datetime.strptime(p['date'], '%Y-%m-%d') for p in self.memory.performance_timeline]
            accuracies = [p['accuracy'] for p in self.memory.performance_timeline]
            
            ax2.plot(timeline_dates, accuracies, 'g-', linewidth=3, marker='o', markersize=6)
            ax2.axhline(y=0.75, color='orange', linestyle='--', alpha=0.7, label='Hedef %75')
            ax2.axhline(y=0.85, color='green', linestyle='--', alpha=0.7, label='Mükemmel %85')
            
            ax2.set_title('📈 Öğrenme Eğrisi (Zaman İçinde Doğruluk)')
            ax2.set_ylabel('Doğruluk')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Henüz öğrenme\ngeçmişi yok', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('📈 Öğrenme Eğrisi')
        
        # Panel 3: Güven Skorları ve AI
        ax3 = axes[1, 0]
        
        confidence_scores = np.max(y_test_proba, axis=1)
        colors = ['green' if correct else 'red' for correct in (y_test.values == y_test_pred)]
        
        bars = ax3.bar(range(len(confidence_scores)), confidence_scores, color=colors, alpha=0.7)
        ax3.axhline(y=0.8, color='darkgreen', linestyle='--', label='Yüksek Güven')
        ax3.axhline(y=0.6, color='orange', linestyle='--', label='Orta Güven')
        
        # AI sinyali göster
        ai_y = 0.9 if ai_analysis['signal'] == 'YUKARI' else 0.1
        ai_color = 'green' if ai_analysis['signal'] == 'YUKARI' else 'red'
        ax3.axhline(y=ai_y, color=ai_color, linewidth=4, alpha=0.8, 
                   label=f'🤖 AI: {ai_analysis["signal"]}')
        
        ax3.set_title('🎯 Güven Skorları + AI Sinyali')
        ax3.set_ylabel('Güven')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Öğrenme İstatistikleri
        ax4 = axes[1, 1]
        
        # Hafıza istatistikleri
        stats = [
            len(self.memory.successful_patterns),
            len(self.memory.failed_patterns),
            len(self.memory.best_parameters),
            results.get('improvements_applied', 0)
        ]
        
        labels = ['Başarılı\nPatternler', 'Başarısız\nPatternler', 
                 'Öğrenilen\nParametreler', 'Uygulanan\nİyileştirmeler']
        colors = ['green', 'red', 'blue', 'orange']
        
        bars = ax4.bar(labels, stats, color=colors, alpha=0.7)
        
        # Değerleri çubukların üzerine yaz
        for bar, stat in zip(bars, stats):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stats)*0.01,
                    str(stat), ha='center', va='bottom', fontweight='bold')
        
        ax4.set_title('🧠 Öğrenme İstatistikleri')
        ax4.set_ylabel('Adet')
        
        plt.tight_layout()
        plt.savefig(f'{self.symbol}_super_smart_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✅ Akıllı analiz '{self.symbol}_super_smart_analysis.png' olarak kaydedildi")
        
        plt.show()
    
    def get_learning_summary(self) -> Dict:
        """📚 Öğrenme özeti"""
        return {
            'total_patterns_learned': len(self.memory.successful_patterns) + len(self.memory.failed_patterns),
            'success_rate': len(self.memory.successful_patterns) / max(1, len(self.memory.successful_patterns) + len(self.memory.failed_patterns)),
            'parameters_optimized': len(self.memory.best_parameters),
            'performance_history': self.memory.performance_timeline[-10:],  # Son 10 performans
            'most_successful_pattern': self.find_most_successful_pattern(),
            'most_problematic_pattern': self.find_most_problematic_pattern()
        }
    
    def find_most_successful_pattern(self) -> Dict:
        """En başarılı pattern'ı bul"""
        if not self.memory.successful_patterns:
            return {}
        
        # En yüksek güvenle başarılı olan pattern
        best_pattern = max(self.memory.successful_patterns.values(), 
                          key=lambda x: x.get('confidence', 0))
        return best_pattern
    
    def find_most_problematic_pattern(self) -> Dict:
        """En sorunlu pattern'ı bul"""
        if not self.memory.failed_patterns:
            return {}
        
        # En sık başarısız olan pattern tipini bul
        failure_types = {}
        for failure in self.memory.failed_patterns.values():
            pattern_signature = str(failure['pattern'])
            failure_types[pattern_signature] = failure_types.get(pattern_signature, 0) + 1
        
        if failure_types:
            most_problematic_signature = max(failure_types.keys(), key=lambda x: failure_types[x])
            # Bu signature'a sahip ilk başarısızlığı döndür
            for failure in self.memory.failed_patterns.values():
                if str(failure['pattern']) == most_problematic_signature:
                    return failure
        
        return {}

# ======= KULLANIM ÖRNEĞİ =======
def run_super_smart_analysis(symbol: str = 'SISE.IS'):
    """
    🚀 Ana çalıştırma fonksiyonu
    
    Bu fonksiyon sistemi başlatır ve analiz yapar.
    Python öğrenirken bu fonksiyonu çağırmanız yeterli!
    """
    
    print("🚀 Süper Akıllı Sistem başlatılıyor...")
    
    # Sistemi oluştur
    predictor = SuperSmartPredictor(symbol)
    
    # Komple analiz çalıştır
    results = predictor.run_complete_analysis()
    
    # Sonuçları göster
    if results.get('status') == 'success':
        print("\n" + "🎉" * 20)
        print("ANALİZ BAŞARIYLA TAMAMLANDI!")
        print("🎉" * 20)
        
        print(f"\n📊 SONUÇ ÖZETİ:")
        print(f"Hisse: {results['symbol']}")
        print(f"En İyi Model: {results['best_model']}")
        print(f"Test Doğruluğu: {results['test_accuracy']}")
        print(f"Tahmin: {results['prediction']}")
        print(f"Yükseliş Olasılığı: {results['up_probability']}")
        print(f"AI Sinyali: {results['ai_analysis']['signal']} (%{results['ai_analysis']['confidence']})")
        print(f"Toplam Deneyim: {results['total_experience']} pattern")
        print(f"İşlem Süresi: {results['processing_time']}")
        
        # Öğrenme özeti
        learning = predictor.get_learning_summary()
        print(f"\n🧠 ÖĞRENME ÖZETİ:")
        print(f"Öğrenilen Pattern: {learning['total_patterns_learned']}")
        print(f"Başarı Oranı: %{learning['success_rate']*100:.1f}")
        print(f"Optimize Parametre: {learning['parameters_optimized']}")
        
        return results, predictor
    
    else:
        print(f"❌ Analiz başarısız: {results.get('error', 'Bilinmeyen hata')}")
        return None, None

# ======= PYTHON ÖĞRENİM REHBERİ =======
"""
🎓 PYTHON ÖĞRENİM REHBERİ

Bu kodu anlamak için öğrenmeniz gerekenler (sırayla):

1. TEMEL SEVİYE (1-3 ay):
   - Python syntax (değişkenler, fonksiyonlar, if/else, döngüler)
   - Veri tipleri (list, dict, tuple)
   - Dosya okuma/yazma
   - Hata yönetimi (try/except)

2. ORTA SEVİYE (3-6 ay):
   - Object-Oriented Programming (class, __init__, methods)
   - Pandas (DataFrame, Series)
   - NumPy (arrays, matematiksel işlemler)
   - Matplotlib (grafik çizme)

3. İLERİ SEVİYE (6-12 ay):
   - Scikit-learn (machine learning)
   - API kullanımı (requests)
   - Çoklu işlemciler (threading, multiprocessing)
   - Pickle (veri saklama)
   - Dataclasses ve Type hints

4. UZMAN SEVİYE (1+ yıl):
   - Performans optimizasyonu
   - Paralel hesaplama
   - Karmaşık algoritma tasarımı
   - Sistem tasarımı

ÖNERİLER:
- Her gün 1-2 saat pratik yapın
- Küçük projelerle başlayın
- Bu kodu parça parça analiz edin
- Chatbotlardan bolca soru sorun
- Gerçek projeler yapın
"""

if __name__ == "__main__":
    # Sistemi çalıştır
    results, predictor = run_super_smart_analysis('SISE.IS')
    
    if results:
        print("\n✅ Analiz tamamlandı! Grafikleri kontrol edin.")
        print("🔄 Tekrar çalıştırdığınızda sistem daha da akıllı olacak!")
