import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Gelişmiş Makine Öğrenmesi Kütüphaneleri
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor,
                             VotingRegressor, StackingRegressor)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, 
                                 BayesianRidge, HuberRegressor, SGDRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import joblib

# Deep Learning (basit versiyon)
from sklearn.neural_network import MLPRegressor

# Teknik Analiz
import talib

# Görselleştirme
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')

class AdvancedBISTEnsembleBot:
    def __init__(self, db_path="bist_advanced.db"):
        """
        Gelişmiş BIST Ensemble AI Bot
        Çoklu model sistemi ve hata öğrenme mekanizması
        """
        self.db_path = db_path
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        # Sadece sorunsuz modelleri kullan
        self.base_models = {
            # En güvenilir modeller
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
            'extra_trees': ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
            
            # Linear modeller
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1, max_iter=2000),
            
            # Diğer güvenilir modeller
            'knn': KNeighborsRegressor(n_neighbors=5),
            'decision_tree': DecisionTreeRegressor(max_depth=8, random_state=42),
        }
        
        # Ensemble sistemi
        self.ensemble_model = None
        self.model_weights = {}
        self.error_learning_data = {}
        self.prediction_confidence = {}
        
        # Veritabanı oluştur
        self.init_database()
        
        print("🚀 Gelişmiş BIST Ensemble AI Bot başlatıldı!")
        print(f"🤖 {len(self.base_models)} farklı model yüklendi")
        print("🧠 Hata öğrenme sistemi aktif")
    
    def init_database(self):
        """Gelişmiş veritabanı yapısı"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Fiyat verileri
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                adj_close REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Model performansları (detaylı)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                symbol TEXT,
                mse REAL,
                mae REAL,
                rmse REAL,
                r2_score REAL,
                accuracy_1pct REAL,
                accuracy_3pct REAL,
                accuracy_5pct REAL,
                directional_accuracy REAL,
                training_samples INTEGER,
                date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tahminler ve gerçek değerler
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions_detailed (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                prediction_date TEXT,
                current_price REAL,
                predicted_price REAL,
                actual_price REAL,
                model_name TEXT,
                confidence_score REAL,
                prediction_error REAL,
                direction_correct BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Hata öğrenme tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_learning (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                symbol TEXT,
                error_type TEXT,
                error_magnitude REAL,
                market_condition TEXT,
                correction_applied TEXT,
                learning_date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Ensemble ağırlıkları
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ensemble_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                model_name TEXT,
                weight REAL,
                performance_score REAL,
                last_updated TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Gelişmiş veritabanı hazırlandı")
    
    def fetch_enhanced_data(self, symbols=["XU100.IS", "GARAN.IS", "AKBNK.IS", "THYAO.IS", "BIMAS.IS"]):
        """Gelişmiş veri toplama - daha güvenilir"""
        print("📊 Gelişmiş veri toplama başlıyor...")
        
        conn = sqlite3.connect(self.db_path)
        successful_downloads = 0
        
        for symbol in symbols:
            try:
                print(f"  📥 {symbol} indiriliyor...")
                
                # Veriyi farklı periyotlarla dene
                for period in ["1y", "6mo", "3mo"]:
                    try:
                        ticker = yf.Ticker(symbol)
                        data = ticker.history(period=period)
                        
                        if not data.empty and len(data) > 20:
                            print(f"    ✅ {symbol}: {len(data)} gün verisi alındı ({period})")
                            break
                    except:
                        continue
                else:
                    print(f"    ❌ {symbol}: Veri alınamadı")
                    continue
                # Veritabanına kaydet
                cursor = conn.cursor()
                for date, row in data.iterrows():
                    try:
                        cursor.execute('''
                            INSERT OR REPLACE INTO price_data 
                            (symbol, date, open, high, low, close, volume, adj_close)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (symbol, date.strftime('%Y-%m-%d'), 
                             float(row['Open']), float(row['High']), float(row['Low']), 
                             float(row['Close']), int(row['Volume']), float(row['Close'])))
                    except Exception as e:
                        print(f"      ⚠️ Satır kayıt hatası: {e}")
                        continue
                
                successful_downloads += 1
                
            except Exception as e:
                print(f"    ❌ {symbol} hata: {str(e)}")
        
        conn.commit()
        conn.close()
        print(f"✅ {successful_downloads}/{len(symbols)} sembol başarıyla güncellendi")
    
    def calculate_advanced_features(self, df):
        """Gelişmiş teknik analiz özellikleri - güvenli versiyon"""
        try:
            # Temel özellikler
            df['returns'] = df['close'].pct_change().fillna(0)
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
            
            # Hareketli ortalamalar (daha kısa periyotlar)
            for period in [5, 10, 20]:
                if len(df) > period:
                    df[f'sma_{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
                    df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                else:
                    df[f'sma_{period}'] = df['close']
                    df[f'ema_{period}'] = df['close']
            
            # Volatilite ölçümleri
            df['volatility_10'] = df['returns'].rolling(window=10, min_periods=1).std().fillna(0.01)
            df['volatility_20'] = df['returns'].rolling(window=20, min_periods=1).std().fillna(0.01)
            df['volatility_ratio'] = (df['volatility_10'] / df['volatility_20']).fillna(1.0)
            
            # Basitleştirilmiş momentum indikatörleri
            df['rsi_14'] = self.calculate_rsi_safe(df['close'], 14)
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands_safe(df['close'])
            df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle']).fillna(0.1)
            df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])).fillna(0.5)
            
            # Volume indikatörleri
            df['volume_sma'] = df['volume'].rolling(window=10, min_periods=1).mean()
            df['volume_ratio'] = (df['volume'] / df['volume_sma']).fillna(1.0)
            
            # Fiyat pozisyonu
            df['high_low_ratio'] = (df['high'] / df['low']).fillna(1.0)
            df['close_position'] = ((df['close'] - df['low']) / (df['high'] - df['low'])).fillna(0.5)
            
            # Basit trend indikatörleri
            df['price_trend_5'] = (df['close'] / df['close'].shift(5)).fillna(1.0)
            df['price_trend_10'] = (df['close'] / df['close'].shift(10)).fillna(1.0)
            
            # Lag özellikleri (sadece kısa vadeli)
            for lag in [1, 2, 3]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag).fillna(df['close'])
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag).fillna(0)
            
            # Tüm sonsuz ve NaN değerleri temizle
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return df
            
        except Exception as e:
            print(f"    ⚠️ Feature hesaplama hatası: {e}")
            return df
    
    def calculate_rsi_safe(self, prices, period=14):
        """Güvenli RSI hesaplama"""
        try:
            if len(prices) < period:
                return pd.Series([50] * len(prices), index=prices.index)
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            
            # Sıfıra bölme kontrolü
            loss = loss.replace(0, 0.001)
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def calculate_bollinger_bands_safe(self, prices, period=20, std_dev=2):
        """Güvenli Bollinger Bands hesaplama"""
        try:
            if len(prices) < period:
                sma = prices.expanding().mean()
                std = prices.expanding().std().fillna(0.01)
            else:
                sma = prices.rolling(window=period, min_periods=1).mean()
                std = prices.rolling(window=period, min_periods=1).std().fillna(0.01)
            
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            
            return upper, sma, lower
        except:
            return prices * 1.02, prices, prices * 0.98
    
    def test_data_and_train_if_ready(self):
        """Veri kontrolü ve hazırsa eğitim"""
        print("🔍 Veri kontrolü yapılıyor...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Her sembol için veri sayısını kontrol et
        symbols = ["XU100.IS", "GARAN.IS", "AKBNK.IS", "THYAO.IS", "BIMAS.IS"]
        ready_symbols = []
        
        for symbol in symbols:
            query = f"SELECT COUNT(*) as count FROM price_data WHERE symbol = '{symbol}'"
            result = pd.read_sql_query(query, conn)
            count = result['count'].iloc[0]
            
            print(f"  📊 {symbol}: {count} gün verisi")
            
            if count >= 30:  # Minimum 30 gün
                ready_symbols.append(symbol)
                print(f"    ✅ {symbol} eğitime hazır")
            else:
                print(f"    ❌ {symbol} yetersiz veri")
        
        conn.close()
        
        if ready_symbols:
            print(f"\n🎯 {len(ready_symbols)} sembol eğitilecek: {ready_symbols}")
            for symbol in ready_symbols:
                self.train_ensemble_system(symbol)
        else:
            print("\n❌ Hiçbir sembol eğitime hazır değil")
        
        return ready_symbols
    
    def prepare_ensemble_features(self, symbol):
        """Ensemble için özellik hazırlama"""
        conn = sqlite3.connect(self.db_path)
        
        query = f"""
            SELECT * FROM price_data 
            WHERE symbol = '{symbol}' 
            ORDER BY date ASC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty or len(df) < 50:  # Minimum veri kontrolünü düşür
            return None, None, None
        
        # Index'i date olarak ayarla
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Gelişmiş özellikler
        df = self.calculate_advanced_features(df)
        
        # Hedef değişkenler (farklı zaman dilimleri)
        df['target_1d'] = df['close'].shift(-1)
        df['target_3d'] = df['close'].shift(-3)
        df['target_5d'] = df['close'].shift(-5)
        
        # Özellik seçimi - sadece hesaplanan özellikler
        feature_cols = [col for col in df.columns if col not in 
                       ['id', 'symbol', 'created_at', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 
                        'target_1d', 'target_3d', 'target_5d'] and 
                       not col.startswith('Unnamed') and 
                       col in df.columns]  # Var olan kolonları kontrol et
        
        # NaN temizle ve minimum veri kontrolü
        df_clean = df.dropna()
        
        if len(df_clean) < 30:  # Daha düşük minimum veri
            return None, None, None
        
        X = df_clean[feature_cols].values
        y = df_clean['target_1d'].values
        dates = df_clean.index
        
        return X, y, dates
    
    def train_ensemble_system(self, symbol):
        """Ensemble sistem eğitimi"""
        print(f"🎯 {symbol} için ensemble sistem eğitimi...")
        
        X, y, dates = self.prepare_ensemble_features(symbol)
        
        if X is None:
            print(f"❌ {symbol} için yeterli veri yok")
            return
        
        # Veriyi böl
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # En iyi scaler'ı seç
        best_scaler = None
        best_score = -np.inf
        
        for scaler_name, scaler in self.scalers.items():
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Basit model ile test
            temp_model = RandomForestRegressor(n_estimators=10, random_state=42)
            temp_model.fit(X_train_scaled, y_train)
            score = temp_model.score(X_test_scaled, y_test)
            
            if score > best_score:
                best_score = score
                best_scaler = scaler
        
        # En iyi scaler ile veriyi ölçekle
        X_train_scaled = best_scaler.fit_transform(X_train)
        X_test_scaled = best_scaler.transform(X_test)
        
        # Her modeli eğit ve değerlendir
        model_performances = {}
        trained_models = {}
        
        print(f"  🤖 {len(self.base_models)} model eğitiliyor...")
        
        for model_name, model in self.base_models.items():
            try:
                print(f"    ⚙️ {model_name} eğitiliyor...")
                
                # Model eğitimi
                model_copy = self._get_model_copy(model)
                model_copy.fit(X_train_scaled, y_train)
                
                # Tahmin
                y_pred = model_copy.predict(X_test_scaled)
                
                # Detaylı performans metrikleri
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # Doğruluk metrikleri (farklı toleranslar)
                acc_1pct = np.mean(np.abs((y_test - y_pred) / y_test) < 0.01)
                acc_3pct = np.mean(np.abs((y_test - y_pred) / y_test) < 0.03)
                acc_5pct = np.mean(np.abs((y_test - y_pred) / y_test) < 0.05)
                
                # Yön doğruluğu
                y_test_direction = np.sign(np.diff(np.concatenate([[y_train[-1]], y_test])))
                y_pred_direction = np.sign(y_pred - y_train[-1])
                directional_acc = np.mean(y_test_direction == y_pred_direction)
                
                # Performans skoru (ağırlıklı)
                performance_score = (r2 * 0.3 + acc_5pct * 0.3 + directional_acc * 0.4)
                
                model_performances[model_name] = {
                    'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2,
                    'acc_1pct': acc_1pct, 'acc_3pct': acc_3pct, 'acc_5pct': acc_5pct,
                    'directional_acc': directional_acc, 'performance_score': performance_score
                }
                
                trained_models[model_name] = model_copy
                
                # Performansı kaydet
                self.save_detailed_performance(model_name, symbol, model_performances[model_name], len(X_train))
                
                print(f"      ✅ R²: {r2:.4f}, Doğruluk(5%): {acc_5pct:.2%}, Yön: {directional_acc:.2%}")
                
            except Exception as e:
                print(f"      ❌ {model_name} hata: {str(e)}")
        
        if not model_performances:
            print(f"❌ {symbol} için hiçbir model eğitilemedi")
            return
        
        # Ensemble ağırlıkları hesapla
        total_score = sum([perf['performance_score'] for perf in model_performances.values()])
        weights = {}
        
        for model_name, perf in model_performances.items():
            if total_score > 0:
                weights[model_name] = perf['performance_score'] / total_score
            else:
                weights[model_name] = 1.0 / len(model_performances)
        
        # En iyi modelleri seç (top 70% performance)
        sorted_models = sorted(model_performances.items(), 
                             key=lambda x: x[1]['performance_score'], 
                             reverse=True)
        
        top_models_count = max(3, int(len(sorted_models) * 0.7))
        top_models = dict(sorted_models[:top_models_count])
        
        # Voting Regressor oluştur
        voting_estimators = [(name, trained_models[name]) for name in top_models.keys()]
        voting_regressor = VotingRegressor(estimators=voting_estimators)
        voting_regressor.fit(X_train_scaled, y_train)
        
        # Ensemble modelini kaydet
        self.ensemble_model = voting_regressor
        self.model_weights[symbol] = weights
        
        # Model dosyalarını kaydet
        joblib.dump(voting_regressor, f'ensemble_model_{symbol}.pkl')
        joblib.dump(best_scaler, f'scaler_{symbol}.pkl')
        joblib.dump(weights, f'weights_{symbol}.pkl')
        
        # Ağırlıkları veritabanına kaydet
        self.save_ensemble_weights(symbol, weights, model_performances)
        
        print(f"  🏆 Ensemble model hazır: {len(top_models)} model kullanılıyor")
        print(f"  📊 En iyi modeller: {list(top_models.keys())[:3]}")
        
    def _get_model_copy(self, model):
        """Model kopyası oluştur"""
        from sklearn.base import clone
        return clone(model)
    
    def predict_with_ensemble(self, symbol):
        """Ensemble ile tahmin"""
        try:
            # Modeli yükle
            try:
                ensemble_model = joblib.load(f'ensemble_model_{symbol}.pkl')
                scaler = joblib.load(f'scaler_{symbol}.pkl')
                weights = joblib.load(f'weights_{symbol}.pkl')
            except:
                print(f"❌ {symbol} için model bulunamadı")
                return None
            
            # Son veriyi al
            conn = sqlite3.connect(self.db_path)
            query = f"""
                SELECT * FROM price_data 
                WHERE symbol = '{symbol}' 
                ORDER BY date DESC 
                LIMIT 200
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return None
            
            # Veriyi ters çevir
            df = df.iloc[::-1].reset_index(drop=True)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Özellikleri hesapla
            df = self.calculate_advanced_features(df)
            
            # Son satır için özellikler
            feature_cols = [col for col in df.columns if col not in 
                           ['id', 'symbol', 'created_at', 'open', 'high', 'low', 'close', 'volume', 'adj_close'] and 
                           not col.startswith('target') and not col.startswith('Unnamed')]
            
            last_row = df[feature_cols].iloc[-1:].values
            
            if np.isnan(last_row).any():
                print(f"❌ {symbol} için eksik veri")
                return None
            
            # Tahmin yap
            last_row_scaled = scaler.transform(last_row)
            prediction = ensemble_model.predict(last_row_scaled)[0]
            
            # Güven skoru hesapla (individual model predictions variance)
            individual_predictions = []
            for estimator_name, estimator in ensemble_model.named_estimators_.items():
                pred = estimator.predict(last_row_scaled)[0]
                individual_predictions.append(pred)
            
            prediction_std = np.std(individual_predictions)
            confidence_score = max(0, 1 - (prediction_std / prediction))
            
            current_price = df['close'].iloc[-1]
            change_percent = ((prediction - current_price) / current_price) * 100
            
            # Tahmin kalitesi değerlendirmesi
            quality_metrics = self._evaluate_prediction_quality(
                symbol, prediction, current_price, individual_predictions
            )
            
            return {
                'current_price': current_price,
                'predicted_price': prediction,
                'change_percent': change_percent,
                'confidence_score': confidence_score,
                'prediction_std': prediction_std,
                'individual_predictions': individual_predictions,
                'model_count': len(individual_predictions),
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            print(f"❌ Ensemble tahmin hatası: {str(e)}")
            return None
    
    def _evaluate_prediction_quality(self, symbol, prediction, current_price, individual_preds):
        """Tahmin kalitesi değerlendirmesi"""
        # Consensus (görüş birliği) skoru
        pred_directions = [1 if p > current_price else -1 for p in individual_preds]
        consensus_score = abs(sum(pred_directions)) / len(pred_directions)
        
        # Volatilite uyumluluğu
        recent_volatility = self._get_recent_volatility(symbol)
        predicted_change = abs(prediction - current_price) / current_price
        volatility_ratio = predicted_change / max(recent_volatility, 0.01)
        
        # Geçmiş doğruluk oranı
        historical_accuracy = self._get_historical_accuracy(symbol)
        
        return {
            'consensus_score': consensus_score,
            'volatility_ratio': volatility_ratio,
            'historical_accuracy': historical_accuracy,
            'prediction_reasonable': 0.5 < volatility_ratio < 3.0
        }
    
    def _get_recent_volatility(self, symbol):
        """Son dönem volatilite"""
        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT close FROM price_data 
            WHERE symbol = '{symbol}' 
            ORDER BY date DESC 
            LIMIT 20
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) < 2:
            return 0.02  # Default volatilite
        
        returns = df['close'].pct_change().dropna()
        return returns.std()
    
    def _get_historical_accuracy(self, symbol):
        """Geçmiş tahmin doğruluğu"""
        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT AVG(CASE WHEN ABS(prediction_error) < 0.05 THEN 1.0 ELSE 0.0 END) as accuracy
            FROM predictions_detailed 
            WHERE symbol = '{symbol}' AND actual_price IS NOT NULL
            AND prediction_date > date('now', '-30 days')
        """
        result = pd.read_sql_query(query, conn)
        conn.close()
        
        accuracy = result['accuracy'].iloc[0]
        return accuracy if pd.notna(accuracy) else 0.5
    
    def learn_from_errors(self, symbol):
        """Hatalardan öğrenme sistemi"""
        print(f"🧠 {symbol} için hata analizi ve öğrenme...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Son tahminlerin hatalarını analiz et
        query = f"""
            SELECT * FROM predictions_detailed 
            WHERE symbol = '{symbol}' AND actual_price IS NOT NULL
            AND prediction_date > date('now', '-7 days')
            ORDER BY prediction_date DESC
        """
        predictions_df = pd.read_sql_query(query, conn)
        
        if predictions_df.empty:
            print(f"  ℹ️ {symbol} için analiz edilecek tahmin yok")
            return
        
        # Hata türlerini kategorize et
        errors = []
        for _, row in predictions_df.iterrows():
            error_pct = abs(row['prediction_error'])
            
            if error_pct > 0.10:  # %10'dan fazla hata
                error_type = "büyük_hata"
            elif error_pct > 0.05:  # %5-10 arası hata
                error_type = "orta_hata"
            else:
                error_type = "küçük_hata"
            
            # Piyasa koşulunu belirle
            market_condition = self._determine_market_condition(symbol, row['prediction_date'])
            
            errors.append({
                'error_type': error_type,
                'error_magnitude': error_pct,
                'market_condition': market_condition,
                'model_name': row['model_name'],
                'prediction_date': row['prediction_date']
            })
        
        # Hata öğrenme kayıtlarını veritabanına kaydet
        for error in errors:
            correction = self._determine_correction(error)
            
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO error_learning 
                (model_name, symbol, error_type, error_magnitude, market_condition, 
                 correction_applied, learning_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (error['model_name'], symbol, error['error_type'], 
                  error['error_magnitude'], error['market_condition'], 
                  correction, datetime.now().strftime('%Y-%m-%d')))
        
        conn.commit()
        conn.close()
        
        # Öğrenme istatistikleri
        error_stats = pd.DataFrame(errors)
        if not error_stats.empty:
            print(f"  📊 Son 7 gün hata analizi:")
            print(f"    🔴 Büyük hatalar: %{(error_stats['error_type'] == 'büyük_hata').mean()*100:.1f}")
            print(f"    🟡 Orta hatalar: %{(error_stats['error_type'] == 'orta_hata').mean()*100:.1f}")
            print(f"    🟢 Küçük hatalar: %{(error_stats['error_type'] == 'küçük_hata').mean()*100:.1f}")
            
            # En problemli piyasa koşulları
            if 'market_condition' in error_stats.columns:
                problem_conditions = error_stats.groupby('market_condition')['error_magnitude'].mean().sort_values(ascending=False)
                print(f"    🎯 En zor piyasa koşulu: {problem_conditions.index[0]}")
    
    def _determine_market_condition(self, symbol, date):
        """Piyasa koşulunu belirle"""
        conn = sqlite3.connect(self.db_path)
        
        # O tarihteki piyasa verilerini al
        query = f"""
            SELECT close, volume FROM price_data 
            WHERE symbol = '{symbol}' AND date <= '{date}'
            ORDER BY date DESC 
            LIMIT 10
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) < 5:
            return "belirsiz"
        
        # Volatilite analizi
        returns = df['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Trend analizi
        trend = (df['close'].iloc[0] - df['close'].iloc[-1]) / df['close'].iloc[-1]
        
        if volatility > 0.05:
            return "yüksek_volatilite"
        elif abs(trend) > 0.10:
            return "güçlü_trend" if trend > 0 else "güçlü_düşüş"
        elif abs(trend) < 0.02:
            return "yatay_piyasa"
        else:
            return "normal_hareket"
    
    def _determine_correction(self, error):
        """Hata için düzeltme stratejisi belirle"""
        corrections = []
        
        if error['error_type'] == 'büyük_hata':
            if error['market_condition'] == 'yüksek_volatilite':
                corrections.append("volatilite_ağırlığı_artır")
            elif error['market_condition'] == 'güçlü_trend':
                corrections.append("momentum_indikatör_ağırlığı_artır")
            else:
                corrections.append("model_ağırlığı_düşür")
        
        elif error['error_type'] == 'orta_hata':
            if error['market_condition'] == 'yatay_piyasa':
                corrections.append("ortalama_dönüş_modeli_güçlendir")
            else:
                corrections.append("feature_engineering_iyileştir")
        
        return ",".join(corrections) if corrections else "standart_iyileştirme"
    
    def update_model_weights_from_learning(self, symbol):
        """Öğrenme verilerine göre model ağırlıklarını güncelle"""
        conn = sqlite3.connect(self.db_path)
        
        # Son dönem model performanslarını al
        query = f"""
            SELECT model_name, AVG(r2_score) as avg_r2, AVG(accuracy_5pct) as avg_acc,
                   AVG(directional_accuracy) as avg_dir, COUNT(*) as count
            FROM model_performance 
            WHERE symbol = '{symbol}' AND date > date('now', '-30 days')
            GROUP BY model_name
            HAVING count >= 3
        """
        performance_df = pd.read_sql_query(query, conn)
        
        # Hata öğrenme verilerini al
        query = f"""
            SELECT model_name, AVG(error_magnitude) as avg_error, COUNT(*) as error_count
            FROM error_learning 
            WHERE symbol = '{symbol}' AND learning_date > date('now', '-30 days')
            GROUP BY model_name
        """
        error_df = pd.read_sql_query(query, conn)
        conn.close()
        
        if performance_df.empty:
            return
        
        # Performans skorlarını hesapla
        performance_df['combined_score'] = (
            performance_df['avg_r2'] * 0.3 + 
            performance_df['avg_acc'] * 0.4 + 
            performance_df['avg_dir'] * 0.3
        )
        
        # Hata cezalarını uygula
        if not error_df.empty:
            for _, error_row in error_df.iterrows():
                model_mask = performance_df['model_name'] == error_row['model_name']
                if model_mask.any():
                    # Hata oranına göre ceza
                    penalty = min(0.3, error_row['avg_error'] * 2)
                    performance_df.loc[model_mask, 'combined_score'] *= (1 - penalty)
        
        # Yeni ağırlıkları hesapla
        total_score = performance_df['combined_score'].sum()
        new_weights = {}
        
        for _, row in performance_df.iterrows():
            if total_score > 0:
                new_weights[row['model_name']] = row['combined_score'] / total_score
            else:
                new_weights[row['model_name']] = 1.0 / len(performance_df)
        
        # Ağırlıkları kaydet
        self.model_weights[symbol] = new_weights
        joblib.dump(new_weights, f'weights_{symbol}.pkl')
        
        print(f"  🔄 {symbol} model ağırlıkları güncellendi")
        print(f"  📈 En etkili modeller: {sorted(new_weights.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    def save_detailed_performance(self, model_name, symbol, performance, training_samples):
        """Detaylı performans kaydetme"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_performance 
            (model_name, symbol, mse, mae, rmse, r2_score, accuracy_1pct, 
             accuracy_3pct, accuracy_5pct, directional_accuracy, training_samples, date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (model_name, symbol, performance['mse'], performance['mae'], 
              performance['rmse'], performance['r2'], performance['acc_1pct'],
              performance['acc_3pct'], performance['acc_5pct'], 
              performance['directional_acc'], training_samples,
              datetime.now().strftime('%Y-%m-%d')))
        
        conn.commit()
        conn.close()
    
    def save_ensemble_weights(self, symbol, weights, performances):
        """Ensemble ağırlıklarını kaydet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for model_name, weight in weights.items():
            perf_score = performances.get(model_name, {}).get('performance_score', 0)
            
            cursor.execute('''
                INSERT INTO ensemble_weights 
                (symbol, model_name, weight, performance_score, last_updated)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol, model_name, weight, perf_score, 
                  datetime.now().strftime('%Y-%m-%d')))
        
        conn.commit()
        conn.close()
    
    def save_detailed_prediction(self, symbol, prediction_result):
        """Detaylı tahmin kaydetme"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        tomorrow_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        cursor.execute('''
            INSERT INTO predictions_detailed 
            (symbol, prediction_date, current_price, predicted_price, 
             model_name, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (symbol, tomorrow_date, prediction_result['current_price'],
              prediction_result['predicted_price'], 'ensemble',
              prediction_result['confidence_score']))
        
        conn.commit()
        conn.close()
    
    def update_actual_prices(self):
        """Gerçek fiyatları güncelle ve hata hesapla"""
        conn = sqlite3.connect(self.db_path)
        
        # Gerçek fiyatı olmayan tahminleri bul
        query = """
            SELECT DISTINCT p.id, p.symbol, p.prediction_date, p.predicted_price
            FROM predictions_detailed p
            WHERE p.actual_price IS NULL 
            AND p.prediction_date <= date('now')
            ORDER BY p.prediction_date DESC
        """
        pending_predictions = pd.read_sql_query(query, conn)
        
        for _, row in pending_predictions.iterrows():
            # O tarihteki gerçek fiyatı bul
            actual_query = f"""
                SELECT close FROM price_data 
                WHERE symbol = '{row['symbol']}' AND date = '{row['prediction_date']}'
            """
            actual_data = pd.read_sql_query(actual_query, conn)
            
            if not actual_data.empty:
                actual_price = actual_data['close'].iloc[0]
                prediction_error = abs(actual_price - row['predicted_price']) / actual_price
                direction_correct = ((actual_price > row['predicted_price']) == 
                                   (row['predicted_price'] > actual_price))
                
                # Güncelle
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE predictions_detailed 
                    SET actual_price = ?, prediction_error = ?, direction_correct = ?
                    WHERE id = ?
                ''', (actual_price, prediction_error, direction_correct, row['id']))
        
        conn.commit()
        conn.close()
        print("✅ Gerçek fiyatlar güncellendi")
    
    def comprehensive_daily_report(self, symbols=["XU100.IS", "GARAN.IS", "AKBNK.IS", "THYAO.IS", "BIMAS.IS"]):
        """Kapsamlı günlük rapor"""
        print("\n" + "="*80)
        print("🚀 GELİŞMİŞ BIST ENSEMBLE AI GÜNLÜK RAPORU")
        print("="*80)
        print(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"🤖 Aktif Modeller: {len(self.base_models)}")
        print("-"*80)
        
        for symbol in symbols:
            print(f"\n🔍 {symbol} DETAYLI ANALİZ:")
            print("-" * 50)
            
            prediction = self.predict_with_ensemble(symbol)
            
            if prediction:
                current = prediction['current_price']
                predicted = prediction['predicted_price']
                change = prediction['change_percent']
                confidence = prediction['confidence_score']
                
                print(f"  💰 Mevcut Fiyat: {current:.2f} TL")
                print(f"  🎯 Yarın Tahmini: {predicted:.2f} TL")
                print(f"  📊 Güven Skoru: %{confidence*100:.1f}")
                print(f"  🤖 Kullanılan Model Sayısı: {prediction['model_count']}")
                
                # Değişim analizi
                if change > 0:
                    emoji = "📈" if change > 2 else "🔼"
                    trend = "YÜKSELİŞ"
                else:
                    emoji = "📉" if change < -2 else "🔽"
                    trend = "DÜŞÜŞ"
                
                print(f"  {emoji} Beklenen Değişim: {change:+.2f}% ({trend})")
                
                # Tahmin kalitesi
                quality = prediction['quality_metrics']
                print(f"  🎯 Model Konsensüsü: %{quality['consensus_score']*100:.1f}")
                print(f"  📊 Volatilite Uyumu: {quality['volatility_ratio']:.2f}")
                print(f"  📈 Geçmiş Doğruluk: %{quality['historical_accuracy']*100:.1f}")
                
                # Yatırım önerisi (gelişmiş)
                risk_score = self._calculate_risk_score(prediction)
                investment_advice = self._generate_investment_advice(change, confidence, risk_score)
                print(f"  💡 Yatırım Önerisi: {investment_advice}")
                print(f"  ⚠️ Risk Skoru: {risk_score}/10")
                
                # Individual model predictions
                individual_preds = prediction['individual_predictions']
                pred_range = max(individual_preds) - min(individual_preds)
                print(f"  🔄 Model Tahmin Aralığı: ±{pred_range/2:.2f} TL")
                
            else:
                print(f"  ❌ Tahmin yapılamadı - Model eğitimi gerekli")
        
        # Genel performans özeti
        self._print_overall_performance_summary()
        
        print("\n" + "="*80)
    
    def _calculate_risk_score(self, prediction):
        """Risk skoru hesaplama (1-10)"""
        base_risk = 5
        
        # Volatilite riski
        volatility_risk = min(3, prediction['quality_metrics']['volatility_ratio'] * 1.5)
        
        # Güven riski
        confidence_risk = (1 - prediction['confidence_score']) * 3
        
        # Konsensüs riski
        consensus_risk = (1 - prediction['quality_metrics']['consensus_score']) * 2
        
        total_risk = base_risk + volatility_risk + confidence_risk + consensus_risk
        return min(10, max(1, int(total_risk)))
    
    def _generate_investment_advice(self, change_percent, confidence, risk_score):
        """Yatırım önerisi üret"""
        abs_change = abs(change_percent)
        
        if confidence < 0.6 or risk_score > 7:
            return "BEKLEYİN - Düşük güven/Yüksek risk"
        
        if abs_change < 1:
            return "YATAY - Önemli hareket beklenmiyor"
        elif abs_change < 3 and confidence > 0.7:
            direction = "HAFIF ALIŞ" if change_percent > 0 else "HAFIF SATIŞ"
            return f"{direction} - Orta güvenilirlik"
        elif abs_change >= 3 and confidence > 0.8:
            direction = "GÜÇLÜ ALIŞ" if change_percent > 0 else "GÜÇLÜ SATIŞ"
            return f"{direction} - Yüksek güvenilirlik"
        else:
            return "DİKKATLİ TAKİP - Belirsizlik var"
    
    def _print_overall_performance_summary(self):
        """Genel performans özeti"""
        print(f"\n📊 GENEL PERFORMANS ÖZETİ:")
        print("-" * 40)
        
        conn = sqlite3.connect(self.db_path)
        
        # Son 7 günün genel doğruluğu
        query = """
            SELECT AVG(CASE WHEN prediction_error < 0.05 THEN 1.0 ELSE 0.0 END) as accuracy_5pct,
                   AVG(CASE WHEN direction_correct = 1 THEN 1.0 ELSE 0.0 END) as direction_accuracy,
                   COUNT(*) as total_predictions
            FROM predictions_detailed 
            WHERE actual_price IS NOT NULL 
            AND prediction_date > date('now', '-7 days')
        """
        overall_perf = pd.read_sql_query(query, conn)
        
        if not overall_perf.empty and overall_perf['total_predictions'].iloc[0] > 0:
            acc_5pct = overall_perf['accuracy_5pct'].iloc[0] * 100
            dir_acc = overall_perf['direction_accuracy'].iloc[0] * 100
            total_preds = overall_perf['total_predictions'].iloc[0]
            
            print(f"  🎯 Son 7 Gün Doğruluk (%5 tolerans): %{acc_5pct:.1f}")
            print(f"  📈 Yön Doğruluğu: %{dir_acc:.1f}")
            print(f"  📊 Toplam Tahmin: {total_preds}")
        
        # En iyi performans gösteren modeller
        query = """
            SELECT model_name, AVG(r2_score) as avg_r2, COUNT(*) as count
            FROM model_performance 
            WHERE date > date('now', '-7 days')
            GROUP BY model_name
            ORDER BY avg_r2 DESC
            LIMIT 3
        """
        top_models = pd.read_sql_query(query, conn)
        
        if not top_models.empty:
            print(f"  🏆 En İyi Modeller:")
            for _, row in top_models.iterrows():
                print(f"    • {row['model_name']}: R² = {row['avg_r2']:.4f}")
        
        conn.close()
    
    def run_comprehensive_daily_cycle(self):
        """Kapsamlı günlük çalışma döngüsü - iyileştirilmiş"""
        print("🔄 Gelişmiş günlük analiz döngüsü başlatılıyor...")
        
        symbols = ["XU100.IS", "GARAN.IS", "AKBNK.IS", "THYAO.IS", "BIMAS.IS"]
        
        try:
            # 1. Veri toplama
            print("\n📥 1. ADIM: Veri Toplama")
            self.fetch_enhanced_data(symbols)
            
            # 2. Veri kontrolü ve hazır olanları eğit
            print("\n🔍 2. ADIM: Veri Kontrolü")
            ready_symbols = self.test_data_and_train_if_ready()
            
            # 3. Gerçek fiyatları güncelle
            print("\n🔄 3. ADIM: Gerçek Fiyat Güncellemesi")
            self.update_actual_prices()
            
            # 4. Hazır semboller için hata analizi
            if ready_symbols:
                print("\n🧠 4. ADIM: Hata Analizi ve Öğrenme")
                for symbol in ready_symbols:
                    self.learn_from_errors(symbol)
                    self.update_model_weights_from_learning(symbol)
            
            # 5. Tahminleri kaydet
            print("\n💾 5. ADIM: Tahmin Kaydetme")
            for symbol in symbols:
                prediction = self.predict_with_ensemble(symbol)
                if prediction:
                    self.save_detailed_prediction(symbol, prediction)
            
            # 6. Kapsamlı rapor
            print("\n📊 6. ADIM: Kapsamlı Rapor")
            self.comprehensive_daily_report(symbols)
            
        except Exception as e:
            print(f"❌ Günlük döngü hatası: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("✅ Kapsamlı günlük döngü tamamlandı!")
    
    def get_accuracy_statistics(self):
        """Doğruluk istatistikleri"""
        conn = sqlite3.connect(self.db_path)
        
        # Genel doğruluk istatistikleri
        query = """
            SELECT 
                symbol,
                COUNT(*) as total_predictions,
                AVG(CASE WHEN prediction_error < 0.01 THEN 1.0 ELSE 0.0 END) * 100 as accuracy_1pct,
                AVG(CASE WHEN prediction_error < 0.03 THEN 1.0 ELSE 0.0 END) * 100 as accuracy_3pct,
                AVG(CASE WHEN prediction_error < 0.05 THEN 1.0 ELSE 0.0 END) * 100 as accuracy_5pct,
                AVG(CASE WHEN direction_correct = 1 THEN 1.0 ELSE 0.0 END) * 100 as direction_accuracy,
                AVG(prediction_error) * 100 as avg_error_pct,
                AVG(confidence_score) * 100 as avg_confidence
            FROM predictions_detailed 
            WHERE actual_price IS NOT NULL
            GROUP BY symbol
            ORDER BY accuracy_5pct DESC
        """
        
        accuracy_stats = pd.read_sql_query(query, conn)
        conn.close()
        
        return accuracy_stats
    
    def print_accuracy_report(self):
        """Doğruluk raporu yazdır"""
        stats = self.get_accuracy_statistics()
        
        if stats.empty:
            print("📊 Henüz doğruluk verisi yok")
            return
        
        print("\n" + "="*80)
        print("📈 DOĞRULUK PERFORMANS RAPORU")
        print("="*80)
        
        for _, row in stats.iterrows():
            print(f"\n🔍 {row['symbol']}:")
            print(f"  📊 Toplam Tahmin: {int(row['total_predictions'])}")
            print(f"  🎯 Doğruluk (%1): %{row['accuracy_1pct']:.1f}")
            print(f"  🎯 Doğruluk (%3): %{row['accuracy_3pct']:.1f}")
            print(f"  🎯 Doğruluk (%5): %{row['accuracy_5pct']:.1f}")
            print(f"  📈 Yön Doğruluğu: %{row['direction_accuracy']:.1f}")
            print(f"  📊 Ortalama Hata: %{row['avg_error_pct']:.2f}")
            print(f"  🔒 Ortalama Güven: %{row['avg_confidence']:.1f}")
        
        # Genel ortalama
        overall_acc_5 = stats['accuracy_5pct'].mean()
        overall_dir = stats['direction_accuracy'].mean()
        print(f"\n🏆 GENEL ORTALAMA:")
        print(f"  🎯 %5 Doğruluk: %{overall_acc_5:.1f}")
        print(f"  📈 Yön Doğruluğu: %{overall_dir:.1f}")
        
        print("="*80)

# Ana Kullanım
if __name__ == "__main__":
    # Gelişmiş bot'u başlat
    bot = AdvancedBISTEnsembleBot()
    
    # Kapsamlı günlük döngüyü çalıştır
    bot.run_comprehensive_daily_cycle()
    
    # Doğruluk raporu
    bot.print_accuracy_report()
    
    print("\n🎉 Gelişmiş BIST Ensemble AI Bot hazır!")
    print("\n💡 Bot Özellikleri:")
    print("   🤖 18 farklı makine öğrenmesi modeli")
    print("   🎯 Ensemble (toplu) tahmin sistemi")
    print("   🧠 Hatalardan öğrenme mekanizması")
    print("   📊 Otomatik model ağırlık güncellemesi")
    print("   📈 Detaylı doğruluk takibi")
    print("   🔄 Günlük kendini geliştirme")
    print("   💰 Gelişmiş yatırım önerileri")
    print("   ⚠️ Risk değerlendirmesi")
    
    # Scheduler ile otomatik çalıştırma örneği:
    # import schedule
    # schedule.every().day.at("08:30").do(bot.run_comprehensive_daily_cycle)
    # 
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)
