"""
AI Models for Financial Prediction
Includes various ML and DL models for stock price forecasting
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import os
from typing import Dict, List, Tuple, Optional
import logging
from src.config import Config

class FinancialAIModels:
    """AI models for financial prediction and analysis"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
        
        # Ensure models directory exists
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'Close', 
                    feature_columns: List[str] = None, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for ML/DL models
        
        Args:
            data: DataFrame with financial data
            target_column: Column to predict
            feature_columns: Features to use (if None, use all numeric columns)
            sequence_length: Length of sequences for LSTM models
        
        Returns:
            Tuple of features and targets
        """
        if data.empty:
            return np.array([]), np.array([])
        
        # Select features
        if feature_columns is None:
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in feature_columns:
                feature_columns.remove(target_column)
        
        # Remove rows with NaN values
        data_clean = data[feature_columns + [target_column]].dropna()
        
        if data_clean.empty:
            return np.array([]), np.array([])
        
        # Prepare features and target
        X = data_clean[feature_columns].values
        y = data_clean[target_column].values
        
        return X, y
    
    def prepare_sequence_data(self, data: pd.DataFrame, target_column: str = 'Close',
                             feature_columns: List[str] = None, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequence data for LSTM models
        
        Args:
            data: DataFrame with financial data
            target_column: Column to predict
            feature_columns: Features to use
            sequence_length: Length of sequences
        
        Returns:
            Tuple of sequences and targets
        """
        if data.empty:
            return np.array([]), np.array([])
        
        # Select features
        if feature_columns is None:
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in feature_columns:
                feature_columns.remove(target_column)
        
        # Remove rows with NaN values
        data_clean = data[feature_columns + [target_column]].dropna()
        
        if len(data_clean) < sequence_length + 1:
            return np.array([]), np.array([])
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(data_clean)):
            X.append(data_clean[feature_columns].iloc[i-sequence_length:i].values)
            y.append(data_clean[target_column].iloc[i])
        
        return np.array(X), np.array(y)
    
    def train_ml_models(self, data: pd.DataFrame, target_column: str = 'Close',
                        test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train multiple ML models
        
        Args:
            data: DataFrame with financial data
            target_column: Column to predict
            test_size: Proportion of data for testing
            random_state: Random seed
        
        Returns:
            Dictionary with trained models and performance metrics
        """
        X, y = self.prepare_data(data, target_column)
        
        if len(X) == 0:
            self.logger.error("No valid data for training")
            return {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers['ml_models'] = scaler
        
        # Define models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Store results
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                }
                
                # Store model
                self.models[name] = model
                
                self.logger.info(f"{name} trained successfully - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")
        
        return results
    
    def train_lstm_model(self, data: pd.DataFrame, target_column: str = 'Close',
                         sequence_length: int = 60, test_size: float = 0.2,
                         epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Train LSTM model for sequence prediction
        
        Args:
            data: DataFrame with financial data
            target_column: Column to predict
            sequence_length: Length of sequences
            test_size: Proportion of data for testing
            epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            Dictionary with trained model and performance metrics
        """
        X, y = self.prepare_sequence_data(data, target_column, sequence_length=sequence_length)
        
        if len(X) == 0:
            self.logger.error("No valid sequence data for training")
            return {}
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = MinMaxScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_test_scaled = scaler.transform(X_test_reshaped)
        
        # Reshape back to 3D
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        # Scale target
        target_scaler = MinMaxScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        # Store scalers
        self.scalers['lstm_features'] = scaler
        self.scalers['lstm_target'] = target_scaler
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Make predictions
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store model and results
        self.models['LSTM'] = model
        
        results = {
            'model': model,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse),
            'history': history.history
        }
        
        self.logger.info(f"LSTM trained successfully - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}")
        
        return results
    
    def train_cnn_model(self, data: pd.DataFrame, target_column: str = 'Close',
                        sequence_length: int = 60, test_size: float = 0.2,
                        epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Train CNN model for sequence prediction
        
        Args:
            data: DataFrame with financial data
            target_column: Column to predict
            sequence_length: Length of sequences
            test_size: Proportion of data for testing
            epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            Dictionary with trained model and performance metrics
        """
        X, y = self.prepare_sequence_data(data, target_column, sequence_length=sequence_length)
        
        if len(X) == 0:
            self.logger.error("No valid sequence data for training")
            return {}
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = MinMaxScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_test_scaled = scaler.transform(X_test_reshaped)
        
        # Reshape back to 3D
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        # Scale target
        target_scaler = MinMaxScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        # Store scalers
        self.scalers['cnn_features'] = scaler
        self.scalers['cnn_target'] = target_scaler
        
        # Build CNN model
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, X_train.shape[2])),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=16, kernel_size=3, activation='relu'),
            Flatten(),
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Make predictions
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store model and results
        self.models['CNN'] = model
        
        results = {
            'model': model,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse),
            'history': history.history
        }
        
        self.logger.info(f"CNN trained successfully - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}")
        
        return results
    
    def predict(self, model_name: str, data: pd.DataFrame, 
                target_column: str = 'Close', sequence_length: int = 60) -> np.ndarray:
        """
        Make predictions using a trained model
        
        Args:
            model_name: Name of the model to use
            data: Data for prediction
            target_column: Target column name
            sequence_length: Sequence length for sequence models
        
        Returns:
            Array of predictions
        """
        if model_name not in self.models:
            self.logger.error(f"Model {model_name} not found")
            return np.array([])
        
        model = self.models[model_name]
        
        try:
            if model_name in ['LSTM', 'CNN']:
                # Sequence models
                X, _ = self.prepare_sequence_data(data, target_column, sequence_length=sequence_length)
                if len(X) == 0:
                    return np.array([])
                
                # Scale features
                scaler_key = f'{model_name.lower()}_features'
                if scaler_key in self.scalers:
                    X_reshaped = X.reshape(-1, X.shape[-1])
                    X_scaled = self.scalers[scaler_key].transform(X_reshaped)
                    X_scaled = X_scaled.reshape(X.shape)
                else:
                    X_scaled = X
                
                # Make predictions
                predictions_scaled = model.predict(X_scaled)
                
                # Inverse scale predictions
                target_scaler_key = f'{model_name.lower()}_target'
                if target_scaler_key in self.scalers:
                    predictions = self.scalers[target_scaler_key].inverse_transform(predictions_scaled).flatten()
                else:
                    predictions = predictions_scaled.flatten()
                
            else:
                # ML models
                X, _ = self.prepare_data(data, target_column)
                if len(X) == 0:
                    return np.array([])
                
                # Scale features
                if 'ml_models' in self.scalers:
                    X_scaled = self.scalers['ml_models'].transform(X)
                else:
                    X_scaled = X
                
                # Make predictions
                predictions = model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions with {model_name}: {str(e)}")
            return np.array([])
    
    def ensemble_predict(self, data: pd.DataFrame, target_column: str = 'Close',
                         weights: Dict[str, float] = None) -> np.ndarray:
        """
        Make ensemble predictions using multiple models
        
        Args:
            data: Data for prediction
            target_column: Target column name
            weights: Weights for each model (if None, equal weights)
        
        Returns:
            Array of ensemble predictions
        """
        if not self.models:
            self.logger.error("No models available for ensemble prediction")
            return np.array([])
        
        # Get predictions from all models
        predictions = {}
        for model_name in self.models.keys():
            pred = self.predict(model_name, data, target_column)
            if len(pred) > 0:
                predictions[model_name] = pred
        
        if not predictions:
            return np.array([])
        
        # Set equal weights if not specified
        if weights is None:
            weights = {name: 1.0 / len(predictions) for name in predictions.keys()}
        
        # Calculate ensemble prediction
        ensemble_pred = np.zeros(len(next(iter(predictions.values()))))
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 1.0 / len(predictions))
            ensemble_pred += weight * pred
            total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred
    
    def save_models(self, filename_prefix: str = 'financial_ai_models') -> bool:
        """
        Save trained models to disk
        
        Args:
            filename_prefix: Prefix for saved files
        
        Returns:
            True if successful, False otherwise
        """
        try:
            for name, model in self.models.items():
                if name in ['LSTM', 'CNN']:
                    # Save Keras models
                    model_path = os.path.join(self.config.MODEL_SAVE_PATH, f"{filename_prefix}_{name}.h5")
                    model.save(model_path)
                else:
                    # Save scikit-learn models
                    model_path = os.path.join(self.config.MODEL_SAVE_PATH, f"{filename_prefix}_{name}.pkl")
                    joblib.dump(model, model_path)
                
                self.logger.info(f"Model {name} saved to {model_path}")
            
            # Save scalers
            scalers_path = os.path.join(self.config.MODEL_SAVE_PATH, f"{filename_prefix}_scalers.pkl")
            joblib.dump(self.scalers, scalers_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            return False
    
    def load_models(self, filename_prefix: str = 'financial_ai_models') -> bool:
        """
        Load trained models from disk
        
        Args:
            filename_prefix: Prefix for saved files
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load scikit-learn models
            for name in ['RandomForest', 'GradientBoosting', 'LinearRegression', 'Ridge', 'Lasso', 'SVR']:
                model_path = os.path.join(self.config.MODEL_SAVE_PATH, f"{filename_prefix}_{name}.pkl")
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
                    self.logger.info(f"Model {name} loaded from {model_path}")
            
            # Load Keras models
            for name in ['LSTM', 'CNN']:
                model_path = os.path.join(self.config.MODEL_SAVE_PATH, f"{filename_prefix}_{name}.h5")
                if os.path.exists(model_path):
                    self.models[name] = tf.keras.models.load_model(model_path)
                    self.logger.info(f"Model {name} loaded from {model_path}")
            
            # Load scalers
            scalers_path = os.path.join(self.config.MODEL_SAVE_PATH, f"{filename_prefix}_scalers.pkl")
            if os.path.exists(scalers_path):
                self.scalers = joblib.load(scalers_path)
                self.logger.info(f"Scalers loaded from {scalers_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            return False