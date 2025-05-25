import argparse
import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
from datetime import datetime
import json
from typing import Optional, Dict, Union


class EmergencyCallPredictor:
    def __init__(self, data_path=None, model_name='emergency_call_model', output_dir='models'):
        """
        Initialize the Emergency Call Predictor
        
        Args:
            data_path (str): Path to the CSV file containing the data
            model_name (str): Base name for saving model files
            output_dir (str): Directory to save model and artifacts
        """
        self.data_path = data_path
        self.model_name = model_name
        self.output_dir = output_dir
        self.model = None
        self.metrics = {}
        self.features = []
        self.target = 'call_count'
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    def _handle_warnings(self):
        """Context manager for controlled warning handling"""
        return warnings.catch_warnings()
           
    def load_and_clean_data(self) -> pd.DataFrame:
        """
        Load and clean the emergency call data with proper warning handling
        
        Returns:
            Cleaned dataframe
            
        Raises:
            ValueError: If required columns not found or data conversion fails
        """
        with self._handle_warnings():
            warnings.filterwarnings('default', category=pd.errors.DtypeWarning)
            warnings.filterwarnings('default', category=UserWarning)
            
            print(f"\n[INFO] Loading data from {self.data_path}")
            
            try:
                # Try to load with explicit dtype specification to avoid mixed-type warnings
                df = pd.read_csv(
                    self.data_path,
                    dtype={
                        'latitude': float,
                        'longitude': float,
                        'lat': float,
                        'lng': float,
                        'lon': float
                    },
                    parse_dates=True,
                    low_memory=False
                )
                
                # Standardize column names
                df.columns = df.columns.str.strip().str.lower()
                
                # Try to identify columns automatically with fallbacks
                self.lat_col = self._detect_column(df, ['latitude', 'lat', 'einsatzortlat'])
                self.lon_col = self._detect_column(df, ['longitude', 'lng', 'lon', 'einsatzortlon'])
                self.time_col = self._detect_column(df, ['timestamp', 'time', 'date', 'tserstesklingelsignal'])
                
                print(f"[INFO] Using columns - Latitude: {self.lat_col}, Longitude: {self.lon_col}, Timestamp: {self.time_col}")
                
                # Data cleaning with warning context
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    
                    # Check for duplicates
                    initial_count = len(df)
                    df = df.drop_duplicates()
                    if len(df) < initial_count:
                        print(f"[INFO] Removed {initial_count - len(df)} duplicate rows")
                    
                    # Check for nulls
                    null_counts = df[[self.lat_col, self.lon_col, self.time_col]].isnull().sum()
                    if null_counts.any():
                        print(f"[INFO] Removing {null_counts.sum()} rows with null values")
                        df = df.dropna(subset=[self.lat_col, self.lon_col, self.time_col])
                    
                    # Convert timestamp with proper warning handling
                    try:
                        df[self.time_col] = pd.to_datetime(
                            df[self.time_col],
                            errors='coerce'
                        )
                        if df[self.time_col].isnull().any():
                            raise ValueError("Timestamp conversion failed for some rows")
                    except Exception as e:
                        warnings.warn(f"Timestamp conversion issues: {str(e)}")
                        raise
                    
                    # Validate numeric columns
                    for col in [self.lat_col, self.lon_col]:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='raise')
                        except ValueError as e:
                            warnings.warn(f"Numeric conversion failed for {col}: {str(e)}")
                            raise
                    
                    return df
                
            except Exception as e:
                raise ValueError(f"Error loading or processing data: {str(e)}")
    
    def _detect_column(self, df: pd.DataFrame, possible_names: list) -> str:
        """Helper to detect column names with fallbacks"""
        for name in possible_names:
            if name in df.columns:
                return name
        raise ValueError(f"Could not find any of {possible_names} in dataframe columns")
    
    def add_temporal_features(self, df, ts_col):
        """
        Add temporal features to the dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            ts_col (str): Name of the timestamp column
            
        Returns:
            pd.DataFrame: Dataframe with added temporal features
        """
        df = df.copy()
        df["hour"] = df[ts_col].dt.hour
        df["dayofweek"] = df[ts_col].dt.weekday
        df["dayofmonth"] = df[ts_col].dt.day
        df["month"] = df[ts_col].dt.month
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        
        # Add season (0=winter, 1=spring, 2=summer, 3=fall)
        season_map = {12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 
                      6:2, 7:2, 8:2, 9:3, 10:3, 11:3}
        df["season"] = df["month"].map(season_map)
        
        return df
    
    def add_spatial_features(self, df, lat_col, lon_col, km=20):
        """
        Add spatial grid features to the dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            lat_col (str): Name of latitude column
            lon_col (str): Name of longitude column
            km (int): Grid size in kilometers
            
        Returns:
            pd.DataFrame: Dataframe with added spatial features
        """
        df = df.copy()
        lat_step = km / 111  # Approx km per degree latitude
        lon_step = km / (111 * np.cos(np.deg2rad(df[lat_col].median())))  # Adjust for longitude
        
        df["lat_grid"] = np.floor(df[lat_col] / lat_step) * lat_step
        df["lon_grid"] = np.floor(df[lon_col] / lon_step) * lon_step
        
        return df
    
    def aggregate_data(self, df, time_block='3h'):
        """
        Aggregate data into time blocks and spatial grids
        
        Args:
            df (pd.DataFrame): Input dataframe
            time_block (str): Pandas time frequency string (e.g., '3H', '6H', 'D')
            
        Returns:
            pd.DataFrame: Aggregated dataframe
        """
        print(f"\n[INFO] Aggregating data into {time_block} time blocks")
        
        # Create time block column
        time_block_col = f"date_{time_block.lower()}"
        df[time_block_col] = df[self.time_col].dt.floor(time_block)
        
        # Add features
        df = self.add_temporal_features(df, time_block_col)
        df = self.add_spatial_features(df, self.lat_col, self.lon_col)
        
        # Define grouping columns
        group_cols = [
            time_block_col, "lat_grid", "lon_grid",
            "hour_sin", "hour_cos", "dayofweek", "season", "month", "dayofmonth"
        ]
        
        # Aggregate
        grouped = (df.groupby(group_cols)
                   .size()
                   .reset_index(name=self.target))
        
        grouped = grouped.sort_values(["lat_grid", "lon_grid", time_block_col])
        
        # Set features for modeling
        self.features = [
            "hour_sin", "hour_cos", "dayofweek", "season",
            "lat_grid", "lon_grid", "month", "dayofmonth"
        ]
        
        return grouped
    
    def train_test_split(self, df, test_size=0.2):
        """
        Split data into train and test sets (time-based)
        
        Args:
            df (pd.DataFrame): Input dataframe
            test_size (float): Proportion of data to use for testing
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("\n[INFO] Performing time-based train-test split")
        
        df = df.sort_values("date_3h").reset_index(drop=True)
        cutoff = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:cutoff]
        test_df = df.iloc[cutoff:]
        
        X_train = train_df[self.features]
        y_train = train_df[self.target]
        X_test = test_df[self.features]
        y_test = test_df[self.target]
        
        print(f"[INFO] Training period: {train_df['date_3h'].min()} to {train_df['date_3h'].max()}")
        print(f"[INFO] Testing period: {test_df['date_3h'].min()} to {test_df['date_3h'].max()}")
        print(f"[INFO] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> LGBMRegressor:
        """
        Train the LightGBM model with proper warning handling
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained LightGBM model
        """
        with self._handle_warnings():
            warnings.filterwarnings('default', category=UserWarning)
            
            print("\n[INFO] Training LightGBM model")
            
            model = LGBMRegressor(
                objective="tweedie",
                tweedie_variance_power=1.2, 
                n_estimators=500,
                learning_rate=0.05,
                max_depth=10,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                force_row_wise=True,
                verbose=-1
            )
            
            # Cross-validation with time series split
            tscv = TimeSeriesSplit(n_splits=3)
            print("[INFO] Performing cross-validation...")
            
            cv_scores = []
            for i, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)  # Ignore LightGBM internal warnings
                    model.fit(
                        X_tr, y_tr, 
                        eval_set=[(X_val, y_val)], 
                        eval_metric='mae'
                    )
                
                y_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                cv_scores.append(mae)
                print(f"  Fold {i+1}: MAE = {mae:.2f}")
            
            print(f"[INFO] Average CV MAE: {np.mean(cv_scores):.2f}")
            
            # Final training
            print("[INFO] Training final model on full training set")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                model.fit(X_train, y_train)
            
            return model
    
    def predict(
        self, 
        model: LGBMRegressor, 
        X: Union[pd.DataFrame, np.ndarray, dict],
        return_confidence: bool = False
    ) -> Union[np.ndarray, tuple]:
        """
        Make predictions with the trained model
        
        Args:
            model: Trained LightGBM model
            X: Input data (DataFrame, array, or dict of features)
            return_confidence: Whether to return confidence intervals
            
        Returns:
            Predictions (and optionally confidence intervals)
            
        Raises:
            ValueError: If input data is invalid or missing required features
        """
        with self._handle_warnings():
            warnings.filterwarnings('default', category=UserWarning)
            
            # Convert input to proper format
            if isinstance(X, dict):
                X = pd.DataFrame([X])
            elif isinstance(X, (list, np.ndarray)):
                if len(X) != len(self.features):
                    raise ValueError(f"Expected {len(self.features)} features, got {len(X)}")
                X = pd.DataFrame([X], columns=self.features)
            
            # Ensure all features are present
            missing_features = set(self.features) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Reorder features to match training order
            X = X[self.features]
            
            # Make predictions
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                preds = model.predict(X)
                
                if return_confidence:
                    # Get prediction intervals (using std of leaf predictions)
                    leaf_preds = model.predict(X, pred_leaf=True)
                    n_trees = model.n_estimators
                    n_data = len(X)
                    
                    # Reshape to (n_data, n_trees)
                    individual_preds = np.zeros((n_data, n_trees))
                    for i in range(n_trees):
                        individual_preds[:, i] = model.predict(X, num_iteration=i+1)
                    
                    std_dev = np.std(individual_preds, axis=1)
                    return preds, (preds - 1.96*std_dev, preds + 1.96*std_dev)
                
                return preds
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load a trained model from disk
        
        Args:
            model_path: Path to model file. If None, uses default path in output_dir
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model loading fails
        """
        if model_path is None:
            model_path = os.path.join(self.output_dir, f"{self.model_name}.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        try:
            with self._handle_warnings():
                warnings.filterwarnings('default', category=UserWarning)
                
                # Load model
                self.model = joblib.load(model_path)
                
                # Load metadata
                meta_path = os.path.join(self.output_dir, f"{self.model_name}_features.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                        self.features = meta.get('features', [])
                        self.target = meta.get('target', 'call_count')
                        self.lat_col = meta.get('lat_col')
                        self.lon_col = meta.get('lon_col')
                        self.time_col = meta.get('time_col')
                
                print(f"[INFO] Successfully loaded model from {model_path}")
                
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            dict: Evaluation metrics
        """
        print("\n[INFO] Evaluating model on test set")
        
        y_pred = model.predict(X_test)
        
        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
            "Mean Actual": y_test.mean(),
            "Mean Predicted": y_pred.mean()
        }
        
        print("\n=== Model Evaluation ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            
        return metrics
    
    def plot_feature_importance(self, model, top_n=10):
        """
        Plot feature importance
        
        Args:
            model: Trained model
            top_n (int): Number of top features to show
        """
        importances = pd.Series(model.feature_importances_, 
                               index=self.features).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances.head(top_n).values,
                    y=importances.head(top_n).index,
                    hue=importances.head(top_n).index,
                    palette='viridis',
                    legend=False)
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, f"{self.model_name}_feature_importance.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved feature importance plot to {plot_path}")
    
    def plot_actual_vs_predicted(self, y_true, y_pred, sample_size=1000):
        """
        Plot actual vs predicted values
        
        Args:
            y_true (pd.Series): Actual values
            y_pred (np.array): Predicted values
            sample_size (int): Number of points to sample for plotting
        """
        if len(y_true) > sample_size:
            idx = np.random.choice(len(y_true), sample_size, replace=False)
            y_true_sampled = y_true.iloc[idx]
            y_pred_sampled = y_pred[idx]
        else:
            y_true_sampled = y_true
            y_pred_sampled = y_pred
            
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_true_sampled, y=y_pred_sampled, alpha=0.6)
        plt.plot([y_true_sampled.min(), y_true_sampled.max()], 
                 [y_true_sampled.min(), y_true_sampled.max()], 
                 'r--')
        plt.xlabel('Actual Call Count')
        plt.ylabel('Predicted Call Count')
        plt.title('Actual vs Predicted Call Counts')
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, f"{self.model_name}_actual_vs_predicted.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved actual vs predicted plot to {plot_path}")
    
    def save_model(self, model):
        """
        Save the trained model and metadata
        
        Args:
            model: Trained model to save
        """
        # Save model files
        model_path = os.path.join(self.output_dir, f"{self.model_name}.pkl")
        joblib.dump(model, model_path)
        
        # Save feature list
        features_path = os.path.join(self.output_dir, f"{self.model_name}_features.json")
        with open(features_path, 'w') as f:
            json.dump({
                'features': self.features,
                'target': self.target,
                'lat_col': self.lat_col,
                'lon_col': self.lon_col,
                'time_col': self.time_col,
                'metrics': self.metrics,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n[INFO] Saved model to {model_path}")
        print(f"[INFO] Saved feature metadata to {features_path}")
    
    def run_pipeline(self, time_block='3h', test_size=0.2):
        """
        Run the complete training pipeline
        
        Args:
            time_block (str): Time frequency for aggregation
            test_size (float): Proportion of data for testing
        """
        try:
            # Step 1: Load and clean data
            df = self.load_and_clean_data()
            
            # Step 2: Feature engineering and aggregation
            aggregated = self.aggregate_data(df, time_block)
            
            # Step 3: Train-test split
            X_train, X_test, y_train, y_test = self.train_test_split(aggregated, test_size)
            
            # Step 4: Train model
            self.model = self.train_model(X_train, y_train)
            
            # Step 5: Evaluate model
            self.metrics = self.evaluate_model(self.model, X_test, y_test)
            
            # Step 6: Visualizations
            self.plot_feature_importance(self.model)
            y_pred = self.model.predict(X_test)
            self.plot_actual_vs_predicted(y_test, y_pred)
            
            # Step 7: Save model
            self.save_model(self.model)
            
            print("\n[INFO] Training pipeline completed successfully!")
            
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {str(e)}")
            raise
    def inference(self, location_datetime: tuple) -> float:
        """
        Simplified inference interface that takes (latitude, longitude, datetime) tuple
        
        Args:
            location_datetime: Tuple of (latitude, longitude, datetime)
            
        Returns:
            Predicted call count for that location and time
            
        Example:
            >>> predictor.inference((49.45, 7.75, datetime(2023, 6, 15, 14)))
            3.8
        """
        if not hasattr(self, 'model') or self.model is None:
            self.load_model()  # Try to load default model if none loaded
            
        lat, lon, dt = location_datetime
        
        # Create feature dictionary
        input_features = {
            "lat_grid": np.floor(lat / (20 / 111)),  # 20km grid
            "lon_grid": np.floor(lon / (20 / (111 * np.cos(np.deg2rad(lat))))),
            "hour_sin": np.sin(2 * np.pi * dt.hour / 24),
            "hour_cos": np.cos(2 * np.pi * dt.hour / 24),
            "dayofweek": dt.weekday(),
            "season": self._get_season(dt.month),
            "month": dt.month,
            "dayofmonth": dt.day
        }
        
        # Make prediction
        prediction = self.predict(self.model, input_features)
        return float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction)
    
    def _get_season(self, month: int) -> int:
        """Helper to get season (0=winter, 1=spring, 2=summer, 3=fall)"""
        if month in [12, 1, 2]:
            return 0
        elif month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        else:
            return 3

def main():
    parser = argparse.ArgumentParser(description='Emergency Call Prediction Training and Inference')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data', type=str, default=r"emergency_calls_data\ReinlandPfalz_calls.csv",
                            help='Path to the input CSV file')
    
    # Infer model_name from data file if not explicitly provided
    # This default will be a placeholder, the actual inference will happen in the script logic if model_name is not set by user
    train_parser.add_argument('--model_name', type=str, 
                            help='Base name for the output model files. Defaults to CSV filename + "_predicted".')
    
    train_parser.add_argument('--output_dir', type=str, default='models',
                            help='Directory to save model and artifacts')
    train_parser.add_argument('--time_block', type=str, default='3h',
                            help='Time block for aggregation (e.g., 3h, 6H, D)')
    train_parser.add_argument('--test_size', type=float, default=0.2,
                            help='Proportion of data to use for testing (0-1)')
    
    # Inference command
    infer_parser = subparsers.add_parser('predict', help='Run inference with a trained model')
    infer_parser.add_argument('--coords', type=str,
                            help='Coordinates and timestamp as "lat,lon,yyyy-mm-dd HH:MM"')
    infer_parser.add_argument('--input', type=str,
                            help='Input data (CSV path or JSON dict) - for batch predictions')
    infer_parser.add_argument('--model_path', type=str,
                            help='Path to trained model (default: models/emergency_call_model.pkl)')
    infer_parser.add_argument('--output', type=str,
                            help='Output format (simple, json, csv) - default: simple')
        
    args = parser.parse_args()
    
    try:
        if args.command == 'train':
            if args.model_name is None:
                    # Extract filename from the data path and construct the model name
                    base_name = os.path.basename(args.data)
                    file_name_without_ext = os.path.splitext(base_name)[0]
                    args.model_name = f"{file_name_without_ext}_predictor"
                    print(f"Model name not specified, inferring from data file: {args.model_name}")
                    
            
            print(f"Using model name: {args.model_name}")
            predictor = EmergencyCallPredictor(
                data_path=args.data,
                model_name=args.model_name,
                output_dir=args.output_dir
            )
            predictor.run_pipeline(
                time_block=args.time_block,
                test_size=args.test_size
            )
        elif args.command == 'predict':
            predictor = EmergencyCallPredictor()
            
            # Load model
            predictor.load_model(args.model_path if args.model_path else None)
            
            # Handle different input types
            if args.coords:
                # Parse coordinate string
                try:
                    coords_parts = args.coords.split(',')
                    if len(coords_parts) != 3:
                        raise ValueError("Expected format: lat,lon,yyyy-mm-dd HH:MM")
                    
                    lat = float(coords_parts[0].strip())
                    lon = float(coords_parts[1].strip())
                    dt = datetime.strptime(coords_parts[2].strip(), '%Y-%m-%d %H:%M')
                except ValueError as e:
                    raise ValueError(f"Invalid coordinate format: {str(e)}")
                
                # Make single prediction
                prediction = predictor.inference((lat, lon, dt))
                
                # Format output
                if args.output == 'json':
                    result = {
                        'latitude': lat,
                        'longitude': lon,
                        'timestamp': dt.isoformat(),
                        'prediction': prediction
                    }
                    print(json.dumps(result, indent=2))
                elif args.output == 'csv':
                    print(f"latitude,longitude,timestamp,prediction")
                    print(f"{lat},{lon},{dt.isoformat()},{prediction}")
                else:  # simple
                    print(f"Predicted emergency calls: {prediction:.1f}")
                    
            elif args.input:
                # Handle batch predictions from file (original functionality)
                results = predictor.run_inference(input_data=args.input)
                
                if args.output == 'json':
                    print(json.dumps(results, indent=2))
                elif args.output == 'csv':
                    if 'predictions' in results:
                        df = pd.DataFrame({
                            'timestamp': results['timestamps'],
                            'latitude': [loc[0] for loc in results['locations']],
                            'longitude': [loc[1] for loc in results['locations']],
                            'prediction': results['predictions']
                        })
                        print(df.to_csv(index=False))
                    else:
                        print("timestamp,latitude,longitude,prediction")
                        print(f"{results['features']['timestamp']},"
                              f"{results['features']['lat_grid']},"
                              f"{results['features']['lon_grid']},"
                              f"{results['prediction']}")
                else:  # simple
                    if 'predictions' in results:
                        print(f"Generated {len(results['predictions'])} predictions")
                        print("First 5 predictions:")
                        for i in range(min(5, len(results['predictions']))):
                            print(f"{results['timestamps'][i]}: {results['predictions'][i]:.1f}")
                    else:
                        print(f"Prediction: {results['prediction']:.1f}")
            else:
                raise ValueError("Must provide either --coords or --input")
                
    except Exception as e:
        print(f"\n[ERROR] Command failed: {str(e)}", file=sys.stderr)
        exit(1)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()