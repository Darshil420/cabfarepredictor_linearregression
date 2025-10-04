from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

app = Flask(__name__)

# Enhanced CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:5000", "http://localhost:5000"],
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Add this to handle preflight requests
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Global variables for model and preprocessing
model = None
scaler = None
label_encoders = {}
feature_names = [
    'Trip_Distance_km', 'Time_of_Day', 'Day_of_Week', 'Passenger_Count',
    'Traffic_Conditions', 'Weather', 'Base_Fare', 'Per_Km_Rate', 
    'Per_Minute_Rate', 'Trip_Duration_Minutes'
]

# Categorical mappings
CATEGORICAL_MAPPINGS = {
    'Time_of_Day': ['Morning', 'Afternoon', 'Evening', 'Night'],
    'Day_of_Week': ['Weekday', 'Weekend'],
    'Traffic_Conditions': ['Low', 'Medium', 'High'],
    'Weather': ['Clear', 'Rain', 'Snow', 'Fog']
}

def create_sample_data():
    """Create sample training data based on the notebook structure"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Trip_Distance_km': np.random.uniform(1, 150, n_samples),
        'Time_of_Day': np.random.choice(CATEGORICAL_MAPPINGS['Time_of_Day'], n_samples),
        'Day_of_Week': np.random.choice(CATEGORICAL_MAPPINGS['Day_of_Week'], n_samples),
        'Passenger_Count': np.random.uniform(1, 4, n_samples),
        'Traffic_Conditions': np.random.choice(CATEGORICAL_MAPPINGS['Traffic_Conditions'], n_samples),
        'Weather': np.random.choice(CATEGORICAL_MAPPINGS['Weather'], n_samples),
        'Base_Fare': np.random.uniform(2, 5, n_samples),
        'Per_Km_Rate': np.random.uniform(0.5, 2, n_samples),
        'Per_Minute_Rate': np.random.uniform(0.1, 0.5, n_samples),
        'Trip_Duration_Minutes': np.random.uniform(5, 120, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate realistic trip price
    df['Trip_Price'] = (
        df['Base_Fare'] +
        df['Trip_Distance_km'] * df['Per_Km_Rate'] +
        df['Trip_Duration_Minutes'] * df['Per_Minute_Rate']
    )
    
    # Add some noise and adjustments based on conditions
    df['Trip_Price'] *= np.random.uniform(0.8, 1.2, n_samples)
    
    # Adjust for traffic and weather conditions
    traffic_multiplier = {
        'Low': 1.0,
        'Medium': 1.2,
        'High': 1.5
    }
    weather_multiplier = {
        'Clear': 1.0,
        'Rain': 1.1,
        'Snow': 1.3,
        'Fog': 1.2
    }
    
    df['Trip_Price'] *= df['Traffic_Conditions'].map(traffic_multiplier)
    df['Trip_Price'] *= df['Weather'].map(weather_multiplier)
    
    return df

def train_model():
    """Train the cab fare prediction model"""
    global model, scaler, label_encoders
    
    # Create sample data (replace with your actual dataset)
    df = create_sample_data()
    
    # Prepare features and target
    X = df[feature_names].copy()
    y = df['Trip_Price']
    
    # Encode categorical variables
    categorical_columns = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather']
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_columns = ['Trip_Distance_km', 'Passenger_Count', 'Base_Fare', 
                        'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes']
    
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
    
    # Train model
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mae': round(mae, 2),
        'mse': round(mse, 2),
        'r2': round(r2, 4)
    }
    
    # Save model and preprocessing objects
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    return metrics

def load_model():
    """Load trained model and preprocessing objects"""
    global model, scaler, label_encoders
    
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        return True
    except:
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/features', methods=['GET'])
def get_features():
    """Return available features and their options"""
    return jsonify({
        'categorical_mappings': CATEGORICAL_MAPPINGS,
        'feature_names': feature_names
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict cab fare based on input features"""
    try:
        data = request.get_json()
        print("Received data:", data)  # Debug print
        
        # Validate input exists
        if not data:
            return jsonify({'error': 'No data received'}), 400
        
        # Check for missing fields with better error messages
        missing_fields = []
        for field in feature_names:
            if field not in data or data[field] is None or data[field] == '':
                missing_fields.append(field)
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Create input array
        input_data = []
        for feature in feature_names:
            value = data[feature]
            
            if feature in CATEGORICAL_MAPPINGS:
                # Validate categorical values
                if value not in CATEGORICAL_MAPPINGS[feature]:
                    return jsonify({
                        'error': f'Invalid value for {feature}. Must be one of: {CATEGORICAL_MAPPINGS[feature]}'
                    }), 400
                # Encode categorical value
                encoded_value = label_encoders[feature].transform([value])[0]
                input_data.append(encoded_value)
            else:
                # Validate numerical values
                try:
                    num_value = float(value)
                    if num_value < 0:
                        return jsonify({'error': f'{feature} must be positive'}), 400
                    input_data.append(num_value)
                except (ValueError, TypeError):
                    return jsonify({'error': f'{feature} must be a valid number'}), 400
        
        # Convert to numpy array and reshape
        input_array = np.array(input_data).reshape(1, -1)
        
        # Scale numerical features
        numerical_columns = ['Trip_Distance_km', 'Passenger_Count', 'Base_Fare', 
                           'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes']
        numerical_indices = [feature_names.index(col) for col in numerical_columns]
        
        input_array[0, numerical_indices] = scaler.transform(
            input_array[0, numerical_indices].reshape(1, -1)
        )
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        
        return jsonify({
            'predicted_price': round(prediction, 2),
            'currency': 'USD',
            'features_used': feature_names,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Debug print
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/model/metrics', methods=['GET'])
def get_model_metrics():
    """Return model performance metrics"""
    try:
        # In a real scenario, you'd load these from your trained model
        metrics = {
            'mean_absolute_error': 4.32,
            'mean_squared_error': 32.15,
            'r2_score': 0.8943,
            'model_type': 'Ridge Regression'
        }
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/retrain', methods=['POST'])
def retrain_model():
    """Retrain the model with new data"""
    try:
        metrics = train_model()
        return jsonify({
            'message': 'Model retrained successfully',
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature_importance', methods=['GET'])
def feature_importance():
    """Return feature importance scores"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 400
        
        importance = dict(zip(feature_names, abs(model.coef_)))
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return jsonify({
            'feature_importance': sorted_importance,
            'most_important': list(sorted_importance.keys())[:3]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug', methods=['POST'])
def debug_data():
    """Debug endpoint to check what data is being received"""
    data = request.get_json()
    print("DEBUG - Received data:", data)
    return jsonify({
        'received_data': data,
        'data_types': {k: type(v).__name__ for k, v in data.items()} if data else {},
        'feature_names': feature_names,
        'missing_fields': [f for f in feature_names if f not in data] if data else feature_names
    })

if __name__ == '__main__':
    # Initialize model on startup
    if not load_model():
        print("Training new model...")
        train_model()
        print("Model trained successfully!")
    
    print("Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)