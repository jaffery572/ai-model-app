"""
Train Model Script for Infrastructure AI
Run this to retrain the model with custom data
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json

def create_training_data(n_samples=10000):
    """Generate synthetic training data"""
    np.random.seed(42)
    
    # Features
    data = {
        'population': np.random.uniform(50, 20000, n_samples),
        'area': np.random.uniform(10, 10000, n_samples),
        'gdp_per_capita': np.random.uniform(1000, 50000, n_samples),
        'urban_population': np.random.uniform(10, 100, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate density
    df['density'] = df['population'] / df['area']
    
    # Create labels based on rules
    conditions = [
        (df['density'] > 5) & (df['gdp_per_capita'] < 10000),
        (df['density'] > 3) & (df['gdp_per_capita'] < 15000),
        (df['density'] > 2) & (df['urban_population'] > 70),
        (df['density'] < 0.5) & (df['gdp_per_capita'] > 20000),
        (df['density'] < 1) & (df['urban_population'] < 30)
    ]
    
    choices = [1, 1, 1, 0, 0]
    
    df['label'] = np.select(conditions, choices, default=0)
    
    # Add some randomness
    mask = np.random.random(n_samples) < 0.1
    df.loc[mask, 'label'] = np.random.choice([0, 1], size=mask.sum())
    
    return df

def train_model():
    """Train and save the model"""
    print("🚀 Creating training data...")
    df = create_training_data(5000)
    
    print("📊 Data Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Prepare features and labels
    X = df[['population', 'area']].values
    y = df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\n🎯 Training model...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\n✅ Training completed!")
    print(f"Training Accuracy: {train_score:.3f}")
    print(f"Testing Accuracy: {test_score:.3f}")
    
    # Save model
    with open('ai_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save model info
    model_info = {
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "training_accuracy": float(train_score),
        "testing_accuracy": float(test_score),
        "features": ["population", "area"],
        "n_estimators": 150,
        "max_depth": 15,
        "created": str(np.datetime64('now'))
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n💾 Model saved as 'ai_model.pkl'")
    print(f"📝 Model info saved as 'model_info.json'")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': ['population', 'area'],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Feature Importance:")
    print(feature_importance)
    
    return model

if __name__ == "__main__":
    print("=" * 50)
    print("🏗️  GLOBAL INFRASTRUCTURE AI - MODEL TRAINING")
    print("=" * 50)
    
    train_model()
    
    print("\n" + "=" * 50)
    print("🎉 Training complete! Ready to deploy.")
    print("=" * 50)
