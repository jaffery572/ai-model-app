import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(42)
n_samples = 1000

# Features: Population (thousands), Area (sq km)
X = np.random.rand(n_samples, 2) * [1000, 500]  # Random data

# Labels: 0 = Sufficient, 1 = Development Needed
y = (X[:, 0] / X[:, 1] > 2).astype(int)  # If population density > 2

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('ai_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
print(f"Accuracy: {model.score(X_test, y_test):.2f}")
