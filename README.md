# Global Infrastructure AI (Streamlit Cloud)

## Files
- app.py
- requirements.txt
- runtime.txt
- .streamlit/config.toml

## Deploy (Streamlit Cloud)
1) Push to GitHub
2) Streamlit Cloud -> New App
3) Branch: main, File: app.py
4) If error: Manage App -> Clear cache -> Reboot

## Notes
- runtime.txt forces python-3.11 (important for scikit-learn / pandas wheels)
- Model training is cloud-safe (default 25k synthetic samples)
- ML model saved to ./models/infrastructure_gb.joblib (ephemeral on some cloud restarts)
