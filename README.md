# Diabetes Prediction Dashboard

A Streamlit app that predicts diabetes risk using an XGBoost model. It’s designed for learning and demonstration, combining interactive predictions with basic fairness analysis.

## Live Demo
Check out the deployed app on Streamlit:  
👉 [Diabetes Prediction Dashboard on Streamlit]()

## Screenshot
Here’s a quick look at the app interface 👇  

![Diabetes Dashboard Screenshot](images/screenshot.png)

## What You Can Do
- Adjust inputs in the sidebar and click Predict to see diabetes risk.

- Explore how the threshold affects results with a simple slider.

- Check the Fairness page to visualize model performance by gender and smoking history.

## How to Run
```bash
  pip install -r requirements.txt
  streamlit run app.py
```

Then open it in your browser — you’ll see a default prediction first, and you can test your own inputs anytime.

## Tech Behind It

- Streamlit for the dashboard

- XGBoost for binary classification

- Altair for visualizations

- Python 3.10+

## Why I Built It

This project started as a way to explore how machine learning can support health-related insights — and how fairness plays a role in prediction models.

## Note

All data here is for demo only, not for medical decisions.

---
Built with curiosity, caffeine, and a love for data