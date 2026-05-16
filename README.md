# Best Time To Call Predictor | Synthetic Data Portfolio Project

## Live Demo

Try the deployed Streamlit app here:
https://shargil-best-time-to-call-predictive-analytics.streamlit.app/


## 1. Project Overview

This project converts an original **descriptive analytics** Best Time To Call analysis into a more mature **predictive analytics** portfolio project.

The original business problem was simple but valuable: for outbound campaigns, identify the call windows where customers are most likely to answer. The original analysis summarized connection rates by hour, weekday, month, weekend/weekday, and time block. This public GitHub version keeps the same business logic and field structure, but replaces confidential client data with synthetic data and adds a machine learning layer that predicts the probability of a successful connection.

> Confidentiality note: all data in this repository is synthetic. No real client names, customer IDs, phone numbers, call IDs, or internal operational records are included.

## 2. Why This Project Matters

Outbound operations often waste time calling customers during poor contact windows. A predictive Best Time To Call model helps operations teams:

- prioritize high-probability call windows,
- improve connection rate,
- reduce wasted dialing attempts,
- support workforce planning,
- create a stronger analytical decision framework than static reporting alone.

## 3. Dataset

The synthetic dataset keeps the same core fields used in the original analysis:

| Field | Description |
|---|---|
| Original Call Date | Date of outbound call attempt |
| Original Call Hour | Hour of call attempt |
| Original Call Minute | Minute of call attempt |
| Total Duration | Simulated total call duration |
| Total Billsec | Simulated billable talk time |
| Outbound Call | Outbound call flag |
| Uniqueid | Synthetic unique call ID |
| Callerid | Synthetic customer/caller ID |
| Last Event | Simulated call-system final event |
| Disposition | Simulated call outcome |
| Agent Name | Synthetic agent/system identifier |

The target variable is created using the original business rule:

```python
Actual Connected = Total Billsec >= 20 AND Disposition == "ANSWERED" AND Agent Name contains "t1.tlp.tso.ms"
```

## 4. Project Structure

```text
btc_predictive_project/
├── data/
│   └── synthetic_btc_outbound_calls.csv
├── models/
│   ├── btc_connect_probability_model.joblib
│   └── model_metrics.json
├── notebooks/
│   └── BTC_Predictive_Analytics_Colab.ipynb
├── outputs/
│   ├── best_hour_by_weekday_recommendations.csv
│   ├── connect_rate_by_agent_timeclass.csv
│   ├── connect_rate_by_hour.csv
│   ├── connect_rate_by_month.csv
│   ├── connect_rate_by_timeclass.csv
│   ├── connect_rate_by_weekday.csv
│   ├── connect_rate_by_weekday_hour.csv
│   ├── feature_importance.csv
│   ├── future_best_time_recommendations.csv
│   └── test_predictions_sample.csv
├── src/
│   ├── generate_synthetic_data.py
│   └── train_model.py
├── streamlit_app/
│   └── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 5. Model Approach

This is a binary classification problem:

- **1 = Actual Connected**
- **0 = Not Connected**

Features used by the model are limited to fields that would be known before making the call, such as:

- call hour,
- call minute,
- weekday,
- month,
- weekend flag,
- time class,
- agent group,
- caller attempt number.

Outcome fields such as `Disposition`, `Total Billsec`, and `Total Duration` are intentionally excluded from the model features because using them would create data leakage.

## 6. How To Run Locally

```bash
# 1. Clone the repo
 git clone https://github.com/YOUR_USERNAME/btc-predictive-analytics.git
 cd btc-predictive-analytics

# 2. Create a virtual environment
 python -m venv .venv

# Windows
 .venv\Scripts\activate

# Mac/Linux
 source .venv/bin/activate

# 3. Install dependencies
 pip install -r requirements.txt

# 4. Generate synthetic data and train model
 python src/train_model.py

# 5. Run Streamlit app
 streamlit run streamlit_app/app.py
```

## 7. How To Run In Google Colab

Open `notebooks/BTC_Predictive_Analytics_Colab.ipynb` and run all cells.

The notebook will:

1. install dependencies,
2. generate synthetic data,
3. train the model,
4. save model artifacts,
5. create output CSVs,
6. show model performance,
7. prepare files for GitHub deployment.

## 8. Streamlit Deployment

The app can be deployed on Streamlit Community Cloud.

Main app file:

```text
streamlit_app/app.py
```

Required files for deployment:

```text
requirements.txt
models/btc_connect_probability_model.joblib
models/model_metrics.json
outputs/future_best_time_recommendations.csv
outputs/feature_importance.csv
```



## 9. Future Improvements

- Adding campaign-level features.
- Adding customer segment features.
- Adding holiday and payday indicators.
- Compare Random Forest, XGBoost, and Logistic Regression.
- Adding MLflow experiment tracking.
- Adding a batch scoring endpoint.
- Adding A/B test design for recommended vs. normal dialing schedule.
