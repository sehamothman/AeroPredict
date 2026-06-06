# ✈️ Flight Delay Prediction & Cost Forecasting

A machine learning pipeline that predicts U.S. flight delays and estimates their financial impact, built by merging real-world airline performance and weather data from 2019 to 2026.

---

## 🧠 What It Does

1. **Predicts whether a flight will be delayed** using a tuned LightGBM classifier trained on ~4.9 million rows of flight records enriched with local weather conditions.
2. **Forecasts the cost of delays** using a two-stage regression pipeline that translates predicted delay minutes into dollar figures.

![Alternative Text Here](visuals/Model.png)
---

## 📊 Data Sources

| Dataset | Source | Coverage |
|---|---|---|
| U.S. Airline On-Time Performance | [BTS via Kaggle](https://www.kaggle.com/datasets/kamalalqedra/bts-ontime-performance-2019-present-csv) | 2019 – 2026 |
| ASOS Weather Observations | [Kaggle](https://www.kaggle.com/datasets/sehamhakimothman/asos-weather-data-2019-2026) | 2019 – 2026 |

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.7128 |
| Overall Accuracy | 71% |
| On-Time F1-Score | 0.80 |
| Delayed F1-Score | 0.47 |
| Cost Forecast Error | ~$40M on $2.38B actual |

> The lower F1 for delayed flights reflects natural class imbalance — delayed flights are the minority class (~25% of records).

Regression Hurdle model results:
![Alternative Text Here](visuals/ReggresionModel_Results.png)


---
## 🛠️ Tech Stack

- **Python** — pandas, scikit-learn, LightGBM
- **Data** — BTS On-Time Performance + ASOS Weather
- **Modeling** — Classification (delay/no delay) + Regression (delay cost)
