# Biomaterial Degradation AI

This project predicts a biomaterial degradation rate constant `k`, then uses an
exponential decay equation to estimate how much mass remains after a given
number of days. It uses a small scikit-learn regression model and a Streamlit
website.

## What the Model Predicts

Original measured target:

- `Mass_Remaining_Percentage`

Model target:

- `Degradation_Rate_k`

Input features:

- `Material_Type`
- `Cellulose_Percentage`
- `Temperature_C`
- `pH_Level`
- `Environment`
- `degree_substitution`

`Days_Elapsed` is not used as a normal model input for `k`. Instead, it is used
inside the exponential decay equation:

```text
Mass remaining percentage = 100 * exp(-k * Days_Elapsed)
```

The app clips predicted `k` values to be non-negative. This physics-informed
constraint prevents predicted mass remaining from increasing as time increases.

The app also reports degradation percentage:

```text
Degradation percentage = 100 - Mass remaining percentage
```

The app reports a simple estimated range using held-out test error for mass
remaining:

```text
Estimated range = predicted mass remaining ± test-set Mass MAE
```

The estimated range is clipped to stay between 0% and 100%. This is easier to
interpret than propagating uncertainty through `k`, which can sometimes create
unhelpful ranges like 0%–100%.

## Project Files

```text
app.py                  Streamlit website
data/degradationdata.xlsx
models/best_model.pkl   Saved best model
models/model_metrics.csv Held-out test metrics for each model
models/test_predictions.csv Actual vs predicted test-set results
models/removed_rate_outliers.csv Fitted-k rows removed by the outlier filter
src/clean_data.py       Loads and cleans the spreadsheet
src/train_model.py      Trains and compares models
src/predict.py          Loads the model and makes predictions
requirements.txt        Python packages
```

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

## Train the Model

The project expects the spreadsheet at:

```text
data/degradationdata.xlsx
```

If you replace the spreadsheet with a new version, keep the same column names and
run the training command again.

Run:

```bash
python src/train_model.py
```

The script compares:

- `DummyRegressor`
- Ridge Regression
- `RandomForestRegressor`
- `ExtraTreesRegressor`
- `GradientBoostingRegressor`

Each model uses a scikit-learn `Pipeline` with:

- `StandardScaler` for numeric features
- `OneHotEncoder(handle_unknown="ignore")` for categorical features

The models are evaluated with:

- MAE
- RMSE
- R²

The main `MAE`, `RMSE`, and `R²` columns in `models/model_metrics.csv` measure
predicted mass remaining percentage because that is the user-facing prediction.
The file also includes technical `K_MAE`, `K_RMSE`, and `K_R2` columns for the
internal degradation-rate model.

The data is split into:

- 80% training data
- 20% test data

The training script first fits `k` values from the measured mass remaining data
using the exponential decay equation. Each machine learning model trains only on
the training set of fitted `k` values. The test set is kept separate and is used
only for evaluation. The best model is selected by lowest RMSE for predicted
mass remaining on the held-out test set and saved to:

```text
models/best_model.pkl
```

The saved model is the best `k` model trained on the 80% training set. It is not
retrained on the test set.

The comparison table is also saved to:

```text
models/model_metrics.csv
```

The test-set actual vs predicted values are saved to:

```text
models/test_predictions.csv
```

## Accuracy Improvements

The current training script uses a balanced model-selection rule:

- It ranks each model by mass MAE, mass RMSE, and mass R².
- It selects the model with the best combined rank, so one metric is not
  improved by making another metric much worse.

It also includes two accuracy-focused changes:

- It adds `ExtraTreesRegressor` and a tuned `RandomForestRegressor`.
- It removes extreme fitted degradation-rate outliers before the train/test
  split using the rule: fitted `k` values above `Q3 + 2 * IQR`.

The raw spreadsheet is not edited. Removed fitted-rate rows are saved to:

```text
models/removed_rate_outliers.csv
```

For the current dataset, this removes 23 fitted-rate rows from 248 fitted
conditions. These removed rows represent unusually fast degradation rates, so
predictions for extremely fast-degrading compost conditions should be treated
with extra caution.

Current held-out performance after these changes:

```text
Best model: RandomForestRegressor
Mass MAE:   6.48 percentage points
Mass RMSE:  9.74 percentage points
Mass R²:    0.865

Technical k MAE:  0.00676
Technical k RMSE: 0.01232
Technical k R²:   0.596
```

## Run the Website

After training, start the Streamlit app:

```bash
python -m streamlit run app.py
```

The website is organized into five tabs:

- Predictor
- Compare Materials
- Model Performance
- Dataset Explorer
- About the Research

It loads `models/best_model.pkl`, accepts material and environment inputs, and
predicts:

- mass remaining percentage
- uncertainty estimate and estimated range
- degradation rate constant `k`
- degradation percentage
- degradation curve over time
- comparison curves for two biomaterial/environment choices

The website also shows held-out test-set MAE, RMSE, and R² from
`models/model_metrics.csv`, plus the full model comparison table.

## Scientific Limitations

This is a beginner-friendly research model, not a laboratory replacement.

- The dataset is small, so predictions may change a lot if more data is added.
- The model learns patterns from the spreadsheet only. It may not work well for
  materials, temperatures, pH levels, environments, or day ranges that are very
  different from the training data.
- The degradation curve is made from separate predictions at each time point.
  Because of that, the curve may not always be perfectly smooth or strictly
  decreasing.
- The uncertainty estimate is based on held-out test error for mass remaining.
  It is a rough typical-error estimate, not a formal confidence interval for a
  specific material or environment.
- The model uses simplified features and does not include every factor that can
  affect degradation, such as humidity, microbial activity, sample thickness,
  crystallinity, additives, or experimental measurement error.
- Results should be checked against real experiments before making scientific or
  engineering conclusions.
