# Biomaterial Degradation AI

This project predicts a biomaterial degradation rate constant `k`, then uses an
exponential decay equation to estimate how much mass remains after a given
number of days. It uses scikit-learn and a Streamlit website.

## What the Model Predicts

Original measured target:

- `Mass_Remaining_Percentage`

Model target:

- `Degradation_Rate_k`

Input features used by the saved model:

- `Material_Type`
- `Cellulose_Percentage`
- `Temperature_C`
- `pH_Level`
- `Environment`
- `degree_substitution`
- `Evaluation_Day`

`Evaluation_Day` comes from the exposure day used to evaluate the fitted
degradation curve. The app passes the user's selected day into this field.

The model predicts an effective `k`, and the app converts it into mass remaining:

```text
Mass remaining percentage = 100 * exp(-k * Days_Elapsed)
```

Predicted `k` values are clipped to be non-negative. This physics-informed
constraint prevents predicted mass remaining from increasing over time.

The app also reports:

```text
Degradation percentage = 100 - Mass remaining percentage
```

## Project Files

```text
app.py                         Streamlit website
data/degradationdata.xlsx      Training spreadsheet
models/best_model.pkl          Saved selected model
models/model_metrics.csv       Final hold-out test metrics for the selected model
models/cv_model_comparison.csv Training-only cross-validation model comparison
models/test_predictions.csv    Actual vs predicted test-set results
models/removed_rate_outliers.csv Training rows removed by the outlier filter
src/clean_data.py              Loads and cleans the spreadsheet
src/train_model.py             Trains, tunes, and evaluates models
src/predict.py                 Loads the model and makes predictions
requirements.txt               Python packages
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
run:

```bash
python src/train_model.py
```

The script compares:

- `DummyRegressor`
- Ridge Regression
- `RandomForestRegressor`
- `GradientBoostingRegressor`

Each model uses a scikit-learn `Pipeline` with:

- `StandardScaler` for numeric features
- `OneHotEncoder(handle_unknown="ignore")` for categorical features
- `TransformedTargetRegressor` with `log1p(k)` to make the skewed degradation
  rates easier to learn

The training workflow avoids test-set leakage:

- First, the fitted-rate dataset is split into 80% training and 20% test data.
- The IQR outlier filter is calculated only from the training target `y_train`.
- Only training rows with fitted `k` above `Q3 + 3.0 * IQR` are removed.
- The hold-out test set is never filtered and is not used for model selection.
- Random forest and gradient boosting hyperparameters are tuned with 5-fold
  cross-validation on the training set only.
- The selected model is evaluated once on the untouched hold-out test set.

`models/cv_model_comparison.csv` contains the training-only model comparison.
`models/model_metrics.csv` contains the final test metrics for the selected
model, so the test set is not used to choose between models.

## Current Performance

Current selected model:

```text
RandomForestRegressor
```

Training-only cross-validation:

```text
CV MAE:   11.84 percentage points
CV RMSE:  19.03 percentage points
CV R²:    0.445
```

Untouched hold-out test set:

```text
Mass MAE:   13.63 percentage points
Mass RMSE:  23.57 percentage points
Mass R²:    0.418

k MAE:      0.03406 1/day
k RMSE:     0.10191 1/day
k R²:       0.0278
```

These honest test numbers are lower than earlier experimental results because
the test set is no longer used for outlier filtering or model selection.

## Test Predictions and Uncertainty

`models/test_predictions.csv` saves actual vs predicted mass remaining for the
untouched test rows.

The uncertainty range uses the selected model's hold-out Mass MAE as a practical
percentage-point band:

```text
lower mass = predicted mass - Mass MAE
upper mass = predicted mass + Mass MAE
```

The lower and upper values are clipped to stay between 0% and 100%. This avoids
the unhelpful 100% upper-bound flatline that can happen when a large `k`
uncertainty clips the lower `k` value to zero. Treat these ranges as rough
typical-error warnings, not formal confidence intervals.

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
- estimated range from flat Mass MAE
- degradation rate constant `k`
- degradation percentage
- degradation curve over time
- comparison curves for two biomaterial/environment choices

## Scientific Limitations

This is a beginner-friendly research model, not a laboratory replacement.

- The dataset is small, so predictions may change a lot if more data is added.
- Many fitted `k` values come from only one or two timepoints, which makes them
  noisy estimates of true degradation behavior.
- The training outlier filter removes extreme fitted-rate rows only from the
  training set. The test set still includes difficult real-world cases.
- Predictions may be unreliable for materials, environments, temperatures, pH
  levels, degrees of substitution, or day ranges far from the training data.
- Real degradation can depend on humidity, microbial activity, sample thickness,
  crystallinity, additives, and measurement error.
- Results should be checked against real experiments before making scientific or
  engineering conclusions.
