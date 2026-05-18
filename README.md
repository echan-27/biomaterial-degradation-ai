# Biomaterial Degradation AI

This project predicts how much mass a biomaterial may have remaining after a
given number of days in a chosen environment. It uses a small scikit-learn
regression model and a Streamlit website.

## What the Model Predicts

Target variable:

- `Mass_Remaining_Percentage`

Input features:

- `Material_Type`
- `Cellulose_Percentage`
- `Temperature_C`
- `pH_Level`
- `Environment`
- `Days_Elapsed`
- `degree_substitution`

The app also reports degradation percentage:

```text
Degradation percentage = 100 - Mass remaining percentage
```

The app reports uncertainty using the best model's held-out test RMSE:

```text
Predicted mass remaining: prediction ± RMSE
Estimated range: prediction - RMSE to prediction + RMSE
```

The estimated range is clipped to stay between 0% and 100%.

## Project Files

```text
app.py                  Streamlit website
data/degradationdata.xlsx
models/best_model.pkl   Saved best model
models/model_metrics.csv Held-out test metrics for each model
models/test_predictions.csv Actual vs predicted test-set results
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
- `GradientBoostingRegressor`

Each model uses a scikit-learn `Pipeline` with:

- `StandardScaler` for numeric features
- `OneHotEncoder(handle_unknown="ignore")` for categorical features

The models are evaluated with:

- MAE
- RMSE
- R²

The data is split into:

- 80% training data
- 20% test data

Each model trains only on the training set. The test set is kept separate and is
used only for evaluation. The best model is selected by lowest RMSE on the
held-out test set and saved to:

```text
models/best_model.pkl
```

The saved model is the best model trained on the 80% training set. It is not
retrained on the test set.

The comparison table is also saved to:

```text
models/model_metrics.csv
```

The test-set actual vs predicted values are saved to:

```text
models/test_predictions.csv
```

## Run the Website

After training, start the Streamlit app:

```bash
python -m streamlit run app.py
```

The website loads `models/best_model.pkl`, accepts material and environment
inputs, and predicts:

- mass remaining percentage
- uncertainty estimate and estimated range
- degradation percentage
- degradation curve over time

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
- The uncertainty estimate is based on overall test-set RMSE. It is a rough
  typical-error estimate, not a formal confidence interval for a specific
  material or environment.
- The model uses simplified features and does not include every factor that can
  affect degradation, such as humidity, microbial activity, sample thickness,
  crystallinity, additives, or experimental measurement error.
- Results should be checked against real experiments before making scientific or
  engineering conclusions.
