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

## Project Files

```text
app.py                  Streamlit website
data/degradationdata.xlsx
models/best_model.pkl   Saved best model
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

The best model is selected by lowest RMSE on the held-out test set and saved to:

```text
models/best_model.pkl
```

The comparison table is also saved to:

```text
models/model_metrics.csv
```

## Run the Website

After training, start the Streamlit app:

```bash
python -m streamlit run app.py
```

The website loads `models/best_model.pkl`, accepts material and environment
inputs, and predicts:

- mass remaining percentage
- degradation percentage
- degradation curve over time

## Scientific Limitations

This is a beginner-friendly research model, not a laboratory replacement.

- The dataset is small, so predictions may change a lot if more data is added.
- The model learns patterns from the spreadsheet only. It may not work well for
  materials, temperatures, pH levels, environments, or day ranges that are very
  different from the training data.
- The degradation curve is made from separate predictions at each time point.
  Because of that, the curve may not always be perfectly smooth or strictly
  decreasing.
- The model uses simplified features and does not include every factor that can
  affect degradation, such as humidity, microbial activity, sample thickness,
  crystallinity, additives, or experimental measurement error.
- Results should be checked against real experiments before making scientific or
  engineering conclusions.
