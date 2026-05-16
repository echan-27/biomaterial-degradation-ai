# Project Instructions

This is a high school research/AI project about predicting biomaterial degradation.

Use beginner-friendly Python code with clear comments.

The website should use Streamlit.

The machine learning model should use scikit-learn, not deep learning, because the dataset is small.

The target variable is Mass_Remaining_Percentage.

Use these features:
- Material_Type
- Cellulose_Percentage
- Temperature_C
- pH_Level
- Environment
- Days_Elapsed
- degree_substitution

Use a scikit-learn Pipeline with:
- StandardScaler for numeric features
- OneHotEncoder(handle_unknown="ignore") for categorical features

Compare:
- DummyRegressor
- Ridge Regression
- RandomForestRegressor
- GradientBoostingRegressor

Evaluate using:
- MAE
- RMSE
- R²

Save the best model to models/best_model.pkl.

The Streamlit app should allow a user to input material/environment values and predict:
- mass remaining percentage
- degradation percentage
- degradation curve over time

Keep the code readable and explain the scientific limitations in the README.