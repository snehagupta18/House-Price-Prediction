This project investigates the effectiveness of various machine learning models for predicting house prices using the California housing dataset.

**Project Goal:**

* Identify the most accurate model for predicting house values based on features like median income, ocean proximity, and average room size.

**Data and Preprocessing:**

* Utilizes the `fetch_california_housing` function from `sklearn.datasets` to load the California housing dataset.
* Transforms the data into a pandas DataFrame for easier manipulation and analysis.
* Explores data distribution and identifies missing values using `.isnull().sum()`.
* Performs a descriptive statistical analysis using `.describe()`.
* Analyzes feature correlations using heatmaps generated with seaborn.

**Model Training and Evaluation:**

* Trains and evaluates five models:
    * XGBRegressor (eXtreme Gradient Boosting)
    * Linear Regression
    * Support Vector Regression (SVR) with RBF kernel
    * Ridge Regression
    * Lasso Regression
* Splits the data into training and testing sets using a 70/30 ratio with `train_test_split`.
* Employs R-squared error and Mean Absolute Error (MAE) to evaluate model performance.
* Visualizes actual vs. predicted prices using scatter plots for each model.

**Key Findings:**

* XGBRegressor achieves the highest accuracy among the evaluated models.
* It exhibits the highest R-squared error (closer to 1 indicates better fit) and the lowest Mean Absolute Error (represents the average difference between predicted and actual values).

**Future Work:**

* Explore feature engineering techniques like dimensionality reduction or creating new features.
* Perform hyperparameter tuning to optimize model performance for each algorithm.
* Consider using ensemble methods that combine predictions from multiple models for potentially better results.
* Develop a user-friendly interface for house price prediction based on the chosen model.

**Dependencies:**

* numpy
* pandas
* matplotlib.pyplot
* seaborn
* scikit-learn (including datasets, model_selection, metrics, linear_model, svm, xgboost)

**Instructions:**

1. Ensure the dependencies are installed.
2. Run the Python script to execute the analysis and predictions.

**Note:**

This project provides a basic framework for house price prediction using machine learning models. The chosen model (XGBRegressor) can be further optimized for better accuracy with additional techniques.
