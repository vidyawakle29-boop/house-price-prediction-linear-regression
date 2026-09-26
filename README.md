# House Price Prediction Using Linear Regression

## Project Overview

This project implements a Machine Learning model to predict house prices using Multiple Linear Regression. The model uses the living area, number of bedrooms, and number of full bathrooms as input features.

## Objective

To build and evaluate a Linear Regression model that predicts house sale prices based on selected housing features.

## Dataset

The project uses the **House Prices - Advanced Regression Techniques** dataset from Kaggle.

- Total records: 1,460
- Target variable: `SalePrice`
- Input features:
  - `GrLivArea` – Above-ground living area
  - `BedroomAbvGr` – Number of bedrooms
  - `FullBath` – Number of full bathrooms

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Methodology

1. Loaded the dataset using Pandas.
2. Selected relevant features for prediction.
3. Checked and handled missing values.
4. Split the dataset into 80% training and 20% testing data.
5. Trained a Multiple Linear Regression model.
6. Generated house price predictions.
7. Evaluated the model using MAE, RMSE, and R².
8. Visualized actual vs. predicted prices.

## Regression Equation

**Predicted SalePrice = 52,261.75 + (104.03 × GrLivArea) − (26,655.17 × BedroomAbvGr) + (30,014.32 × FullBath)**

## Model Performance

| Metric | Result |
|---|---:|
| MAE | 35,788.06 |
| RMSE | 52,975.72 |
| R² Score | 63.41% |

## Visualization

The project includes an **Actual vs. Predicted House Prices** visualization created using Matplotlib.

## Conclusion

The Linear Regression model achieved an R² score of 63.41% on the test dataset. This project provided practical experience in data preprocessing, feature selection, model training, prediction, evaluation, and data visualization.

## Author

**Vidya Ratan Wakle**

B.Tech Electronics & Communication Engineering
