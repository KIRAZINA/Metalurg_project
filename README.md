# Metalurg_project
Steel Composition Analysis
Predicts silicon content in steel post-processing using linear regression and decision tree models. Features data preprocessing, OLS analysis, and correlation matrix visualization. Uses chemical compositions, additives, and process parameters from source_data.xls. Generates regression plots and evaluates model performance with MSE and R-squared.
Overview
This project analyzes and predicts the silicon content (steel_Si_after) in steel after processing using machine learning techniques. It leverages a dataset containing chemical compositions (e.g., carbon, manganese), additives (e.g., lime, FeSi65), and process parameters (e.g., temperature, argon flow) to build predictive models. The code is optimized for clarity and performance, making it suitable for demonstrating data science skills.
Features

Data Preprocessing: Loads and cleans data from source_data.xls, handling missing values and converting to numeric format.
Linear Regression: Trains a model to predict silicon content and visualizes results with regression plots.
OLS Analysis: Performs statistical analysis using Ordinary Least Squares to assess model significance.
Decision Tree Regression: Implements a decision tree model with evaluation metrics (MSE, R²).
Visualization: Includes regression plots and a correlation matrix heatmap for data exploration.
Best Configuration: Identifies and displays the optimal steel composition based on model predictions.

Installation

Clone the repository:git clone https://github.com/KIRAZINA/Metalurg_project.git
cd Metalurg_project


Create a virtual environment and activate it:python -m venv venv
.\venv\Scripts\Activate  # On Windows


Install dependencies:pip install pandas seaborn matplotlib scikit-learn statsmodels xlrd


Ensure source_data.xls is in the project directory.

Usage
Run the script to perform analysis and generate visualizations:
python main.py


The script will output model performance metrics, OLS summary, and the best steel configuration.
Plots will display regression results and correlation matrix.

Results

Linear Regression: Provides predictions with a regression plot.
Decision Tree Regression: Achieves an R² of approximately 0.94 and identifies the best configuration.
OLS Analysis: Offers statistical insights, though multicollinearity may require further data refinement.
Best Configuration: Outputs an optimal formula (e.g., Si=0.75%, C=0.20%, Mn=1.50%) based on closest prediction to actual values.

Dependencies

Python 3.x
pandas
seaborn
matplotlib
scikit-learn
statsmodels
xlrd (for .xls file support)

Future Improvements

Explore additional models (e.g., Random Forest, Gradient Boosting).
Implement feature selection to reduce multicollinearity.
Add cross-validation for robust model evaluation.
Support .xlsx files with openpyxl for broader compatibility.

