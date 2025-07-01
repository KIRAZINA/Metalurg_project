import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Constants
FILE_PATH = 'source_data.xls'
TARGET = 'steel_Si_after'
PREDICTORS = [
    'steel_C_after', 'steel_Mn_after', 'steel_P_after',
    'steel_Cr_after', 'steel_Ni_after', 'steel_Cu_after', 'steel_Al_after',
    'steel_N_after', 'steel_Ti_after', 'steel_Zr_after', 'steel_V_after',
    'steel_Mo_after', 'steel_Nb_after', 'steel_Sn_after', 'steel_As_after',
    'steel_Ca_after', 'slag_CaO_after', 'slag_SiO2_after', 'slag_MgO_after',
    'slag_FeO_after', 'slag_Al2O3_after', 'slag_P2O5_after', 'slag_Fe2O3_after',
    'slag_MnO_after', 'slag_S_after', 'slag_basicity_after', 'add_FeCa_before',
    'add_FeSi65_before', 'add_FeMn_before', 'add_FeSiMn_before', 'add_lime_before',
    'add_smelting_before', 'add_AlP_before', 'add_SiCa_before', 'add_Al12_before',
    'add_Al13_before', 'add_FeB_before', 'add_feldspar_before', 'add_FeTi_before',
    'add_AlGr_before', 'sulfur_reduction_ratio'
]  # Removed 'steel_Si_after' to avoid multicollinearity


# Data preprocessing function
def load_and_preprocess_data(file_path):
    """Load and preprocess Excel data for steel composition analysis."""
    try:
        data = pd.read_excel(file_path, header=3, usecols='B:CN')
        data.columns = [
            'sample_number', 'steel_C_before', 'steel_Mn_before', 'steel_Si_before', 'steel_S_before',
            'steel_P_before', 'steel_Cr_before', 'steel_Ni_before', 'steel_Cu_before', 'steel_Al_before',
            'steel_N_before', 'steel_Ti_before', 'steel_Zr_before', 'steel_V_before', 'steel_Mo_before',
            'steel_Nb_before', 'steel_Sn_before', 'steel_As_before', 'steel_Ca_before',
            'slag_CaO_before', 'slag_SiO2_before', 'slag_MgO_before', 'slag_FeO_before',
            'slag_Al2O3_before', 'slag_P2O5_before', 'slag_Fe2O3_before', 'slag_MnO_before',
            'slag_S_before', 'slag_basicity_before', 'add_FeCa_before', 'add_FeSi65_before',
            'add_FeMn_before', 'add_FeSiMn_before', 'add_lime_before', 'add_smelting_before',
            'add_AlP_before', 'add_SiCa_before', 'add_Al12_before', 'add_Al13_before',
            'add_FeB_before', 'add_feldspar_before', 'add_FeTi_before', 'add_AlGr_before',
            'sulfur_reduction_ratio', 'steel_C_after', 'steel_Mn_after', 'steel_Si_after',
            'steel_S_after', 'steel_P_after', 'steel_Cr_after', 'steel_Ni_after', 'steel_Cu_after',
            'steel_Al_after', 'steel_N_after', 'steel_Ti_after', 'steel_Zr_after', 'steel_V_after',
            'steel_Mo_after', 'steel_Nb_after', 'steel_Sn_after', 'steel_As_after', 'steel_Ca_after',
            'slag_CaO_after', 'slag_SiO2_after', 'slag_MgO_after', 'slag_FeO_after',
            'slag_Al2O3_after', 'slag_P2O5_after', 'slag_Fe2O3_after', 'slag_MnO_after',
            'slag_S_after', 'slag_basicity_after', 'total_weight', 'cutting_total',
            'heat_number', 'upk_number', 'processing_time', 'bottom_blowing_time',
            'argon_flow_total', 'blow_flow1', 'blow_flow2', 'slag_height_incoming',
            'slag_height_outgoing', 'slag_color_incoming', 'slag_color_outgoing',
            'metal_temp_first', 'metal_temp_last', 'heat_temp_first', 'heat_temp_last',
            'heating_duration', 'energy_consumption'
        ]
        data_numeric = data.apply(pd.to_numeric, errors='coerce')
        data_numeric = data_numeric.dropna(axis=1, thresh=len(data_numeric) * 0.5)
        data_numeric = data_numeric.fillna(data_numeric.mean())
        return data_numeric
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return None


# Model training function
def train_linear_regression(X_train, y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# Visualization function
def plot_predictions(y_true, y_pred, xlabel, ylabel, title):
    """Plot regression predictions with a regression line."""
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_true, y=y_pred, line_kws={"color": "red"})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# OLS analysis function
def analyze_ols_model(X, y):
    """Perform OLS regression analysis and print summary."""
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())


# Function to find best configuration
def find_best_configuration(data, y_test, y_pred_test):
    """Find the sample with the best predicted silicon content based on R² closeness."""
    try:
        # Find index where prediction is closest to actual value
        best_idx = (y_test - y_pred_test).abs().idxmin()
        best_config = data.iloc[best_idx][PREDICTORS + [TARGET]].to_dict()  # Convert to dict

        # Extract scalar values for formula
        si = best_config[TARGET]
        c = best_config.get('steel_C_after', 0.0)
        mn = best_config.get('steel_Mn_after', 0.0)

        # Construct formula
        formula = (f"Optimal Steel Composition: Si={si:.2f}%, C={c:.2f}%, Mn={mn:.2f}%, "
                   "with other elements optimized based on data.")
        print("\nBest Configuration for Steel Quality:")
        print(formula)
        print("Full composition details:", best_config)
    except Exception as e:
        print(f"Error finding best configuration: {e}")


# Main execution
def main():
    """Main function to execute the steel composition analysis pipeline."""
    # Load and preprocess data
    data_numeric = load_and_preprocess_data(FILE_PATH)
    if data_numeric is None:
        return

    # Prepare data for modeling
    X = data_numeric[PREDICTORS]
    y = data_numeric[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    linear_model = train_linear_regression(X_train, y_train)
    y_pred = linear_model.predict(X)
    plot_predictions(y, y_pred, 'Actual Silicon Content', 'Predicted Silicon Content',
                     'Linear Regression Predictions')
    analyze_ols_model(X, y)

    # Correlation Matrix
    corr_matrix = data_numeric[PREDICTORS + [TARGET]].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

    # Decision Tree Regression
    tree_model = DecisionTreeRegressor(random_state=42)
    tree_model.fit(X_train, y_train)
    y_pred_test = tree_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    print("\nDecision Tree Regression Results")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    plot_predictions(y_test, y_pred_test, 'Actual Silicon Content', 'Predicted Silicon Content',
                     'Decision Tree Regression Predictions')

    # Find and display best configuration
    find_best_configuration(data_numeric, y_test, y_pred_test)


if __name__ == "__main__":
    main()
