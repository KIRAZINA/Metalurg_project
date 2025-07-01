import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_excel(file_path, header=3, usecols='B:CN')
    data.columns = [
        'sample_number',
        'steel_C_before', 'steel_Mn_before', 'steel_Si_before', 'steel_S_before',
        'steel_P_before', 'steel_Cr_before', 'steel_Ni_before', 'steel_Cu_before',
        'steel_Al_before', 'steel_N_before', 'steel_Ti_before', 'steel_Zr_before',
        'steel_V_before', 'steel_Mo_before', 'steel_Nb_before', 'steel_Sn_before',
        'steel_As_before', 'steel_Ca_before',
        'slag_CaO_before', 'slag_SiO2_before', 'slag_MgO_before', 'slag_FeO_before',
        'slag_Al2O3_before', 'slag_P2O5_before', 'slag_Fe2O3_before', 'slag_MnO_before',
        'slag_S_before', 'slag_basicity_before',
        'add_FeCa_before', 'add_FeSi65_before', 'add_FeMn_before', 'add_FeSiMn_before',
        'add_lime_before', 'add_smelting_before', 'add_AlP_before', 'add_SiCa_before',
        'add_Al12_before', 'add_Al13_before', 'add_FeB_before', 'add_feldspar_before',
        'add_FeTi_before', 'add_AlGr_before',
        'sulfur_reduction_ratio',
        'steel_C_after', 'steel_Mn_after', 'steel_Si_after', 'steel_S_after',
        'steel_P_after', 'steel_Cr_after', 'steel_Ni_after', 'steel_Cu_after',
        'steel_Al_after', 'steel_N_after', 'steel_Ti_after', 'steel_Zr_after',
        'steel_V_after', 'steel_Mo_after', 'steel_Nb_after', 'steel_Sn_after',
        'steel_As_after', 'steel_Ca_after',
        'slag_CaO_after', 'slag_SiO2_after', 'slag_MgO_after', 'slag_FeO_after',
        'slag_Al2O3_after', 'slag_P2O5_after', 'slag_Fe2O3_after', 'slag_MnO_after',
        'slag_S_after', 'slag_basicity_after',
        'total_weight', 'cutting_total', 'heat_number', 'upk_number',
        'processing_time', 'bottom_blowing_time', 'argon_flow_total',
        'blow_flow1', 'blow_flow2', 'slag_height_incoming', 'slag_height_outgoing',
        'slag_color_incoming', 'slag_color_outgoing', 'metal_temp_first',
        'metal_temp_last', 'heat_temp_first', 'heat_temp_last', 'heating_duration',
        'energy_consumption'
    ]
    data_numeric = data.apply(pd.to_numeric, errors='coerce')
    data_numeric = data_numeric.dropna(axis=1, thresh=len(data_numeric) * 0.5)
    data_numeric = data_numeric.fillna(data_numeric.mean())
    return data_numeric

# Function to train linear regression model
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to plot predictions
def plot_predictions(y_true, y_pred, xlabel, ylabel, title):
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_true, y=y_pred, line_kws={"color": "red"})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# Function to analyze OLS model
def analyze_ols_model(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())

# Main code
file_path = 'source_data.xls'
data_numeric = load_and_preprocess_data(file_path)

target_after = 'steel_Si_after'
predictors = [
    'steel_C_after', 'steel_Mn_after', 'steel_Si_after', 'steel_P_after',
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
]

X = data_numeric[predictors]
y = data_numeric[target_after]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
linear_model = train_linear_regression(X_train, y_train)
y_pred = linear_model.predict(X)

# Plot predicted values
plot_predictions(y, y_pred, 'Actual Silicon Content in Steel after UPK Processing',
                 'Predicted Silicon Content in Steel after UPK Processing',
                 'Linear Regression Predictions of Silicon Content in Steel after UPK Processing')

# Analyze OLS model
analyze_ols_model(X, y)

# Plot correlation matrix
corr_matrix = data_numeric[predictors + [target_after]].corr()
plt.figure(figsize=(20, 15))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of All Variables')
plt.show()

# Train decision tree model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_test = tree_model.predict(X_test)

# Evaluate decision tree model
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print("\nDecision Tree Regression Predictions of Silicon Content in Steel after UPK Processing")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot decision tree model predictions
plot_predictions(y_test, y_pred_test, 'Actual Silicon Content in Steel after UPK Processing',
                 'Predicted Silicon Content in Steel after UPK Processing',
                 'Decision Tree Regression Predictions of Silicon Content in Steel after UPK Processing')
