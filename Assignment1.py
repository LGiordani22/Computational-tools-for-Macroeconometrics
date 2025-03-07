import pandas as pd

#First of all we load we import the dataset
file_path = "C:/Users/giord/OneDrive/Desktop/Computational tools for macroeconometrics/current (1).csv"

#We rename the file
df = pd.read_csv(file_path)


# We remove the first line with transformation codes 
df_cleaned = df.drop(index=0)
df_cleaned.reset_index(drop=True, inplace=True)

# Now we convert the sasdate column into datatime format 
df_cleaned['sasdate'] = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')

# With this comand we control the first rows of the dataframe 
print(df_cleaned.head())

print(df_cleaned.shape)  #We control the total number of rows and columns and we get 792,127 so the dataframe has been sucsessfully imported
print(df_cleaned.columns) #We control columns names
print(df_cleaned.dtypes) #With this comand we control if the columns have been imported in the right format: we get sasdate as "datetime64" so it means that is in datatime format and the other columns are "float64" so they are in numeric format

# Extract transformation codes
transformation_codes = df.iloc[0, 1:].to_frame().reset_index()
transformation_codes.columns = ['Series', 'Transformation_Code']

# df.iloc[0, 1:] → Selects the first row (index 0), excluding the first column (sasdate). This row contains the transformation codes.
# .to_frame() → Converts the extracted series into a DataFrame.
# .reset_index() → Turns the column names into a separate column called Series.
# transformation_codes.columns = ['Series', 'Transformation_Code'] → Renames the columns for better clarity.


#Now we have a DataFrame called transformation_codes with two columns:

#Series → The names of the economic variables.
#Transformation_Code → The transformation code that needs to be applied to each variable.

#To display this new dataframe we use the comand 
print(transformation_codes) #As we can see in this new dataframe we have 126 rows (because sasdate is exluded) and 2 columns

#Now let's analyse what are the transformnation codes

# Code 1: No transformation → The variable is already stationary, so no changes are needed.

# This means that the variable does not have a trend or changing variance over time.
# In this case, we do not need to transform the data.
#💡Example: Short-term interest rates, unemployment rates (often stationary in raw form).

# Code 2: First difference (ΔX_t = X_t - X_{t-1}) → Used for non-stationary series with a unit root, removing trends.

#Used for variables with a linear trend (random walk process).
#If a time series is non-stationary but becomes stationary after subtracting the previous value, we apply a first difference transformation.
#This removes linear trends and makes the series more stable.
#💡Example: GDP, stock prices, money supply.

# Code 3: Second difference (Δ²X_t = ΔX_t - ΔX_{t-1}) → Removes quadratic trends from the data.

#Used when first differencing is not enough (stronger trends).
#Some series have a quadratic trend (e.g., GDP growth), meaning that even after first differencing, they are still non-stationary.
#In this case, we apply a second difference.
#💡Example: GDP level, industrial production index.

# Code 4: Log transformation (log(X_t)) → Stabilizes variance, often used for financial or economic data with exponential growth.

#Used to stabilize variance (heteroskedasticity).
#If a time series has an increasing variance over time (e.g., stock prices, inflation), taking the log makes it more stable.
#This transformation is common in finance and macroeconomics.
#💡Example: Stock market indices, price indices (CPI, PPI), real GDP.

# Code 5: First difference of log (Δlog(X_t) = log(X_t) - log(X_{t-1})) → Equivalent to the percentage change of the variable.

#Used when a variable follows an exponential growth pattern.
#The first difference of a log-transformed series approximates the percentage change (growth rate).
#This is especially useful for variables like GDP, stock prices, and inflation rates.
#💡Example: GDP growth, inflation rate, productivity growth.

# Code 6: Second difference of log (Δ²log(X_t)) → Used for series with strong exponential trends.

#Used when the first difference of log is not enough to make a series stationary.
#If Δlog(X_t) is still non-stationary, we difference it again.
#This removes nonlinear trends and stabilizes highly volatile variables.
#💡Example: Inflation acceleration, productivity growth rate.

# Code 7: Approximate percentage change (Δ((X_t / X_{t-1}) - 1)) → A refined way to measure growth rates in certain economic indicators.

#Used as an alternative to the first difference of log.
#Instead of using Δlog(X_t), this transformation calculates the relative change in a variable.
#It is useful when we want an approximation of percentage changes but without logarithms.
#💡Example: Inflation rate, exchange rate fluctuations.


#Function to apply transformations based on the transformation code
#This function takes a time series (series) and a transformation code (code).
#It applies the appropriate transformation based on the code (from 1 to 7).
#If an invalid code is provided, it raises an error.

import numpy as np # ✅ Import NumPy

def apply_transformation(series, code):
    if code == 1:
        return series  # No transformation
    elif code == 2:
        return series.diff()  # First difference
    elif code == 3:
        return series.diff().diff()  # Second difference
    elif code == 4:
        return np.log(series)  # Log transformation
    elif code == 5:
        return np.log(series).diff()  # First difference of log
    elif code == 6:
        return np.log(series).diff().diff()  # Second difference of log
    elif code == 7:
        return series.pct_change()  # Approximate percentage change
    else:
        raise ValueError("Invalid transformation code")

#Applying the transformations to each column in df_cleaned based on transformation_codes, this function has diffent purposes
#1️⃣ Iterates over each series (column name) and its corresponding transformation code from the transformation_codes DataFrame.
#The loop for series_name, code in transformation_codes.values: goes through all economic variables and their assigned transformation codes.
#2️⃣ Converts the column values to float to avoid type errors.
#Some values might be read as strings, so astype(float) ensures the data is numeric before applying transformations.
#3️⃣ Applies the apply_transformation function to transform the data based on the specified code.
#If a variable has code 2, the first difference is computed.
#If it has code 5, the log difference is applied.
#The function ensures that each variable is transformed correctly.

for series_name, code in transformation_codes.values:
    df_cleaned[series_name] = apply_transformation(df_cleaned[series_name].astype(float), float(code))

#We now remove the first two rows 
#Many of the transformations introduce missing values (NaN) at the beginning of the time series.
#💡Examples:
#First difference (ΔX_t = X_t - X_{t-1})
#The first observation will be missing because there's no previous value to subtract.
#Second difference (Δ²X_t = ΔX_t - ΔX_{t-1})
#The first two observations will be missing.
#Log transformation + differences (log(X_t) - log(X_{t-1}))
#Also introduces NaN values in the first row(s).
#By removing the first two rows, we eliminate these missing values and ensure that the dataset starts with valid data.
df_cleaned = df_cleaned[2:]  

#After removing rows, the index will still reference the original row numbers (e.g., it might start from 2 instead of 0).
#reset_index(drop=True, inplace=True) does two things:
#drop=True → Prevents the old index from being added as a new column.
#inplace=True → Modifies the DataFrame directly instead of creating a copy.
df_cleaned.reset_index(drop=True, inplace=True)

#To check the transformed data 
df_cleaned.head() #As we can see by running this comand the 0th observastion starts from March 1959 and not anymore from January 1959
#with the above comand we had modified the file df_cleaned and we did not create a different copy


#1️⃣Now we want to create some graphs of some variables of our dataframe
import matplotlib.pyplot as plt #This is the main plotting library in Python.
import matplotlib.dates as mdates #Helps format the dates on the x-axis when dealing with time-series data.

#2️⃣We consider three series (INDPRO, CPIAUCSL, TB3MS) and assign them human-readable names (“Industrial Production”, “Inflation (CPI)”, “3-month Treasury Bill rate.”).
series_to_plot = ['INDPRO', 'CPIAUCSL', 'TB3MS'] #List of economic indicators we want to plot from the dataframe.
series_names = ['Industrial Production', 'Inflation (CPI)', '3-month Treasury Bill rate'] #We use this comand to give to the variables human-friendly names for each indicator (used in titles and legends).
# Note that the order of the lists matches so that each name corresponds to its variable.

#3️⃣We now create a figure with three (len(series_to_plot)) subplots arranged vertically. The figure size is 8x15 inches.
fig, axs = plt.subplots(len(series_to_plot), 1, figsize=(8, 15))
# plt.subplots(rows, columns, figsize=(width, height)) → Creates a figure with multiple subplots.
# len(series_to_plot) → Number of rows (since we have 3 time series, we create 3 subplots).
# 1 → We create a single-column layout (subplots stacked vertically).
# figsize=(8, 15) → Sets the size of the entire figure (8 inches wide, 15 inches tall).

# 🔹We check if axs is a list
axs = np.atleast_1d(axs)
# 🔹We control the list 
print(len(series_to_plot), len(series_names))  #They should have the same lenght

#We Run this long comand from point 4️⃣ to 1️⃣1️⃣
                     
                                                                             #4️⃣We check if the series exists in each series df_cleaned DataFrame columns.  
for ax, series_name, plot_title in zip(axs, series_to_plot, series_names):   # zip(axs, series_to_plot, series_names) → Loops over the subplots (ax), dataset columns (series_name), and their human-friendly names (plot_title).                                                                           
    if series_name in df_cleaned.columns:                                    # #if series_name in df_cleaned.columns: → Ensures we only plot series that exist in the dataset (avoids errors).
        
                                                                             #5️⃣We convert the sasdate column to datetime format (not necessary, since sasdate was converter earlier). 
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')     #This comand ensures that the sasdate column is recognized as a date (useful for proper plotting).
        
                                                                             #6️⃣We plot each series against the sasdate on the corresponding subplot, labeling the plot with its human-readable name. 
        ax.plot(dates, df_cleaned[series_name], label=plot_title)            #The general comand is ax.plot(x_values, y_values, label='Legend Name') Where: 
                                                                             # x_values = dates (time on the x-axis).
                                                                             # y_values = df_cleaned[series_name] (economic indicator on the y-axis).
                                                                             # label=plot_title → Uses the human-readable name for the legend.
        
                                                                             #7️⃣We now format the x-axis to display ticks and label the x-axis with dates taken every five years. 
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))               #This comand places major ticks (x-axis labels) every 5 years.    
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))             #This comand formats the x-axis ticks to show only the year (YYYY)

                                                                             #8️⃣Each subplot is titled with the name of the economic indicator.
        ax.set_title(plot_title)                                             #This comand adds a title to the subplot using the human-readable name.
        
                                                                             #9️⃣We label the x-axis “Year,” and the y-axis “Transformed Value,” to indicate that the data was transformed before plotting.
        ax.set_xlabel('Year')                                                #This comand labels the x-axis as "Year."
        ax.set_ylabel('Transformed Value')                                   #This comand labels the y-axis to indicate that the values were transformed before plotting.

                                                                             #🔟A legend is added to the upper left of each subplot for clarity.
        ax.legend(loc='upper left')                                          #This comand places the legend in the top-left corner of each subplot.

                                                                             #1️⃣1️⃣We rotate the x-axis labels by 45 degrees to prevent overlap and improve legibility.
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')    #This comand rotates the year labels 45 degrees to prevent overlap. (ha='right') aligns the labels to the right for better readability.

    else:
        ax.set_visible(False)                                                #This final comand hides plots for which the data is not availabl



#1️⃣2️⃣We use plt.tight_layout() automatically adjusts subplot parameters to give specified padding and avoid overlap.
plt.tight_layout()

#1️⃣3️⃣ We use plt.show() Displays the Final Figure
plt.show()


#Now we see another important usage of these data, in particular we focus on 

#🔴Forecasting in Time Series
#Forecasting in time series analysis involves using historical data to predict future values. 
#The objective is to model the conditional expectation of a time series based on past observations.

#🔴Direct Forecasts
#Direct forecasting involves modeling the target variable directly at the desired forecast horizon. 
#Unlike iterative approaches, which forecast one step ahead and then use those forecasts as inputs for subsequent steps, 
#direct forecasting directly models the relationship between past observations and future value.

#🔴ARX Models
#Autoregressive Moving with predictors (ARX) models are a class of univariate time series models that extend ARMA models by incorporating exogenous (independent) variables. 
#These models are formulated as follows:

#❕❕❕
#Compact form:
#Y_{t+h} = α + Σ (φ_i * Y_{t-i}) + Σ Σ (θ_{s,j} * X_{t-s,j}) + ε_{t+h}

#Expanded form:
#Y_{t+h} = α + φ_0 Y_t + φ_1 Y_{t-1} + φ_2 Y_{t-2} + ... + φ_p Y_{t-p} 
#         + θ_{0,1} X_{t,1} + θ_{1,1} X_{t-1,1} + ... + θ_{p,1} X_{t-p,1} 
#         + ... + θ_{0,k} X_{t,k} + ... + θ_{p,k} X_{t-p,k} + ε_{t+h}

#Where:
#1️⃣α: Constant term (intercept). A constant term representing the fixed component of the model.
#2️⃣ϕᵢ (Autoregressive coefficients): Measure the influence of past values of Yₜ on its future values.
#   ϕ₁ Yₜ₋₁ indicates that the value of Y in period t-1 affects Yₜ₊ₕ.
#   ϕ₂ Yₜ₋₂ represents the effect of Y from two periods ago.
#   This extends up to p past periods.



#3️⃣Y_{t-i}: Past values of the target variable (e.g., INDPRO, log-difference of industrial production).
#4️⃣Xₜ,ⱼ (Exogenous variables): External factors influencing Yₜ. (e.g., CPIAUSL for inflation, TB3MS for interest rates).
#   Each exogenous variable Xₜ,ⱼ has a set of coefficients θₛⱼ that determine its effect.
#   θ₀,1 Xₜ,1 represents the effect of X₁ at time t.
#   θ₁,1 Xₜ₋₁,1 represents the effect of X₁ with a lag of 1 period.
#   This applies to all k exogenous variables and all p lags.

#5️⃣θ_{s,j}: Coefficients for exogenous variables, capturing their lagged effects.
#6️⃣εₜ₊ₕ (Error term): Captures uncertainty and variations not explained by the model.
 
#🎯🎯🎯
#Example: Predicting industrial production (INDPRO) using inflation (CPIAUSL) and 3-month T-Bill rates (TB3MS).
#Target: INDPRO (log-difference, approximating month-to-month percentage change in industrial production).
#Predictors: CPIAUSL (inflation) and TB3MS (interest rates).
#Model captures how past industrial production, inflation, and interest rates influence future industrial production.

#⚠️⚠️⚠️
#The Use of Past Values is very useful because:
#Past values of the target variable Yₜ₋₁, Yₜ₋₂, ..., Yₜ₋ₚ help capture temporal dependencies. If ϕᵢ coefficients are significant, past values of Y help predict future values.
#Exogenous variables Xₜ,ⱼ include external factors affecting Yₜ. Using past values of X helps model delayed effects (e.g., economic policies may affect Y after some time).


#Data is assumed to range from time period t = 1 to t = T, where T is the last time period in the dataset.
#For example, in the df_cleaned dataset, T corresponds to December 2024.

#Forecasting with ARX
#The ARX model is used for forecasting, where we aim to predict the future value of a variable Y_t+h
#based on past values of Y_t and exogenous variables X_t. The model is:


#Expanded Formula for Forecasting:
# Y_(T+h) = α + φ₀ * Y_T + φ₁ * Y_(T-1) + ... + φp * Y_(T-p) + θ₀,1 * X_T,1 + θ₁,1 * X_(T-1),1 + ... + θp,1 * X_(T-p),1 + ... 
#
# Compact Formula for Forecasting:
# Y_(T+h) = α + Σ (φi * Y_(T-i)) + Σ (θs,j * X_(T-s,j)) for i=0 to p and s=0 to p, j=1 to k

#Matrix Formulation
#The ARX model can be written in matrix form for linear regression:
# y = X * β + u


# Where:
# y is the target vector (size T x 1), 
# X is the matrix of predictor variables (size T x (1 + p + k * p)), 
# β is the vector of coefficients (size (1 + p) * (1 + k)), 
# u is the error term (size T x 1).


# 📊 ARX Model Matrix Dimensions Explanation

# 🏗️ Explanation of Beta vector size β (Coefficient Matrix)
# β is the vector of coefficients used to estimate Y.
# It has a shape of ((1 + p + k * p), 1), where:
#🔹Each row corresponds to a coefficient multiplying a predictor in X.
#🔹The first coefficient is for the intercept.
#🔹The next p coefficients correspond to the lags of Y.
#🔹The remaining k * p coefficients correspond to the lags of the exogenous variables.

# 🎯 y (Target Variable)
# y is the dependent variable we aim to predict.
#🔹It has a shape of (T, 1), where:
#🔹T is the total number of time periods in the dataset.
#🔹It is a column vector with one value per time period.

# 📈 X (Predictor Matrix)
# X contains the predictors, including past values of Y and exogenous variables X.
# It has a shape of (T, 1 + p + k * p), where:
#🔹1: Represents the intercept (constant term α).
#🔹p: The number of lags of Y (Y_T, Y_(T-1), ..., Y_(T-p)).
#🔹k: The number of exogenous variables.
#🔹k * p: The number of lags for each exogenous variable (X_T, X_(T-1), ..., X_(T-p)).

# 🔢 Matrix Equation:
# The model follows the equation: y = X * β + u
# - y (T, 1) = 📈 X (T, 1 + p + k*p) * 🟩 β (1 + p + k*p, 1) + error (T, 1)

#🎯🎯🎯
#Simple example to understand the model 
#Assuming that Y represents industrial production(indpro) we want to use 3 past observations of indpro and two exogenous variables (X1 and X2), where:
#X1 represents inflation
#X2 represents the 3-month Treasury bill rate 
#Both exogenous variables also have 3 past observations.

# 1️⃣ Expanded Form of the ARX Model Equation
# The ARX model predicts Y_T+h based on past values of Y and exogenous variables X.
# Y_(T+h) = α + φ₁ * Y_(T-1) + φ₂ * Y_(T-2) + φ₃ * Y_(T-3) 
#           + θ₁,1 * X_(T-1),1 + θ₂,1 * X_(T-2),1 + θ₃,1 * X_(T-3),1 
#           + θ₁,2 * X_(T-1),2 + θ₂,2 * X_(T-2),2 + θ₃,2 * X_(T-3),2 + u_T


# 2️⃣ Matrix Formulation
# The model can be rewritten in matrix form:
# Y = X * β + u

# ✔️ Y matrix (dependent variable)
# Y is a column vector containing the historical values of Y (size 3x1).
# Y = [ Y_3 ]  
#     [ Y_4 ]
#     [ Y_5 ]

# ✔️ X matrix (independent variables)
# X includes a column of ones (intercept - the 1s are multiplied for α value in Beta matrix), past values of Y, and exogenous variables X1 and X2.
# X matrix (independent variables). Shape: (3,10)
# X includes:
#  - A column of ones (intercept)
#  - Three past values of Y
#  - Three past values of X1 (inflation)
#  - Three past values of X2 (TB3MTS)

# X= [1, Y_3, Y_2, Y_1, X_3_1, X_2_1, X_1_1, X_3_2, X_2_2, X_1_2],  # For observation 1
#    [1, Y_4, Y_3, Y_2, X_4_1, X_3_1, X_2_1, X_4_2, X_3_2, X_2_2],  # For observation 2
#    [1, Y_5, Y_4, Y_3, X_5_1, X_4_1, X_3_1, X_5_2, X_4_2, X_3_2]   # For observation 3

# ✔️ β vector (coefficients)
# This column vector contains coefficients for each term in X (size 10x1).
# β = [ α   ]   # Intercept
#     [ φ₁   ]   # Lag 1 of Y
#     [ φ₂   ]   # Lag 2 of Y
#     [ φ₃   ]   # Lag 3 of Y
#     [ θ₁,1 ]   # Lag 1 of X1
#     [ θ₂,1 ]   # Lag 2 of X1
#     [ θ₃,1 ]   # Lag 3 of X1
#     [ θ₁,2 ]   # Lag 1 of X2
#     [ θ₂,2 ]   # Lag 2 of X2
#     [ θ₃,2 ]   # Lag 3 of X2

# ✔️ Error term (u)
# This captures random variations (size 3x1).
# u = [ u_3 ]
#     [ u_4 ]
#     [ u_5 ]

#Let's see some future prediction 

Yraw = df_cleaned['INDPRO'] #Yraw contains the target variable 'INDPRO' (industrial production)
#Yraw contains 790 observations, one for every month starting from march 1959 (we removed the first two months)
Xraw = df_cleaned[['CPIAUCSL', 'TB3MS']] #Xraw contains two exogenous variables: CPIAUCSL (Inflation) and TB3MS (Treasury Bills)
#Xraw contains 790 observations x 2 columns, one for every month starting from march 1959

# Set the number of lags (p) and leads (h)
num_lags  = 4  ## p = 4 lags --- num_lags determines how many past values are used to predict the future value (in this case 4 past values).
num_leads = 1  ## h = 1, predicts the next period (T+1) 
#In time series forecasting, h represents the forecast horizon or the number of steps ahead we are predicting. 
#For example, if we are predicting the value of a variable for the next period (T+1), then h = 1.

# Initialize an empty DataFrame to store the lagged variables
X = pd.DataFrame()

#🔢The X matrix contains all the lagged variables (both the dependent and exogenous ones) and a column of ones to account for the intercept term in the model.
#In this code, X represents the predictor matrix that will be used to build the model for forecasting.
#It consists of:

# - Lagged values of Y (INDPRO): These are previous values of the target variable (INDPRO) shifted by different time periods, which are used to predict future values of Y.
# - Lagged values of exogenous variables (X1 = CPIAUCSL, X2 = TB3MS): These are lagged values of external variables (e.g., inflation, Treasury Bills rate) that could help explain variations in the target variable Y.
#In short, X contains the history of both Y and exogenous variables to predict the future values of Y.





# Loop through each lag (0 to 4) and add lagged values of the target variable 'INDPRO'
#This part of the code is looping through different lag values to create lagged features for the target variable INDPRO (industrial production):
# for lag in range(0, num_lags + 1):  #This loop iterates over lag values from 0 to num_lags (in this case, 4).
# X[f'{col}_lag{lag}'] = Yraw.shift(lag)  #For each lag, it shifts the values of the target variable (Yraw which is INDPRO) by lag positions. The shifted values are then stored as new columns in the X DataFrame, named as INDPRO_lag0, INDPRO_lag1, ..., up to INDPRO_lag4.
#This creates new features that are the past values of INDPRO, which can be used for forecasting.
col = 'INDPRO'
for lag in range(0, num_lags + 1):  # Loop from lag 0 to lag 4
    X[f'{col}_lag{lag}'] = Yraw.shift(lag)  # Shift the values of Y by the lag and add them as columns in X


# Loop through each exogenous variable (CPIAUCSL and TB3MS) and create lagged versions
# for col in Xraw.columns: This loop iterates over the columns of the exogenous variables Xraw (which are 'CPIAUCSL' and 'TB3MS').
# for lag in range(0, num_lags + 1): This nested loop runs for each lag value (from 0 to 4, as num_lags is set to 4).
# X[f'{col}_lag{lag}'] = Xraw[col].shift(lag): For each exogenous variable (e.g., CPIAUCSL), the values are shifted by the current lag and added as a new column in the X DataFrame, with a label indicating the variable and the lag (e.g., 'CPIAUCSL_lag0').
for col in Xraw.columns:  # Loop through 'CPIAUCSL' and 'TB3MS'
    for lag in range(0, num_lags + 1):  # For each lag from 0 to 4
        X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)  # Shift the exogenous variables by the lag and add them to X

# Add a column of ones (intercept term) to the DataFrame X for the constant alpha
X.insert(0, 'Ones', np.ones(len(X)))  # Insert a column of 1s in the first position for the intercept (α) in Beta matrix

# Display the first few rows of the DataFrame X to check the structure
X.head()  
#The resulting X matrix will have a number of rows equal to the number of observations 
#minus the number of lags (since lagging shifts data), 
#and the number of columns will include one column for the intercept, multiple columns for lagged values of INDPRO (target variable), 
#and lagged values for the exogenous variables like inflation (CPIAUCSL) and 3-month Treasury (TB3MS). 
#Each exogenous variable will have lagged columns corresponding to each past time point, 
#so the total number of columns will be --- 1 + (1 + num_lags (p)) * (1 + number of exogenous variables (k))= 1+(1+4)(1+2)=16


#Note that in X the first p=4 rows of the X matrix have missing values because we are creating lagged values of the variables (e.g., INDPRO, CPIAUCSL, and TB3MS). 
#For each lag (0 to 4), the data needs previous rows of values to compute the lagged value. Therefore, the first 4 rows cannot have any lagged values (since they do not exist yet), 
#so those rows are filled with NaN (missing values). Once we have enough data (after the first 4 rows), the matrix will be fully populated.
#As we can see in fact after the first lag we have value for INDPRO_lag0 but we still do not have data for INDPRO_lag1 
#we need to wait 4 periods to compute these lags
#The reason why INDPRO_lag0 has values while INDPRO_lag1 might not have values is due to how the shift() function works. 
#INDPRO_lag0 represents the current period’s value (i.e., no shift), so it directly copies the current value of INDPRO for each row.
#However, INDPRO_lag1 shifts the data by one period backward, so for the first row, there's no previous value for the lag, which results in a NaN (missing value) for that observation.

#Now we create the vector y
y = Yraw.shift(-num_leads)
y

#⚠️⚠️⚠️
#The y = Yraw.shift(-num_leads) code creates the target vector y by shifting the Yraw (the target variable) by a specified number of periods forward (num_leads). This allows us to align Yraw with future values, for example, predicting the future values of Yraw based on past data.
#The reason for missing values at the last h positions is that there is no data beyond the current period to shift forward (i.e., the forecast period). For example, if num_leads = 1, the last row of the dataset can't shift forward because it’s the last data point available. Therefore, the final values of y will be NaN.
#Lastly, the last row of X is kept to use for forecasting, as it contains all the necessary information to predict future values.
#In this context, the goal is not to predict the future directly but to evaluate how well the model can replicate the actual data of Y (the target variable) based on the past values of Y and the exogenous variables (such as inflation and treasury bills).
#By using past data (lags) to predict Y, we are essentially trying to verify if our model can approximate the real data we have for Y. 
#This process is known as model validation.

import numpy as np  # Import numpy for array manipulations

# Save the last row of X (used later for forecasting)
# .iloc[-1:] selects the last row of the DataFrame
# .values converts it into a NumPy array
X_T = X.iloc[-1:].values  

# Remove missing values from y
# .iloc[num_lags:-num_leads] selects rows from 'p' (num_lags) to the last h rows removed (num_leads)
# .values converts it into a NumPy array
y = y.iloc[num_lags:-num_leads].values  

# Remove missing values from X
# We apply the same slicing as we did for y to ensure they match in length
X = X.iloc[num_lags:-num_leads].values  

# Now:
# - X_T (1 row) will be used for forecasting
# - y is the cleaned target variable, without missing values
# - X is the cleaned matrix of predictors, matching the shape of y

X_T #As we can see we have 15 values 


#📈📈📈Why do we remove values from y?
#We remove the first num_lags rows of y because:
#The first p rows contain missing values due to lagging.
#We remove the last num_leads rows because shifting y forward by h creates missing values at the end.
#This ensures y aligns properly with X, so they have the same number of rows.

#📈📈📈Why do we take only the last row of X?
#The last row of X (X_T = X.iloc[-1:].values) contains the most recent known values.
#This is needed to predict the next time step (Y_T+h), as forecasting requires the latest available inputs.


#💡💡💡💡💡💡💡💡💡💡💡💡
#Estimation of Beta hat: let's start with some theory ⚠️
#Ordinary Least Squares (OLS) Estimation

#In this part of the script we derive the formula for estimating the coefficient vector (beta) using OLS.
#We start from the Residual Sum of Squares (RSS), express it in matrix form, differentiate, 
#and solve for beta.

# ➡️ Define RSS (Residual Sum of Squares)
#The Residual Sum of Squares (RSS) measures the difference between observed values (Y) 
#and the predicted values Yhat (Xβ):

#RSS = Σ (y_i - Yhat_i)^2
#    = Σ (y_i - (β_0 + β_1 * x_i1 + β_2 * x_i2 + ...))^2

#As we have seen in a linear regression model, the dependent variable Y is modeled as:
# ➡️ Y = Xβ + u

#where:
#    - Y is the vector of observed values 
#    - X is the matrix of independent variables 
#    - β is the vector of regression coefficients 
#    - u is the vector of residuals 

#Rearranging for the residuals:
#    u = Y - Xβ    #Where Y is the vector of yi that we find in the data 
                   #Where Xβ gives us the vector for the Yhats. The Yhats are the values that the model predicts
                   #The difference Y-Xβ gives us the difference between the real observation and the predicted one

#Since the Residual Sum of Squares (RSS) measures the sum of squared residuals:

#    RSS = Σ (y_i - Yhat_i)^2
#        = Σ (y_i - (β_0 + β_1 * x_i1 + β_2 * x_i2 + ...))^2

#Since u = Y - Xβ, we substitute:

#    RSS = Σ (ui)^2
#    RSS = Σ (Y_i - X_i β)^2  #Where (Y_i - X_i β) represents the residual for observation i (ui).
#    RSS = (Y-Xβ)^2   #We can write the RSS without the sum operator if we use vector and matrices.
                      #Remember that Y is the vector containing all yi (real observations in data)
                      #Remember that Xβ is the matrix containing all Yhats predicted by the model 
#🚨🚨🚨
#Let's specify that the form RSS = (Y-Xβ)^2 = (Y-Xβ)(Y-Xβ) does not have sense 
#This beacuse mathematically speaking we cannot multiply (Y - Xβ)(Y - Xβ)

#🚫 1. **Dimensions of the Vectors and Matrices:**

#   - Y is an n × 1 column vector (n rows, 1 column).
#   - Xβ is also an n × 1 column vector (obtained by multipling X matrix and Beta vector)   
#     Thus, both Y and Xβ are column vectors with the same dimensions, n × 1.

#🚫 2. **Matrix Multiplication Rules:**

#    When performing matrix multiplication, the **inner dimensions must match**. So, in order to multiply two matrices or vectors, the number of columns in the first matrix/vector must equal the number of rows in the second matrix/vector.

#    - (Y - Xβ) is a column vector of dimension n × 1.
#    - (Y - Xβ) is another column vector of the same dimension n × 1.

#     Since matrix multiplication requires the number of columns in the first vector to match the number of rows in the second vector, **two column vectors of dimension n × 1 cannot be multiplied directly**.

# 🏆 3. **The Role of Transposition:**

#To make the multiplication valid, we need to transpose one of the vectors to change it from a column vector (n × 1) into a row vector (1 × n).

#    So, we take the transpose of (Y - Xβ), which becomes (Y - Xβ)ᵀ, and now we have:
    
#    - (Y - Xβ)ᵀ, which is a 1 × n row vector.
#    - (Y - Xβ), which remains an n × 1 column vector.

#    Now the inner dimensions match: (1 × n) * (n × 1), and the result is a scalar value (1 × 1) which represents the sum of squared residuals (RSS).

#🎉🎉🎉
#Conclusion:
#To generalize RSS in matrix notation, we use the transpose operation:

#    RSS = (Y - Xβ)'(Y - Xβ)

#This expression represents the sum of squared residuals because:
#    - (Y - Xβ) is an (n×1) vector of residuals.
#    - (Y - Xβ)' is a (1×n) row vector (transposed version of (Y - Xβ)).
#    - The multiplication (Y - Xβ)'(Y - Xβ) results in a (1×1) scalar, 
#      which is the sum of squared residuals.
#In matrix form, we can write this as:
#RSS = Σ (y_i - Yhat_i)^2 = (Y - Xβ)ᵀ (Y - Xβ)

#✅ To minimize the RSS we 
#1 ➡️ Expand the expression:
#   RSS = (Y - Xβ)'(Y - Xβ)
#       = Y'Y - 2β'X'Y + β'X'Xβ

# 2 ➡️ To find the minimum RSS, we take the derivative with respect to β and set it to zero.
#Differentiating w.r.t. β:
#    d(RSS)/dβ = -2X'Y + 2X'Xβ
#Setting this to zero:
#    X'Xβ = X'Y
#Actually we would have needed also the second derivative of RSS to be positive in order to have a local minimum
#The first derivative equals zero is the condition also for a local maximum
#If we have a convex function the second derivative is always positive so we just need the first derivative to be zero in order to minimize the function
#The RSS is a convex function

#Solving for β we get: 
#    β = (X'X)^(-1) X'Y  🥇


#Now we can apply this things  above to predict our Beta hat. 
from numpy.linalg import solve
#Remember that 
#X is the design matrix (with predictors).
#Y is the target vector (the observed values, in this case, the industrial production values).
#X.T is the transpose of 𝑋
#(X.T X)^-1 is the inverse of (X.T X), which is used to find the best-fitting line.

# Solving for the OLS estimator beta: (X'X)^{-1} X'Y
beta_ols = solve(X.T @ X, X.T @ y) #Where: 
#X.T @ X calculates X.T X
#X.T @ y calculates X.T Y
#solve() computes (X.T X)^-1 (X.T Y) which gives Beta OLS (β = (X'X)^(-1) X'Y)


#One-Step Ahead Forecast: ## Produce the One step ahead forecast
# The forecast for future values is calculated by multiplying X_T with the beta_ols coefficients.
# This gives the predicted values of the target variable (INDPRO).
# Since we want the percentage change, we multiply by 100 to scale the result.

forecast = X_T @ beta_ols * 100

#Print the forecast

# The forecast will be an array of predicted values for INDPRO in percentage terms.
# Each value corresponds to a one-step ahead forecast based on the predictors in X_T.
print(forecast)

# This is the percentage change in INDPRO predicted for future periods based on the model.
## % change month-to-month INDPRO

#The variable forecast contains now the one-step ahead (forecast) of INDPRO. 
#Since INDPRO has been transformed in logarithmic differences, we are forecasting the percentage change (and multiplying by 100 gives the forecast in percentage points).
#To obtain the h-step ahead forecast, we must repeat all the above steps using a different h.

#📊📊📊
#Assessing Forecast's quality
#One thing we could do to assess the forecast’s quality is to wait for the new data on industrial production and see how big the forecasting error is. 
#However, this evaluation would not be appropriate because we need to evaluate the forecast as if it were repeatedly used to forecast future values of the target variables.
#To properly assess the model and its ability to forecast INDPRO, we must keep producing forecasts and calculating the errors as new data arrive. 
#This procedure would take time as we must wait for many months to have a series of errors that is large enough.

#A different approach is to do what is called a Real-time evaluation. 
#A Real-time evaluation procedure consists of putting ourselves in the shoes of a forecaster who has been using the forecasting model for a long time.

#In practice, that is what are the steps to follow to do a Real-time evaluation of the model:
#0️⃣Set T such that the last observation of df coincides with December 1999;
#1️⃣Estimate the model using the data up to T
#2️⃣Produce Ŷ_{T+1}, Ŷ_{T+2}, ..., Ŷ_{T+H}
#3️⃣Since we have the actual data for January, February, ..., we can calculate the
#forecasting errors of our model
# ê_{T+h} = Ŷ_{T+h} - Y_{T+h},   h = 1, ..., H.

#4️⃣Set T = T + 1 and do all the steps above.

#🔄🔄🔄
#The process results are a series of forecasting errors we can evaluate using several
#metrics. The most commonly used is the MSFE, which is defined as:
# MSFE_h = (1/J) * Σ (ê_{T+j+h}^2) for j=1 to J

#Where J is the number of errors we collected through our real-time evaluation.

# This assignment asks us to perform a real-time evaluation assessment of our simple
# forecasting model and calculate the MSFE for steps h = 1, 4, 8.

#Let's see an example of code to do a real time evaluation 
def calculate_forecast(df_cleaned, p=4, H=[1, 4, 8], end_date='12/1/1999', target='INDPRO', xvars=['CPIAUCSL', 'TB3MS']):
   
#Let's see the steps from 0️⃣ to 4️⃣


    # Step 0️⃣: Subset the dataset to only include observations up to the given `end_date` T
    rt_df = df_cleaned[df_cleaned['sasdate'] <= pd.Timestamp(end_date)]
#In this step 
#- We subset the dataset to include only data up to the given `end_date` (December 1999 in this case).
#- This simulates a real-time forecasting environment where we can only use past data.

    # Step 1️⃣: Extract actual values for the target variable at future time steps (T+h)
    Y_actual = []
    for h in H:
        future_date = pd.Timestamp(end_date) + pd.DateOffset(months=h)  # Identify the future date T+h
        # Get the actual value of the target variable at T+h and multiply by 100 for scaling
        Y_actual.append(df_cleaned[df_cleaned['sasdate'] == future_date][target] * 100)

    # Extract the target variable time series up to T
    Yraw = rt_df[target]

    # Extract independent variables (predictors) up to T
    Xraw = rt_df[xvars]
#In this step 
#- Since we are evaluating the forecast's accuracy, we extract the actual values of the target variable at time T+H (i.e., January 2000, February 2000, etc.).
#- These are stored in the list `Y_actual` and later used to compute forecasting errors.

    # Step 2️⃣: Create the feature matrix (X) including lags of the target and independent variables
    X = pd.DataFrame()

    # Add lagged values of the target variable (Y)
    for lag in range(p):
        X[f'{target}_lag{lag}'] = Yraw.shift(lag)  # Shift the series by 'lag' periods

    # Add lagged values of the independent variables
    for col in Xraw.columns:
        for lag in range(p):
            X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)  # Shift each predictor by 'lag' periods

    # Add a column of ones for the intercept in the regression model
    X.insert(0, 'Ones', np.ones(len(X)))

    # Extract the last available row of X, which corresponds to time T (used for forecasting)
    X_T = X.iloc[-1:].values  # Convert the row to a NumPy array

#- We construct a feature matrix X that includes:
#  - Lagged values of the target variable (to account for past trends).
#  - Lagged values of independent variables (e.g., inflation rate, interest rate).
#  - A column of ones for the intercept in the regression model.

    # Step 3️⃣: Train the forecasting model using data up to T and generate predictions
    Yhat = []  # List to store forecasts for different horizons (H)
    for h in H:
        # Shift the target variable back by h periods to align with predictors
        y_h = Yraw.shift(-h)

        # Select only rows where we have valid data for both X and y
        y = y_h.iloc[p:-h].values  # Dependent variable for OLS estimation
        X_ = X.iloc[p:-h].values   # Corresponding independent variables

        # Solve for the Ordinary Least Squares (OLS) estimator: (X'X)^(-1) X'Y
        beta_ols = np.linalg.solve(X_.T @ X_, X_.T @ y)

        # Compute the forecast for T+h (percentage change of INDPRO month-to-month)
        Yhat.append(X_T @ beta_ols * 100)
#In this step 
#- We use an Ordinary Least Squares (OLS) regression model:
#  - β = (X'X)^(-1)X'Y, which minimizes the sum of squared errors.
#- We compute forecasts for different horizons H = {1,4,8}

    # Step 4️⃣: Compute the forecasting errors
    # The error is calculated as: ê_{T+h} = Ŷ_{T+h} - Y_{T+h}
    return np.array(Y_actual) - np.array(Yhat)
#In this step
#- The error for each forecast is calculated as:
#  ê_{T+h} = Ŷ_{T+h} - Y_{T+h}
#- These errors are returned as the output of the function.

#💡💡💡We calculate now the RMSFE

#Running real-time evaluation over multiple time periods

t0 = pd.Timestamp('12/1/1999')  # Step 0: Set the initial training period to December 1999

# Initialize empty lists to store results
e = []  # List to store forecasting errors for different time steps
T = []  # List to keep track of time periods for each evaluation
Y_actuals = []  # List to store actual values of the target variable for comparison
Y_forecasts = []  # List to store predicted (forecasted) values

#Perform the forecasting process over 10 iterations (10 months)
for j in range(10):  
    t0 = t0 + pd.DateOffset(months=1)  # Move forward one month for real-time evaluation
    print(f'Using data up to {t0}')  # Display the current evaluation period
   
    # Call the forecasting function to generate predictions and errors
    ehat, Y_act, Y_hat = calculate_forecast(df_cleaned, p=4, H=[1, 4, 8], end_date=t0)
   
    # Store the results
    e.append(ehat)  # Append forecast errors for this time period
    Y_actuals.append(Y_act)  # Append actual observed values
    Y_forecasts.append(Y_hat)  # Append predicted values
    T.append(t0)  # Store the corresponding time period

# Convert the collected forecast errors into a Pandas DataFrame for analysis
edf = pd.DataFrame(e)

# Calculate the RMSFE (Root Mean Squared Forecasting Error)
# RMSFE is a common metric for evaluating forecast accuracy:
# It computes the square root of the mean squared error across all real-time evaluations.
rmsfe = np.sqrt(edf.apply(np.square).mean())  # Square errors, compute mean, then take square root
print("RMSFE:", rmsfe)  # Display the calculated RMSFE values

#Here we rewrite the previous code but with some modifications in order to asses, with some graphs, the goodness of the model 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_forecast(df_cleaned, p=4, H=[1, 4, 8], end_date='12/1/1999', target='INDPRO', xvars=['CPIAUCSL', 'TB3MS']):
    """
    This function calculates forecast errors, actual values, and predicted values for real-time evaluation.

    Parameters:
    - df_cleaned: Preprocessed DataFrame
    - p: Number of lags to include in the model
    - H: Forecast horizons
    - end_date: Last available date for forecasting
    - target: Target variable to predict
    - xvars: Independent variables

    Returns:
    - forecasting_errors: Forecast errors
    - actual_values: Actual observed values
    - predicted_values: Forecasted values
    """

    # 0️⃣ Subset the dataset up to the given `end_date`
    rt_df = df_cleaned[df_cleaned['sasdate'] <= pd.Timestamp(end_date)]

    # 1️⃣ Extract actual values for the target variable at T+h
    Y_actual = []
    for h in H:
        future_date = pd.Timestamp(end_date) + pd.DateOffset(months=h)
        Y_actual.append(df_cleaned[df_cleaned['sasdate'] == future_date][target] * 100)

    # Extract historical values of the target variable and independent variables
    Yraw = rt_df[target]
    Xraw = rt_df[xvars]

    # 2️⃣ Create the feature matrix (X) with lags
    X = pd.DataFrame()
    for lag in range(p):
        X[f'{target}_lag{lag}'] = Yraw.shift(lag)

    for col in Xraw.columns:
        for lag in range(p):
            X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)

    # Add a column of ones for the intercept
    X.insert(0, 'Ones', np.ones(len(X)))

    # Extract the last row of X for forecasting
    X_T = X.iloc[-1:].values

    # 3️⃣ Train the model and generate forecasts
    Yhat = []
    for h in H:
        y_h = Yraw.shift(-h)
        y = y_h.iloc[p:-h].values
        X_ = X.iloc[p:-h].values

        # Ordinary Least Squares (OLS) estimation
        beta_ols = np.linalg.solve(X_.T @ X_, X_.T @ y)

        # Compute the forecast for T+h
        Yhat.append(X_T @ beta_ols * 100)

    # 4️⃣ Compute forecasting errors
    forecast_errors = np.array(Y_actual) - np.array(Yhat)

    return forecast_errors, np.array(Y_actual), np.array(Yhat)  # 🔴 Now also returns actual & predicted values
                                                                # We only needed this modification but in order to add it we must run again all the previous code 
# ⚡ Real-time evaluation
t0 = pd.Timestamp('12/1/1999')

# Lists to store results
e = []
T = []
Y_actuals = []  # 🔴 Added list to store actual values
Y_forecasts = []  # 🔴 Added list to store predicted values

# Run the forecasting process for 10 months
for j in range(10):
    t0 = t0 + pd.DateOffset(months=1)
    print(f'Using data up to {t0}')
    
    # Call the forecasting function
    ehat, Y_act, Y_hat = calculate_forecast(df_cleaned, p=4, H=[1, 4, 8], end_date=t0)

    # Store results
    e.append(ehat.flatten())
    Y_actuals.append(Y_act.flatten())  # 🔴 Now storing actual values
    Y_forecasts.append(Y_hat.flatten())  # 🔴 Now storing predicted values
    T.append(t0)

# Convert stored results into DataFrames for analysis
errors_df = pd.DataFrame(e, columns=['H1', 'H4', 'H8'])  # 🔴 Named columns for clarity
actual_df = pd.DataFrame(Y_actuals, columns=['H1', 'H4', 'H8'])  # 🔴 Actual values DataFrame
predicted_df = pd.DataFrame(Y_forecasts, columns=['H1', 'H4', 'H8'])  # 🔴 Predicted values DataFrame

# Compute Root Mean Squared Forecast Error (RMSFE)
rmsfe = np.sqrt(errors_df.apply(np.square).mean())  # 🔴 Calculation unchanged, but errors_df is now clearer
print("\nRoot Mean Squared Forecast Error (RMSFE):")
print(rmsfe)

# 📈 Plot Actual vs Forecasted values for H=1
plt.figure(figsize=(10, 5))
plt.plot(T, actual_df['H1'], marker='o', linestyle='-', label="Actual (H=1)")
plt.plot(T, predicted_df['H1'], marker='s', linestyle='--', label="Forecasted (H=1)")

plt.xlabel("Time")
plt.ylabel("Industrial Production Index (scaled)")
plt.title("Real-Time Forecasting: Actual vs Predicted (H=1)")
plt.legend()
plt.grid()
plt.show()

# 📉 Plot Actual vs Forecasted values for H=4
plt.figure(figsize=(10, 5))
plt.plot(T, actual_df['H4'], marker='o', linestyle='-', label="Actual (H=4)")
plt.plot(T, predicted_df['H4'], marker='s', linestyle='--', label="Forecasted (H=4)")

plt.xlabel("Time")
plt.ylabel("Industrial Production Index (scaled)")
plt.title("Real-Time Forecasting: Actual vs Predicted (H=4)")
plt.legend()
plt.grid()
plt.show()

# 📊 Plot Actual vs Forecasted values for H=8
plt.figure(figsize=(10, 5))
plt.plot(T, actual_df['H8'], marker='o', linestyle='-', label="Actual (H=8)")
plt.plot(T, predicted_df['H8'], marker='s', linestyle='--', label="Forecasted (H=8)")

plt.xlabel("Time")
plt.ylabel("Industrial Production Index (scaled)")
plt.title("Real-Time Forecasting: Actual vs Predicted (H=8)")
plt.legend()
plt.grid()
plt.show()
