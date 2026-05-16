# Synthetic Vietnam Banking Customer Churn Analysis

## Overview
- This dataset contains synthetic Vietnam retail banking customer records. 
- The data includes demographic, financial, credit, and behavioral information along with churn labels and risk insights. 
- It is designed for:
   - Churn prediction
   - Customer segmentation
   - Risk scoring
   - Behavioral analytics
   - Data science education and experimentation
   - All records are fully synthetic and generated programmatically using realistic financial behavior rules.

## Dataset Information

### Source & Size
- **Dataset**: [Kaggle Synthetic Banking Customer Dataset (Vietnam)](https://www.kaggle.com/datasets/tranhuunhan/vietnam-bank-churn-dataset-2025)
- **Records**: 80,000 customer records
- **Features**: 22 columns
- **Missing Values**: None

### Key Features Included
- **Demographic**: full _name,  gender, age,occupation, address, origin_province, married
- **Financial Records**: credit_sco, balance, monthly_ir, nums_card, nums_service
- **Customer Lifecycle**: tenure_ye, last_active_date, last_transaction_month, created_date
- **Loyalty**: active_member, engagement_score, loyalty_level, digital_behavior
- **Risk segment**: risk_score, risk_segment, cluster_group
- **Target**: Churn label (Exit column)

## Analysis Workflow

### 1. Data Loading
- Load dataset from `../data/vietnam_bank_churn_dataset.csv`
- Display dataset shape and memory usage
- Print head, info, and summary statistics
- Check null and duplicated values
- Identify data types: integer, float, object, and boolean

### 2. Data Cleaning & Preparation
- **Column Renaming**: Convert short column names to readable formats
  - `credit_sco` → `credit_score`
  - `occupation` → `profession`
  - etc.
- **Data Type Fixes**: Convert `last_active_date` and `created_date` to datetime format

- **Drop Columns**: 
  - id, name, address is id like unique columns
  - engagement score, loyalty level, risk score, and other calculated after this can create data leakage and affect model find patterns quickly


### 3. Feature Classification
- Separate numeric and categorical features
- Identify unique value counts per feature
- Translate profession column to readable values

### 4. Target Variable Analysis
- **Churn Imbalance**: ~82% customers are retained (did not exit)
- **18% Churn Rate**: Indicates class imbalance in dataset
- **Implication**: Use precision, recall, and F1-score metrics instead of accuracy alone


### 5. Data Quality Insights
- No missing values detected
- Dataset contains no null values - clean dataset
- Data types: integer, float, object, and boolean

### 6. Feature Distributions & Patterns

#### Numeric Features Insight
- **Balance**: Highly skewed distribution with extreme outliers
- **Monthly Income**: Highly skewed, some customers with very large income values
- **Tenure Years**: Relatively short customer relationships overall
- **Number of Cards**: Most customers hold 2–3 cards
- **Number of Services**: Customers typically use multiple services


### 7. Correlation Analysis
- Correlation matrix for numeric features
- Heatmap visualization showing feature relationships
- Identification of highly correlated features
- Feature importance relative to churn prediction

## Visualizations

- Pie chart of churn distribution (Target distribution)
- Distribution plots for numeric features
- Box plots for outlier detection
- Bar charts for categorical features
- Correlation heatmap for feature relationships
- Feature Analysis by target

## Key Findings

- **Imbalanced Target**: 82% retained vs. 18% churned customers
- **No Missing Data**: Dataset is complete with no null values
- **Right-Skewed Features**: Balance and income show extreme outliers
- **Calculated Features**: Some features (engagement score, customer segment) are pre-calculated and should be removed for realistic modeling
- **Short Relationships**: Average tenure is relatively short, indicating newer customer base

## Data Preprocessesing

- **Boolean Encoding**: Map exit and active member columns (True/False) for processing

### Feature Engineering
- Created `recency_date` for understanding how long this customer don't do any operation
- Created `account_age_days` for understanding how long this customer exist for bank
- And `created_month`, `created_weekday` for if there any relation with seasonality
- Finally divided groups for account age as `account_age_groups`
 - **Checked Correlation**: between new features and target
  - `recency`has more relation compared to others with `3.6` percent

### Feature Selection
**Assign X and y**
- Divide data 2 part first with target(y) other not(X)
- Also dropped not essential columns

**Split data train and test**
- 80% train, 20% test, with random state=42

**Look shape of train and test sets**
- Check if there any wrong

**Define numeric and categoric input**
- For encoding and scaling

**Assign encoder and scaler**
- Fit and transform for train data, just transform test data

## Model Training

## Model Selection

**Choosed 4 model for testing**
- Logistic Regression
- Random Forest
- XGBoost
- Neural Network

**Fitted models and checked resluts**
- Choosed Logistic Regression

Actually from our results we can see where our presicion is low than recall it means our model predict churners 80% but also gives us false alarms 

And where recall is low, presicion is high it means our model gives low false alarm but not find churners

For business case finding churners is more important thats why we will choose model where our recall is high

**BEST MODEL**: Logistic Regression

### Run app file
- Open terminal and run `streamlit run app.py`


### Consider
1. **Imports**: Load required libraries
2. **Load Data**: Read dataset from CSV
3. **Rename Columns**: Apply readable column names
4. **Data Type Fixes**: Convert data types and handle dates


## Data Considerations

- **Synthetic Data**: This is simulated data; patterns may not reflect real-world scenarios exactly
- **Feature Engineering Required**: Remove pre-calculated features (engagement score, customer segment) before model training
- **Class Imbalance**: Account for imbalance when training models (use class weights, stratified splits, etc.)
- **Outliers**: Balance and income features have extreme values handle with scaling


## Files Directory:
- Churn
  - data
    - `vietnam_bank_churn_dataset`
  - docs:
    - `data_dictionary.md`
    - `README.md`: This file
  - model:
    - `model.pkl`
    - `scaler.pkl` 
    - `threshold.pkl`
  - notebooks:
    - `dataprofiling.ipynb`: Additional data profiling analysis
    - `EDA.ipynb`: Main EDA notebook (this file)
  - reports:
    - `profiling.html`: profiling file, open in browser
  


## Notes
- All analyses and visualizations are embedded in the notebook cells
- Data source: `../data/_vietnam_bank_churn_dataset.csv`
- Used precision, recall, and F1-score for model evaluation due to class imbalance
- Considered stratified train-test split to maintain class distribution


### Libraries Used
- Make sure this libraries is installed
- If not installed write `pip install ___` (required library to space)in terminal and run
Essential libraries is in  `../docs/requirements.txt`
