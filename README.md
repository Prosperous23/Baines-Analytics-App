Transaction Analysis and Churn Prediction
An application designed for transaction analysis and customer churn prediction. This project integrates user authentication, data visualization, and machine learning to help financial institutions analyze transactional data and predict customer behavior.

Features
1. User Authentication
Login and Sign-Up:
Users can log in or create an account.
Passwords are securely managed and stored in a JSON file.
2. Data Upload and Analysis
File Upload:
Upload CSV files for analysis.
Data Overview:
View and summarize key metrics like total debit and credit.
Date Filtering:
Filter transactions within a specified date range.
Customer Insights:
Identify top customers by debit and credit.
Transaction Trends:
Analyze transaction trends over time using line charts.
Monthly Summaries:
View total monthly debit and credit transactions.
Channel Analysis:
Visualize transaction counts by channel.
3. Customer Churn Prediction
Feature Engineering:
Includes metrics like days since last transaction, total debit/credit, transaction count, and channel diversity.
Random Forest Classifier:
A pre-trained machine learning model predicts churn probabilities.
Customizable Model:
Replace the dataset or model file to adapt to your needs.
Requirements
To run this project, you need:

Python 3.7+
Required Python libraries:
streamlit
pandas
scikit-learn
seaborn
matplotlib
