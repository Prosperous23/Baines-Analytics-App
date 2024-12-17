import streamlit as st
import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib

# File to store user data
USER_DATA_FILE = "user_data.json"

# Function to load user data from a file
def load_users():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as file:
            return json.load(file)
    return {"user1": "password1", "user2": "password2"}  # Default users

# Function to save user data to a file
def save_users(users):
    with open(USER_DATA_FILE, "w") as file:
        json.dump(users, file)

# Load users into session state
if 'users' not in st.session_state:
    st.session_state.users = load_users()

# Check if user is authenticated
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

if 'page' not in st.session_state:
    st.session_state.page = "login"  # Default page

# Login Page Function
def login_page():
    st.title("Login Page")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.page = "main"  # Navigate to main page
        else:
            st.error("Invalid username or password")

    if st.button("Go to Sign Up"):
        st.session_state.page = "signup"  # Navigate to Sign-Up Page

# Sign-Up Page Function
def signup_page():
    st.title("Sign-Up Page")
    new_username = st.text_input("Choose a Username", key="signup_username")
    new_password = st.text_input("Choose a Password", type="password", key="signup_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")

    if st.button("Sign Up"):
        if new_username in st.session_state.users:
            st.error("Username already exists. Please choose a different one.")
        elif new_password != confirm_password:
            st.error("Passwords do not match. Please try again.")
        elif new_username.strip() == "" or new_password.strip() == "":
            st.error("Username and Password cannot be empty.")
        else:
            # Add user and save to file
            st.session_state.users[new_username] = new_password
            save_users(st.session_state.users)
            st.success("Account created successfully! You can now log in.")
            st.session_state.page = "login"  # Navigate to Login Page

    if st.button("Back to Login"):
        st.session_state.page = "login"  # Navigate back to Login Page
import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report


# Main Page Function (after login)
def main_page():
    # App Configuration
    st.set_page_config(
        page_title="E-Channel Transaction Analysis",
        page_icon="📊",
        layout="wide"
    )

    # Sidebar for Navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Use the options below:")
    # Select Options to Analyze/predict
    product_dataset = st.sidebar.selectbox("Kindly select Analyzer", ("E-channel Report", "Blinkcash Transaction"))

    uploaded_file = st.sidebar.file_uploader("Feed me my data", type=["csv"])
    # Add Logout Button to Sidebar
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.page = "login"  # Navigate to Login Page

    # Main Title
    st.title("📊Transaction Analysis and Churn Prediction")

    # Check if file is uploaded
    baines = pd.DataFrame()  # Initialize an empty DataFrame for baines
    if product_dataset == "E-channel Report":
        if uploaded_file:
            # Loading Data with Spinner
            with st.spinner("Loading data..."):
                baines = pd.read_csv(uploaded_file, skiprows=5, usecols=range(16))
                baines.rename(columns={
                    'Financial Date': 'Financial_Date',
                    'Transaction Date': 'Transaction_Date',
                    'Account Number': 'Account_Number',
                    'Account Name': 'Account_Name',
                    'Branch Name': 'Branch_Name',
                    'Posting Ref. No.': 'Posting_Ref_No.',
                    'Instrument Number': 'Instrument_Number',
                    'Entry Code': 'Entry_Code',
                    'Transaction Type': 'Transaction_Type',
                    'Trnx Channel': 'Trnx_Channel',
                }, inplace=True)

        # Check if baines is empty
        if baines.empty:
            st.warning("Please upload a valid CSV file.")

        else:
            # Data Overview
            st.markdown("### 📋 Data Overview")
            st.dataframe(baines.head())

            # Summary Statistics
            st.markdown("### 📈 Summary Statistics")
            col1, col2 = st.columns(2)
            with col1:
                total_debit = baines['Debit'].sum()
                st.metric(label="Total Debit", value=f"{total_debit:,.2f}")
            with col2:
                total_credit = baines['Credit'].sum()
                st.metric(label="Total Credit", value=f"{total_credit:,.2f}")
                # Top Customers
            #st.dataframe(baines.info)
            st.markdown("### 🏅 Top Customers by Debit and Credit")
            with st.expander("Top Debit Customers"):
                # Ensure Account_Number is treated as a string
                baines['Account_Number'] = baines['Account_Number'].astype(str)

                # Remove rows where Account_Number is less than 10 digits
                baines = baines[baines['Account_Number'].str.len() == 10]

                # Now perform the groupby operation
                top_debit_customers = baines.groupby(['Account_Name', 'Account_Number'])['Debit'].sum().sort_values(
                    ascending=False).head(10)

                # Display the result
                st.write("Top 10 Debit Customers:")

                st.dataframe(top_debit_customers)
            with st.expander("Top Credit Customers"):
                # Ensure Account_Number is treated as a string
                baines['Account_Number'] = baines['Account_Number'].astype(str)

                # Remove rows where Account_Number is less than 10 digits
                baines = baines[baines['Account_Number'].str.len() == 10]

                # Now perform the groupby operation
                top_credit_customers = baines.groupby(['Account_Name', 'Account_Number'])[
                    'Credit'].sum().sort_values(
                    ascending=False).head(10)

                # Display the result
                st.write("Top 10 credit Customers:")

                st.dataframe(top_credit_customers)

            # Ensure Transaction_Date is in datetime format
            baines['Transaction_Date'] = pd.to_datetime(baines['Transaction_Date'], errors='coerce')

            # Remove rows with invalid dates if any
            baines = baines.dropna(subset=['Transaction_Date'])

            # Filter out rows where Account_Number is not 10 digits
            baines = baines[baines['Account_Number'].apply(lambda x: len(str(x)) == 10)]

            # Calculate minimum and maximum dates
            min_date = baines['Transaction_Date'].min().date()
            max_date = baines['Transaction_Date'].max().date()

            st.markdown("#### Filter Transactions by Date Range")

            # Date range input with a unique key
            date_range = st.date_input(
                "Select a date range:",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date,
                key="unique_date_range"  # Unique key added
            )

            # Ensure the user selects a range (start and end dates)
            if len(date_range) == 2:
                start_date, end_date = date_range

                if start_date > end_date:
                    st.error("Error: Start date must be before end date.")
                else:
                    # Filter data based on the selected date range
                    filtered_data = baines[
                        (baines['Transaction_Date'].dt.date >= start_date) &
                        (baines['Transaction_Date'].dt.date <= end_date)
                        ]

                    # Group by Account_Name and Account_Number, count the transactions, and sum the debit and credit values
                    customer_stats = filtered_data.groupby(['Account_Name', 'Account_Number']).agg(
                        Transaction_Count=('Transaction_Date', 'size'),
                        Total_Debit=('Debit', 'sum'),
                        Total_Credit=('Credit', 'sum')
                    ).sort_values(by='Transaction_Count', ascending=False)

                    # Get top 10 customers with most transactions
                    top_10_customers = customer_stats.head(10)

                    # Display the table with the customer details
                    st.markdown("#### Top 10 Customers with Most Transactions:")
                    st.dataframe(top_10_customers.reset_index())

            else:
                st.warning("Please select both a start date and an end date.")

                # Ensure Transaction_Date is in datetime format
                baines['Transaction_Date'] = pd.to_datetime(baines['Transaction_Date'], errors='coerce')

                # Remove rows with invalid dates if any
                baines = baines.dropna(subset=['Transaction_Date'])

                # Calculate minimum and maximum dates
                min_date = baines['Transaction_Date'].min().date()
                max_date = baines['Transaction_Date'].max().date()

            st.markdown("#### Transaction Count by Date Range")

            # Date range input
            date_range = st.date_input(
                "Select a date range:",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date
                )

            # Ensure the user selects a range (start and end dates)
            if len(date_range) == 2:
                start_date, end_date = date_range

                if start_date > end_date:
                        st.error("Error: Start date must be before end date.")
                else:
                        # Filter data based on the selected date range
                    filtered_data = baines[
                        (baines['Transaction_Date'].dt.date >= start_date) &
                        (baines['Transaction_Date'].dt.date <= end_date)
                            ]

                    # Group by date and count transactions
                    transactions_per_day = filtered_data.groupby(filtered_data['Transaction_Date'].dt.date).size()

                    # Convert to DataFrame for better formatting
                    transactions_per_day_df = transactions_per_day.reset_index(name='Transaction Count')
                    transactions_per_day_df.rename(columns={'Transaction_Date': 'Date'}, inplace=True)

                    # Display the line chart

                    transactions_per_day_df['Date'] = pd.to_datetime(transactions_per_day_df['Date'])
                    transactions_per_day_df['Date'] = pd.to_datetime(transactions_per_day_df['Date'])

                    # Sort values by date (ensure correct order)
                    transactions_per_day_df = transactions_per_day_df.sort_values('Date')

                    # Streamlit app layout
                    st.title("Transaction Per Day Chart")

                    st.line_chart(data=transactions_per_day_df.set_index('Date')['Transaction Count'])
                    # Find the day with the highest and lowest transaction count
                    highest_day = transactions_per_day_df.loc[transactions_per_day_df['Transaction Count'].idxmax()]
                    lowest_day = transactions_per_day_df.loc[transactions_per_day_df['Transaction Count'].idxmin()]

                    # Display the highest and lowest transaction days
                    st.markdown(
                        f"###### (i). From the chart above the Day with Highest Transactions: {highest_day['Date']} - {highest_day['Transaction Count']} transactions")
                    st.markdown(
                            f"###### (ii). From the chart above the Day with Lowest Transactions: {lowest_day['Date']} - {lowest_day['Transaction Count']} transactions")

            else:
                st.warning("Please select both a start date and an end date.")

            # Monthly Totals
            st.markdown("### 🗓️ Monthly Debit and Credit Totals")
            baines['Transaction_Month'] = baines['Transaction_Date'].dt.to_period('M')
            monthly_totals = baines.groupby('Transaction_Month')[['Debit', 'Credit']].sum()
            st.bar_chart(monthly_totals)

            # Transaction Channels
            st.markdown("### 📡 Transactions by Channel")
            channel_counts = baines['Trnx_Channel'].value_counts()
            st.bar_chart(channel_counts)

            # Churn Prediction model, the data here should be saved into be saved in your data in github
            st.markdown("### 🔮 Customer Churn Prediction")

            def churn_prediction():
                # Define file path
                file_path = "E-Channel Transaction Report.csv"
    
                # Check if the file exists
                if os.path.exists(file_path):
                    # Load the file
                    bainess = pd.read_csv(file_path, skiprows=5, usecols=range(16))
                    st.write("File loaded successfully!")
                    return bainess
                else:
                    # Handle missing file
                    st.error(f"File not found: {file_path}. Please upload it.")
                    st.stop()
                # Rename columns for easier processing
                bainess.rename(columns={
                    'Financial Date': 'Financial_Date',
                    'Transaction Date': 'Transaction_Date',
                    'Account Number': 'Account_Number',
                    'Account Name': 'Account_Name',
                    'Branch Name': 'Branch_Name',
                    'Posting Ref. No.': 'Posting_Ref_No.',
                    'Instrument Number': 'Instrument_Number',
                    'Entry Code': 'Entry_Code',
                    'Transaction Type': 'Transaction_Type',
                    'Trnx Channel': 'Trnx_Channel',
                }, inplace=True)

                # Convert 'Transaction_Date' to datetime format
                bainess['Transaction_Date'] = pd.to_datetime(bainess['Transaction_Date'], errors='coerce')

                # Get the latest transaction date for feature engineering
                latest_date = bainess['Transaction_Date'].max()

                # Feature Engineering
                customer_features = bainess.groupby(['Account_Number', 'Account_Name']).agg({
                    'Debit': 'sum',
                    'Credit': 'sum',
                    'Transaction_Date': lambda x: (latest_date - x.max()).days,  # Days since last transaction
                    'Account_Number': 'count',  # Total number of transactions
                    'Trnx_Channel': lambda x: x.nunique(),  # Number of unique transaction channels used
                    'Transaction_Type': lambda x: x.nunique(),  # Number of different transaction types
                }).rename(columns={
                    'Debit': 'Total_Debit',
                    'Credit': 'Total_Credit',
                    'Transaction_Date': 'Last_Transaction_Days',
                    'Account_Number': 'Transaction_Count',
                    'Trnx_Channel': 'Unique_Channels',
                    'Transaction_Type': 'Unique_Transaction_Types'
                })

                # Refining Churn Definition: If no activity in 30 days, mark as churned
                customer_features['Churn'] = (customer_features['Last_Transaction_Days'] > 30).astype(int)

                # Feature Scaling: Standardizing the numerical features
                scaler = StandardScaler()
                numerical_features = ['Total_Debit', 'Total_Credit', 'Transaction_Count', 'Last_Transaction_Days',
                                      'Unique_Channels', 'Unique_Transaction_Types']
                customer_features[numerical_features] = scaler.fit_transform(customer_features[numerical_features])

                # Train-test Split
                X = customer_features.drop('Churn', axis=1)
                y = customer_features['Churn']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Model Training: Using RandomForestClassifier
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for AUC

                # Model Evaluation using Streamlit UI elements
                #st.markdown("### 🧪 Model Evaluation")
                #st.write("Accuracy:", accuracy_score(y_test, y_pred))
                #st.write("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))
                #st.write("Confusion Matrix:")
                #st.write(confusion_matrix(y_test, y_pred))
                #st.write("Classification Report:")
                #st.write(classification_report(y_test, y_pred))

                # Save Model Button (using Streamlit)
                if st.button("Save Model"):
                    joblib.dump(model, 'churn_model.pkl')
                    st.success("Model saved as churn_model.pkl")

                return model, customer_features

                # Usage: To be called within a Streamlit app
                # model, customer_features = churn_prediction()

            # Churn Prediction model
            #feature engineering


            latest_date = baines['Transaction_Date'].max()
            live_customer_features = baines.groupby(['Account_Number', 'Account_Name']).agg({
                    'Debit': 'sum',
                    'Credit': 'sum',
                    'Transaction_Date': lambda x: (latest_date - x.max()).days,
                    'Account_Number': 'count',
                    'Trnx_Channel': lambda x: x.nunique(),# Number of unique transaction channels used
                    'Transaction_Type': lambda x: x.nunique(),  # Number of different transaction types

                }).rename(columns={
                    'Debit': 'Total_Debit',
                    'Credit': 'Total_Credit',
                    'Transaction_Date': 'Last_Transaction_Days',
                    'Account_Number': 'Transaction_Count',
                    'Trnx_Channel': 'Unique_Channels',
                    'Transaction_Type': 'Unique_Transaction_Types'
                })

            live_customer_features['Churn'] = (live_customer_features['Last_Transaction_Days'] > 30).astype(int)  # Example: if no activity in 30 days, churned

            # Scaling and Model Training
            scaler = StandardScaler()
            numerical_features = ['Total_Debit', 'Total_Credit', 'Transaction_Count', 'Last_Transaction_Days', 'Unique_Channels', 'Unique_Transaction_Types']
            live_customer_features[numerical_features] = scaler.fit_transform(live_customer_features[numerical_features])
            # Drop the target column if present
            if 'Churn' in live_customer_features.columns:
                live_data = live_customer_features.drop(columns=['Churn'])
            else:
                live_data = live_customer_features
            # Load the saved model
            model = joblib.load('churn_model.pkl')
            # Use moddel for new live data
            predictions = model.predict(live_data)
            live_customer_features['Prediction'] = predictions
            live_customer_features['Churn_Status'] = live_customer_features['Prediction'].apply(
                lambda x: 'Churned' if x == 1 else 'Active')
             #import icad report
            icad = pd.read_csv("1012ICAD Report.csv", skiprows=2 ,usecols=range(21))
            #st.dataframe(icad.head())
            #needed column from the icad report
            neededicad = icad[['ACCOUNTNAME','ACCOUNTNUMBER','EMAIL','PHONENUMBER']]
            #st.dataframe(neededicad.info)
            # Data Overview

            # Predict on New Data

            if st.button("Click to Predict"):


                # Filter and Display Churned Customers
                churned_customers = live_customer_features[live_customer_features['Churn_Status'] == 'Churned']


                churned_customer = pd.merge(churned_customers,neededicad,left_on='Account_Number',right_on='ACCOUNTNUMBER',how='inner')
                st.dataframe(churned_customers)
                #st.write(churned_customers.count())
                st.dataframe(churned_customer)
                #st.write( churned_customer.count())

                # st.write("Prediction for Sample Data:", "Churned" if prediction[0] == 1 else "Active")

        # Filtering and Analysis Code (Top customers, Date ranges, etc.)...

            # Churn Prediction Model (if implemented correctly)

    elif product_dataset == "Blinkcash Transaction":
        st.write("Blinkcash Transaction")


# Navigation Logic
if st.session_state.authenticated:
    st.session_state.page = "main"

if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "signup":
    signup_page()
else:
    login_page()

