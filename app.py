# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.title("Advanced Credit Card Fraud Detection")

# Upload cleaned dataset
uploaded_file = st.file_uploader("Upload your cleaned credit card CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Summary")
    st.write(df.describe())
    st.write("Missing values:", df.isnull().sum())

    # Visualizations
    st.subheader("Data Visualizations")
    
    # Pie chart for Class distribution
    st.write("### Fraud vs Legit Transactions")
    class_counts = df['Class'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(class_counts, labels=['Legit', 'Fraud'], autopct='%1.1f%%', startangle=90, colors=['#4CAF50','#F44336'])
    ax1.axis('equal')
    st.pyplot(fig1)
    
    # Bar chart for numeric features
    st.write("### Feature Distribution")
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.drop('Class')
    selected_feature = st.selectbox("Select feature for bar chart", numeric_features)
    fig2, ax2 = plt.subplots()
    sns.histplot(df[selected_feature], kde=True, ax=ax2, color='skyblue', bins=30)
    ax2.set_title(f"Distribution of {selected_feature}")
    st.pyplot(fig2)

    # Correlation heatmap
    st.write("### Feature Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

    # Assume 'Class' is the target column
    if "Class" not in df.columns:
        st.error("Error: Dataset must have a 'Class' column (0 = Legit, 1 = Fraud)")
    else:
        feature_cols = df.columns.drop("Class")
        target_col = "Class"

        if st.button("Train Random Forest Model"):
            X = df[feature_cols]
            y = df[target_col]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrics
            st.subheader("Model Performance")
            st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
            st.text(classification_report(y_test, y_pred))

            st.subheader("Predict New Transaction")
            user_input = {}
            for feature in feature_cols:
                user_input[feature] = st.number_input(f"Enter {feature} value")

            if st.button("Predict Transaction"):
                input_df = pd.DataFrame([user_input])
                prediction = model.predict(input_df)
                if prediction[0] == 1:
                    st.error("⚠️ This transaction is Fraudulent!")
                else:
                    st.success("✅ This transaction is Legitimate.")
else:
    st.info("Please upload your cleaned credit card CSV file to proceed.")
