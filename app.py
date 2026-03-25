
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

st.title("💄 Personalized Makeup Analytics Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx","csv"])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith("xlsx") else pd.read_csv(uploaded_file)
    st.write("Preview Data", df.head())

    # Encode categorical
    df_enc = df.copy()
    le_dict = {}
    for col in df_enc.columns:
        if df_enc[col].dtype == 'object':
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            le_dict[col] = le

    # Split
    if "Interested" in df_enc.columns:
        X = df_enc.drop("Interested", axis=1)
        y = df_enc["Interested"]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        # Classification
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.subheader("📊 Classification Metrics")
        st.write("Accuracy:", accuracy_score(y_test,y_pred))
        st.write("Precision:", precision_score(y_test,y_pred,average='weighted'))
        st.write("Recall:", recall_score(y_test,y_pred,average='weighted'))
        st.write("F1 Score:", f1_score(y_test,y_pred,average='weighted'))

        # ROC Curve
        y_prob = clf.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob[:,1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.legend()
        st.pyplot(fig)

        # Feature importance
        st.subheader("Feature Importance")
        importances = pd.Series(clf.feature_importances_, index=X.columns)
        st.bar_chart(importances.sort_values(ascending=False))

    # Regression
    if "Monthly_Spend" in df_enc.columns:
        X = df_enc.drop("Monthly_Spend", axis=1)
        y = df_enc["Monthly_Spend"]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        reg = RandomForestRegressor()
        reg.fit(X_train, y_train)

        st.subheader("💰 Regression Prediction Sample")
        st.write(reg.predict(X_test[:5]))

    # Clustering
    st.subheader("👥 Customer Segmentation")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_enc)
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(scaled)
    df["Cluster"] = clusters
    st.write(df["Cluster"].value_counts())

    # Association Rules
    st.subheader("🛒 Association Rules")
    binary_df = df_enc.applymap(lambda x: 1 if x>0 else 0)
    freq = apriori(binary_df, min_support=0.1, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.5)
    st.write(rules[['antecedents','consequents','support','confidence','lift']])
