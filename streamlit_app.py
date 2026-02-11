import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix, roc_curve
)

# Set page config
st.set_page_config(page_title="Machine Learning Assignment 2 - Phishing Website Detection", layout="wide")

# CSS

st.markdown("""
<style>

/* main app background */
.stApp {
    background-color: #0e1117;
}

/* main title */
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #00FFD1;
    text-align: center;
    margin-bottom: 10px;
}

/* section headers */
.section-title {
    font-size: 26px;
    font-weight: 600;
    color: #1f77ff;
    border-left: 6px solid #1f77ff;
    padding-left: 12px;
    margin-top: 25px;
    margin-bottom: 10px;
}

/* sub headers */
.sub-title {
    font-size: 20px;
    font-weight: 600;
    color: #F9A826;
    margin-top: 20px;
}

/* metric text */
.metric-box {
    font-size: 18px;
    font-weight: 500;
    color: white;
    background-color: #1c1f26;
    padding: 8px 12px;
    border-radius: 8px;
    margin-bottom: 5px;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Phishing Website Detection</div>', unsafe_allow_html=True)

st.markdown("""
<div class="section-title">About the Project</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background-color:#1c1f26; padding:15px; border-radius:10px; color:white; font-size:16px;'>

Hi, Welcome to my phishing website detection system! 

This application detects whether a website is 
<span style='color:#ff4b4b; font-weight:bold;'>phishing</span> 
or 
<span style='color:#00e676; font-weight:bold;'>legitimate</span> 
using machine learning models.

<span style='color:#ff4b4b; font-weight:bold;'>Phishing Websites</span> are fraudulent pages designed to steal sensitive data such as passwords, banking details, and personal information by imitating trusted websites.

The dataset used in this project contains structural and behavioral features of URLs and webpages such as:

• URL length and structure  
• Presence of HTTPS  
• Number of subdomains  
• Redirect behavior  
• Domain age  
• Web traffic indicators  
• DNS and security signals  

Each record represents one website and is labeled as:

<span style='color:#00e676; font-weight:bold;'>1 → Legitimate Website</span><br>
<span style='color:#ff4b4b; font-weight:bold;'>0 → Phishing Website</span>

Six machine learning models are trained and compared:

1. Logistic Regression
2. Decision Tree
3. KNN
4. Naive Bayes
5. Random Forest
6. XGBoost

Upload a test CSV file and select a model to view:

• Evaluation metrics  
• Performance visualization  
• Confusion matrix  
• ROC curve  

This interface demonstrates an end-to-end phishing detection pipeline including model training, evaluation, and deployment.

</div>
""", unsafe_allow_html=True)


# Download test data

st.markdown("### Download sample test data")
with open("test_data.csv", "rb") as f:
    st.download_button("Download Test CSV", f, "test_data.csv")

# Model mapping

model_files = {
    "Logistic Regression": "logistic.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

model_name = st.selectbox("Select Model", list(model_files.keys()), key="model")


# After data Upload

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    y = y.replace(-1, 0)

    # load selected model

    model = joblib.load(f"models/{model_files[model_name]}")
    preds = model.predict(X)

    # metrics
    
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    mcc = matthews_corrcoef(y, preds)

    try:
        probs = model.predict_proba(X)[:,1]
        auc = roc_auc_score(y, probs)
    except:
        probs = None
        auc = 0

    st.markdown('<div class="sub-title">Evaluation Metrics for Selected Model</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-box">Accuracy: {acc:.4f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-box">Precision: {prec:.4f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-box">Recall: {rec:.4f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-box">F1 Score: {f1:.4f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-box">MCC: {mcc:.4f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-box">AUC: {auc:.4f}</div>', unsafe_allow_html=True)



    # Performance Graph for Selected Model

    st.subheader("Performance Metrics Graph for Selected Model")

    metrics_dict = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "MCC": mcc,
        "AUC": auc
    }

    metrics_df = pd.DataFrame(
        list(metrics_dict.items()),
        columns=["Metric","Value"]
    )

    fig_perf, ax_perf = plt.subplots()

    bars = ax_perf.bar(metrics_df["Metric"], metrics_df["Value"])

    ax_perf.set_ylim(0, 1)
    ax_perf.set_ylabel("Score")
    ax_perf.set_title(f"{model_name} Performance")

    plt.xticks(rotation=30)

    st.pyplot(fig_perf)


    # Confusion Matrix
    
    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
    cm = confusion_matrix(y, preds)

    fig1, ax1 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    st.pyplot(fig1)

    
    # ROC Curve
    
    if probs is not None:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y, probs)

        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax2.plot([0, 1], [0, 1], '--')
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend()

        st.pyplot(fig2)
