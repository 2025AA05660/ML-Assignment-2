<h1>Phishing Website Detection using Machine Learning</h1>

<h2>Problem Statement</h2>

Phishing attacks are one of the most common cybersecurity threats, where fraudulent websites mimic legitimate platforms to steal sensitive user information such as passwords, banking credentials, and personal data.

The goal of this project is to build multiple machine learning classification models that can automatically detect whether a website is phishing or legitimate based on URL and website features.
An interactive Streamlit web application is developed to demonstrate model performance and allow real-time testing using uploaded data.


<h2>Dataset Description</h2>

Dataset: Phishing Websites Dataset (UCI)

Number of Instances: ~11,000

Number of Features: 30

Type: Binary Classification

Each row represents one website and includes features such as:

URL length, Presence of IP address, HTTPS usage, Number of subdomains, Redirect behavior, Domain age, DNS record, Web traffic, Security indicators

<i><b>Target Labels:</b></i>

1 → Legitimate website

0 → Phishing website

<h3>Machine Learning Models Implemented</h3>

All models are trained and evaluated on the same dataset.

1. Logistic Regression

2. Decision Tree Classifier

3. K-Nearest Neighbors

4. Naive Bayes (Gaussian)

5. Random Forest (Ensemble)

6. XGBoost (Ensemble)


<h4>Evaluation Metrics</h4>

Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- AUC Score

<h3>Model Performance Comparison</h3>

| Model               | Accuracy  | Precision | Recall  | F1 Score | MCC     | AUC    |
| ------------------- | --------- | --------- | ------- | -------- | ------  | ------ |
| Logistic Regression | 0.9433    | 0.9298    | 0.9695  | 0.9493   | 0.8862  | 0.9864 |
| Decision Tree       | 0.9667    | 0.9639    | 0.9756  | 0.9697   | 0.9327  | 0.9963 |
| KNN                 | 0.7433    | 0.7430    | 0.8110  | 0.7755   | 0.4797  | 0.8190 |
| Naive Bayes         | 0.9033    | 0.9355    | 0.8841  | 0.9091   | 0.8075  | 0.9730 |
| Random Forest       | 0.9567    | 0.9415    | 0.9817  | 0.9612   | 0.9132  | 0.9973 |
| XGBoost             | 0.9733    | 0.9643    | 0.9878  | 0.9759   | 0.9464  | 0.9984 |

<h3>Observations</h3>

| Model Name            | Model Performance Observations                                                                      |
| ------------------    | --------------------------------------------------------------------------------------- |
| Logistic Regression   | Performs well due to linear separability of several phishing indicators                 |
| Decision Tree         | Captures non-linear relationships but can overfit if depth is high                      |
| KNN                   | Gives stable performance with scaled features but slower prediction time                |
| Naive Bayes           | Fast and simple but assumes feature independence, reducing accuracy slightly            |
| Random Forest         | Improves performance by combining multiple trees and reducing overfitting               |
| XGBoost               | Provides the best performance due to boosting and handling complex feature interactions |


<h2>Conclusion</h2>
The ensemble methods, Random Forest and XGBoost delivered the strongest and most consistent performance across nearly all evaluation metrics.

Among them, XGBoost emerged as the best-performing model. It achieved the highest AUC and MCC values, indicating superior ability to distinguish between phishing and legitimate websites while maintaining balanced performance across both classes. 


XGBoost works well on this dataset because it builds trees sequentially and corrects previous errors using gradient boosting, allowing it to capture complex feature interactions such as URL structure patterns, security indicators, and domain-related signals.


The deployed Streamlit application demonstrates how these models can be integrated into a practical system for real-time phishing detection and performance comparison.
