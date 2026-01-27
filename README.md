# Insurance-Claim-Prediction-Model
#  **Insurance Claim Prediction — Machine Learning Project**

This project focuses on predicting whether a building will file at least one insurance claim during its insured period based on its structural and contextual characteristics. The final model outputs claim probabilities that can assist insurers in risk identification and underwriting decisions.


##  **Project Objective**

Predict the probability that a building will file at least one claim (Claim = 1) versus no claim (Claim = 0), and evaluate which model provides the best performance for claim detection in the presence of class imbalance.



##  **Dataset Overview**

* Total Records: 7,160 buildings
* Target Variable: `Claim` (1 = claim occurred, 0 = no claim)
* Input Features: Building characteristics, settlement information, and insured period

The dataset required cleaning, preprocessing, and feature engineering before modeling.



## 🔧 **Steps Carried Out**

### **1. Data Loading & Initial Inspection**

* Loaded dataset and inspected structure, datatypes, and missing values
* Identified imbalance in target variable (~22.8% claims vs 77.2% non-claims)



### **2. Data Cleaning & Preprocessing**

* Handled missing values in both numerical and categorical columns
* Cleaned inconsistent entries (e.g., window count placeholders)
* Corrected column types and standardized categorical variables
* Dropped non-informative identifiers



### **3. Exploratory Data Analysis (EDA)**

* Explored distributions of key features
* Examined relationships between features and claims
* Computed claim rates across building types, settlement zones, and other attributes



### **4. Feature Engineering**

Engineered domain-informed features to improve predictive signal:

* `Geo_Risk` — average claim rate by geographical code
* `Type_Risk` — average claim rate by building type
* `Age_Dimension` — interaction between age and dimension
* `Window_Density` — windows relative to dimension

These new features improved correlation with the target and improved model performance.



### **5. Encoding & Scaling**

* Performed one-hot encoding for categorical variables
* Applied scaling to numerical features using StandardScaler
* Built preprocessing pipelines for cleaner workflow



### **6. Modeling & Evaluation**

Trained and compared multiple models:

* Logistic Regression
* Random Forest
* KNN
* Gradient Boosting
* XGBoost (baseline & tuned)
* SVM

Due to class imbalance, evaluation prioritized:

* Recall (Claim class)
* ROC-AUC
* F1-score

Accuracy was deprioritized as it is misleading for imbalanced data.



### **7. Threshold Adjustment**

Adjusted prediction threshold from default 0.50 to 0.35 to reduce false negatives (missed claims) which are costlier in insurance applications.



### **8. Hyperparameter Tuning**

Performed tuning on XGBoost using RandomizedSearchCV to optimize ROC-AUC:

* Achieved strong generalization with Test AUC ≈ 0.80
* Best Recall for claim class ≈ 0.86 (with tuned threshold)



### **9. Final Model Selection**

* Tuned XGBoost selected as final model based on Recall, AUC, and F1



##  **Results Summary**

* Best Model: **Tuned XGBoost**
* Test Recall (Claim Class): ~0.86
* ROC-AUC: ~0.80
* Model demonstrates strong ability to rank buildings by risk while minimizing missed claims.

---

##  **Business Relevance**

This model can support:

* Underwriting decisions
* Premium adjustments
* Risk segmentation
* Monitoring high-risk buildings

False positives carry operational cost; false negatives carry financial exposure. The model prioritizes reducing false negatives, which aligns with insurance risk preferences.


##  **Project Achievement**

* Completed end-to-end ML workflow
* Improved performance through feature engineering & tuning
* Delivered interpretable probability-based risk output
* Addressed imbalanced classification challenge effectively
