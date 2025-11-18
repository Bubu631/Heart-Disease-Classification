Heart Disease Prediction: Handling Imbalanced Data ❤️
Project Overview
This project focuses on a critical medical diagnostic task: predicting heart disease (a binary classification problem). The core challenge is the severe Class Imbalance in the dataset, where healthy samples significantly outnumber positive cases. A standard model would achieve misleadingly high accuracy by simply ignoring the minority class.

Therefore, this project strictly prioritizes Recall (Sensitivity) over Accuracy to minimize False Negatives (missed diagnoses), which is the most critical factor in medical screening.

Key Objectives
The project structure was designed to systematically address the challenges of mixed data types and class imbalance:

Advanced Preprocessing: Implementing a specialized encoding strategy to preserve feature integrity.

Ordinality: Using OrdinalEncoder with manual ranking (e.g., Health ratings) to preserve the intrinsic logical order of features.

Nominality: Using OneHotEncoder(drop='first') to prevent the Dummy Variable Trap (multicollinearity) for unordered features (like Race).

Leakage Prevention: Strictly separating training and testing data during Scaling and Transformation parameters (StandardScaler) to prevent data leakage.

Handling Imbalance: Utilizing Stratified Splitting and Class Weights ('balanced' or scale_pos_weight) to force models to learn from the sparse minority class.

Hyperparameter Tuning: Employing GridSearchCV to optimize models explicitly for the Recall metric, ensuring the best possible sensitivity.

🛠️ Tech Stack & Methods
Python Libraries: Pandas, NumPy, Scikit-Learn, XGBoost, RandomForest

Data Preparation: OneHotEncoder, OrdinalEncoder, StandardScaler, StratifiedKFold

Modeling Strategy: Comparing Linear (Logistic Regression) and Non-Linear (XGBoost, Random Forest) classifiers.

Optimization: GridSearchCV with scoring='recall'.

📊 Workflow & Comparative Results
Encoding and Imbalance Strategy
The Data Encoding Strategy separated features into Binary, Ordinal, and Nominal types, applying the most appropriate encoding for each.

The Imbalance Strategy was implemented by applying class_weight='balanced' in Logistic Regression/Random Forest and scale_pos_weight=class\_ratio in XGBoost, effectively increasing the penalty for missing a positive case.
