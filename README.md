Risk-Assessment-Using-Alternative-Data

# Aim:
This system aims to revolutionize credit risk assessment by integrating traditional data sources, improving accuracy, and expanding access to credit for underserved populations.

# ALGORITHM:
We used a Random Forest Classifier to predict credit risk based on various features. The model combines multiple decision trees to improve accuracy and reduce overfitting. Data preprocessing included handling missing values, encoding categorical features, and standardizing numerical data. The model's performance was evaluated using classification metrics, and SHAP was employed to visualize feature importance for model interpretability. This approach provides both accurate predictions and insights into the key factors influencing decisions.

code :

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap

# Load the dataset
data = pd.read_csv('german_credit_data.csv')

# Display basic info about the dataset
print("Dataset Info:")
data.info()


# Preprocess the data
# Handle missing values
data = data.dropna()


# Encode categorical features
categorical_columns = data.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])


X = data.drop('Checking account', axis=1)  # Assuming 'CreditRisk' is the target column
y = data['Checking account']


print(data.columns)

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Explainability using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

```



Output:

![image](https://github.com/user-attachments/assets/9f547aa0-7a07-48ad-a8e8-737ef8475df2)








