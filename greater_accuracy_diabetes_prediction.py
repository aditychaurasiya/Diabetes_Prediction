import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv(r"diabetes (1).csv")
print(df.head())
print(df.shape)
print(df.groupby('Outcome').mean())

corr_matrix = df.corr()
plt.figure(figsize=(12, 8))

# Create a heatmap using seaborn
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

# Show the plot
plt.title('Correlation Heatmap')
plt.show()

independent_variables = df.drop(['Outcome'], axis=1)

# Calculate the correlation matrix
correlation_matrix = independent_variables.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Create a heatmap using seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

# Show the plot
plt.title('Correlation Heatmap for Independent Variables')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

x = independent_variables
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

scaler = MinMaxScaler()
x_train_normalized = scaler.fit_transform(x_train)
x_test_normalized = scaler.transform(x_test)

model = LogisticRegression()  # to avoid max iteration error
model.fit(x_train_scaled, y_train)
prediction = model.predict(x_test_scaled)
accuracy1 = accuracy_score(prediction, y_test)
print(f"{accuracy1 * 100} %")

model = LogisticRegression()  # to avoid max iteration error
model.fit(x_train_normalized, y_train)
prediction = model.predict(x_test_normalized)
accuracy2 = accuracy_score(prediction, y_test)
print(f"{accuracy2 * 100} %")

reg = LogisticRegression()
model.fit(x_train,y_train
          )
accuracy3 = accuracy_score(prediction, y_test)
print(f"{accuracy3 * 100} %")




from sklearn.metrics import classification_report

# Print classification report
print("Classification Report:")
print(classification_report(y_test, prediction))


"""OUTPUT
greater_accuracy_diabetes_prediction.py 
   Pregnancies  Glucose  BloodPressure  ...  DiabetesPedigreeFunction  Age  Outcome
0            6      148             72  ...                     0.627   50        1
1            1       85             66  ...                     0.351   31        0
2            8      183             64  ...                     0.672   32        1
3            1       89             66  ...                     0.167   21        0
4            0      137             40  ...                     2.288   33        1

[5 rows x 9 columns]
(768, 9)
         Pregnancies     Glucose  ...  DiabetesPedigreeFunction        Age
Outcome                           ...                                     
0           3.298000  109.980000  ...                  0.429734  31.190000
1           4.865672  141.257463  ...                  0.550500  37.067164

[2 rows x 8 columns]
80.51948051948052 %
81.81818181818183 %
81.81818181818183 %
Classification Report:
              precision    recall  f1-score   support

           0       0.83      0.91      0.87       102
           1       0.79      0.63      0.70        52

    accuracy                           0.82       154
   macro avg       0.81      0.77      0.79       154
weighted avg       0.82      0.82      0.81       154
"""
