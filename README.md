# EX-06 : Implementation of Decision Tree Classifier Model for Predicting Employee Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.Import the required libraries.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: J.JENISHA
RegisterNumber:  212222230056
```
```python
import pandas as pd
df=pd.read_csv("Employee.csv")

df.head()
df.info()
df.isnull().sum()
df["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])
df.head()

x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=df["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy : ",accuracy)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
#### Dataset
![Screenshot 2024-05-05 130130](https://github.com/Jenishajustin/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405070/b1401c21-428b-4905-be86-274cd8baab66)

#### Dataset info
<img src="https://github.com/Jenishajustin/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405070/3c6a69da-3564-4e49-8622-e32ad02e63b5" height=300 width=400>

#### Label Encoding for string values
<img src="https://github.com/Jenishajustin/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405070/543ada14-73a3-403b-b7cf-56fdca6741fd" height=200 width=800>
<br>
<img src="https://github.com/Jenishajustin/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405070/59912ecf-4d98-4ed0-8dde-e3ad080eca98" height=200 width=800>

#### Accuracy
![Screenshot 2024-05-05 130542](https://github.com/Jenishajustin/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405070/91148d16-82de-45fb-b61a-7b220b8bc116)

#### Prediction
![Screenshot 2024-05-05 130621](https://github.com/Jenishajustin/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405070/3f115c42-d4f6-4d0d-a8bc-3a379e8ef265)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
