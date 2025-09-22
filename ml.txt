prg1
import csv
def loaddata(filename):
    with open(filename,'r') as f:
        reader=csv.reader(f)
        data=list(reader)
        header=data[0]
        instances=data[1:]
    return header,instances
def finds(data):
    for instance in data:
        if instance[-1].lower()=="yes":
            hypo=instance[:-1]
            break
    else:
        None
    for instance in data:
        if instance[-1].lower()=="yes":
            for i in range(len(hypo)):
                if hypo[i]!=instance[i]:
                    hypo[i]='?'
    return hypo
filename='training.csv'
header,data=loaddata(filename)
print('attributes',header)
print(' ')
print('training data')
for row in data:
    print(row)
hypo=finds(data)
if hypo:
    print('most specific hypothesis by find-s')
    print(hypo)
else:
    print('no positive training egs in training data')

------------------------------------------------------------------------------------------------------------------------------------------
prg2
import pandas as pd
import numpy as np
def loaddata(filename):
    data=pd.read_csv(filename)
    print(data)
    concepts=data.iloc[:,:-1].values
    target=data.iloc[:,-1].values
    return concepts,target
def candidate(concepts,target):
    nf=len(concepts[0])
    s=concepts[0].copy()
    g=[["?" for _ in range(nf)]for _ in range(nf)]
    for i,eg in enumerate(concepts):
        if target[i].lower()=="yes":
            for x in range(nf):
                if s[x]!=eg[x]:
                    s[x]="?"
                    g[x][x]="?"
        else:
            for x in range(nf):
                if s[x]!=eg[x]:
                    g[x][x]=s[x]
                else:
                    g[x][x]="?"
    g=[h for h in g if any(attr!="?" for attr in h)]
    return s,g
filename='training1.csv'
concepts,target=loaddata(filename)
s,g=candidate(concepts,target)
print("most specific hypothesis using candidate")
print(s)
print("most general hypothesis using candidate")
print(g)

------------------------------------------------------------------------------------------------------------------------------------------
prg3
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris=load_iris()
x=iris.data
y=iris.target
print(iris)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy:2f}")

new_sample=[[5.1,2.5,4.6,1.5]]
predictclassindex=clf.predict(new_sample)[0]
predictclassname=iris.target_names[predictclassindex]
print(f"predicted class for the new sample {new_sample} is:{predictclassname}")

------------------------------------------------------------------------------------------------------------------------------------------
prg4
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
iris = load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu',
max_iter=1000, random_state=1)
model.fit(x_train, y_train)
y_pred = model.predict (x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print (classification_report(y_test, y_pred,
target_names=iris.target_names))
------------------------------------------------------------------------------------------------------------------------------------------
prg5
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,precision_score,recall_score
data=pd.read_csv('data.csv')
print(data)
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x=x.copy()
leoutlook=LabelEncoder()
x.outlook=leoutlook.fit_transform(x.outlook)
letem=LabelEncoder()
x.tem=letem.fit_transform(x.tem)
print("\nnow the training output is\n ",x)
lept=LabelEncoder()
y=lept.fit_transform(y)
print("\nnow the train output is\n",y)
from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.20)
classifier=GaussianNB()
classifier.fit(xtr,ytr)
print("accuracy is:",accuracy_score(classifier.predict(xte),yte))
print("recall :",recall_score(classifier.predict(xte),yte))
print("precision:",precision_score(classifier.predict(xte),yte))

------------------------------------------------------------------------------------------------------------------------------------------
prg6
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_score,recall_score
data=pd.read_csv('document.csv')
texts=data['text'].values
labels=data['label'].values
vectorizer=CountVectorizer()
x=vectorizer.fit_transform(texts)
xtr,xte,ytr,yte=train_test_split(x,labels,test_size=0.3,random_state=42)
model=MultinomialNB()
model.fit(xtr,ytr)
ypred=model.predict(xte)
accuracy=accuracy_score(yte,ypred)
precision=precision_score(yte,ypred,pos_label='positive')
recall=recall_score(yte,ypred,pos_label='positive')
print("test results:")
for text,true_label,pred_label in zip(vectorizer.inverse_transform(xte),yte,ypred):
    print(f"text:{ ' '.join(text) } | true: { true_label } | predicted: { pred_label }")
print("\nmetrics:")
print(f"accuracy: {accuracy:.2f}")
print(f"precision: {precision:.2f}")
print(f"recall: {recall:.2f}")

------------------------------------------------------------------------------------------------------------------------------------------

prg7
import seaborn as sns
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
iris=sns.load_dataset('iris')
s=iris[iris['species']=='setosa']
v=iris[iris['species']=='versicolor']
s_length=s['petal_length']
v_length=v['petal_length']
t_stat,p_value=stats.ttest_ind(s_length,v_length)
alpha=0.05
if p_value<alpha:
 print("Reject the null hypothesis")
 print("There is a significant difference between the
petal length of Iris setosa and Iris versicolor.")
else:
 print("Fail to reject the null hypothesis")
 print("There is no significant difference between the
petal length of Iris setosa and Iris versicolor.")
model=ols('sepal_length ~ C(species)',data=iris).fit()
anova_table=sm.stats.anova_lm(model,typ=2)
print("\nOne-Way ANOVA Results(statsmodels):")
print(anova_table)
p_value=anova_table['PR(>F)'].iloc[0]
alpha=0.05
if p_value<alpha:
 print("Reject the null hypothesis: There is a significant
difference between sepal length among the species.")
else:
 print("Fail to reject the null hypothesis: No significant
difference between sepal length among the species.")
------------------------------------------------------------------------------------------------------------------------------------------
prg8
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data = load_iris()
x=data.data
y=data.target
y= pd.get_dummies (y).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=20,
random_state=4)
learning_rate = 0.1
iterations = 5000
N = y_train.size
input_size = 4
hidden_size=2
output_size =3
results = pd.DataFrame ({"accuracy":[0]})
np.random.seed(10)
W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))
W2 = np.random.normal (scale=0.5, size=(hidden_size ,
output_size))
def sigmoid(x):
 return 1 /(1 + np.exp(-x))
def accuracy(y_pred, y_true):
 acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
 return acc.mean()
for itr in range(iterations):
 Z1 = np.dot (x_train, W1)
 A1 = sigmoid(Z1)
 Z2 = np.dot (A1, W2)
 A2 = sigmoid(Z2)
 acc = accuracy (A2, y_train)
 new_row = pd.DataFrame({"accuracy":[acc]})
 results = pd.concat ([results, new_row], ignore_index=True)
 E1 = A2 - y_train
 dW1 = E1 * A2 *(1 - A2)
 E2 = np.dot (dW1, W2.T)
 dW2= E2 * A1 *(1 - A1)
 W2_update = np.dot(A1.T, dW1)/N
 W1_update = np.dot (x_train.T, dW2) / N
 W2 = W2 - learning_rate * W2_update
 W1 =W1 - learning_rate * W1_update
Z1 = np.dot (x_test, W1)
A1 = sigmoid(Z1)
Z2 = np.dot (A1, W2)
A2 = sigmoid(Z2)
acc = accuracy (A2, y_test)
print("Accuracy:{}".format(acc))
results.accuracy.plot (title="Accuracy")
plt.show()
------------------------------------------------------------------------------------------------------------------------------------------
prg9
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import datasets
iris=datasets.load_iris()
print("iris dataset loaded..")
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=0)
for i in range(len(iris.target_names)):
    print("label",i,"_",str(iris.target_names[i]))
classifier=KNeighborsClassifier(n_neighbors=2)
classifier.fit(x_train,y_train)
ypred=classifier.predict(x_test)
print("results of classification using k-nn with k=2")
for r in range(0,len(x_test)):
    print("sample:",str(x_test[r]),"actual label:",str(y_test[r]),"predicted label:",str(ypred[r]))
print("/nclassification accuracy:",classifier.score(x_test,y_test))
print("/nconfusion matrix:",metrics.confusion_matrix(y_test,ypred))
------------------------------------------------------------------------------------------------------------------------------------------
prg10
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
diabetes=load_diabetes()
x=diabetes.data
y=diabetes.target
x_single=x[:,2].reshape(-1,1)
x_train,x_test,y_train,y_test=train_test_split(x_single,y,t
est_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
print(f"Mean Squared Error on test set:{mse:.2f}")
print(f"Intercept:{model.intercept_:.2f}")
print(f"Cofficient:{model.coef_[0]:.2f}")
plt.scatter(x_test,y_test,alpha=0.6,label='TestData')
plt.plot(x_test,y_pred,color='red',label='Predicted
regression line')
plt.xlabel('BMI feature')
plt.ylabel('Disease Progression')
plt.title('Linear Regression on diabetes dataset')
plt.legend()
plt.show()
new_BMI_values=np.array([[0.05],[0.10],[-0.02]])
new_predictions=model.predict(new_BMI_values)
for bmi,pred in
zip(new_BMI_values.flatten(),new_predictions):
 print(f"Predicted disease progression for
BMI={bmi:.2f}:{pred:.2f}")

