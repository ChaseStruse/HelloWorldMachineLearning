#Guide from : https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def printData(dataset):
    print('Dataset (instances, attributes) : ' + str(dataset.shape))

    print('Datasets actual data: ')
    print(dataset.head(20))

    print(' ')
    print(dataset.describe())

    print(' ')
    print(dataset.groupby('class').size())
    
def plotData(dataset):
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    pyplot.show()

    dataset.hist()
    pyplot.show()

    scatter_matrix(dataset)
    pyplot.show()
    
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

data = dataset.values
X = data[:,0:4]
y = data[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	#print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
 

model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print('YValidation: ', Y_validation)
print('Predictions: ', predictions)
print('Accuracy Score: ', accuracy_score(Y_validation, predictions))
print('Confusion Matrix: ')
print(confusion_matrix(Y_validation, predictions))
print('Classification Report: ')
print(classification_report(Y_validation, predictions))