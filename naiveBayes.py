from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy

seed = 7
numpy.random.seed(seed)
# load the iris datasets
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[: ,0:8]
Y = dataset[: ,8]
# create model
model = GaussianNB()
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())