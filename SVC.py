from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
import numpy

seed = 7
numpy.random.seed(seed)
# load the iris datasets
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[: ,0:8]
Y = dataset[: ,8]
X4 = scale(X)
# create model
model = SVC(random_state=seed)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
param_grid = dict(C=[0.7,1],gamma=[0.006,0.004,0.005])
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=kfold)
grid_result = grid.fit(X4, Y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))