from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l1_l2
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy

seed = 7
numpy.random.seed(seed)
# load the datasets
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[: ,0:8]
Y = dataset[: ,8]
X4 = scale(X)
def create_model(reg = l1_l2(l1=0.01, l2=0.01),opt='Nadam'):
	# create model
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=reg, input_dim=X.shape[1]))
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
# create model
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
epochs = [100]
batches = [5,10]
optimizers = ['RMSprop','Nadam','Adam','Adamax','Adagrad','Adadelta']

# evaluate using 10-fold cross validation
param_grid = dict(epochs=epochs, batch_size=batches, opt=optimizers)
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=kfold)
grid_result = grid.fit(X4, Y)


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))