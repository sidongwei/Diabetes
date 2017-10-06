# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from keras.regularizers import l1_l2
import numpy

# Function to create model, required for KerasClassifier
def create_model(neuron1=15, neuron2=4, opt='Adam',act='relu',reg = l1_l2(l1=0.01, l2=0.01)):
    # create model
    model = Sequential()
    model.add(Dense(neuron1, input_dim=8, kernel_regularizer=reg, activation=act))
    model.add(Dense(neuron2, activation=act))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[: ,0:8]
X3 = scale(X)
Y = dataset[: ,8]

# create model
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
epochs = [10,20,30,40,50,60,70,80,90,100]
batches = [10]
optimizers = ['Nadam']
initializers = ['TruncatedNormal','glorot_uniform','Orthogonal','glorot_normal']
[cell1,cell2] = [[15],[4]]
activisers = ['relu','sigmoid', 'linear', 'tanh']

# choosing parameters to be optimized
param_grid = dict(epochs=epochs, batch_size=batches, opt=optimizers)
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=kfold)
grid_result = grid.fit(X3, Y)


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

'''
# comparing the effects of different characters
for i in range(8):
    results = cross_val_score(model, X3.T[i], Y, cv=kfold)
    print("%f with feature %r" % (results.mean(), i))
'''