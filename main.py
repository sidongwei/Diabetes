from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l1_l2
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
import numpy

# using random seed
seed = 7
numpy.random.seed(seed)

# load the datasets
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = X2 = dataset[: ,0:8]
Y = dataset[: ,8]

# indicate the missing data
for i in range(X.shape[0]):
    for j in range(1,8):
        if X[i][j]==0:
            X2[i][j]=-1
# preprocessing the data
Imp = Imputer(missing_values=-1,strategy='mean')
X3=Imp.fit_transform(X=X2,y=Y)
X4=scale(X)
X5=normalize(X,norm='l2')
X6=scale(X3)
X7=normalize(X3)
# build different learning modules
D_T = DecisionTreeClassifier(random_state=seed)
S_V_C = SVC(random_state=seed)
G_N_B = GaussianNB()
reg = l1_l2(l1=0.01, l2=0.01)    #regularization for logistic regression
def create_model():
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=reg, input_dim=X.shape[1]))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model
L_R = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
def create_model_2():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
N_N = KerasClassifier(build_fn=create_model_2, epochs=150, batch_size=10, verbose=0)

data = dict(original=X, impute=X3, scale=X4, normalize=X5, im_sc=X6, im_nor=X7)
method = dict(DT=D_T,SVC=S_V_C,GNB=G_N_B,LR=L_R,NN=N_N)

#using 10-splits cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# outputting data
for i in data:
    for j in method:
        results = cross_val_score(method[j], data[i], Y, cv=kfold)
        print("%f using %r and %r"%(results.mean(),i,j))
