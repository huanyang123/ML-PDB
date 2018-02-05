'''
----------------------------------------------------------------------------------------------------
A general utility function to tune parameters for various classification and regression ML algorithms
----------------------------------------------------------------------------------------------------
By Huanwang Shawn Yang  (2016-08-15)

How to use it?
1. import this module to your python code, for example: import ml_tune_para as tune
2. run the module : 
2a. if it is a classification problem, use  tune.auto_tune_classifier(X, Y)
2b. if it is a regression problem, use tune.tune_regressor(X, Y)

After running, the module will generate a log file 'tuned_para.log' containing the scores, best parameters
and the algorithm with the best parameters. You can use the best one (ranked at first) for prediction.

Here X and Y are the arrays or dataframe. They must be clean before using the module.
X contains the features and Y contains the target.

'''

# below for regression
from  sklearn.ensemble import RandomForestRegressor
from  sklearn.ensemble import ExtraTreesRegressor
from  sklearn.ensemble.weight_boosting import AdaBoostRegressor
from  sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from  sklearn.neighbors.regression import KNeighborsRegressor
from  sklearn.neighbors.regression import RadiusNeighborsRegressor #
from  sklearn.linear_model.base import LinearRegression
from  sklearn.linear_model import SGDRegressor
from  sklearn.linear_model import Ridge
from  sklearn.linear_model import Lasso
from  sklearn.linear_model import ElasticNet
from  sklearn.linear_model import BayesianRidge
from  sklearn.tree.tree import DecisionTreeRegressor
from  sklearn.neural_network import MLPRegressor
from  sklearn.svm.classes import SVR
from  sklearn.svm.classes import LinearSVR


# below for classification
from  sklearn.ensemble import RandomForestClassifier
from  sklearn.ensemble.weight_boosting import AdaBoostClassifier
from  sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from  sklearn.ensemble import ExtraTreesClassifier
from  sklearn.tree.tree import DecisionTreeClassifier
from  sklearn.neighbors.classification import KNeighborsClassifier #often used
from  sklearn.neighbors.classification import RadiusNeighborsClassifier  # if data not uniformly sampled
from  sklearn.linear_model import LogisticRegression
from  sklearn.linear_model import Perceptron
from  sklearn.linear_model import SGDClassifier

from  sklearn.svm.classes import LinearSVC
from  sklearn.svm.classes import SVC
from  sklearn.naive_bayes import GaussianNB
from  sklearn.naive_bayes import BernoulliNB  #for binary/boolean features

from  sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from  sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from  sklearn.gaussian_process import GaussianProcessClassifier
from  sklearn.neural_network import MLPClassifier


# library below is used for generating the results ...
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import roc_auc_score   #for classifier
from sklearn.metrics import classification_report   #for classifier
from sklearn.metrics import confusion_matrix   #for classifier
from sklearn.metrics import accuracy_score   #for classifier
from sklearn.metrics import mean_squared_error, r2_score  #for regressor

from sklearn import model_selection
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

####-------------------------------------------------
all_models=[]  #global
####-------------------------------------------------
def auto_tune_regressor(X, Y):
    ''' Tuning ML algorithms for numerical data 
    '''

    regressor=[
    'RandomForestRegressor'
   ,'AdaBoostRegressor'
   ,'GradientBoostingRegressor'
   ,'ExtraTreesRegressor'
   ,'DecisionTreeRegressor'
   ,'KNeighborsRegressor'
   ,'MLPRegressor'
   ,'LinearRegression'
   ,'Ridge'
   ,'Lasso'
   ,'ElasticNet'
   ,'BayesianRidge'
#   ,'SVR'
#   ,'LinearSVR'
#   ,'SGDRegressor'

    ]

    for model in regressor:
        tune_regressor(model, X, Y)

    write_model2file()

####------------------------------------------------#-
def auto_tune_classifier(X, Y):
    ''' Tuning ML algorithms for categorical data
    '''

    classifier=[
     'RandomForestClassifier'
    ,'AdaBoostClassifier'
    ,'GradientBoostingClassifier'
    ,'ExtraTreesClassifier'
    ,'DecisionTreeClassifier'
    ,'KNeighborsClassifier'
    ,'LogisticRegression'
    ,'GaussianNB'
    ,'BernoulliNB'
    ,'LinearDiscriminantAnalysis'
    ,'QuadraticDiscriminantAnalysis'
    ,'MLPClassifier'
    ,'SGDClassifier'
    ,'LinearSVC'   
#    ,'SVC'   #takes too much memory for large files   
#    ,'Perceptron'
#    ,'GaussianProcessClassifier'  #take all the memory for large data set! remove it
#    ,'RadiusNeighborsClassifier'  #
     ]

    for model in classifier:
        tune_classifier(model, X, Y)
    
    write_model2file()

####------------------------------------------------#-
def write_model2file():
    ''' write the sorted list to a file tuned_para.log
    '''

    fw=open('tuned_para.log','w')

    best=sorted(all_models, key=lambda x: x[0], reverse=True)

    for score, hyper_para, model in best:
        fw.write('score= %f\n' %score)
        fw.write('hyper_para= %s\n' %hyper_para)
        fw.write('model= %s\n\n' %model)

    fw.close()
    print '\nThe file (tuned_para.log) contains the sorted score, hyper_parameters, models\n'
####------------------------------------------------#-
def write_values(grid):
    '''
    '''
    print('\nbest_score=', grid.best_score_)
    print('best_params=', grid.best_params_)
    print('best_estimator=', grid.best_estimator_)

    all_models.append([grid.best_score_, grid.best_params_, grid.best_estimator_])

#    for x in  grid.grid_scores_ : print x  # print all the combinations

####-------------------------------------------------
def run_GridSearchCV(model, X, Y, param_grid, type):
   #include default
    run_para=[param_grid, dict()] 
    if len(param_grid)==0 :run_para=[dict()]
    run_para=[param_grid] #
    for param in run_para :
        if type == 'reg' :  #regression
            grid = GridSearchCV(model, param, cv=10,  n_jobs=-1) #use all cpu
        elif type =='class':
            grid = GridSearchCV(model, param, cv=10, scoring='accuracy', n_jobs=-1) #use all cpu
        else:
            print 'Error: please indicate "reg" or "class" for GridSearchCV'
            return
        grid.fit(X, Y)
        write_values(grid)

####-------------------------------------------------
def tune_regressor(model, X, Y):
    ''' Tune various popular regressors 
    '''
    if model=='RandomForestRegressor':
       print '\nTuning hyperparameters for sklearn.ensemble.RandomForestRegressor ...'
       hyper_para=dict(criterion=['mse', 'mae'], max_depth=[8,6,None], n_estimators=[120],
                  max_features=['auto','sqrt'])

       model = RandomForestRegressor()
       run_GridSearchCV(model, X, Y, hyper_para, 'reg') 

    elif model=='ExtraTreesRegressor':
       print '\nTuning hyperparameters for sklearn.ensemble.ExtraTreesRegressor ...'
       hyper_para=dict(criterion=[ 'mse','mae'], max_depth=[8,6, None], n_estimators=[120],
                  max_features=['auto','sqrt'])

       model = ExtraTreesRegressor()
       run_GridSearchCV(model, X, Y, hyper_para, 'reg')

    elif model=='AdaBoostRegressor':
       print '\nTuning hyperparameters for sklearn.ensemble.weight_boosting.AdaBoostRegressor ...'
       hyper_para=dict(n_estimators=[80],learning_rate=[1.0,0.7], loss=['linear', 'square', 'exponential'])

       model = AdaBoostRegressor()
       run_GridSearchCV(model, X, Y, hyper_para, 'reg') 

    elif model=='GradientBoostingRegressor':
       print '\nTuning hyperparameters for sklearn.ensemble.gradient_boosting.GradientBoostingRegressor ...'
       hyper_para=dict(n_estimators=[120],learning_rate=[0.1,0.05], loss=['ls', 'lad', 'huber'],
                  max_features= [ 'auto','sqrt'])

       model = GradientBoostingRegressor()
       run_GridSearchCV(model, X, Y, hyper_para, 'reg') 

    elif model=='DecisionTreeRegressor':
       print '\nTuning hyperparameters for sklearn.tree.tree.DecisionTreeRegressor ...'
       hyper_para=dict(splitter=['best', 'random'], max_depth=[5,4,3,2,None], max_features=
                  ['auto','sqrt'])
       model = DecisionTreeRegressor()
       run_GridSearchCV(model, X, Y, hyper_para, 'reg') 

    elif model=='KNeighborsRegressor':
       print '\nTuning hyperparameters for sklearn.neighbors.regression.KNeighborsRegressor ...'
       hyper_para=dict(n_neighbors=list(range(1, 30)), weights=['uniform', 'distance'])
       model = KNeighborsRegressor()  #by default
       run_GridSearchCV(model, X, Y, hyper_para, 'reg') 
   
    elif model=='MLPRegressor':
       print '\nTuning hyperparameters for sklearn.neural_network.MLPRegressor ...'
       hyper_para=dict(solver=['lbfgs', 'sgd', 'adam'])
       model = MLPRegressor()
       run_GridSearchCV(model, X, Y, hyper_para, 'reg')

    elif model=='SGDRegressor':
       print '\nTuning hyperparameters for sklearn.linear_model.SGDRegressor ...'
       hyper_para=dict(loss=['squared_loss', 'huber'])
       model = SGDRegressor()
       run_GridSearchCV(model, X, Y, hyper_para, 'reg')

    elif model=='LinearRegression':
       print '\nTuning hyperparameters for LinearRegression ...'
       hyper_para=dict()
       model = LinearRegression()  #by default
       run_GridSearchCV(model, X, Y, hyper_para, 'reg')

    elif model=='Ridge':
       print '\nTuning hyperparameters for sklearn.linear_model.Ridge ...'
       hyper_para=dict(solver=['auto'], alpha=[1.0, 2.0])
       model = Ridge()
       run_GridSearchCV(model, X, Y, hyper_para, 'reg')

    elif model=='Lasso':
       print '\nTuning hyperparameters for sklearn.linear_model.Lasso ...'
       hyper_para=dict(alpha=[1.0])
       model = Lasso()
       run_GridSearchCV(model, X, Y, hyper_para, 'reg')
    
    elif model=='ElasticNet':
       print '\nTuning hyperparameters for sklearn.linear_model.ElasticNet ...'
       hyper_para=dict( l1_ratio=[0, 0.3, 0.5, 0.7, 1.0])
       model = ElasticNet()
       run_GridSearchCV(model, X, Y, hyper_para, 'reg')
    

    elif model=='BayesianRidge':
       print '\nTuning hyperparameters for sklearn.linear_model.BayesianRidge ...'
       hyper_para=dict()
       model = BayesianRidge()
       run_GridSearchCV(model, X, Y, hyper_para, 'reg')
    

    elif model=='SVR':
       print '\nTuning hyperparameters for sklearn.linear_model.SVR ...'
       hyper_para=dict()
       model = SVR()
       run_GridSearchCV(model, X, Y, hyper_para, 'reg')
    
    elif model=='LinearSVR':
       print '\nTuning hyperparameters for sklearn.linear_model.LinearSVR ...'
       hyper_para=dict()
       model = LinearSVR()
       run_GridSearchCV(model, X, Y, hyper_para, 'reg')
 



####----------------------------------------
def tune_classifier(model, X, Y):
    ''' Tune various popular ML algorithm for classification 
    '''

    if model=='RandomForestClassifier':
       print '\nTuning hyperparameters for sklearn.ensemble.RandomForestClassifier ...'
       hyper_para=dict(criterion=['gini', 'entropy'], max_depth=[6,4, None], n_estimators=[100],
                  max_features=[ 'auto','sqrt'])

       model = RandomForestClassifier()
       run_GridSearchCV(model, X, Y, hyper_para, 'class') 

    elif model=='ExtraTreesClassifier':
       print '\nTuning hyperparameters for sklearn.ensemble.ExtraTreesClassifier ...'
       hyper_para=dict(criterion=['gini', 'entropy'], max_depth=[6,4,None], n_estimators=[100],
                  max_features=[ 'auto','sqrt'])

       model = ExtraTreesClassifier()
       run_GridSearchCV(model, X, Y, hyper_para, 'class')

    elif model=='AdaBoostClassifier':
       print '\nTuning hyperparameters for sklearn.ensemble.weight_boosting.AdaBoostClassifier ...'
       hyper_para=dict(n_estimators=[100], algorithm=['SAMME', 'SAMME.R'])

       model = AdaBoostClassifier()
       run_GridSearchCV(model, X, Y, hyper_para, 'class') 

    elif model=='GradientBoostingClassifier':
       print '\nTuning hyperparameters for sklearn.ensemble.gradient_boosting.GradientBoostingClassifier ...'
       hyper_para=dict(n_estimators=[100],learning_rate=[1.0], loss=['deviance'],
                  max_features= ['auto','sqrt'], max_depth=[5,4,3,1])

       model = GradientBoostingClassifier()
       run_GridSearchCV(model, X, Y, hyper_para, 'class') 

    elif model=='DecisionTreeClassifier':
       print '\nTuning hyperparameters for sklearn.tree.tree.DecisionTreeClassifier ...'
       hyper_para=dict(splitter=['best', 'random'], max_depth=[5,4,3,None], max_features=
                  ['auto','sqrt', 'log2', None])
       model = DecisionTreeClassifier()
       run_GridSearchCV(model, X, Y, hyper_para, 'class') 

    elif model=='KNeighborsClassifier':
       print '\nTuning hyperparameters for sklearn.neighbors.classification.KNeighborsClassifier ...'

       hyper_para = dict(n_neighbors=list(range(1, 30)), weights=['uniform', 'distance'])
       model=KNeighborsClassifier()  #by default
       run_GridSearchCV(model, X, Y, hyper_para, 'class') 

    elif model=='RadiusNeighborsClassifier':
       print '\nTuning hyperparameters for sklearn.neighbors.classification.RadiusNeighborsClassifier ...'

       hyper_para = dict(radius=[0.5, 1, 2, 3], weights=[ 'distance'])
       model=RadiusNeighborsClassifier()  #by default
       run_GridSearchCV(model, X, Y, hyper_para, 'class')

    elif model=='LogisticRegression':
       print '\nTuning hyperparameters for sklearn.linear_model.LogisticRegression ...'
       hyper_para=dict(penalty=['l2'], class_weight=[None, 'balanced'], 
                  solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
       model = LogisticRegression()
       run_GridSearchCV(model, X, Y, hyper_para, 'class') 

    elif model=='LinearDiscriminantAnalysis':
       print '\nTuning hyperparameters for sklearn.discriminant_analysis.LinearDiscriminantAnalysis ...'
       hyper_para=dict(solver=['svd', 'lsqr', 'eigen'])

       model = LinearDiscriminantAnalysis()
       run_GridSearchCV(model, X, Y, hyper_para, 'class')

    elif model=='SVC':
       print '\nTuning hyperparameters for sklearn.SVM.SVC ...'
       hyper_para=dict(kernel=['linear', 'poly', 'rbf'], decision_function_shape=[ 'ovo', 'ovr'])
       model = SVC()
       run_GridSearchCV(model, X, Y, hyper_para, 'class')

    elif model=='LinearSVC':
       print '\nTuning hyperparameters for sklearn.SVM.LinearSVC ...'
       hyper_para=dict(penalty=[ 'l2'], loss=['hinge', 'squared_hinge'])
       model = LinearSVC()
       run_GridSearchCV(model, X, Y, hyper_para, 'class')


    elif model=='BernoulliNB':
       print '\nTuning hyperparameters for sklearn.naive_bayes.BernoulliNB ...'
       hyper_para=dict()
       model = BernoulliNB()
       run_GridSearchCV(model, X, Y, hyper_para, 'class')

    elif model=='GaussianNB':
       print '\nTuning hyperparameters for sklearn.naive_bayes.GaussianNB ...'
       hyper_para=dict()
       model = GaussianNB()
       run_GridSearchCV(model, X, Y, hyper_para, 'class')

       '''
       print '\nNaive Beyes: Using Kfold cross-validation only (no tune hyperparameters)......'
       kfold = model_selection.KFold(n_splits=10, random_state=7)
       cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
       msg = "\n%s: \n\naccuracy=%f (std=%f)" % (model, cv_results.mean(), cv_results.std())
       print(msg)
       '''

    elif model=='Perceptron':
       print '\nTuning hyperparameters for sklearn.linear_model.Perceptron ...'
       hyper_para=dict()
       model = Perceptron()
       run_GridSearchCV(model, X, Y, hyper_para, 'class')

    elif model=='SGDClassifier':
       print '\nTuning hyperparameters for sklearn.linear_model.SGDClassifier ...'
       hyper_para=dict(loss=['hinge', 'log', 'modified_huber'], penalty=['l2', 'l1', 'elasticnet'])
       model = SGDClassifier()
       run_GridSearchCV(model, X, Y, hyper_para, 'class')


    elif model=='QuadraticDiscriminantAnalysis':
       print '\nTuning hyperparameters for sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis ...'
       hyper_para=dict()

       model = QuadraticDiscriminantAnalysis()
       run_GridSearchCV(model, X, Y, hyper_para, 'class')
 
    elif model=='GaussianProcessClassifier':
       print '\nTuning hyperparameters for sklearn.gaussian_process.GaussianProcessClassifier ...'
       hyper_para=dict()
       model = GaussianProcessClassifier()
       run_GridSearchCV(model, X, Y, hyper_para, 'class')

    elif model=='MLPClassifier':
       print '\nTuning hyperparameters for sklearn.neural_network.MLPClassifier ...'
       hyper_para=dict(solver=['lbfgs', 'sgd', 'adam'])
       model = MLPClassifier()
       run_GridSearchCV(model, X, Y, hyper_para, 'class')


##----------------------------------------
def ploting(list1, list2, grid, xlab, ylab):
    grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
    print(grid_mean_scores)
    plt.plot(list1, list2)
    plt.xlabel('%s'%xlab)
    plt.ylabel('%s'%ylab)
    plt.show()

##----------------------------------------
#Classification results
def write_result_class(X_test, y_test, y_pred, model):
    '''X_test: test set containing features only
       y_test: test set containing target only
       y_pred: the predicted values corresponding to the y_test
       model:  the model used to train the data (X_train)
    '''

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt 

#    for i in range(len(y_pred)): print 'predicted=%-20s  target=%-20s' %(y_pred[i],y_test[i])

    model_roc_auc = roc_auc_score(y_test, model.predict(X_test))
    print 'Classification accuracy=', model.score(X_test, y_test)
    print 'Classification AUC_ROC= ', model_roc_auc
    print 'Confusion_matrix=\n', confusion_matrix(y_test, y_pred)
    print '\nClassification_report=\n', classification_report(y_test, y_pred)

    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='AUC_ROC (area = %0.2f)' % model_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('AUC_ROC')
    plt.show()

##----------------------------------------
def write_result_regr(pred, Y_test, model):

    print '\nUsing %s for prediction' %model
 #   print '\npredicted=', pred
    print '\nr2_score=', r2_score(Y_test, pred)
    print '\nmean_squared_error=', mean_squared_error(Y_test, pred)
    print '\nroot_mean_squared_error=', np.sqrt(mean_squared_error(Y_test, pred))

##----------------------------------------
def data_scale(X_train, X_test):
    '''
    Stochastic Gradient Descent is sensitive to feature scaling, so it is highly recommended 
    to scale your data. For example, scale each attribute on the input vector X to [0,1] 
    or [-1,+1], or standardize it to have mean 0 and variance 1.
    '''
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)  # Don't cheat - fit only on training data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)  # apply same transformation to test data
    return X_train, X_test

##----------------------------------------

def upsample_minority(df):
    '''df: data frame;   the predictor (value=0, 1) must be labeled as target 
    '''

    from sklearn.utils import resample
    import pandas as pd

    print df.target.value_counts() 

# Separate majority and minority classes
    df_majority = df[df.target ==0]
    df_minority = df[df.target ==1]
#    print df_majority , df_minority

# Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=576,    # to match majority class
                                 random_state=123) # reproducible results


# Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
    print df_upsampled.target.value_counts()

# Separate input features (X) and target variable (y)
    Y = df_upsampled.target
    X = df_upsampled.drop('target', axis=1)

    return X, Y
