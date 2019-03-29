from sklearn.linear_model import LogisticRegression
from read_data import *
from sklearn.metrics import classification_report, confusion_matrix 
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate, validation_curve
from sklearn import metrics
from hoeffding import *

def logit(train_x,train_y,test_x,test_y):
    print("\nLogistic Regression Outputs")
    print("========================================================")    
    # Create functino for logistic regression
    clf = LogisticRegression(class_weight={-1:1.5,1:1})

    # Fitting train data to train the model
    model = clf.fit(train_x, train_y)

    # Use test data to test the accuracy of our model
    score = model.predict(test_x)
    testerror = 1 - model.score(test_x,test_y)
    trainerror = 1 - model.score(train_x,train_y)
    print('Training Error: %2.3f' % trainerror)
    print('Testing Error: %2.3f' % testerror)
    # Create a table to visualize our accuracy and scores
    print("\nScore matrix: ")
    print(confusion_matrix(test_y,score))

    # Create report to show precision recall f-score and support
    print("Classification Report:")
    print(classification_report(test_y,score))

    # Try k-fold cross validation on 10 folds
    (xvals,yvals) = read_data(1)

    score_accuracy = cross_validate(clf,xvals,yvals,cv=10,scoring='accuracy',return_train_score=True)
    #score_auc = cross_val_score(clf,xvals,yvals,cv=10,scoring='roc_auc')
    print('K-fold cross validation results:')
    print("Average train accuracy is %2.3f" % score_accuracy['train_score'].mean())
    print("Average test accuracy is %2.3f \n " % score_accuracy['test_score'].mean())

    # Hoeffding's
    CI_L , CI_U = hoeffding(testerror,0.05)
    print("Hoeffding's Confidence interval for Logistic Regression is:")
    print(CI_L,CI_U)

    # For k-folds
    kerisk = 1 - score_accuracy['test_score'].mean()
    kCI_L, kCI_U = hoeffding(kerisk, 0.05)
    print("\nHoeffding's Confidence interval for LR after k-folds is:")
    print(kCI_L,kCI_U)

    return (CI_L,CI_U)
