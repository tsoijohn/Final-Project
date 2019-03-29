from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from hoeffding import *

def random_forest(train_x,train_y,test_x,test_y):
    rf = RandomForestClassifier(n_estimators=30,class_weight={1:1,-1:1.5})
    rf.fit(train_x,train_y)
    y_pred = rf.predict(test_x)
    print("\nRandom Forrest Outputs")
    print("========================================================")
    print(confusion_matrix(test_y,y_pred))  
    print(classification_report(test_y,y_pred))
    trainerror = 1- rf.score(train_x,train_y)
    print("Train Error:", trainerror)
    testerror = 1- rf.score(test_x, test_y)
    print('Test Error:', testerror)

    # Hoeffding's
    CI_L , CI_U = hoeffding(testerror,0.05)
    print("\nHoeffding's Confidence interval for Random Forest is:")
    print(CI_L,CI_U)

    return (CI_L,CI_U)
