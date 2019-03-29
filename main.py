from logit import *
from read_data import *
from svm import * 
from hoeffding import *
from random_forest import *

if __name__ == "__main__":
    '''
    Run line by line to avoid confusion of ouptut
    '''
    # Read data
    (train_x,train_y,test_x,test_y) = read_data()

    # Run SVM algorithm
    CI_SVM = svm(train_x,train_y,test_x,test_y)

    # Logist Regression algorithm
    CI_LR = logit(train_x,train_y,test_x,test_y)

    # Random Forrest
    CI_RF = random_forest(train_x,train_y,test_x,test_y)

    print("\n\nFinal Results")
    print("==================================================")

    print("\nHoeffding's Confidence interval for SVM is:")
    print(CI_SVM)

    print("\nHoeffding's Confidence interval for LR is:")
    print(CI_LR)

    print("\nHoeffding's Confidence interval for Random Forest is:")
    print(CI_RF)
