from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def read_data(whole = None):
    data = genfromtxt('spambase.data', delimiter=',')
    (train,test) = train_test_split(data,test_size=0.8)
    train_x = train[:,:-1]
    train_y = train[:,-1]
    train_y[train_y == 0] = -1
    test_x = test[:,:-1]
    test_y = test[:,-1]
    test_y[test_y == 0] = -1
    sc = StandardScaler()
    train_x = sc.fit_transform(train_x)
    test_x = sc.fit_transform(test_x)
    if whole == 1:
        xvals = data[:,:-1]
        yvals = data[:,-1]
        yvals[yvals == 0] = -1
        return xvals,yvals
    return train_x,train_y,test_x,test_y
