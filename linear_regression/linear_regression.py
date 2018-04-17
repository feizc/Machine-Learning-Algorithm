"""
    广义线性模型
    LinearRegression
    费政聪
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model,cross_validation

def load_data():
    '''
    加载用于回归问题的数据集
    '''
    diabetes = datasets.load_diabetes()
    return cross_validation.train_test_split(datasets.data,diabetes.target,test_size=0.25,random_state=0) 
def test_LinearRegression(*data):
    '''
    测试 LinearRegression 的用法
    '''
    X_train,X_test,y_train,y_test=data
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept %.2f'%(regr.coef_,regr.intercept_))
    print("Residual sum of squares: %.2f"% np.mean((regr.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))
if __name__=='__main__':
    X_train,X_test,y_train,y_test=load_data() 
    test_LinearRegression(X_train,X_test,y_train,y_test) # 调用 test_LinearRegression
