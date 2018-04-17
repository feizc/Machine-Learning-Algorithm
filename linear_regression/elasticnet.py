
"""
    ElasticNet 回归
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
    return cross_validation.train_test_split(datasets.data,diabetes.target,
		test_size=0.25,random_state=0) 

def test_ElasticNet(*data):
    '''
    测试 ElasticNet 的用法

    '''
    X_train,X_test,y_train,y_test=data
    regr = linear_model.ElasticNet()
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept %.2f'%(regr.coef_,regr.intercept_))
    print("Residual sum of squares: %.2f"% np.mean((regr.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))
def test_ElasticNet_alpha_rho(*data):
    '''
    测试 ElasticNet 的预测性能随 alpha 和 l1_ratio 的影响

    '''
    X_train,X_test,y_train,y_test=data
    alphas=np.logspace(-2,2)
    rhos=np.linspace(0.01,1)
    scores=[]
    for alpha in alphas:
            for rho in rhos:
                regr = linear_model.ElasticNet(alpha=alpha,l1_ratio=rho)
                regr.fit(X_train, y_train)
                scores.append(regr.score(X_test, y_test))
    alphas, rhos = np.meshgrid(alphas, rhos)
    scores=np.array(scores).reshape(alphas.shape)
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig=plt.figure()
    ax=Axes3D(fig)
    surf = ax.plot_surface(alphas, rhos, scores, rstride=1, cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\rho$")
    ax.set_zlabel("score")
    ax.set_title("ElasticNet")
    plt.show()
if __name__=='__main__':
    X_train,X_test,y_train,y_test=load_data() # 产生用于回归问题的数据集
    test_ElasticNet(X_train,X_test,y_train,y_test) # 调用 test_ElasticNet
    # test_ElasticNet_alpha_rho(X_train,X_test,y_train,y_test) # 调用 test_ElasticNet_alpha_rho