import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
import csv


def ldaLearn(X,y): # problem 1 (akannan4)
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
   
    # Handouts B.3 page 6      

    N,d = X.shape
    labels = y.reshape(y.size)
    classes = np.unique(labels)
    no_of_classes = classes.shape[0]
    k = no_of_classes

    means = np.zeros((d, k))

    for cl in range(k):
        XTarget = X[labels == classes[cl]]
        means[:, cl] = np.mean(XTarget, axis=0)

    covmat = np.cov(X.T)

    return means,covmat

def qdaLearn(X,y): # problem 1 (akannan4)
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # Handouts B.3 page 6

    N,d = X.shape
    labels = y.reshape(y.size)
    classes = np.unique(labels)
    no_of_classes = classes.shape[0]
    k = no_of_classes

    means = np.zeros((d, k))
    covmats = [np.zeros((d,d))]* k

    for cl in range(k):
        XTarget = X[labels == classes[cl]]
        means[:, cl] = np.mean(XTarget, axis=0)
        covmats[cl] = np.cov(XTarget.T)

    
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest): # problem 1 (akannan4)
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # Handouts B.3 page 6

    N,d = Xtest.shape
    _,k = means.shape
    det_covmat = np.linalg.det(covmat)
    inv_covmat = np.linalg.inv(covmat)

    p = np.zeros((N, k))

    for cl in range(k):
        divisor = (np.power(np.pi * 2,d/2) * (np.power(det_covmat, 0.5)))
        c1 = Xtest - means[:, cl]
        c2 = np.dot(inv_covmat, c1.T)
        c3 = np.sum(c1 * c2.T, axis = 1)
        dividend = np.exp(-0.5 * c3)
        p[:, cl] = dividend / divisor

    ypred = np.argmax(p, 1)
    ypred += 1
    ytest = ytest.reshape(ytest.size)

    acc = np.mean(ypred == ytest) * 100

    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest): # problem 1 (akannan4)
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # Handouts B.3 page 6

    N,d = Xtest.shape
    _,k = means.shape


    p = np.zeros((N, k))

    for cl in range(k):
        det_covmat = np.linalg.det(covmats[cl])
        inv_covmat = np.linalg.inv(covmats[cl])
        divisor = (np.power(np.pi * 2,d/2) * (np.power(det_covmat, 0.5)))
        c1 = Xtest - means[:, cl]
        c2 = np.dot(inv_covmat, c1.T)
        c3 = np.sum(c1 * c2.T, axis = 1)
        dividend = np.exp(-0.5 * c3)
        p[:, cl] = dividend / divisor

    ypred = np.argmax(p, 1)
    ypred += 1
    ytest = ytest.reshape(ytest.size)

    acc = np.mean(ypred == ytest) * 100

    return acc,ypred


def learnOLERegression(X,y): # problem 2 (arjunsun)
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                

    # w = (X^T.X)^(-1).(X^T.Y)  

    Xtran=np.transpose(X)
    w=np.linalg.inv(np.dot(Xtran,X))
    w=np.dot(np.dot(w,Xtran),y)
    
    return w

def learnRidgeRegression(X,y,lambd): # problem 3 (arjunsun)
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # w = (((X^T.X) + lambd * identity(d)) ^ -1 ).(X^T.Y)

    Xtran=np.transpose(X)
    i=np.identity(X.shape[1])
    w=np.linalg.inv(np.dot(Xtran,X)+np.dot(lambd,i))
    w=np.dot(np.dot(w,Xtran),y)

    return w


def testOLERegression(w,Xtest,ytest): # problem 2 (akannan4)
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    #RMSE = (SQRT(SUM((ytest^T - (w^T.Xtest^T))^2)/N))

    wTran=np.transpose(w)
    Xtran=np.transpose(Xtest)
    yTran=np.transpose(ytest)
    N=Xtest.shape[0]
    rmse=np.sum(np.square(np.subtract(yTran,np.dot(wTran,Xtran))))
    rmse=np.sqrt(rmse/N)

    return rmse

def regressionObjVal(w, X, y, lambd): # problem 4 (sammokka)

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  
    
    # error = equation (4) in project-description

    w = w.reshape(65,1)

    a= y-np.dot(X,w)
    error = (0.5*np.dot(a.transpose(),a)) + 0.5*lambd* np.dot(w.transpose(),w)

    a = np.dot(X.T,X)
    b = np.dot(a,w)

    c = np.dot(X.T,y)

    d  = lambd * w

    error_grad = (b - c + d).flatten()

    
    return error, error_grad


def mapNonLinear(x,p): # problem 5 (sammokka)
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                   
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - dimensions : (N x (p+1))  
    # where N is number of rows of X                                                       
    # IMPLEMENT THIS METHOD


    N= x.shape[0]
    Xd = np.empty((N,p+1))

    for i in range(p+1):
        Xd[:,i] = np.power(x,i)

    return Xd

# Main script

# Problem 1
print('Problem 1 \n')
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,pred = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,pred = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')
plt.savefig('problem1_lda.jpg')
print('LDA plot saved to problem1_lda.jpg')
plt.clf()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')
plt.savefig('problem1_qda.jpg')
print('QDA plot saved to problem1_qda.jpg \n\n')
plt.clf()


# Problem 2
print('Problem 2 \n')
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle_train = testOLERegression(w,X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i_train = testOLERegression(w_i,X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept for training data '+str(mle_train))
print('RMSE with intercept for training data '+str(mle_i_train))
print('RMSE without intercept for testing data '+str(mle))
print('RMSE with intercept for testing data '+str(mle_i))
print('\n')


# Problem 3
print('Problem 3 \n')

k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3_test = np.zeros((k,1))
rmses3_train = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3_test[i] = testOLERegression(w_l,Xtest_i,ytest)
    rmses3_train[i] = testOLERegression(w_l,X_i,y)
    i = i + 1

lambda_opt_3 = lambdas[np.argmin(rmses3_test)]

w_l = learnRidgeRegression(X_i,y,lambda_opt_3) # weights learnt by ridge regression with optimal lambda

plt.plot(np.linspace(0,64,num = 65),w_i)
plt.plot(np.linspace(0,64,num = 65),w_l)
plt.legend(('OLE regression weights', 'Ridge regression weights \n(with optimal lambda 0.06)'))

plt.title('Relative magnitude of weights learnt using OLE and RIDGE regression')
plt.savefig('problem3-magnitude.jpg')
print('Relative magnitude of weights plot saved to problem3-magnitude.jpg \n\n')
plt.clf()

plt.plot(lambda_opt_3,np.min(rmses3_test),'xr')
plt.plot(lambdas,rmses3_test)
plt.plot(lambdas,rmses3_train)
plt.xlabel('lambda')
plt.ylabel('rmse')
plt.legend(('Optimal lambda '+str(lambda_opt_3),'Testing data','Training data'), loc = 'lower right')
plt.title('Ridge regression (without gradient descent)')
plt.savefig('problem3.jpg')
print('Ridge regression plot saved to problem3.jpg \n\n')
plt.clf()


# Problem 4
print('Problem 4 \n')

k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4_test = np.zeros((k,1))
rmses4_train = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4_test[i] = testOLERegression(w_l,Xtest_i,ytest)
    rmses4_train[i] = testOLERegression(w_l,X_i,y)

    i = i + 1


lambda_opt_4 = lambdas[np.argmin(rmses4_test)]

plt.plot(lambda_opt_4,np.min(rmses4_test),'xr')
plt.plot(lambdas,rmses4_test)
plt.plot(lambdas,rmses4_train)
plt.xlabel('lambda')
plt.ylabel('rmse')
plt.legend(('Optimal lambda '+str(lambda_opt_4),'Testing data','Training data'), loc = 'lower right')
plt.title('Ridge regression (with gradient descent)')
plt.savefig('problem4.jpg')
print('Gradient descent based ridge regression plot saved to problem4.jpg \n\n')
plt.clf()


# Problem 5
print('Problem 5 \n')

pmax = 7
lambda_opt = lambdas[np.argmin(rmses4_test)]
rmses5_test = np.zeros((pmax,2))
rmses5_train = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5_test[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5_test[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    rmses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    rmses5_train[p,1] = testOLERegression(w_d2,Xd,y)


plt.xlabel('p')
plt.ylabel('RMSE')
plt.plot(range(pmax),rmses5_test)
plt.legend(('No Regularization ( on testing data )','Regularization ( on testing data )'))
plt.savefig('problem5-testing.jpg')
plt.clf()

plt.xlabel('p')
plt.ylabel('RMSE')
plt.plot(range(pmax),rmses5_train)
plt.legend(('No Regularization ( on training data )','Regularization ( on training data )'))
plt.savefig('problem5-training.jpg')
plt.clf()
print('Non-linear regression plots saved to problem5-training.jpg and problem5-testing.jpg')




