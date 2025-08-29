import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import copy,math

data=pd.read_csv('/Users/moahmedharith/Downloads/classification_dataset.csv')
# print(data.head(10))
X_train=data[['Hours_Studied','Sleep_Hours','Previous_Score']].values
y_train=data["Pass"].values
X_mean=np.mean(X_train,axis=0)
X_std=np.std(X_train,axis=0)
X_train=(X_train-X_mean)/X_std
w = np.zeros(X_train.shape[1], dtype=float)
b = 0.0
# print(X_train)
# print(y_train)
def predict(X,w,b):
    f_wb=1/(1+np.exp(-(np.dot(X,w)+b)))
    prediction=1 if f_wb>=0.5 else 0
    return prediction
def sigmoid(z):
    g_z=1/(1+np.exp(-z))
    return g_z
def C0st_function(X,y,w,b):
    # m => number of Eamples 
    # Set Cost = 0
    # for loop in it the cost function 
    # in for loop set f_wb=sigmoid(np.dot(x[i],w)+b)
    # cost function J(w,b)=-(1/m)*(sumof(y[i]*log(f(w,b))+(1-y[i])*log(1-f(w,b)))
    # return the cost
    m=X.shape[0]
    cost=0
    for i in range(m):
        f_wb_i=sigmoid(np.dot(X[i],w)+b)
        cost+=(y[i]*np.log(f_wb_i)+(1-y[i])*np.log(1-f_wb_i))
    cost/=(-1*m)
    return cost
def compute_gradient(X,y,w,b):
    # m,n => X.shape
    # det dj_dw=np.zeros((n,)) , dj_db=0
    # in for loop (1.f_wb 2. err 3 for loop for dj_dw to apply x[i,j]=> x_j^i x[0][1] => fature 1 row 0 give you the value of this )
    # out for loop dj_dw/=m,dj_db/=m
    # return dj_dw,dj_db
    m,n=X.shape
    dj_dw=np.zeros((n,))
    dj_db=0
    for i in range(m):
        z=np.dot(X[i],w)+b
        f_wb=sigmoid(z)
        err=f_wb-y[i]
        for j in range(n):
            dj_dw[j]+=err*X[i,j]
        dj_db+=err
    dj_dw/=m
    dj_db/=m
    return dj_dw,dj_db
def gradient_descent(X,y,w_in,b_in,cost_f,compute_g,alpha,num_iters):
    # set w equal to copy.deepcopy(w_in) to take a deep copy of w b = b_in j_history=[]
    # for loop => inside it we will do this 1 => dj_dw,dj_db= compute_g=(X,y,w,b) 2 => w-=alpah*dj_dw,b-=dj_db
    # inside for loop i<10000: j_history.append(cost_f) if i%math.ceil(num_iters/10)==0: print iteration and cost on every iteration
    # return w,b,j_history
    w=copy.deepcopy(w_in)
    b=b_in
    j_history=[]
    for i in range(num_iters):
        dj_dw,dj_db=compute_g(X,y,w,b)
        w-=alpha*dj_dw
        b-=alpha*dj_db
        if i <10000:
            j_history.append(cost_f(X,y,w,b))
        if i%math.ceil(num_iters/10)==0:
            print(f"Iteration {i:4d} and Cost : {j_history[-1]:8.2f}")
    return w,b,j_history
w_init=np.zeros_like(w,dtype=float)
b_init=0
alpha=0.03
iterations=2000
w_final , b_final , j_hist=gradient_descent(X_train,y_train,w_init,b_init,C0st_function,compute_gradient,alpha,iterations)
print(f"w,b are found w = {w_final} , b = {b_final}")
# new student data
new_data=np.array([8,7,75])
new_data_scaled=(new_data-X_mean)/X_std
prediction_for_new_value=predict(new_data_scaled,w_final,b_final)
print(f"Probality of passing = {prediction_for_new_value}")
# plot Cost function 
plt.plot(j_hist)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function")
plt.show()
# 2. Predicted vs Actual
y_pred = []
for i in range(len(X_train)):
    y_pred.append(predict(X_train[i],w_final,b_final))
plt.figure(figsize=(8,5))
plt.scatter(range(len(y_train)), y_train, label="Actual", marker='o')
plt.scatter(range(len(y_pred)), y_pred, label="Predicted", marker='x')
plt.xlabel("Samples")
plt.ylabel("Pass (1) / Fail (0)")
plt.title("Predicted vs Actual")
plt.legend()
plt.show()
