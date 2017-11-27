# -*——coding:utf8-*-
from quadratic_solver import *
from linear_solver import *

import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse.linalg import LinearOperator, eigsh,eigs
import time
import traceback
import sys
import datetime
import os


def initial(context_num,d_dim):
    U=[1./context_num]*context_num
    W_old=sp.csr_matrix(np.random.uniform(low=-0.5/d_dim, high=0.5/d_dim, size=d_dim))
    W_old=W_old.T
    Z=sp.csr_matrix((d_dim,d_dim),dtype='f')
    return U,W_old,Z

def save2pkl(iter_count,flod_name,datapath,Z,W):
    fo=open(datapath+flod_name+'_iter_'+str(iter_count)+'_save_weight.pkl','wb')
    pickle.dump(W,fo)
    pickle.dump(Z, fo)
    fo.close()
    
def train(boosting_iters,X_uv,X_uf,linear_epoc,batch_size,eta,a_1,a_3,lambda_epsilon,context_num):
    '''
    :param boosting_iters
    :param X_uv context-observed 数据,csr_matrix 矩阵，每一行一个样本
    :param X_uf context-non_observed 数据,csr_matrix 矩阵，每一行一个样本
    :param linear_epoc 计算线性项的时候,迭代的周期数
    :param batch_size  计算线性项的时候,batch 的大小
    :param a_1 线性正则项参数
    :param a_3 二次项正则参数
    :param eta 这指的是在使用梯度下降计算线性项的时候，步长的大小。（随后改进：这个参数应该随训练的进行而减小）
    :param lambda_epsilon 二次项超参数，控制lambda_t 的精度 e.g 0.001,0.01
    :param context_num  这里的context_指的是在计算Z的时候，使用的样本总数，与线性项计算无关，与真正的上下文无关
    '''
    d_dim=X_uv.shape[1]
    U,W_old,Z=initial(context_num,d_dim)
    
    # create a folder to save the weight
    modelPath='models_'+str(datetime.datetime.now().month)+'_'+str(datetime.datetime.now().day)+'_'
    modelPath+=str(datetime.datetime.now().hour)
    datapath='/home/zju/dgl/dataset/recommend/ml-100k/'
    modelPath=datapath+modelPath+'/'
    os.makedirs(modelPath)
    
    print 'sum(U):',np.sum(U)
    
    for iter_count in range(boosting_iters):
        print '#############boosting_iter:',iter_count,'#########################'

        linearSolver=LinearSolver(batch_size,linear_epoc,X_uv,X_uf,Z,a_1,eta)

        W=linearSolver.fit()
        print 'W is finished:',W.shape
        # 更新Z
        z_t,eigenval=QuadraticSolver.getComponentZ_eigval(context_num, U,d_dim,X_uv,X_uf)
        print 'Z_t eigenval is finished:',Z.shape,eigenval
        if np.abs(eigenval) < a_3:
            break;
        print 'lambdaSearch for t#####################'
        lambda_t=QuadraticSolver.lambdaSearch(context_num,z_t,U,a_3,lambda_epsilon,[0,100],W,W_old,X_uv,X_uf)
        print 'lambda_t is finished:',lambda_t
        U=QuadraticSolver.updateU(context_num,U,z_t,lambda_t,W,W_old,X_uv,X_uf)
        print 'update U  is finished，the sum(u)=:',np.sum(U)
        W_old=W
        Z+=lambda_t*z_t
        
        # To do : change to genearal
        save2pkl(iter_count,'u1',modelPath,Z,W)
        ## 这里输出损失total loss ,boosting 的第n次
        
    return W,Z


