# -*-coding:utf8-*-

import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse.linalg import LinearOperator, eigsh,eigs
import time
import traceback

class QuadraticSolver():

    def __init__(self):
        pass

    @staticmethod
    def getA_c(index, X_uv, X_uf):
        '''
        index： the index of sample
        :return: (x_ci)*(x_ci^T)-(x_cj)*(x_cj^T)
        (x_ci) is the observed pair(context,item_i),while (x_cj) is the non-observed pair(context,item_j)
        '''
        x_ci = X_uv[index].T
        x_cj = X_uf[index].T
        return (x_ci * x_ci.T) - (x_cj * x_cj.T)
    
    @staticmethod
    def getAccA_c(context_num, U, s_dim, X_uv, X_uf):
        '''
        :param context_num: total number of train_data
        :param U: the weight of A_r
        :param s_dim: 输入向量的维度
        :return: sum(u_rA_r) it is a matrix
        '''
        
        A = sp.csr_matrix((s_dim, s_dim), dtype='f')
        for c in range(context_num):
            try:
                Ac = QuadraticSolver.getA_c(c, X_uv, X_uf)
                U[c] * Ac
                A += U[c] * Ac
            except Exception:
                #print 'AC:', Ac.shape, type(Ac)
                print 'A:', A.shape, type(A)
                print 'U', type(U)
                print 'U[c]', U[c], type(U[c])
                print traceback.format_exc()
                raise Exception('getAccA_c 又出异常了')
        return A
    
    @staticmethod
    def getEigenvector(A):
        '''
        :param A: the matrix to be decomposed，A should be symmetric
        :return:
        '''
        # todo: use power method
        eigenval, eigenvec = eigsh(A, k=1)
        return eigenval, eigenvec
    
    @staticmethod
    def getGamma(W, X_uv, X_uf):
        '''
        :param W the transform of W, is a column vector
        :param X_uv should be a row matrix or a row vecotr
        :param X_uf should be a row matrix or a row vecotr
        '''
        B = X_uv - X_uf
        temp = safe_sparse_dot(B, W).data
        return np.exp(-temp)
    
    @staticmethod
    def getGammaRelativeRatio(index, W, W_last,X_ci,X_cj):
        '''
         线性项增缩比gar=gamma_r^j/gamma_r^(j-1)
        index： the index of sample
        :return:
        (x_ci) is the observed pair(context,item_i),while (x_cj) is the non-observed pair(context,item_j)
        '''
        x_ci = X_ci[index]
        x_cj = X_cj[index]
        gamma = QuadraticSolver.getGamma(W, x_ci, x_cj)
        gamma_old = QuadraticSolver.getGamma(W_last, x_ci, x_cj)
        return gamma / gamma_old
    
    @staticmethod
    def getComponentZ_eigval(context_num, U, s_dim, X_uv, X_uf):
        '''
        :param s_dim: 输入向量的维度
        return: vv^T、eigenval
        '''
        A = QuadraticSolver.getAccA_c(context_num, U, s_dim, X_uv, X_uf)
        eigenval, eigenvec = QuadraticSolver.getEigenvector(A)
        #     print eigenvec.shape
        return safe_sparse_dot(eigenvec, eigenvec.T), eigenval
    
    
    @staticmethod
    def get_Hc(Ac, z_t):
        '''
        Note：Z_t 是一个对称矩阵
        '''
        return np.sum(np.diag(safe_sparse_dot(Ac, z_t)))
    
    
    @staticmethod
    def lambdaSearch(context_num, z_t, U, a_3, lambda_epsilon, inital_interval, W, W_old, X_uv, X_uf):
        def costfun_lambda_t(context_num, z_t, a_3, lambda_t, W, W_old, X_uv, X_uf):
            '''
             本部分对应p.s.m learning with boosting 公式(8)
            '''
            cost_lambda_t = 0.0
            for c in range(context_num):
                Ac = QuadraticSolver.getA_c(c, X_uv, X_uf)
                # gamma_ration
                gr = QuadraticSolver.getGammaRelativeRatio(c, W, W_old,X_uv,X_uf)
                Hc = QuadraticSolver.get_Hc(Ac, z_t)
                cost_lambda_t += (Hc - a_3) * gr * U[c] * np.exp(-lambda_t * Hc)
            return cost_lambda_t

        def bi_search(lambda_l, lambda_u, epsilon):
            lambda_mid = 0.0
            while True:
                lambda_mid = 0.5 * (lambda_l + lambda_u)
                cost_lambda_t = costfun_lambda_t(context_num, z_t, a_3, lambda_mid, W, W_old, X_uv, X_uf)
                if cost_lambda_t > 0:
                    lambda_l = lambda_mid
                else:
                    lambda_u = lambda_mid
                if lambda_u - lambda_l < epsilon:  # 这里要小心，太小的浮点运算可能不准确。
                    #                     print lambda_l,lambda_u,epsilon
                    break
                    #                 print '[lambda_l,lambda_u]:', lambda_l,lambda_u
            return lambda_mid

        return bi_search(inital_interval[0], inital_interval[1], lambda_epsilon)
    
    @staticmethod
    def updateU(context_num, U, z_t, lambda_t, W, W_old, X_uv, X_uf):
        '''
         这里会改变U的原始数据类型,如：
         原始：U[1]=12
         计算后：U[1]=[12]
        '''
        for c in range(context_num):
            Ac = QuadraticSolver.getA_c(c, X_uv, X_uf)
            gr = QuadraticSolver.getGammaRelativeRatio(c, W, W_old,X_uv,X_uf)
            Hc = QuadraticSolver.get_Hc(Ac=Ac, z_t=z_t)
            # 注意：等式右边，计算结果不是一个值，而是一个大小为1的数组，所以要加上[0]
            U[c] = (gr * U[c] * np.exp(-lambda_t * Hc))[0]
        # 别忘了归一化
        U = U / (np.sum(U))
        return U




