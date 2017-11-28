# -*-coding:utf8-*-

import pickle
from boost2block import *
import time
import sys
import argparse

def load_data_file(train_data_file):
    '''
    从文件加载处理好的数据
    '''
    fi = open(train_data_file, 'rb')
    X_ci = pickle.load(fi)
    X_cj = pickle.load(fi)
    fi.close()
    X_ci = sp.csr_matrix(X_ci)
    X_cj = sp.csr_matrix(X_cj)
    return X_ci,X_cj


if __name__=='__main__':
    datapath='/home/zju/dgl/dataset/recommend/ml-100k/'
    

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-flod_name', help='Training flod file. e.g u1', dest='flod_name', default='u1')   
    parser.add_argument('-iters', help='boosting iters', dest='iters', default=6,type=int)  
    parser.add_argument('-lepoc', help='linear epoc', dest='lepoc', default=15, type=int)   
    parser.add_argument('-batch-size', help='batch size when trian linear term', dest='batch_size', default=1000, type=int)     
    parser.add_argument('-eta', help='linear learning rate', dest='eta', default=0.01, type=float)
    parser.add_argument('-alpha1', help='hyper param of regulizer of linear term', dest='alpha_1', default=0.01, type=float)
    parser.add_argument('-alpha3', help='hyper param of regulizer of quadratic term', dest='alpha_3', default=0.001, type=float)   
    parser.add_argument('-debug', help='1,if true;else 0', dest='isDebug', default=0, type=int) 
    parser.add_argument('-epsilon', help='precision of lambda ', dest='epsilon', default=0.001, type=float)
    args = parser.parse_args()
    
    
    print '############训练参数#############'
    
    print args
    
    print '############Begin#############'
    
    flod_name=args.flod_name
    train_data_file = datapath+flod_name+'_b.pkl'    
    X_ci, X_cj=load_data_file(train_data_file)
    
    context_num=X_ci.shape[0]
    if args.isDebug != 0:
        context_num = args.isDebug
        
    start=time.time()
    W,Z=train(boosting_iters=args.iters, X_uv=X_ci, X_uf=X_cj, linear_epoc=args.lepoc, batch_size=args.batch_size, eta=args.eta,
            a_1=args.alpha_1, a_3=args.alpha_3, lambda_epsilon=args.epsilon, context_num=context_num)
    print 'Training end,total time:',(time.time()-start)/60,'min'
    print 'Done!'
