# -*-coding:utf8-*-

import pickle
from boost2block import *


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
    
    flod_name='u1'
    train_data_file = datapath+flod_name+'_b.pkl'
    X_ci, X_cj=load_data_file(train_data_file)
    context_num=X_ci.shape[0]
    W,Z=train(boosting_iters=10, X_uv=X_ci, X_uf=X_cj, linear_epoc=10, batch_size=1000, eta=1,
            a_1=0.01, a_3=0.001, lambda_epsilon=0.1, context_num=10)
    fo=open(datapath+flod_name+'_save_weight.pkl','wb')
    pickle.dump(W,fo)
    pickle.dump(Z, fo)
    fo.close()
    print 'Done!'