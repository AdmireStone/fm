#-*-coding:utf8-*-
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse.linalg import LinearOperator, eigsh,eigs
import time
import traceback
import sys
import argparse

def getContextset_and_Test_data(test_data_file):
    df_test_data=pd.read_table(names=['userID','itemID','rating','timestamp'],
                               filepath_or_buffer=test_data_file)
    contex_set=df_test_data['userID'].drop_duplicates().values
    return contex_set,df_test_data


def getItemset(item_data_file):
    df_item = pd.read_table(header=None, filepath_or_buffer=item_data_file, sep='|')

    item_set = set(df_item[0].values)
    return item_set

def getUser_and_item_onehot(u_user_fi,u_item_fi):
    '''
    return pandas.sparseframe
    '''
    u_user=pd.read_table(header=None,filepath_or_buffer=u_user_fi,sep='|')
    u_item=pd.read_table(header=None,filepath_or_buffer=u_item_fi,sep='|')
    userID_sparseOnehot=pd.get_dummies(u_user[0],sparse=True)
    itemID_sparseOnehot=pd.get_dummies(u_item[0],sparse=True)
    return userID_sparseOnehot,itemID_sparseOnehot

# 已测试
def getUserOnehot_sp_vec(userID,userIdOneHot):
    '''
    return (1,n) csr_matrix
    '''
    oht_userID=sp.csc_matrix(userIdOneHot.loc[userID-1].to_dense().values,
                         dtype='f')
    return oht_userID

def getitemOnehot_sp_vec(itemID,itemIDOneHot):
    '''
    return (1,n) csr_matrix
    '''
    oht_movieID=sp.csc_matrix(itemIDOneHot.loc[itemID-1].to_dense().values,
                          dtype='f')
    return oht_movieID

# 已测试
def getContextOneHot_sp_vec(context,userIdOneHot):
    '''
     该方法需要重载，不同问题中,上下文的定义不一样
    :param context:
    :return:
    '''
    return getUserOnehot_sp_vec(context,userIdOneHot)

# 已测试
def get_c_tag_matrix(items_set,context_sp_vec,itemIDsOneHot,total_dim):
    '''
    :param tags_set
    :param context should be tuple,(userID,movieID)
    :return a sparse matrix，csr_matrix,in which each row is the onehot(userID,movieID,tagID)
    '''
    sparse_matrix=None
    # 1. 首先获取userID，movieID，tagID的 index
    # 2. 根据各个index 获取对应onehot 编码
    # 3. 拼接onehot 编码
    # 将拼接好的追加到矩阵中
    for itemID in items_set:
        item_sp_vec=getitemOnehot_sp_vec(itemID,itemIDsOneHot)
        x=sp.hstack([context_sp_vec,item_sp_vec])
        sparse_matrix=sp.vstack([sparse_matrix,x])

    sparse_matrix=sp.csc_matrix(sparse_matrix)
    return sparse_matrix

# 已测试
def getRelevenceScore(X,W,Z):
#         raise Exception('注意加上线性项')
    '''
    :param X the input matrx
    :param W the linear weight,the shape is (dim,1)
    :return the shape is like [ 1.49672429  1.36277656]
    '''
    # the shape of linear_term is (1,n), csr_matrix
    linear_term=safe_sparse_dot(X,W).T
    # the shape of qusdratic_term is (1,n), matrix
    qusdratic_term=(safe_sparse_dot(safe_sparse_dot(X, Z), X.T)).diagonal()
    return np.asarray((linear_term+qusdratic_term))[0]

def getSingleContextAUC(s_c_ob,s_c_non_ob):
    acc=0.0
    for s_ci in s_c_ob:
        acc+=np.sum(s_c_non_ob<s_ci)
    return acc/(len(s_c_ob)*len(s_c_non_ob))


def evalu(W,Z,test_data, contex_set,items_set, userIdOneHot, itemIDsOneHot, total_dim,isDebug):
        
    count = 0

    global_accuracy = 0.
    # context_set 是 当前测试数据集合中userID的集合
    for context in contex_set:
        # 1. 对每个用户，划分正负样本
        positive_items = set(test_data[(test_data['userID'] == context)]['itemID'].values)

        # non-observed tags
        negative_items = items_set - positive_items

        context_sp_vec = getContextOneHot_sp_vec(context, userIdOneHot)

        get_c_tag_matrix(items_set, context_sp_vec, itemIDsOneHot, total_dim)

        observed_c_tag_matrix = get_c_tag_matrix(positive_items, context_sp_vec,
                                                 itemIDsOneHot, total_dim)
        non_observed_c_tag_matrix = get_c_tag_matrix(negative_items, context_sp_vec,
                                                     itemIDsOneHot, total_dim)

        # 计算评分
        s_c_ob = getRelevenceScore(observed_c_tag_matrix, W, Z)
        s_c_non_ob = getRelevenceScore(non_observed_c_tag_matrix, W, Z)

        # 统计
        acc = getSingleContextAUC(s_c_ob, s_c_non_ob)
        global_accuracy += acc

        count += 1

        ###########
        if isDebug>0:
            print("\rHandling context %d of %d" % (count, len(contex_set)))
            print 'context:', context,' ','curr_acc:', acc,' ','avg_acc：', global_accuracy / count
            if int(count) >= int(isDebug):
                break                
        ##############


    global_accuracy = global_accuracy / count

    return global_accuracy


def loadWeight(weight_file):
    fo = open(weight_file, 'rb')
    W = pickle.load(fo)
    Z = pickle.load(fo)
    fo.close()
    return W,Z

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-iter_num', help='the iter num,uesd to fetch weight_file all weight files should be in the same folder', dest='iter_num', default=0)
       
    parser.add_argument('-eval_num', help='the context number to be evaluated ', dest='eval_num', default=1,type=int)
    parser.add_argument('-datapath', help='the place where the original data files,like u.user,u1.base, are placed ', dest='datapath', default='/home/zju/dgl/dataset/recommend/ml-100k/')
    
    parser.add_argument('-weightpath', help='the place where all the weight files created during certain training process are placed ', dest='weightpath', required=True)
    
    parser.add_argument('-flod_name', help='the test flod name,like u1,u2,u3..', dest='flod_name', required=True)
    
    
    args = parser.parse_args()
    
    isDebug=args.eval_num
    datapath = args.datapath
    weight_file=args.weightpath+'u1_iter_'+args.iter_num+'_save_weight.pkl'   
    flod_name = args.flod_name
    test_data_file = datapath + flod_name + '.test'
    
    user_data_file = datapath + 'u.user'
    item_data_file = datapath + 'u.item'
      
    
    print '##############Evaluation: 参数.....##############'
    
    print args
    
    print '##############Evaluation: loading data.....##############'

    contex_set, df_test_data = getContextset_and_Test_data(test_data_file)
    items_set = getItemset(item_data_file)
    userIdOneHot, itemIDsOneHot = getUser_and_item_onehot(user_data_file, item_data_file)
    total_dim = userIdOneHot.shape[1] + itemIDsOneHot.shape[1]
    W, Z=loadWeight(weight_file)

    print '##############Evaluation: evaluating data.....##############'
    auc=evalu(W,Z,df_test_data, contex_set, items_set,
            userIdOneHot, itemIDsOneHot, total_dim,isDebug=isDebug)

    print '##############Final AUC for iter-'+args.iter_num+':'+str(auc)+'##############'






