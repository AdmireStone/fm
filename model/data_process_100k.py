# -*-coding:utf8-*-
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
import sys
import time



def get_non_interact_items(userID,u_data,min_tagID,max_tagID):
    # oberved items
    c_items_set=set(u_data[u_data['userID']==userID]['itemID'].values)
    
    # 从所有负标签中随机的取出一定数目负标签
    c_negative_items=[]
    set_size=len(c_items_set)
    for _ in range(set_size):     
        while True:
           itemID=np.random.randint(min_tagID,max_tagID) 
           if (itemID not in c_items_set):
               c_negative_items.append(itemID)
               c_items_set.add(itemID) # 防止重复
               break;
    return c_negative_items

def getSytheticTrainData(org_data,user_set,u_data,min_tagID,max_tagID):
    sythetic_data=org_data
    sythetic_data['negItemID']=0    
    for userID in user_set:
        c_neg_items=get_non_interact_items(userID,u_data,min_tagID,max_tagID)
        # 在原始数据中去除包含userID的所有的记录的index
        contex_indecise=sythetic_data[(sythetic_data['userID']==userID)].index.values
        count=0
        for index in contex_indecise:
            sythetic_data.loc[index]['negItemID']=c_neg_items[count]
            count+=1
    return sythetic_data
    
def createTraininData(train_file,userIdOneHot,itemIDOneHot,u_data,user_set,min_tagID,max_tagID,isDebug):
    '''
    userID 和itemID 都是1开始的
    train_file is like u1.base,u1.test and so on.
    if isDebug is true,then just use 
    '''
    slide_data=pd.read_table(names=['userID','itemID','rating','timestamp'],filepath_or_buffer=train_file)
    del slide_data['rating'],slide_data['timestamp']
    d_userID=userIdOneHot.shape[1]
    d_itemID=itemIDOneHot.shape[1]
    print 'sythetic_data...'
    sythetic_data=getSytheticTrainData(slide_data,user_set,u_data,min_tagID,max_tagID)
    print 'sythetic_data ok'
    def createX_ciX_cj(slide_data):
        X_ci=None
        X_cj=None
        total_num=sythetic_data.shape[0]
        for index in sythetic_data.index:
            userID=sythetic_data.loc[index]['userID']
            movieID=sythetic_data.loc[index]['itemID']
            neg_movieID=sythetic_data.loc[index]['negItemID']
            
            oht_userID=sp.csc_matrix(userIdOneHot.loc[userID-1].to_dense().values,dtype='f')
            oht_movieID=sp.csc_matrix(itemIDOneHot.loc[movieID-1].to_dense().values,dtype='f')
            oht_neg_ItemID=sp.csc_matrix(itemIDOneHot.loc[neg_movieID-1].to_dense().values,dtype='f')
            
            x_ci=sp.hstack([oht_userID,oht_movieID])
            if X_ci == None:
                X_ci=x_ci
            else:
                X_ci=sp.vstack([X_ci,x_ci])
    
          ### 负样本生成
        
           #生成x_cj
            x_cj=sp.hstack([oht_userID,oht_neg_ItemID])
            if X_cj == None:
                X_cj=x_cj
            else:
                X_cj=sp.vstack([X_cj,x_cj])
                
            if isDebug:        
                if index > 100:
                    break
                    
            sys.stdout.write("\rHandling context %d of %d" % (index+1,total_num))
            sys.stdout.flush()    
            
        return X_ci,X_cj  
    X_ci,X_cj=createX_ciX_cj(slide_data)
    return X_ci,X_cj
    

def save2pkl(file_name,obj1,obj2):
    fo=open(file_name,'wb')
    pickle.dump(obj1,fo)
    pickle.dump(obj2,fo)
    fo.close()


def data_process(flod_name):
    '''
    return b_X_ci,b_X_cj,t_X_ci,t_X_cj
    '''
    datapath='/home/zju/dgl/dataset/recommend/ml-100k/'
    u_user_fi=datapath+'u.user'
    u_item_fi=datapath+'u.item'
    u_user=pd.read_table(header=None,filepath_or_buffer=u_user_fi,sep='|')
    u_item=pd.read_table(header=None,filepath_or_buffer=u_item_fi,sep='|')
    userID_sparseOnehot=pd.get_dummies(u_user[0],sparse=True)
    itemID_sparseOnehot=pd.get_dummies(u_item[0],sparse=True)
    u_data=pd.read_table(names=['userID','itemID','rating','timestamp'],filepath_or_buffer=datapath+'u.data')
    user_set=set(u_user[0].values)
    
    print '##########################New Data Set Generation:',flod_name,'###########################'
    base_file=datapath+flod_name+'.base'
    test_file=datapath+flod_name+'.test'
    print 'Begin process:'+flod_name+'.base'+'.............'
    
    start=time.time()
    
    b_X_ci,b_X_cj=createTraininData(base_file,userID_sparseOnehot,itemID_sparseOnehot,
                                    u_data,user_set,1,1682,isDebug=False)
    print 'b_X_ci:',b_X_ci.shape,type(b_X_ci)
    print 'b_X_cj:',b_X_cj.shape,type(b_X_cj)
    print '耗时:',(start-time.time())/60,'min'
    
    b_file_name=datapath+flod_name+'_b.pkl'
    save2pkl(b_file_name,b_X_ci,b_X_cj)
  
    print 'Begin process:'+flod_name+'.test'+'.............'
    start=time.time()
    
    t_X_ci,t_X_cj=createTraininData(test_file,userID_sparseOnehot,itemID_sparseOnehot,
                                    u_data,user_set,1,1682,isDebug=False)
    
    print 't_X_ci:',t_X_ci.shape,type(t_X_ci)
    print 't_X_cj:',t_X_cj.shape,type(t_X_cj)
    print '耗时:',(start-time.time())/60,'min'
    
    t_file_name=datapath+flod_name+'_t.pkl'
    save2pkl(t_file_name,t_X_ci,t_X_cj)
    print '##########################Generation End###########################'
      
  
if __name__=='__main__':
    # To DO: 删掉对test数据的处理，因为测试的时候不需要使用合成数据
    data_process('u1')
  
  
  