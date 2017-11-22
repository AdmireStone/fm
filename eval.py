#-*-coding:utf8-*-

'''
copy from with_linerterm_metric

'''
import gc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse.linalg import LinearOperator, eigsh,eigs

class Evaluation():
    
    def init_context_set(self,user_moive_tag):
        contex_set_df=user_moive_tag[['userID','movieID']].drop_duplicates()
        contex_set=set() #(userID,moviveID)
        for index in contex_set_df.index:
            tuple=(contex_set_df.loc[index]['userID'],contex_set_df.loc[index]['movieID'])
            contex_set.add(tuple)
        return contex_set

    def __init__(self,tags_file,user_movies_tag_file,sparseOnehot_file,W,Z):
        '''
        :param W the linear term weight
        :param Z the quadratic weight
            
        '''

        # 生成tags
        self.tags=pd.read_table(tags_file)
        self.tags_set=set(self.tags['id'].values)
        self.user_moive_tag=pd.read_table(user_movies_tag_file)
        del self.user_moive_tag['timestamp']
        
        print 'init contex_set'
        # 生成context_set
        self.contex_set=self.init_context_set(self.user_moive_tag)    

        # 加载数据
        fi=open(sparseOnehot_file,'rb')
        self.userID_sparseOnehot=pickle.load(fi)
        self.movieID_sparseOnehot=pickle.load(fi)
        self.tags_sparseOnehot=pickle.load(fi)

        print 'init sparseOnehot'
        #注意df_userID_unique和df_movieID_unique是Series,不是DataFrame
        self.df_userID_unique=self.user_moive_tag['userID'].drop_duplicates()
        self.df_movieID_unique=self.user_moive_tag['movieID'].drop_duplicates()
    
        self.d_tagID=self.tags_sparseOnehot.shape[1]
        self.d_userID=self.userID_sparseOnehot.shape[1]
        self.d_movieID=self.movieID_sparseOnehot.shape[1]
        self.total_dim=self.d_userID+self.d_movieID+self.d_tagID
        
        self.global_auc=0.0
        
        self.W=W
        self.Z=Z
        
        print 'init complete'
        
    def getUserOnehot(self,userID):

        index=self.df_userID_unique[self.df_userID_unique==userID].index.values[0]
        return self.userID_sparseOnehot.loc[index].values

    def getMovieOnehot(self,movieID):

        index=self.df_movieID_unique[self.df_movieID_unique==movieID].index.values[0]
        return self.movieID_sparseOnehot.loc[index].values

    def getTagOnehot(self,tagID):
        index=self.tags[self.tags['id']==tagID].index.values[0]
        return self.tags_sparseOnehot.loc[index].values
    
    def get_c_tag_matrix(self,tags_set,user_sp_vec,movie_sp_vec,total_dim):
        '''
        :param tags_set
        :param context should be tuple,(userID,movieID)
        :return a sparse matrix，in which each row is the onehot(userID,movieID,tagID)
        '''
        sparse_matrix=None
        # 1. 首先获取userID，movieID，tagID的 index
        # 2. 根据各个index 获取对应onehot 编码
        # 3. 拼接onehot 编码
        # 将拼接好的追加到矩阵中
        for tagID in tags_set:
            tagOneHot=getTagOnehot(tagID)       
            tag_sp_vec=sp.csc_matrix(tagOneHot.to_dense(),(1, d_tagID))
            x=sp.hstack([user_sp_vec,movie_sp_vec,tag_sp_vec])        
            sparse_matrix=sp.vstack([sparse_matrix,x])

        sparse_matrix=sp.csc_matrix(sparse_matrix)   
        return sparse_matrix


    def getRelevenceScore(self,X,W,Z):
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

    def getSingleAUC(self,s_c_ob,s_c_non_ob):
        acc=0.0
        for s_ci in s_c_ob:
            acc+=np.sum(s_c_ob<s_ci)
        print 'acc:',acc,'len(s_c_ob)*len(s_c_non_ob)',len(s_c_ob)*len(s_c_non_ob)
        return acc/(len(s_c_ob)*len(s_c_non_ob))
    
    
    def evalu(self):
        
        count=0
        for context in self.contex_set:
            # oberved tags
             positive_tags=set(self.user_moive_tag[(self.user_moive_tag['userID']==context[0]) & 
                              (self.user_moive_tag['movieID']==context[1])]['tagID'].values)

                                                                                           
            #positive_tags=set(self.user_moive_tag[(self.user_moive_tag['userID']==127) & 
            #                   (self.user_moive_tag['movieID']==2080)]['tagID'].values)


            # non-observed tags
            negative_tags=self.tags_set-positive_tags

            # 下面两个是pandas.sparse.SparseArray
            userOneHot=self.getUserOnehot(context[0])
            movieOneHot=self.getMovieOnehot(context[1])

            # 转换为sparse 矩阵
            user_sp_vec=sp.csc_matrix(userOneHot.to_dense(),(1, self.d_userID))
            movie_sp_vec=sp.csc_matrix(movieOneHot.to_dense(),(1, self.d_movieID))
            
                        
            observed_c_tag_matrix = self.get_c_tag_matrix(positive_tags,user_sp_vec,movie_sp_vec,self.total_dim)  
            non_observed_c_tag_matrix = self.get_c_tag_matrix(negative_tags,user_sp_vec,movie_sp_vec,self.total_dim)

            
            # 计算评分
            s_c_ob=self.getRelevenceScore(observed_c_tag_matrix,W,Z)
            s_c_non_ob=self.getRelevenceScore(non_observed_c_tag_matrix,W,Z)

            # 统计    
            self.global_auc+=self.getSingleAUC(s_c_ob=s_c_ob,s_c_non_ob=s_c_non_ob)
            
            ##################
#            self.global_auc+=self.getSingleAUC(s_c_ob=s_c_ob,s_c_non_ob=s_c_ob)
#            count+=1
#            if count >10:
#                break
#            print 'processing:',context,count,s_c_ob,self.global_auc
            ##################
                   
        self.global_auc=self.global_auc/len(contex_set)    
        return self.global_auc
    


def load_weight(weight_file):
    fo=open(weight_file,'wb')
    W=pickle.load(fo)
    Z=pickle.load(fo)
    return W,Z

if __name__=='__main__':
    
    weight_file=''
    W,Z=load_weight(weight_file)
    evaluation=Evaluation(tags_file,user_movies_tag_file,sparseOnehot_file,W,Z)
    print evaluation.evalu()