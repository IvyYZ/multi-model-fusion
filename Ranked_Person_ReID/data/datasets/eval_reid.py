# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import pdb
from collections import OrderedDict



def eval_func(distmat, q_pids, g_pids, q_camids, g_camids,q_names,g_names, max_rank=200):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    flag=3
    if (flag==0):
        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]
    
            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)
    
            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue
    
            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1
    
            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.
    
            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)
    
        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)
        
    elif(flag==1):
        mAP=0
        all_cmc=[]
        data=[] 
        for i in range(num_q):
            query_name=q_names[i]            
            gallery_name=g_names[indices[i]]
            #pdb.set_trace()
            data.append({query_name:g_names[indices[i]][:200]})
            #json_str=json.dumps(data)
        #with open('/home/xiaodui/zy_re_id/PCB_RPP_for_reID-master/PCB5/test_a_result.json','w') as f: 
           # pdb.set_trace()
            #json.dump(data,f)
           # f.write(json.dumps(data,cls=MyEncoder))
        f=open('/home/xiaodui/match_results/test_a_Ranlist_1111_1.txt','w')      
        f.write(str(data))
        f.close()
        
    elif(flag==2):
        
        dict1={}
        #dict2={}
        list1=[]
        indices2=np.argsort(g_names,axis=0)
       
        for i in range(num_q):
            dict2=OrderedDict()           
            #query_name= q_names[i]
           # pdb.set_trace()
            query_name= str(q_pids[i])+'_'+str(q_camids[i])+'_'+str(q_names[i])
            #q_camid = q_camids[q_idx]            
            #gallery_name=gallery_ids[indices[i]]
            list2=[]
            list3=[] 
            list1.append(query_name)
            indices2=np.argsort(g_names,axis=0)
            for j in range(len(g_pids)):
                #pdb.set_trace()
                #list2.append(g_names[indices[i]][j])
                gallery_name=str(g_pids[indices2[j]])+'_'+str(g_camids[indices2[j]])+'_'+str(g_names[indices2[j]])           
                #gallery_name=str(g_pids[indices[i]][j])+'_'+str(g_camids[indices[i]][j])+'_'+str(g_names[indices[i]][j])              
                list2.append(gallery_name)
                #list3.append(distmat[i][j])
                #list3.append(distmat[i][indices[i]][j])
                list3.append(distmat[i][indices2[j]])
                dict2[list2[j]]=list3[j]
            #pdb.set_trace()
            dict1[list1[i]]=dict2
            #pdb.set_trace()
            #data.append({query_name:gallery_names[indices[i]][:200]})  
        dict_m=sorted(dict1.items(),key=lambda d:d[0])
        f6 = open("Rank_dist_t.txt",'a')
        f6.write(str(dict_m)+'\n')
        f6.close() 
        
    else:
   # compute cmc curve for each query
        import multi_model
        #pdb.set_trace()
        distmat,q_pids,q_camids,qnames,qnames_m,g_pids,g_camids,gnames,fgnames=multi_model.get_result()#qnames,gnames,distmat
        #pdb.set_trace()
        distmat = np.array(distmat)
        q_pids=np.array(q_pids)
        q_camids=np.array(q_camids)
        g_pids=np.array(g_pids)
        g_camids=np.array(g_camids)        
        
        num_q,num_g=distmat.shape
        indices=np.argsort(distmat,axis=1)
       
        
        matches = (g_pids== q_pids[:, np.newaxis]).astype(np.int32)
        
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]
    
            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            #remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            remove = (g_pids[q_idx][order] == q_pid) & (g_camids[q_idx][order] == q_camid)
            keep = np.invert(remove)
    
            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue
    
            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1
    
            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.
    
            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)
    
        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)
  

    return all_cmc, mAP
