# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import pdb


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
    
    flag=1
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
    else:
        mAP=0
        all_cmc=[0]
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
        f=open('/home/xiaodui/match_results/test_b_Ranlist_1128.txt','w')      
        f.write(str(data))
        f.close()
               

    return all_cmc, mAP
