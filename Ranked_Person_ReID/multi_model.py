from collections import OrderedDict
import pdb
import json
import numpy as np
import torch


fr1=open("Rank_dist_tA.txt",'r+')
dic_R = eval(fr1.read())

fr2= open("MHN_dist_tA.txt",'r+') #MGN_dist.txt Rank_dist.txt,PCB_dist.txt,MHN_dist.txt
dic_H = eval(fr2.read())   #读取的str转换为字典

fr3 = open("MGN_dist_tA.txt",'r+') #MGN_dist.txt Rank_dist.txt,PCB_dist.txt,MHN_dist.txt
dic_G = eval(fr3.read())   #读取的str转换为字典

def get_dist(dic_txt):  #read .txt (type:list),then transfer to different list
    q_names=[]
    g_names=[]
    dists=[]
    
    for dic in dic_txt:

        key=dic[0]
        dic2=dic[1]
        q_names.append(key)
        
        g_name=[]
        dist=[]
        
        for key2,d in dic2.items():
            #print(key2)
            
            g_name.append(key2)
            dist.append(d)
        g_names.append(g_name)
        dists.append(dist)
    return q_names,g_names,dists

def dist_process(dist):
    dists2=[]
    for dd in dist:
        max_d=max(dd)
        dist2=[]
        for i in range(len(dd)):
            norm_d=dd[i]/max_d
            dist2.append(norm_d)
        dists2.append(dist2)
        
    return dists2

def dist_process2(dist):
    dists2=[]
    for dd in dist:
    #dd=dist
        max_d=max(dd)
        min_d=min(dd)
        dd=torch.Tensor(dd)
        dist2=[]
        pdb.set_trace()
        norm_d=torch.mul(torch.add(dd,(-min_d)),(1/(max_d-min_d)))
        dist2=norm_d
        dists2.append(dist2)
        
    return dists2

def dist_process3(dist):
    dists2=[]
    for dd in dist:
    #dd=dist
        max_d=max(dd)
        min_d=min(dd)
        dd=torch.Tensor(dd)
        dist2=[]
        #pdb.set_trace()
        norm_d=torch.add(-(torch.mul(torch.add(dd,(-min(dd))),(1/(max_d-min_d)))),1)
        dist2=norm_d
        dists2.append(dist2)
        
    return dists2

def dist_process2_d(dist):
    dd=dist
    max_d=max(dd)
    min_d=min(dd)
    dd=torch.Tensor(dd)
    norm_d=torch.mul(torch.add(dd,(-min_d)),(1/(max_d-min_d)))
        
    return norm_d

def dist_process3_d(dist):
    dd=dist
    max_d=max(dd)
    min_d=min(dd)
    dd=torch.Tensor(dd)
    norm_d=torch.add(-(torch.mul(torch.add(dd,(-min(dd))),(1/(max_d-min_d)))),1)      
    return norm_d

    
def compare_d(dist1,dist2,g_names1,g_names2,m,n): #m=len(q_names) n=len(top200)
    dists3=[]
    fgnames=[]
    alph=1.0 #mgn
    beta=0 #rank
    for i in range(m):
        dist3=[]
        fgname=[]
        for j in range(n):
            if ((dist1[i][j]*alph)<=(dist2[i][j]*beta)):
                better_d=dist1[i][j]
                better_name=g_names1[i][j]
            else:
                better_d=dist2[i][j]
                better_name=g_names2[i][j]                
                
                
            dist3.append(better_d)   
            fgname.append(better_name)    
            
        dists3.append(dist3)
        fgnames.append(fgname) 
    return dists3,fgnames

    
def compare_d2(dist1,dist2,q_names,g_names1,g_names2,m,n): #m=len(q_names) n=len(top200) compare dist return q-id,gid
    dists3,fgnames=[],[]
    q_ids,q_cams,q_files,g_ids,g_cams,gfiles=[],[],[],[],[],[]
    
    gooddist=dist1
    w1=0
    w2=1
    alph=1-w1#mgn
    beta=1-w2#rank 
    for i in range(m):
        
        qq=q_names[i].split('_')
        q_ids.append(qq[0])
        q_cams.append(qq[1])
        q_files.append(qq[2])

        dist3,fgname=[],[]
        g_id,g_cam,gfile=[],[],[]
        for j in range(n):
            if (j>=0):
            
                if ((dist1[i][j]*alph)<=(dist2[i][j]*beta)):
                    better_d=dist1[i][j]
                    better_name=g_names1[i][j]
                else:
                    better_d=dist2[i][j]
                    better_name=g_names2[i][j]   
            else:
                better_d=dist1[i][j]
                better_name=g_names1[i][j]                 
                
                
            dist3.append(better_d)   
            fgname.append(better_name)
            gg=better_name.split('_')
            g_id.append(gg[0])
            if(gg[1].isdigit()):
                g_cam.append(gg[1])
            else:
                g_cam.append(gg[1][1])
            gfile.append(gg[2])
            
        dists3.append(dist3)
        fgnames.append(fgname) 
        g_ids.append(g_id)
        g_cams.append(g_cam)
        gfiles.append(gfile)
    return dists3,q_ids,q_cams,q_files,g_ids,g_cams,gfiles,fgnames

def compare_d4(dist1,dist2,q_names,g_names1,g_names2,m,n): #m=len(q_names) n=len(top200) compare dist return q-id,gid
    dists3,fgnames=[],[]
    q_ids,q_cams,q_files,g_ids,g_cams,gfiles=[],[],[],[],[],[]

        
    dista=dist_process2(dist1)
    distb=dist_process3(dist2)
        
    
    gooddist=dist1
    w1=1
    w2=0.8
    alph=1
    beta=0
    for i in range(m):
        
        qq=q_names[i].split('_')
        q_ids.append(qq[0])
        q_cams.append(qq[1])
        q_files.append(qq[2])

        dist3,fgname=[],[]
        g_id,g_cam,gfile=[],[],[]
        for j in range(n):  
            dist_f=dista[i][j]*alph+distb[i][j]*beta
            #pdb.set_trace()                                        
            dist3.append(dist_f)
        dists3.append(dist3)
        index=np.argsort(dist3,axis=0)
#        pdb.set_trace()
        for j in range(n):
            better_name=g_names1[i][index[j]]
            fgname.append(better_name)
            gg=better_name.split('_')
            g_id.append(gg[0])
            if(gg[1].isdigit()):
                g_cam.append(gg[1])
            else:
                g_cam.append(gg[1][1])
            gfile.append(gg[2])
            
        
        fgnames.append(fgname) 
        g_ids.append(g_id)
        g_cams.append(g_cam)
        gfiles.append(gfile)
    return dists3,q_ids,q_cams,q_files,g_ids,g_cams,gfiles,fgnames



def normalize_curve(score_sorted, ref, topN):
    score_norm = score_sorted - ref
    score_norm = (score_norm - min(score_norm[:topN]) + 0.000000000001) / \
                 (max(score_norm[:topN]) - min(score_norm[:topN]) + 0.000000000001)
    return score_norm

def get_weights(dist1,dist2,m,n): #m=len(q_names) n=len(top200) compare dist return q-id,gid            
    #  select 1-knn reference curves in reference curves
    #  build a knn references matrix
 
    import numpy as np
    import matplotlib.pyplot as plt
    
    weight=[]

    for i in range(m):

        score1_sorted = sorted(dist1[i], reverse=True)
        score2_sorted = sorted(dist2[i], reverse=True)
        
        i1= np.argsort(dist1)
        i2= np.argsort(dist2)
        
        
        ref_Rank=np.array(dist1)
        ref_Rank_selected = ref_Rank[0,:]
        
        ref_M=np.array(dist2)
        ref_M_selected = ref_M[0,:]
        
        knn = 10
        topN=335
        
        for tt in range(71):
            ref_Rank_selected = np.vstack((ref_Rank_selected, ref_Rank[tt]))
       # print (ref_Rank_selected.shape)
    
        for tt in range(71):
            ref_M_selected = np.vstack((ref_M_selected, ref_M[tt]))

        ref_Rank_final = np.mean(ref_Rank_selected, axis=0)
        #print('Rank reference score curve final', ref_Rank_final.shape)
    
        ref_M_final = np.mean(ref_M_selected, axis=0)
        #print ('MGN reference score curve final', ref_M_final.shape)
 
        #  normalize the test score curves with calculated reference
        score1_norm = normalize_curve(np.array(score1_sorted), ref_Rank_final, topN)
        score2_norm = normalize_curve(np.array(score2_sorted), ref_M_final, topN)
    
        #  calculate the area under the normalized score curve
        minN=1
        
        area1 = sum(score1_norm[minN:topN]) + 0.000000000001
        area2 = sum(score2_norm[minN:topN]) + 0.000000000001
        
        #  calculate the weights for each feature
        a = [(1 / area1) / (1 / area1 + 1 / area2), (1 / area2) / (1 / area1 + 1 / area2)]
        weight.append(a)
    print ('weights done!')
    return weight

def compare_d3(dist1,dist2,dist3,q_names,g_names1,g_names2,g_names3,m,n): #ists_r,dists_h,dists_g
    
    dists4,fgnames,data=[],[],[]
    q_ids,q_cams,q_files,g_ids,g_cams,gfiles=[],[],[],[],[],[]
    weight=get_weights(dist2,dist3,m,n)

    for i in range(m):
        a=weight[i]
        w2=a[0]
        w3=a[1]
        
        qq=q_names[i].split('_')
        q_ids.append(qq[0])
        q_cams.append(qq[1])
        q_files.append(qq[2])
        
        dista=dist_process2_d(dist1[i])
        distb=dist_process3_d(dist2[i])
        distc=dist_process2_d(dist3[i])

        dist4,fgname=[],[]
        g_id,g_cam,gfile=[],[],[]        
        
        dist_f=dist1
        w1=0.81
        #dist_f[i] =  dista**w1*distb**((1-w1)*w2) * distc**((1-w1)*w3)
        dist_f[i]=torch.add(torch.mul(dista,w1),torch.add(torch.mul(distb,(1-w1)*w2),torch.mul(distc,(1-w1)*w3)))
        index=np.argsort(dist_f[i]) #[:200])
        for mm in range(200):
           # pdb.set_trace()
            better_d=dist_f[i][index[mm]]
            better_name=g_names1[i][index[mm]]
                  
            #print(better_d)    
            dist4.append(better_d)   
            fgname.append(better_name)
            gg=better_name.split('_')
            g_id.append(gg[0])
            if(gg[1].isdigit()):
                g_cam.append(gg[1])
            else:
                g_cam.append(gg[1][1])
            gfile.append(gg[2])
            
        dists4.append(dist4)
        fgnames.append(fgname) 
        g_ids.append(g_id)
        g_cams.append(g_cam)
        gfiles.append(gfile)
        data.append({qq[2]:gfile})
        f=open('test_fusion_A.txt','w')      
        f.write(str(data))
        f.close()
   # pdb.set_trace()
    return dists4,q_ids,q_cams,q_files,g_ids,g_cams,gfiles,fgnames
                
    
def get_result():
    qnames_r,gnames_r,dists_r=get_dist(dic_R)   
    qnames_h,gnames_h,dists_h=get_dist(dic_H)
    qnames_g,gnames_g,dists_g=get_dist(dic_G)

    dists3,q_ids,q_cams,q_files,g_ids,g_cams,gfiles,fgnames=compare_d3(dists_r,dists_h,dists_g,qnames_r,gnames_r,gnames_h,gnames_g,len(qnames_r),200)
    return dists3,q_ids,q_cams,q_files,qnames_r,g_ids,g_cams,gfiles,fgnames

fr1.close()
fr2.close()
