# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 18:36:40 2021

@author: marcf
"""

import numpy as np
#linear algebra
from aux_fun import intersection,null_space
#Quiver and matrices in fields
from aux_fun import  E_in_out,Quiver,eye_mat,Field
import networkx as nx


#build ear decomposition of a SG quiver
def find_ear_decompo(Q):
    E_in,E_out=E_in_out(Q)
    v_ear=-1*np.ones(Q.n,dtype=int)#all vertices not seen 
    e_ear=-1*np.ones(len(Q.E),dtype=int)# all edges not seen
    
    #find a cycle
    start=0
    v_ear[start]=0
    e_ear[E_out[start][0]]=0
    v=  Q.E[ E_out[start][0]][1]
    l=[start]#visited vertices
    while(v_ear[v]!=0):
        l.append(v)
        #mark as seen
        v_ear[v]=0
        e_ear[E_out[v][0]]=0
        #go to next vertex
        v=  Q.E[ E_out[v][0]][1]
    
    r=v#first point of the cycle
    #mark as unseen vertices not in the cycle
    for u in l[:l.index(r)]:
        v_ear[u]=-1
        e_ear[E_out[u][0]]=-1
    #build ear decomposition
    ear_num=0#last built ear
    while(len(np.where(v_ear==-1)[0])!=0):
        #edge outgoing from decompo
        e0=None
        for i_e,e in enumerate(Q.E):
            if v_ear[e[0]]>=0 and v_ear[e[1]]==-1:
               e0=i_e
               break
        w=Q.E[e0][1]
        #shortest pasth w-> ear_num
        distances=(2*Q.n+1)*np.ones(Q.n)
        previous_edge=[[] for _ in range(Q.n)]#shortest paths from w to x
        d=0#distance to w
        distances[w]=d
        previous_edge[w]=e0
        e_ear[e0]=ear_num+1
        arrived=False
        while(not arrived):
            for w_cur in  np.where(distances==d)[0]:
                for e_cur in E_out[w_cur]:
                    #if new vertex in the ear decompo choose this path
                    if not(arrived) and v_ear[Q.E[e_cur][1]]!=-1:
                        arrived=True
                        #retrieve the path inductively
                        e_backwards=e_cur
                        while (v_ear[Q.E[e_backwards][0]]==-1):
                            v_ear[Q.E[e_backwards][0]]=ear_num+1
                            e_ear[e_backwards]=ear_num+1
                            e_backwards=previous_edge[Q.E[e_backwards][0]]
                        ear_num+=1
                        break
                    #continue search
                    if distances[Q.E[e_cur][1]]>d+1:
                        distances[Q.E[e_cur][1]]=d+1
                        previous_edge[Q.E[e_cur][1]]=e_cur
                
                if arrived:
                    break
            if arrived:
                break
            d+=1
    for e in np.where(e_ear==-1)[0]:
        e_ear[e]=ear_num+1
        ear_num+=1
    return r,v_ear,e_ear          


#strongly connected quiver to out-tree
def SG_to_tree(Q,only_K):
    field=Q.field#retrieve field
    #trivial quiver
    if len(Q.E)==0:
        if only_K:
            return eye_mat(Q.Av[0],field),Q,0
        return Q
    #choose an ear decomposition
    r,v_ear,e_ear=find_ear_decompo(Q)
    #find terminal edges
    ter_edges=[]
    non_ter_edges=[]
    for i_e,e in enumerate(Q.E):
        if e_ear[i_e]>v_ear[e[1]] or e[1]==r:
            ter_edges.append(i_e)
        else:
            non_ter_edges.append(i_e)

    #build out-tree
    T=Quiver(Q.n,[Q.E[i_e] for i_e in non_ter_edges],Q.Av,[Q.Ae[i_e] for i_e in non_ter_edges],field)
    E_in_T,E_out_T=E_in_out(T)
    #build paths from root
    phi=[[] for _ in range (Q.n)]
    phi[r]=eye_mat(Q.Av[r],field)
    stack=[r]
    while not len(stack)==0:
        v=stack.pop()
        for i_e in E_out_T[v]:
            phi[T.E[i_e][1]]=np.dot(T.Ae[i_e],phi[v])
            stack.append(T.E[i_e][1])
    #compute K
    K=eye_mat(Q.Av[r],field)
    for epsilon in ter_edges:
        K=intersection(K,null_space(phi[Q.E[epsilon][1]]-np.dot(Q.Ae[epsilon],phi[Q.E[epsilon][0]]),field),field=field)
    if only_K:
        return K,T,r
    #return as a quiver
    T.Av[r]=len(K[0])
    for i_e in E_out_T[r]:
        T.Ae[i_e]=np.dot(T.Ae[i_e],K)
    return T        
        
    
#generate examples of SC Quiver with either:
#   -Erdos-Renyi and connecting the SC quiver
#   -Circular graph+random additional edges
def generate_SG_graphs(range_test,n_test,gen_method='ER'):
    k=5
    Q_test_all=[]
    #generate examples
    for n in range_test:        
        Q_test=[]
        for _ in range(n_test):
            #Graph generation
            #erdos-renyi
            E=[]
            G=nx.gnp_random_graph(n, 2./(n**0.8), seed=None, directed=True)
            SG_compo = [compo for compo in nx.strongly_connected_components(G)]
            for x in SG_compo:
                for y in SG_compo:
                    G.add_edge(list(x)[0],list(y)[0])
            E=[[e[0],e[1]] for e in G.edges]
            #circular
            if gen_method=='circular':
                m=int(np.power(n,1.5))+2
                E=[[x,(x+1)%n] for x in range(n)]                
                for _ in range(m-n):
                    E.append([np.random.randint(n),np.random.randint(n)])
            #quiver generation
            AE=[np.eye(k,dtype=int)+np.array(np.floor((1.+1./((1+len(E))*2*k))*np.random.random((k,k))),int) for _ in E]        
            np.random.shuffle(E)
            Q_test.append(Quiver(n,E,list(k*np.ones(n)),AE,Field('R')))
        Q_test_all.append(Q_test)
    return Q_test_all

