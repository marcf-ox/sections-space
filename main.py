
#linear algebra
import numpy as np
import scipy.linalg
from aux_fun import proj, inj, intersection,null_space,sum_subspaces,extract_basis
#quiver and matrix in fields
from aux_fun import Field,zeros_mat,eye_mat,Quiver
#display
from aux_fun import print_limit,test_plot, test_plot_by_m
#steps of advanced algo
from strongly_connected import generate_SG_graphs,SG_to_tree
from arboreal import arboreal_out,generate_acyclic
from acycli_red import acyclic_red
#graph
import networkx as nx

import matplotlib.pyplot as plt


epsilon=1e-12

temp_list=[]


    

def sections_naive(Q):
    field=Q.field
    #compute dimensions in sum of Av
    partial_sum_Av=[0]
    for i in range(Q.n):
        partial_sum_Av.append(partial_sum_Av[-1]+Q.Av[i])
    Gamma= eye_mat(partial_sum_Av[-1],field)
    
    for i_e,e in enumerate(Q.E):
        pi_se=proj(partial_sum_Av[e[0]],Q.Av[e[0]],partial_sum_Av[-1],field)
        pi_te=proj(partial_sum_Av[e[1]],Q.Av[e[1]],partial_sum_Av[-1],field)
        #print("mu",null(pi_te-np.matmul(Q.Ae[i_e],pi_se)))
        Gamma = intersection(Gamma,null_space(pi_te-np.matmul(Q.Ae[i_e],pi_se),field),field=field)
        #temp_list.append(len(Gamma[0]))
    return len(Gamma[0])

def cosections_naive(Q):
    field=Q.field
    #compute dimensions in sum of Av
    partial_sum_Av=[0]
    for i in range(Q.n):
        partial_sum_Av.append(partial_sum_Av[-1]+Q.Av[i])
    Delta= zeros_mat(partial_sum_Av[-1],1,field)
    
    for i_e,e in enumerate(Q.E):
        i_se=inj(partial_sum_Av[e[0]],Q.Av[e[0]],partial_sum_Av[-1],field)
        i_te=inj(partial_sum_Av[e[1]],Q.Av[e[1]],partial_sum_Av[-1],field)
        Delta = sum_subspaces(Delta,extract_basis(i_se-np.dot(i_te,Q.Ae[i_e]),field),field)
    return partial_sum_Av[-1]-len(Delta[0])  

def compute_sections(Q,CC_separated=True):
    field=Q.field
    #all connected components together
    if not(CC_separated):
        Q_star,Av_star=acyclic_red(Q)
        return arboreal_out(Q_star,True)     
    #weakly connected components
    G=nx.DiGraph()
    G.add_nodes_from(range(Q.n))
    G.add_edges_from(Q.E)
    weak_compos,sub_nodes,sub_edges=[],[],[]
    ind_inv=[[-1,-1] for _ in range(Q.n)] #reordering
    for i_compo,compo in enumerate(nx.weakly_connected_components(G)):
        sub_nodes.append(np.array(list(compo)))
        sub_edges.append([i_e for i_e,e in enumerate(Q.E) if (e[0] in compo and e[1] in compo)])
        for i_v,v in enumerate(sub_nodes[-1]):
            ind_inv[v]=[i_compo,i_v]
        renamed_edges=[[ind_inv[Q.E[i][0]][1],ind_inv[Q.E[i][1]][1]] for i in sub_edges[-1]]
        weak_compos.append(Quiver(len(sub_nodes[-1]),renamed_edges,
                                  [Q.Av[x] for x in sub_nodes[-1]],[Q.Ae[x] for x  in sub_edges[-1]],field))
    #section computation by weakly connected compo
    lim,maps=[],[]
    for Q_c in weak_compos:
        Q_star,Av_star=acyclic_red(Q_c)
        lim_c,maps_c=arboreal_out(Q_star,True) 
        for i in range(len(maps_c)):
            if maps_c[i].shape[0]*maps_c[i].shape[1]==0:
                maps_c[i]=zeros_mat(Q_c.Av[i],lim_c,field)
            else:
                maps_c[i]=np.dot(Av_star[i], maps_c[i])
        lim.append(lim_c)
        maps.append(maps_c)
    lim_final=sum(lim)
    maps_final=[]
    for v in range(Q.n):
        i_compo=ind_inv[v][0]
        blocks=[zeros_mat(Q.Av[v],sum(lim[:i_compo]),field), maps[i_compo][ind_inv[v][1]],zeros_mat(Q.Av[v],sum(lim[i_compo+1:]),field)]
        maps_final.append(np.concatenate(blocks,axis=1))
    return lim_final,maps_final


   
def generate_random_quiver(range_test,n_test,by_m,field):
    k=5
    Q_test_all=[]
    #generate examples
    for n_or_m in range_test:
        n=n_or_m
        m=int(2*n**1.2)        
        if by_m:
            m=n_or_m
            n=25
        Q_test=[]
        for _ in range(n_test):
            #erdos-renyi
            E=[]
            G=nx.gnp_random_graph(n, m/float(n*(n-1)), seed=None, directed=True)
            E=[[e[0],e[1]] for e in G.edges]
            eta=1./(2*k*(1+len(E)))
            if field.descr in ['R','C']:
                AE=[np.eye(k,dtype=float)+np.floor((1+eta)*np.random.random((k,k)))for _ in E]        
            else:
                AE=[eye_mat(k,field) for _ in E]
                for i_E in range(len(E)):
                    if np.random.random()<eta*k*k:
                        AE[i_E][np.random.randint(k)][np.random.randint(k)]=field.one
            np.random.shuffle(E)
            Q_test.append(Quiver(n,E,list(k*np.ones(n,dtype=int)),AE,field))
        Q_test_all.append(Q_test)
    return Q_test_all

from cfractions import Fraction
#create quiver from networkx graph 
#maps as "map" edge data
#dimension of spaces of singletons as "dim" node data
def nx_graph_to_quiver(G,field):
    G=nx.convert_node_labels_to_integers(G)
    n=len(list(G.nodes()))
    E,AE=[],[]
    spaces= np.zeros(n)
    for e in G.edges.keys():
        E.append(list(e))
        AE.append(G.edges[e]["map"])
        spaces[e[0]]= AE[-1].shape[1]
        spaces[e[1]]= AE[-1].shape[0]
    #dimension of singletons
    for i,dim_i in G.nodes("dim"):
        if dim_i!=None:
            spaces[i]= dim_i
    return Quiver(n, E,list(spaces),AE,field)
        




field0= Field("Q")
edges = [("v"+str(i),"v"+str((i+1)%4), {"map": np.array([Fraction(1,i%2+1)]).reshape((1,1))}) for i in range(4)]
  
G= nx.from_edgelist(edges)
G.add_node("v5", dim=2)
nx.draw(G,arrows=True,with_labels=True)
Q=nx_graph_to_quiver(G,field0)

compute_sections(Q)