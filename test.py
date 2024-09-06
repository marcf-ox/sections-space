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
from main import compute_sections,sections_naive
#graph
import networkx as nx

import matplotlib.pyplot as plt


#test on Erdos Renzi dense
def test_SG():
    range_test=range(5,305,5)
    n_test=15
    n_max_naive=45
    ear_fun=lambda Q: len(SG_to_tree(Q,True)[0][0])
    test_plot(generate_SG_graphs(range_test,n_test),sections_naive,
              ear_fun,range_test,n_max_naive,n_test,False,title="computation time for a SC quiver (k=5,ER p=2/n^0.8)")
    
#test on acyclic graph
def test_arboreal():
    range_test=range(5,105,5)
    n_test=15
    n_max_naive=45
    adv_fun= lambda Q: arboreal_out(Q,False)
    test_plot(generate_acyclic(range_test,n_test),sections_naive,adv_fun,
              range_test,n_max_naive,n_test,False,title="computation time for an acyclic quiver (k=5,m=2n^1.2)")

#test on a generalised Kronecker quiver
def test_arboreal_equalizer():
    #Equaliser
    k=20
    M=np.floor(1.01*np.random.random((k,k)))
    Q=Quiver(2,[[0,1],[0,1]],[k,k],[np.eye(k,dtype=float),np.eye(k,dtype=float) +  M])
    
    assert(len(scipy.linalg.null_space(M)[0])==sections_naive(Q))
    assert(len(scipy.linalg.null_space(M)[0])==arboreal_out(Q))

#computation of classical limits
def test_classics():
    print("=====================================================")
    print("pullback")
    Q=Quiver(3,[[0,2],[1,2]],[3,2,2],[np.array([[1,0,1],[1,1,0]]),np.eye(2)],Field('R'))
    print(Q)
    print_limit(compute_sections(Q))
    print("=====================================================")
    print("equalizer")
    Q=Quiver(2,[[0,1],[0,1]],[3,3],[np.array([[1,0,1],[0,1,0],[0,1,1]]),np.array([[1,1,0],[0,1,0],[0,0,2]])],Field('R'))
    print(Q)
    print_limit(compute_sections(Q))
    print("=====================================================")
    print("different fields")
    for field in [Field("Q"),Field("F_2")]:
        mat_11_temp=eye_mat(2,field)
        mat_11_temp[0][0]= field.one+field.one+field.one
        Q=Quiver(2,[[0,1],[1,1]],[2,2],[eye_mat(2,field),mat_11_temp],field)
        print(Q)
        print_limit(compute_sections(Q,field),field)

#performance test for large random graphs
def test_general():
    range_test=list(range(2,10,2))+list(range(10,75,5))
    n_test=3
    adv_fun=lambda Q: compute_sections(Q)[0]
    test_plot(generate_random_quiver(range_test,n_test,False,Field('F_2')),sections_naive,adv_fun,range_test,7,n_test,log=True,title="computation time for a general quiver (k=5,ER p=2/n^0.8)")
    plt.show()
    
#test for the complex field
def test_C():
    field=Field("C")
    E=[[7, 17], [18, 20], [22, 24],  [1,1], [23, 19], [16, 2], [2, 11], [23, 11], [9, 3],  [3, 2], [12, 3], [15, 19], [22, 18], [20, 9], [6, 0], [23, 10], [3, 10], [10, 20], [7, 5], [7, 23], [6, 14],   [4, 24], [1, 6], 
        [5, 18], [22, 7],  [6, 5],   [19, 6],  [19, 8],[12, 18],[15,21],[7,10],[18,15], [11, 19],  [22, 9]]
    for _ in range(10):
        E.append([np.random.randint(25),np.random.randint(25)])
    G=nx.DiGraph()
    G.add_nodes_from(range(61))
    for i_e,e in enumerate(E):
        color2='b'
        if i_e in [0,len(E)-1,len(E)-12]:
            color2='r'
        G.add_edge(e[0],e[1],color=color2)
    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    #nx.draw(G,with_labels=True,edge_color=colors)
    AE=[eye_mat(5,field) for _ in E]
    AE[0][1][4]=1
    AE[-1][0][0]=2
    AE[-4][4][2]=1
    AE[2][4][2]=1j
    AE[6][4][2]=1
    n=25
    k=5
    
    Q=Quiver(n,E,[k for _ in range(n)],AE,field)
    
    
    #print(Q)
    print("adv")
    print(compute_sections(Q)[0])
    print("naive")
    print("=",sections_naive(Q))


   
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
    
test_general()
