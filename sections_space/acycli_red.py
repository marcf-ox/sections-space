#linear algebra
from sections_space.aux_fun import intersection,row_echelon,col_echelon,solve_triangular,null_space
#Quiver and matrices in fields
from sections_space.aux_fun import Quiver,E_in_out,zeros_mat,eye_mat,is_all_zero_mat
from sections_space.strongly_connected import SG_to_tree
import numpy as np
import networkx as nx
import copy
epsilon=1e-12

# compute the inverse image of M restricted to Im(M)\cap K
def inverse_image(M,K,field):
    #column echelon form
    Img_M=col_echelon(M,field)[0]
    #eliminate zero columns
    if field.descr in ['R','C']:
        non_zero_cols=np.where(np.max(np.abs(Img_M),axis=0)>epsilon)[0]
    else:
        non_zero_cols=[i  for i,M_col_i in enumerate(list(Img_M.transpose())) if not(is_all_zero_mat(M_col_i,field)) ]
    Img_M=Img_M[:,non_zero_cols]
    #basis of Im(M)\cap K
    Img_inter=intersection(Img_M,K,field)
    #solve Mx=y for y in Im(M)\cap K
    x=inverse_image_vect(M,Img_inter,field)
    ker = null_space(M,field)#add a basis of ker(M)
    return np.concatenate((x, ker),axis=1)
    
#solve Mx=y
def inverse_image_vect(M,y,field):
    #empty matrix
    if M.shape[1]*M.shape[0]==0:
        return np.array([]).reshape(M.shape[1],0)       
    #column echelon form of the augmented matrix
    Augmented_mat,pivots=row_echelon(np.concatenate((M,y),axis=1),field)
    M_ech=Augmented_mat[:,:len(M[0])]
    #no solution
    if len(pivots)>0 and pivots[-1]>= np.shape(M)[1]:
        return np.array([]).reshape(np.shape(M)[0],0)
    #tranform into square invertible triangular matrix and solve
    non_zero_rows = np.array([i for i in range(len(M_ech)) if not(is_all_zero_mat(M_ech[i], field))],dtype=int)
    x_part= solve_triangular(M_ech[non_zero_rows][:,np.array(pivots,dtype=int)],Augmented_mat[non_zero_rows][:,M.shape[1]:] ,field)
    #reintegrate 0 rows and non pivots
    x=zeros_mat(M.shape[1],x_part.shape[1],field)
    for i_p,p in enumerate(pivots):
        x[p]=x_part[i_p]
    return x
    
    

def acyclic_red(Q):
    field=Q.field
    #find strongly connected components
    G=nx.DiGraph()
    G.add_nodes_from(range(Q.n))
    G.add_edges_from(Q.E)
    SG_compo,sub_nodes,sub_edges=[],[],[]
    ind_inv=[[-1,-1] for _ in range(Q.n)]#keep track of the decomposition
    for i_compo,compo in enumerate(nx.strongly_connected_components(G)):
        sub_nodes.append(np.array(list(compo)))
        sub_edges.append([i_e for i_e,e in enumerate(Q.E) if (e[0] in compo and e[1] in compo)])
        
        for i_v,v in enumerate(sub_nodes[-1]):
            ind_inv[v]=[i_compo,i_v]
        renamed_edges=[[ind_inv[Q.E[i][0]][1],ind_inv[Q.E[i][1]][1]] for i in sub_edges[-1]]
        SG_compo.append(Quiver(len(sub_nodes[-1]),renamed_edges,
                                  [Q.Av[x] for x in sub_nodes[-1]],[Q.Ae[x] for x  in sub_edges[-1]],field))
    Q_star=Q
    #apply SG_to_tree to each SC component
    roots,E_star,Av_star,AE_star=[],[],[eye_mat(k,field) for k in Q.Av],[]
    for i_R,R in enumerate(SG_compo):
        K,T,r=SG_to_tree(R,True)
        roots.append(sub_nodes[i_R][r])
        E_star.extend([[sub_nodes[i_R][e[0]],sub_nodes[i_R][e[1]]] for e in T.E])
        Av_star[roots[-1]]=K
        AE_star.extend(T.Ae)
    #add edges not in a stronggly connected component
    for i_e,e in enumerate(Q.E):
        if ind_inv[e[0]][0]!=ind_inv[e[1]][0]:
            E_star.append(e)
            AE_star.append(Q.Ae[i_e])
    #transform Av,Ae into a valid representation Av*,Ae*
    Q_star=Quiver(Q.n,E_star,Q.Av,AE_star,field)
    E_in,E_out=E_in_out(Q_star)
    for i_R,R in enumerate(SG_compo):    
        stack=[roots[i_R]]
        while len(stack)!=0:
            v=stack.pop()
            for i_e in E_in[v]:
                Av_star[Q_star.E[i_e][0]]=intersection(Av_star[Q_star.E[i_e][0]],inverse_image(copy.deepcopy(Q_star.Ae[i_e]),Av_star[v],field),field  )
                if not(Q_star.E[i_e][0] in stack):
                    stack.append(Q_star.E[i_e][0])
    # transform from subspaces of Av to field^d
    Av_star_dim=[len(A[0]) for A in Av_star]
    Ae_star=[inverse_image_vect(Av_star[Q_star.E[i_y][1]], np.dot(y,Av_star[Q_star.E[i_y][0]]),field) for (i_y,y) in enumerate(Q_star.Ae)]
    for i in range(len(Ae_star)):
        if Ae_star[i].shape[0]*Ae_star[i].shape[1]==0:
            Ae_star[i]=np.zeros((Av_star_dim[Q_star.E[i][1]],Av_star_dim[Q_star.E[i][0]]))
    Q_star.Av=Av_star_dim
    Q_star.Ae=Ae_star
    return Q_star,Av_star
        
