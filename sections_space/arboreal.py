#linear algebra
from aux_fun import intersection,proj,null_space,flatten_zero
#quiver and matrix in fields
from aux_fun import E_in_out,Quiver,shift_vertices,zeros_mat,eye_mat,Field
import numpy as np

## Reordering vertices so that edges are increasing
def swap_int(x,u,v):
    if x==u:
        return v
    if x==v:
        return u
    return x
        

def swap_vertices(Q,u,v):
    Q.Av[u],Q.Av[v]=Q.Av[v],Q.Av[u]
    Q.E=[[swap_int(e[0],u,v),swap_int(e[1],u,v)] for e in Q.E]


def order_vertices(Q):
    order=list(range(Q.n))
    i_e=0
    while i_e<len(Q.E):
        cur_edge=Q.E[i_e]
        if cur_edge[0]>cur_edge[1]:
            swap_vertices(Q,cur_edge[0],cur_edge[1])
            order=[swap_int(x, cur_edge[0], cur_edge[1]) for x in order]
            #print("swap",cur_edge[0],cur_edge[1],order)
            i_e=-1
        i_e+=1
    return order

## sections of an acyclic quiver
def arboreal_out(Q,maps):
    #solve separately the trivial quiver for speed when m<<n
    if Q.n==1 and len(Q.E)==0:
        if maps:
            return Q.Av[0], [eye_mat(Q.Av[0],Q.field)]
        return Q.Av[0]
    #reorder V to make edges increasing
    order=order_vertices(Q)
    #add a root
    Q=shift_vertices(Q)
    order=[x+1 for x in order]
    
    field=Q.field
    Phi=[np.array([]) for _ in range(Q.n)]
    phi=[np.array([]) for _ in range(Q.n)]
    E_in,E_out=E_in_out(Q)
    #computing minimal and maximal vertices
    V_min,V_max=[],[]
    for v in range(1,Q.n):
        if E_out[v]==[]:
            V_max.append(v)
        if E_in[v]==[]:
            V_min.append(v)
    Av_min=[Q.Av[v] for v in V_min]
    sum_Av_min=sum(Av_min)
    #special case: root of dim 0
    if sum_Av_min==0:
        if maps:
            return 0, [np.array([]).reshape(Q.Av[i],0) for i in range(1,Q.n)]
        else:
            return 0
    # representation of the root
    Q.Av[0]=sum_Av_min
    Phi[0]= eye_mat(sum_Av_min,field)
    phi[0]= eye_mat(sum_Av_min,field)
    #maps root-> minimal vertices
    sum_Av_min_p=0
    for i in range(len(V_min)):
        Phi[V_min[i]] = eye_mat(sum_Av_min,field)
        phi[V_min[i]] = proj(sum_Av_min_p,Av_min[i],sum_Av_min,field)    
        sum_Av_min_p+=Av_min[i]  
    #computation if sections by going down the graph
    for v in range(1,Q.n):
        for e in E_out[v]:
            u=Q.E[e][1]#current vertex
            #if u not seen
            if len(Phi[u])==0:
                Phi[u]=Phi[v]
                #dimension 0
                if Q.Ae[e].shape[1]==0:
                    phi[u]=zeros_mat(Q.Av[u],phi[v].shape[1],field)
                else:
                    phi[u]= np.dot(Q.Ae[e],phi[v])
            else:
                if Q.Ae[e].shape[1]==0:#dim 0
                    equali=phi[u]#difference of functions in equalizer
                else:
                    equali=phi[u]-np.dot(Q.Ae[e],phi[v])
                equali= flatten_zero(equali,field)
                if Q.Av[u]!=0:    
                    Phi[u]=intersection(null_space(equali,field),intersection(Phi[v],Phi[u],field),field)
    #compute total flow space
    result_space = Phi[0]
    for v in V_max:
        result_space=intersection(result_space,Phi[v],field)
    if maps:
        #return dim(Av*) and isomorphism field^dim(Av*)-> subspace of Av
        result_maps=[]
        for v in range(0,Q.n-1):
            result_maps.append(np.dot(phi[order[v]],result_space))
        return  result_space.shape[1],result_maps
    return result_space.shape[1]



#generate random acyclic quivers by only adding increasing edges
def generate_acyclic(range_test,n_test,symetric=True):

    Q_test_all=[]
    k=5 #dim (Av)
    
    for n in range_test:
        m=2*int(np.power(n,1.2))#number of edges
        Q_test=[]
        for _ in range(n_test):
            E,AE=[],[]
            for _ in range(m):
                if symetric:
                    s,target=np.random.randint(0,n-1),np.random.randint(0,n-1)
                    while s>=target:
                        s,target=np.random.randint(0,n-1),np.random.randint(0,n-1)        
                if not symetric:
                    s=np.random.randint(0,n-2)
                    target=s+np.random.randint(1,n-s)
                E.append([s,target])
                AE.append(np.eye(k,dtype=float)+np.floor( (1+1./(2*k*(1+len(E)))) *np.random.random((k,k))))        
            Q_test.append(Quiver(n,E,list(k*np.ones(n,dtype=int)),AE,Field('R')))
        Q_test_all.append(Q_test)
    return Q_test_all
    


