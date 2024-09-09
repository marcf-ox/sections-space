# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 19:50:59 2021

@author: marcf
"""
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import copy
from cfractions import Fraction
from time import time
Q_pb=[]


#maximum computation error
epsilon=1e-12

## COMPUTATIONS IN ALL FIELDS

#field F_2
class F_2():
    #definition
    is_one=False
    def __init__(self, is_one):
        self.is_one=is_one
    #override operators
    def __add__(self,b):
        return F_2(self.is_one^b.is_one)
    def __sub__(self,b):
        return self+b
    def __mul__(self,b):
        return F_2(self.is_one&b.is_one)
    def __truediv__(self,b):
        if not(b.is_one):
            raise ValueError("Divide by 0")
        else:
            return self
    def __neg__(self):
        return self
    def __eq__(self,b):
        return isinstance(b, F_2) and self.is_one==b.is_one
    #display
    def __str__(self):
        return "cl("+str(int(self.is_one))+")"
    def __repr__(self):
        return "cl("+str(int(self.is_one))+")"

#general field definition
class Field:
    descr='O'
    zero=None
    one=None
    def __init__(self,*args):
        # usual fields R, C, F_2, Q
        if len(args)==1:
            self.descr=args[0]
            if args[0]=='C':
                self.zero=0.+0j
                self.one=1.+0j
            elif args[0]=='R':
                self.zero=0.
                self.one=1.
            elif args[0]=='F_2':
                self.descr=args[0]
                self.zero=F_2(0)
                self.one=F_2(1)
            elif args[0]=='Q':
                self.descr=args[0]
                self.zero=Fraction(0)
                self.one=Fraction(1)
            else:
                raise ValueError("No predefined field "+args[0])
        #user-defined fields
        else:
            self.zero=args[0]
            self.one=args[1]
            if len(args)==3:
                self.descr=args[2]

#Replacing numpy operations if the field is not R or C
#np.zeros
def zeros_mat(n,m,field):
    if field.descr=='R': 
        return np.zeros((n,m))
    if field.descr=='C': 
        return np.zeros((n,m),dtype=complex)
    return np.array([[field.zero for _ in range(m)]for _ in range(n)]).reshape((n,m))
        
#np.eye
def eye_mat(n,field):
    if field.descr =='R':
        return np.eye(n)
    if field.descr=='C':
        return np.eye(n,dtype=complex)
    I=zeros_mat(n,n,field)
    for i in range(n):
            I[i][i]=field.one
    return I            
            
def is_all_zero_mat(M,field):
    if field.descr in ['R','C']:
        return np.max(np.abs(M))<epsilon
    return all([m==field.zero for m in M.flatten()])

def is_all_zero_elem(x,field):
    return is_all_zero_mat(np.array([x]),field)
#Remove computation errors
def flatten_zero(U,field):
    if field.descr in ['R','C']:
        V=U.flatten()
        V[np.where(np.abs(V)<epsilon)[0]]=0.
        return V.reshape(U.shape)
    return U

## QUIVER
#definition
class Quiver:
  field='R'
  def __init__(self, n,edges,rep_spaces,rep_maps,field):
    self.n = n #|V|
    self.E = edges 
    self.Av=np.array(rep_spaces,dtype=int) #dimension of A_v
    self.Ae=rep_maps
    self.field=field
  #display
  def __str__(self):
      if self.field.descr in ['R','C']:
          Ae_rounded=[np.round(ae,3) for ae in self.Ae]
          return "n="+str(self.n)+" E="+str(self.E)+" Av="+str(self.Av)+"\n Ae="+str(Ae_rounded)
      return "n="+str(self.n)+" E="+str(self.E)+" Av="+str(self.Av)+"\n Ae="+str([self.Ae])    
  def __repr__(self):
      if self.field.descr in ['R','C']:
          return "n="+str(self.n)+" E="+str(self.E)+" Av="+str(self.Av)+"\n Ae="+str([np.round(ae,3) for ae in self.Ae])
      return "n="+str(self.n)+" E="+str(self.E)+" Av="+str(self.Av)+"\n Ae="+str(self.Ae)
 
#add new vertices with order 0,1,...
def shift_vertices(Q,shi=1):
    E=[]
    for e in Q.E:
        E.append([e[0]+1,e[1]+1])
    return Quiver(Q.n+1,E,[-1]+list(Q.Av),Q.Ae,Q.field)
    
def sub_quiver(Q,S):
    is_E_in_S=[ e[0] in S and e[1] in S for e in Q.E]
    edges=[]
    for e in np.array(Q.E)[is_E_in_S]:
        edges.append([S.index(e[0]), S.index(e[1])])
    rep_spaces=list( np.array(Q.Av)[S] )
    rep_maps=[ae for i_ae,ae in enumerate(Q.Ae) if is_E_in_S[i_ae]]
    return Quiver(len(S), edges, rep_spaces, rep_maps, Q.field)


#sort edges by starting/arriving extremity
def E_in_out(Q):
    E_in=[[] for _ in range(Q.n)]
    E_out=[[] for _ in range(Q.n)]
    for i_e,e in enumerate(Q.E):
        E_in[e[1]].append(i_e)
        E_out[e[0]].append(i_e)
    return E_in,E_out
   
 
## LINEAR ALGEBRA
    

# Row echelon form (Gaussion pivot)
def row_echelon(M_input,field): 
    M=copy.deepcopy(M_input)
    #empty matrix
    if M.shape[0]*M.shape[1]==1:
        if is_all_zero_elem([M[0][0]],field):
            return M,[]
        return M,[0]
    
    pivots=[]
    #create one new echelon
    def echelonify(next_pivot_row, col):
        #choose best row to pivot
        if field.descr in ['R','C']:
            best_row= next_pivot_row+np.argmax(np.abs(M[next_pivot_row:,col]))
        else:
            non_zero_rows_sub=np.where(M[next_pivot_row:,col]!=field.zero)[0]
            if len(non_zero_rows_sub)==0:
                best_row=next_pivot_row
            else:
                best_row=next_pivot_row+non_zero_rows_sub[0]
        #swap rows
        if not is_all_zero_elem([M[best_row][col]],field):
            rw=np.copy(M[next_pivot_row])
            M[next_pivot_row]=np.copy(M[best_row])
            M[best_row]=rw
            rw=np.copy(M[next_pivot_row])
            pivots.append(col)
        else: # the column col is null
            return next_pivot_row
        #echelonify the matrix
        for j, row in enumerate(M[(next_pivot_row+1):]):
          M[j+next_pivot_row+1] = row - np.array([ row[col] / rw[col] ]  )* rw
        return next_pivot_row+1
    
    next_pivot_row=0#nb of pivoted rows +1
    for i in range(M.shape[1]):#column to pivot
      if next_pivot_row>=M.shape[0]:#all possible rows pivoted
          break
      next_pivot_row=echelonify(next_pivot_row, i)
    #remove some computation errors
    M=flatten_zero(M,field)
    return np.array(M),pivots

#put in column echelon form
def col_echelon(M,field):
    row_ech= row_echelon(np.transpose(M),field)
    return np.transpose(row_ech[0]),row_ech[1]
     
   
#compute kernel of M
def null_space(M,field):
    # if field is R or C: SVD
    if field.descr in ['R','C']:
        return scipy.linalg.null_space(M,rcond=epsilon)
    # otherwise column echelon of the augmanted matrix
    M=flatten_zero(M,field)
    aug_mat=flatten_zero(col_echelon(np.concatenate([M,eye_mat(M.shape[1],field)]),field)[0],field)
    #column of the kernel base
    zero_col_top=[is_all_zero_mat(aug_mat[:M.shape[0],i], field) for i in range(M.shape[1])]
    return aug_mat[M.shape[0]:,np.array(zero_col_top)]
    
        
#intersection of two families U and V by computing the kernel of 
#( U)
#(-V)
def intersection(U,V,field):
    U=flatten_zero(U,field)
    V=flatten_zero(V,field)
    M=np.concatenate((U,-V),axis=1)
    #empty matrix
    if np.shape(M)[0]*np.shape(M)[1]==0:
        return np.array([]).reshape(np.shape(M))
    u=null_space(M,field)[:np.shape(U)[1]]
    return np.dot(U,u)
    
def extract_basis_old(M,field):
    if M.shape[1]==0:
        return M
    if field.descr in['R','C']:
        return scipy.linalg.orth(M,rcond=epsilon)

def extract_basis(M,field):
    if M.shape[0]* M.shape[1]==0:
        return M.reshape((M.shape[0],0))
    if field.descr in['R','C']:
        return scipy.linalg.orth(M,rcond=epsilon)
    col_ech,pivots=col_echelon(M, field)
    return col_ech[:,:len(pivots)].reshape((M.shape[0], len(pivots)))

def sum_subspaces(U,V,field):
    U=flatten_zero(U,field)
    V=flatten_zero(V,field)
    M=np.concatenate((U,V),axis=1)
    return extract_basis(M,field)

# matrix of a projection from dim tot to dim b
def proj(a,b,tot,field,B=None):
    if B==None:
        B=eye_mat(b,field)
    return np.concatenate((zeros_mat(b,a,field),B,zeros_mat(b,tot-a-b,field) ),axis=1)

def inj(a,b,tot,field):
    return proj(a,b,tot,field).transpose()

# transform a matrix from row echelon form to diagonal
def ech_to_diag_row(T_input,field):
    T=copy.deepcopy(T_input)
    #P_pivots s.t. T*P_pivot diag
    pivots=[]
    for i in range(min(T.shape)):
        col_piv=i
        while col_piv < T.shape[1] and is_all_zero_elem(T[i][col_piv], field) :
            col_piv+=1
        if col_piv<T.shape[1]:
            pivots.append(col_piv)
        else:
            pivots.append(-1)
    for i in range(min(T.shape)):
        if pivots[i]!=-1:      
            T[i]=T[i]/T[i][pivots[i]]
    for i in range(min(T.shape)):
        if pivots[i]!=-1: 
            for i_2 in range(i):
                T[i_2]= T[i_2] - np.array([T[i_2][pivots[i]]/T[i][pivots[i]]])*T[i]                  
    return T

# transform a matrix from column echelon form to diagonal    
def ech_to_diag_col(T_input,field):
    return np.transpose(ech_to_diag_row(np.transpose(copy.deepcopy(T_input)),field))

# solve Mx=y with M triangular (square)
def solve_triangular(M,y,field):
    if y.shape[1]==0:
        return np.array([]).reshape(M.shape[0],0)
    
    if field.descr in ['R','C']:
        return scipy.linalg.solve_triangular(M,y)
    
    aug_mat=ech_to_diag_row(np.concatenate([M,y],axis=1),field)
    y_ech=aug_mat[:,M.shape[1]:]
    for i in range(M.shape[0]):
        y_ech[i]=y_ech[i]/aug_mat[i][i]
    return y_ech
 
## DISPLAY TOOLS

# plot the (averaged) time of the naive and advanced function on quivers sorted by Q.n
def test_plot(Q_test,naive_fun,advanced_fun,range_test,n_max_naive,n_test,log=False,title="",fig=None):
    range_test_small=[n for n in range_test if n<n_max_naive] # range of test for the slower naive function
    times_ear=[]
    times_naive=[]
    for i_n,n in enumerate(range_test):
        #time for advanced
        t0=time()
        res_ear=[]
        for Q in Q_test[i_n]:
            res_ear.append(advanced_fun(Q))
        times_ear.append((time()-t0)/float(n_test))
        #time naive
        t0=time()
        if n<n_max_naive:
            for i_Q,Q in enumerate(Q_test[i_n]):
                #print(naive_fun(Q))
                if (naive_fun(Q)!=res_ear[i_Q]):
                    print(Q)
                    print([np.allclose(A, np.eye(len(A))) for A in Q.Ae])
                    raise NameError("naive neq adv")
            times_naive.append((time()-t0)/float(n_test))
    #plotting
    if fig==None:
        fig =plt.Figure()
    plt.plot(range_test_small,times_naive,label="naive "+ str(Q_test[0][0].field.descr))
    plt.plot(range_test,times_ear,label="advanced "+ str(Q_test[0][0].field.descr))
    plt.xlabel("n")
    plt.ylabel("time(s)")
    if log:
        def to_log_reshape(l,is_X=False):
            if is_X:
                 return [[np.log(x),1] for x in l]
            return [[np.log(x)] for x in l]           
        coefs_naive=np.linalg.lstsq(to_log_reshape(range_test_small,True), to_log_reshape(times_naive),rcond=None)[0].flatten()
        coefs_adv=np.linalg.lstsq(to_log_reshape(range_test,True), to_log_reshape(times_ear),rcond=None)[0].flatten()   
        plt.plot(range_test,np.exp(coefs_adv[1])*np.power(np.array(range_test),coefs_adv[0]), '--',label="n^"+str(np.round(coefs_adv[0],1)))
        plt.plot(range_test_small,np.exp(coefs_naive[1])*np.power(np.array(range_test_small),coefs_naive[0]), '--', label="n^"+str(np.round(coefs_naive[0],1)))
        plt.xscale('log')
        plt.yscale('log')
    plt.legend()
    plt.title(title)
    return fig
    #plt.show()
    
    
    
# plot the (averaged) time of the naive and advanced function on quivers sorted by |E|
def test_plot_by_m(Q_test,naive_fun,advanced_fun,advanced_fun_2,range_test,n_test,title=""):
    times_ear=[]
    times_adv_2=[]
    times_naive=[]
    for i_m,m in enumerate(range_test):
        #time for advanced
        t0=time()
        res_ear=[]
        for Q in Q_test[i_m]:
            res_ear.append(advanced_fun(Q))
        times_ear.append((time()-t0)/float(n_test))
        #time for advanced 2
        t0=time()
        res_ear2=[]
        for Q in Q_test[i_m]:
            res_ear2.append(advanced_fun_2(Q))
        times_adv_2.append((time()-t0)/float(n_test))
        #time naive
        t0=time()
        for i_Q,Q in enumerate(Q_test[i_m]):
            #print(naive_fun(Q))
            naive_res=naive_fun(Q)
            if (naive_res!=res_ear[i_Q] or naive_res!=res_ear2[i_Q]):
                print(naive_fun(Q),res_ear[i_Q] ,res_ear2[i_Q])
                print(len(Q.E))
                raise NameError("naive neq adv")
        times_naive.append((time()-t0)/float(n_test))
    #plotting
    plt.plot(range_test,times_naive,label="naive")
    plt.plot(range_test,times_ear,label="advanced")
    plt.plot(range_test,times_adv_2,label="advanced all CC together")
    plt.xlabel("m")
    plt.ylabel("time(s)")
    plt.legend()
    plt.title(title)

    #plt.show()
    

#display of a limit with its maps
def print_limit(lim_and_maps,field=Field('R')):
    lim,maps=lim_and_maps
    #the limit
    print("lim="+field.descr+"^"+str(lim))
    #choosing a isomorphic limit with simpler maps (isomorphism given by P_pivot)
    maps_prod_dim=[maps[i].shape[0]*maps[i].shape[1] for i in range(len(maps))]
    i=np.argmax(np.array(maps_prod_dim))   
    aug_map=np.concatenate([maps[i],eye_mat(maps[i].shape[1],field)])
    P_pivot=ech_to_diag_col(col_echelon(aug_map,field)[0],field)[len(maps[i]):]
    for v in range(len(maps)):        
        print("lim ->",v,"\n",flatten_zero(np.dot(maps[v],P_pivot) ,field))
    

