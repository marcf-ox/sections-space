# sections-space

This package provides a simple implementation of the algorithm described in <a href="https://link.springer.com/article/10.1007/s10208-022-09563-x">Principal Components Along Quiver Representations</a> by A. Seigal, H. Harrington and V. Nanda to compute the space of sections of a quiver representation -- see the notebook tutorial for more details.

## Installation

It may be installed via

```
pip install git+https://github.com/marcf-ox/sections-space
```

This requires Python >=3.8.0, Numpy>=1.23, Scipy>=1.10.0, cfractions>=2.2.0, Networkx>=2.6.2, Matplotlib>=3.3.4, Markdown>=3.4.1

## Exampple

'''
field= Field("R")
#create a cyclic graph
edges = [("v"+str(i),"v"+str((i+1)%4), {"map": np.array([2**(2*(i%2)-1)]).reshape((1,1))}) for i in range(4)]
G= nx.from_edgelist(edges,create_using=nx.DiGraph())
#convert to Quiver
Q=nx_graph_to_quiver(G,field)
#compute the dimension d of the   space  of sections and dim d and the projections of a base onto each component
d,projs=compute_sections(Q)
'''
