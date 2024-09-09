# sections-space

This package provides a simple implementation of the algorithm described in <a href="https://link.springer.com/article/10.1007/s10208-022-09563-x">Principal Components Along Quiver Representations</a> by A. Seigal, H. Harrington and V. Nanda to compute the space of sections of a quiver representation.

## Installation

It may be installed via

```
pip install git+https://github.com/marcf-ox/sections-space
```

This requires Python >=3.8.0, Numpy>=1.23, Scipy>=1.10.0, cfractions>=2.2.0, Networkx>=2.6.2, Matplotlib>=3.3.4, Markdown>=3.4.1

## Example

 A detailed Jupyter tutorial notebook is available at:

<a href = https://github.com/marcf-ox/sections-space/blob/main/notebook/tuto.ipynb> Tutorial notebook </a>

A first example of use of this package is given below:

```
from sections_space import Field, nx_graph_to_quiver,compute_sections
import numpy as np
import networkx as nx

#choose base field
field= Field("R")

#create a cyclic graph
edges = [(i,(i+1)%4, {"map": np.ones((1,1))}) for i in range(4)]
G= nx.from_edgelist(edges,create_using=nx.DiGraph())

#convert to Quiver
Q=nx_graph_to_quiver(G,field)

#compute the dimension d of the   space  of sections and the projections of a base onto each component
d,projs=compute_sections(Q)
```

