import numpy as np
import matplotlib.pyplot as plt


n_pts=100
np.random.seed(0) # will give same random values each time
top_region=np.array([np.random.normal(10,2,n_pts), np.random.normal(12,2,n_pts)]).T # for both x1 and x2 coordinates maxium density area near 10 and 12 respectively ....2 is the standard deviation
bottom_region= np.array([np.random.normal(5,2, n_pts), np.random.normal(6,2, n_pts)]).T
_, ax= plt.subplots(figsize=(4,4)) # allows to  display multiple plots in same figure
ax.scatter(top_region[:,0], top_region[:,1], color='r')
ax.scatter(bottom_region[:,0], bottom_region[:,1], color='b')

plt.show()
