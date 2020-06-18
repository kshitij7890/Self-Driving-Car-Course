import numpy as np
import matplotlib.pyplot as plt


def draw(x1,x2):
    ln = plt.plot(x1,x2)

def sigmoid(score):
    return 1/(1 + np.exp(-score))

n_pts=100
np.random.seed(0) # will give same random values each time
bias=np.ones(n_pts)
top_region=np.array([np.random.normal(10,2,n_pts), np.random.normal(12,2,n_pts),bias]).T # for both x1 and x2 coordinates maxium density area near 10 and 12 respectively ....2 is the standard deviation
bottom_region= np.array([np.random.normal(5,2, n_pts), np.random.normal(6,2, n_pts),bias]).T
all_points = np.vstack((top_region,bottom_region))
w1 = -0.2
w2 = -0.35
b = 3.5
line_parameters = np.matrix([w1,w2,b]).T#transpose to make it compatible for matrix multiplication
x1 = np.array([ bottom_region[:,0].min() , top_region[:, 0].max() ]) # we'll get two extreme horizontal coordinates
# w1x1 + w2x2 +b = 0
x2= -b/w2 + (x1*(-w1/w2)) # by rearranging # we'll get two extreme vrtical coordinates
linear_combination = all_points*line_parameters
probabilities = sigmoid(linear_combination) # we'll receive probablities of each point ...if in positive region(down the line) higher probabilities and vice versa



_, ax= plt.subplots(figsize=(4,4)) # allows to  display multiple plots in same figure
ax.scatter(top_region[:,0], top_region[:,1], color='r')
ax.scatter(bottom_region[:,0], bottom_region[:,1], color='b')
draw(x1,x2)
plt.show()
