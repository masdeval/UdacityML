# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

if __name__ == "__main__":
    print("Hello World")


"""Softmax."""
import numpy as np

scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])

# a 3D array (two stacked 2D arrays)                   
c = np.array( [ [ [0,  1,  2],[ 10, 12, 13] ],
                [ [100,101,102],[110,112,113] ]
                
                ] )                   


print(c[0])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
    pass  # TODO: Compute and return softmax(x)


print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
#plt.show()

x=1000000000
for i in range(1,1000000):
    x = x + 0.000001
x = x - 1000000000
print(x)