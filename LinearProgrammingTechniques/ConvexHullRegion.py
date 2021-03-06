#! /usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Zhenzhong Gao at Inuiguchi Laboratory'

# from tkinter import W
import numpy as np
from scipy.spatial import ConvexHull

def ConvexHullRegion(A, b, c, **kw):
    # Check the dimension of the inequality constraints region
    #
    #       {c in R^n: A @ c <= b}
    #
    # A and b should be numpy.array
    if not (isinstance(A, np.ndarray) and isinstance(b, np.ndarray) \
            and isinstance(c, np.ndarray)):
        raise TypeError('A, b and center should be in numpy.array form')

    m, n = A.shape
    if b.shape[0] != m or c.shape[0] != n:
        raise ValueError('The dimension of input variables are wrong')

    # Define the number of vertices to define the convex hull region
    if 'number' not in kw:
        num_c = (n+1)
    else:
        num_c = kw['number']
    
    num_v = num_c

    # Generate a list of vertices around the center
    center = c; width = np.diag(np.absolute(c))
    points = np.random.multivariate_normal(center, width/10, num_v)

    # Remove the vertices not in the region
    tmp = A @ points.T <= np.matrix(b).T @ np.ones([1,num_v])
    tmp = np.all(tmp, axis=0)
    tmp = np.where(tmp == True)[1]
    points = points[tmp,:]

    # Construct the convex hull by the derive vertices and obtain
    # the valid points and the corresponding constraints
    while True:
        hull = ConvexHull(points)
        constraints = hull.equations
        if constraints.shape[0] >= num_c:
            break
        else:
            while True:
                p = np.random.multivariate_normal(center, width/10, 1)
                if np.all(A @ p.T <= np.matrix(b).T):
                    points = np.r_[points, p]
                    break

    # It is noted that the equations in the class satisfy the following
    # conditions:
    #
    #       tmp.equations @ [vertices,1] <= 0
    #
    D = constraints[:, :-1]
    d = -constraints[:, -1]

    return D,d


if __name__ == '__main__':
    """
    The main process to generate a convex hull region inside another
    region defined by a series of linear constraints.
    
    The input variables forms a region constrained by a series of constraints
    with the following inequalities:

        {x in R(n)matrix matrix : A @ x <= b}

    The output variables also forms a region constrained by a series of 
    constraints with the following inquealities:

        {x in R(n): D @ x <= d}.

    The function is used by 

        D, d = ConvexHullRegion(A, b, center, number=None)

    The parameters follows

    Input:  A --> np.array with R(m * n)
            b --> np.array with R(m)
            center --> np.array with R(n)
            number --> int
    Output: D --> np.array with R(l * p)
            d --> np.array with R(p)

    It is noted that c is given as the candidate objective vector for generating
    a series of vertices in defining the convex hull. 

    number=None can be given for specifying the number of valid constraints in 
    defining the convex hull.
    """
    
    A = np.array([[1/9,-1/3],
                  [-4/9,1/3]])
    b = np.array([0,0])
    center = np.array([1,1])

    D,d = ConvexHullRegion(A,b,center, number=5)

    print(D @ center <= d)

    # Check the drawn region by plotting the figure:w
    import matplotlib.pyplot as plt
    fig = plt.figure()

    x = np.linspace(0,5,10)
    for i in range(D.shape[0]):
        y = (D[i,0] * x - d[i]) / -D[i,1]
        plt.plot(x,y,'r')

    for j in range(A.shape[0]):
        y = - A[j,0]/A[j,1] * x
        plt.plot(x,y,'b')

    plt.xlim([0,5])
    plt.ylim([0,5])
    plt.show()

