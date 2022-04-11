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
        raise ValueError('The dimensions of A, b and c do not correspond')

    # Define the number of constraints in the convex hull region
    if 'constraint_num' in kw:
        constraint_num = kw['constraint_num']
    else:
        constraint_num = round(1.1 * (n+1))
    # Define the success of the the convex hull region in the given one
    if 'succ' in kw:
        succ = kw['succ']
    else:
        succ = True
        
    """
    the main process
    """
    # Generate a list of vertices around the center
    center = c; width = np.diag(np.absolute(c)); No = 5*n
    points = np.random.multivariate_normal(center, width/5, No)

    # Remove the vertices not in the region
    if succ:
        tmp = A @ points.T <= np.matrix(b).T @ np.ones([1,No])
        tmp = np.all(tmp, axis=0)
        tmp = np.where(tmp == True)[1]
        points = points[tmp,:]
    points = points[:constraint_num,:]

    # Construct the convex hull by the derive vertices and obtain
    # the valid points and the corresponding constraints
    tmp = ConvexHull(points)

    # It is noted that the equations in the class satisfy the following
    # conditions:
    #
    #       tmp.equations @ [vertices,1] <= 0
    #
    res = tmp.equations

    return res[:, :-1], -res[:, -1]


if __name__ == '__main__':
    """
    The main process to generate a convex hull region inside another
    region defined by a series of linear constraints.

    The function is utilised by 

        ConvexHullRegion(A, b, c, constraints = None(int), succ = None(bool))

    Input:  A --> np.array with R(m * n)
            b --> np.array with R(m)
    Output: D --> np.array with R(l * p)
            d --> np.array with R(p)

    The input variables forms a region constrained by a series of constraints
    with the following inequalities:

        {x in R(n)matrix matrix : A @ x <= b}.

    The output variables also forms a region constrained by a series of 
    constraints with the following inquealities:

        {x in R(n): D @ x <= d}.

    """
    
    A = np.array([[1/9,-1/3],
                  [-4/9,1/3]])
    b = np.array([0,0])
    center = np.array([1,1])

    D,d = ConvexHullRegion(A,b,center,constraint_num=round(1.1*A.shape[1] + 3))

    # Check if the center is in the convex hull
    print(D @ center <= d)

    # Check the drawn region by plotting the figure:w
    import matplotlib.pyplot as plt
    fig = plt.figure()

    x = np.linspace(0,5,10)
    for i in range(D.shape[0]):
        y = (D[i,0] * x - d[i]) / -D[i,1]
        plt.plot(x,y,'r')

    for j in range(A.shape[0]):
        y = (A[j,0] * x - b[j]) / -A[j,1]
        plt.plot(x,y,'b')

    plt.xlim([0,5])
    plt.ylim([0,5])
    plt.show()