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
        number = 10 * (n+1)
    else:
        number = kw[number]

    # Generate a list of vertices around the center
    center = c; width = np.diag(np.absolute(c))
    points = np.random.multivariate_normal(center, width/10, number)

    # Remove the vertices not in the region
    tmp = A @ points.T <= np.matrix(b).T @ np.ones([1,number])
    tmp = np.all(tmp, axis=0)
    tmp = np.where(tmp == True)[1]
    points = points[tmp,:]

    # Construct the convex hull by the derive vertices and obtain
    # the valid points and the corresponding constraints
    tmp = ConvexHull(points)
    D = tmp.equations[:, :-1]
    d = tmp.equations[:, -1]

    return D,d


if __name__ == '__main__':
    """
    The main process to generate a convex hull region inside another
    region defined by a series of linear constraints.

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

    D,d = ConvexHullRegion(A,b,center)

    # Check the drawn region by plotting the figure:w
    import matplotlib.pyplot as plt
    fig = plt.figure()

    x = np.linspace(0,5,10)
    for i in range(D.shape[0]):
        y = (D[i,0] * x + d[i]) / -D[i,1]
        plt.plot(x,y,'r')

    for j in range(A.shape[0]):
        y = - A[j,0]/A[j,1] * x
        plt.plot(x,y,'b')

    plt.xlim([0,5])
    plt.ylim([0,5])
    plt.show()

