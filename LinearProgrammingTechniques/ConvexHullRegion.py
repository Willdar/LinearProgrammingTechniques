#! /usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Zhenzhong Gao at Inuiguchi Laboratory'

import numpy as np
from sci.spatial import ConvexHull

def ConvexHullRegion(A, b, **kw):
    # Check the dimension of the inequality constraints region
    m, n = A.shape
    if b.shape != [m,1]:
        raise ValueError('The size of the constraints is not correct')

    # Define the center to generate a list of vertices
    if 'center' not in kw:
        center = np.ones(n)
    else:
        center = kw[center]

    # Define the number of vertices to define the convex hull region
    if 'number' not in kw:
        number = 50
    else:
        number = kw[number]

    # Generate a list of vertices around the center
    rng = np.random.default_rng(8823)
    points = rng.random((3 * number, n))
    scale = np.linalg.norm(center)
    points = scale * 2 * (points - 0.5)

    # remove the vertices not in the region



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
    pass
