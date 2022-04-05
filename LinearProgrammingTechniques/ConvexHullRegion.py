#! /usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Zhenzhong Gao at Inuiguchi Laboratory'

import numpy as np
from sci.spatial import ConvexHull

def ConvexHullRegion(A, b):



if __name__ == '__main__':
    """
    The main process to generate a convex hull region inside another
    region defined by a series of linear constraints.

    Input:  A --> np.matrix with R(m * n)
            b --> np.array with R(m) or np.matrix with R(m * 1)
    Output: D --> np.matrix with R(l * p)
            d --> np.array with R(p)

    The input variables forms a region constrained by a series of constraints
    with the following inequalities:

        {x in R(n): A @ x <= b}.

    The output variables also forms a region constrained by a series of 
    constraints with the following inquealities:

        {x in R(n): D @ x <= d}.
    """
    pass
