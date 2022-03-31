#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'Optimality Assurance Cone'

__author__ = 'Zhenzhong Gao at Inuiguchi Laboratory'

import numpy as np
from scipy.optimize import linprog

threshold = 1e-5

def OptimalityAssuranceCone(mode, **kw):
    """
    The main process to calculate the optimality assurance cone.
    """

    if mode != 'min' and mode != 'max':
        raise TypeError('The type of LP problem is unknown, \
            please input "min" or "max"')
    elif ('A' or 'b' or 'c') not in kw:
        raise NameError('The parameters are not fully given,', \
            'please give "A", "b" for the equality constraints,', \
            'and "c" for the objective function.')
    else:
        A = kw['A']; b = kw['b']; c = kw['c']
        if mode == 'max': c = -c

    # Reform the dimension of the objective function vector and find a reasonable
    # one for obtaining a feasible solution
    m, n = A.shape

    # Solve the linear programming problem
    res = linprog(c, A_eq=A, b_eq=b, method='simplex')
    if not res.success:
        print('The LP problem has no solution because of', res.status)
        return False, []
    else:
        x = res.x
        # If the procedure succeed, Find the basic index set 
        Basis = np.where(x >= threshold)[0].tolist()
    
    # Define a function to solve the subtraction of array
    def SubArray(id, L):
        return list(filter(lambda i: i not in id, range(L)))

    A_B = A[np.ix_(range(m), Basis)]
    A_N = A[np.ix_(range(m), SubArray(Basis, n))]
    tmp = np.concatenate(((-np.linalg.inv(A_B)*A_N).T, np.identity(n-m)), axis=1)
    Index_u = Basis + SubArray(Basis, n)
    idx = np.empty_like(Index_u)
    idx[Index_u] = np.arange(n)
    M = tmp[:, idx]

    if mode == 'max': M = -M
    return True, M

if __name__ == '__main__':
    """

    The optimality assurance cone of a non-degenerated basic feasible solution 
    for a linear programming problem, which is in the form of 

        max(min) c@x, s.t. A@x == b, x >= 0,

    where the function is used by 

        res = OptimalityAssuranceCone('max'(or 'min'), A = None, b = None, c = None),

    where parameters:
        A: 2-D array or matrix
        b: 1-D array or matrix with only one column
        c: 1-D array or matrix with only one column

    and returns:
        [0]. True or False
        [1]. The optimality assurance cone by 
            M@c <=(in maximization problem) or >=(in minimization problem) 0

    The relative methodology as well as the illustrative figures can be found 
    in the following paper:
        
        https://link.springer.com/article/10.1007/s10700-022-09383-2

    """

    A = np.matrix([
        [3,4,1,0,0],
        [3,1,0,1,0],
        [0,1,0,0,1]
    ])
    b = np.matrix([42, 24, 9]).T
    c = np.matrix([1,1,0,0,0]).T
    res = OptimalityAssuranceCone('max', A = A, b = b, c = c)
    print(res)
