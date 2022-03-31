#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'Robust Optimality Analysis with Oblique Fuzzy Vector'

__author__ = 'Zhenzhong Gao at Inuiguchi Laboratory'

import numpy as np
from OptimalityAssuranceCone import OptimalityAssuranceCone

class LR(object):
    def __init__(self, aL, aR, alpha, beta, L, R, L_inv, R_inv, M):
        """
        aL, aR, alpha, beta --> List
        """
        self.aL,    self.aR     = aL,       aR
        self.alpha, self.beta   = alpha,    beta
        """
        L, R            --> Function List: the reference function
        L_inv, R_inv    --> Inverse Function List: the inverse reference function
        M               --> Matrix: the matrix of optimality assurance cone
        """
        self.L,     self.R      = L, R
        self.L_inv, self.R_inv  = L_inv, R_inv
        self.M                  = M

        self.LR = range(len(aL))
        
    def Solver(self):
        """
        Default Functions and Values
        """
        LF = lambda r: 1-r

        def ext(l):
            a = [((self.aR[i]+self.R_inv[i](l)*self.beta[i]) + \
                (self.aL[i]-self.L_inv[i](l)*self.alpha[i]))/2 for i in self.LR]
            s = [((self.aR[i]+self.R_inv[i](l)*self.beta[i]) - \
                (self.aL[i]-self.L_inv[i](l)*self.alpha[i]))/2 for i in self.LR]
            return a, s

        delta = 1e-3
        thres = 1e-3

        """
        The iteration steps
        """
        # Step 1 --> Initialize h_H
        a, s = ext(1) 
        if all([s[i]<=0 for i in self.LR]): 
            s = [thres]*len(s) # Address the situation that spread equal to 0
        tmp = Sym(a, s, LF, self.M)
        tau = tmp.Solver()
        if tau < 1:
            return f'No Necessarily Optimal Solution'
        else:
            l = [self.L[i]((self.aL[i] - (a[i]-tau*s[i]))/self.alpha[i]) for i in self.LR]
            r = [self.R[i](((a[i]+tau*s[i]) - self.aR[i])/self.beta[i]) for i in self.LR]
            h_M = max(l+r)
            if h_M <= 0:
                return f'h = 1'
            else:
                h_H = h_M

        # Step 2 --> Initialize h_L
        a, s = ext(0)
        tmp = Sym(a, s, LF, self.M)
        tau = tmp.Solver()
        if tau >= 1:
            return f'h = 1'
        else:
            l = [self.L[i]((self.aL[i] - (a[i]-tau*s[i]))/self.alpha[i]) for i in self.LR]
            r = [self.R[i](((a[i]+tau*s[i]) - self.aR[i])/self.beta[i]) for i in self.LR]
            h_M = max(l+r)
            h_H = min(h_M,h_H)
            h_L = 0

        # Step 3 --> Iteration
        i = 1
        while h_H - h_L >= delta:
            h = (h_H + h_L) / 2
            # print(i,h)
            a, s = ext(h)
            tmp = Sym(a, s, LF, self.M)
            tau = tmp.Solver()

            if tau <= 1:
                h_L = h
            else:
                l = [self.L[i]((self.aL[i] - (a[i]-tau*s[i]))/self.alpha[i]) for i in self.LR]
                r = [self.R[i](((a[i]+tau*s[i]) - self.aR[i])/self.beta[i]) for i in self.LR]
                tmp = max(l+r)
                if tmp > h:
                    raise Exception('Wrong Answer')
                else:
                    h_H = tmp
            i += 1
        return f'h = {1-h_H}'
   

class LL(object):
    def __init__(self, aL, aR, alpha, beta, L, M):
        """
        aL, aR, alpha, beta --> List
        """
        # self.aL,    self.aR     = np.matrix(aL).T,      np.matrix(aR).T
        # self.alpha, self.beta   = np.matrix(alpha).T,   np.matrix(beta).T
        self.aL,    self.aR     = aL,       aR
        self.alpha, self.beta   = alpha,    beta
        """
        L --> Function:     the reference function;
        M --> Matrix:       the matrix of optimality assurance cone
        """
        self.L,     self.M      = L, M.tolist()

    def Solver(self):
        k, w, r = [0.0]*len(self.M), [0.0]*len(self.M), [0.0]*len(self.M)

        for i in range(len(self.M)):
            for j in range(len(self.M[i])):
                if self.M[i][j] > 0:
                    w[i] += self.M[i][j] * self.alpha[j]
                    r[i] += self.M[i][j] * self.aL[j]
                elif self.M[i][j] < 0:
                    w[i] -= self.M[i][j] * self.beta[j]
                    r[i] += self.M[i][j] * self.aR[j]
                else:
                    pass
            k[i] = r[i] / w[i]
        
        # print(k)
        return 1 - self.L(min(k))

class Sym(LL):
    def __init__(self, a, alpha, L, M):
        super().__init__(a, a, alpha, alpha, L, M)

class ObliqueLR(LR):
    def __init__(self, aL, aR, alpha, beta, L, R, M, D):
        super().__init__(aL, aR, alpha, beta, L, R, M*np.linalg.inv(D))

class ObliqueLL(LL):
    def __init__(self, aL, aR, alpha, beta, L, M, D):
        super().__init__(aL, aR, alpha, beta, L, M*np.linalg.inv(D))

class ObliqueSym(Sym):
    def __init__(self, a, alpha, L, M, D):
        super().__init__(a, a, alpha, alpha, L, M*np.linalg.inv(D))


if __name__ == '__main__':
    """
    
    The robust optimality analysis to a linear programming problem
    in the following form:
    
        max(min) c@x, s.t. A@x == b, x >= 0,

    where A and b are constant matrix and vector to construct the 
    feasible set of the problem. However, c is a objective vector 
    composed of fuzzy numbers. 

    To treat the problem, we use LL, LR, Oblique LL and LR fuzzy
    numbers to represent the uncertainties. The only output is 
    the optimality degree, which is a number from 0 to 1.

    The main function is a class, where the solver would give the
    correct answer. The examples can be found below.

    The details can be found in the following paper:

        https://link.springer.com/article/10.1007/s10700-022-09383-2
        
    """
    # Define the equality constraints at first with
    #   A@x == b
    A = np.matrix([
        [3,4,1,0,0],
        [3,1,0,1,0],
        [0,1,0,0,1]
    ])
    b = np.matrix([42, 24, 9]).T

    # A typical reference function, similar to the interval one.
    L = lambda r: 1-r if r>=0 and r<=1 else 0

    # The oblique matrix D
    D = np.matrix([\
        [10, 1],\
        [-8, 15]\
        ])

    # Assert the test case
    TestCase = 'LL'

    if TestCase == 'Sym':
        """
        c_1 = [23,23,8,8]
        c_2 = [14,14,5,5]
        """
        a = [23, 14]
        alpha = [8, 5]

        a = np.matrix(a).T
        tmp = A.shape[1] - a.shape[0]
        if tmp > 0: c = np.r_[a, np.zeros([tmp,1])]

        res = OptimalityAssuranceCone('max', A = A, b = b, c = c)

        if res[0]:
            M = res[1][:, :a.shape[0]]
            tmpCase = Sym(a, alpha, L, M)
            print(tmpCase.Solver())
    
    elif TestCase == 'ObliqueLL':
        """
        aL, aR, alpha, beta, L, M, D
        d_1 = [399,399,140,18]
        d_2 = [139,139,204,112]
        """
        aL = [399,139]
        aR = aL
        alpha = [140,204]
        beta = [18,112]

        a = np.matrix(aR).T
        tmp = A.shape[1] - a.shape[0]
        if tmp > 0: c = np.r_[a, np.zeros([tmp,1])]

        res = OptimalityAssuranceCone('max', A = A, b = b, c = c)

        if res[0]:
            M = res[1][:, :a.shape[0]]
            tmpCase = ObliqueLL(aL, aR, alpha, beta, L, M, D)
            print(tmpCase.Solver())

    elif TestCase == 'LL':
        """
        aL, aR, alpha, beta, L, M
        """
        aL = [23, 14]
        aR = aL
        alpha = [8, 5]
        beta = [4, 2]
        # beta = alpha
        L = lambda x: (x-1)**2 if x >= 0 and x <= 1 else 0
        # L = lambda x: 1-x if x >= 0 and x <= 1 else 0

        a = np.matrix(aR).T
        tmp = A.shape[1] - a.shape[0]
        if tmp > 0: c = np.r_[a, np.zeros([tmp,1])]

        res = OptimalityAssuranceCone('max', A = A, b = b, c = c)

        if res[0]:
            M = res[1][:, :a.shape[0]]
            tmpCase = LL(aL, aR, alpha, beta, L, M)
            print(tmpCase.Solver())

    elif TestCase == 'LR':
        """
        aL, aR, alpha, beta, L, R, M 
        c_1 = [23,23,8,8]
        c_2 = [14,14,5,5]
        """
        aL = [23,14]
        aR = aL
        alpha = [8,5]
        beta = [8,5]
        L = [\
            lambda x: -x*x+1 if x >= 0 and x <= 1 else 0,\
            lambda x: -x*x+1 if x >= 0 and x <= 1 else 0]
        R = [\
            lambda x: -x*x+1 if x >= 0 and x <= 1 else 0,\
            lambda x: (x-1)**2 if x >= 0 and x <= 1 else 0]

        L_inv = [lambda x: (1-x)**0.5, lambda x: (1-x)**0.5]
        R_inv = [lambda x: (1-x)**0.5, lambda x: 1-x**0.5]

        a = np.matrix(aL).T
        tmp = A.shape[1] - a.shape[0]
        if tmp > 0: c = np.r_[a, np.zeros([tmp,1])]

        res = OptimalityAssuranceCone('max', A = A, b = b, c = c)

        if res[0]:
            M = res[1][:, :a.shape[0]]
            tmpCase = LR(aL, aR, alpha, beta, L, R, L_inv, R_inv, M)
            print(tmpCase.Solver())
