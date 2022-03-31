#! /usr/bin/env python3
# -*- coding: utf-8 -*-

' Optimality Analysis with Interactive Uncertainties '

__author__ = 'Zhenzhong Gao at Inuiguchi Laboratory'

import sys
import numpy as np
import time
from scipy.optimize import linprog as LinearProgrammingSolver
from collections import Counter
from itertools import combinations
from datetime import datetime

threshold = 1e-5
np.set_printoptions(linewidth=400, threshold=sys.maxsize, suppress=True, \
    formatter={'float': lambda x: "{0:0.3f}".format(x)})

class OptimalityAnalysis(object):
    def __init__(self, mode, **kw) -> None:
             # Check the input
        if mode != 'min' and mode != 'max':
            raise TypeError('The type of LP problem is unknown, please input "min" or "max"')
        if ('Aeq' or 'beq' or 'Gub' or 'gub') not in kw:
            raise NameError('The parameters are not fully given,', \
                'please give "Aeq", "beq" for the LP equality constraints,', \
                'and the "Gub" and "gub" for the constrainted inexactness', \
                'satisfying Gub * c <= gub.')
        else:
            self.mode = mode
            self.A, self.b = kw['Aeq'], kw['beq']
            self.G, self.g = kw['Gub'], kw['gub']

        # Check the correctness of input data in dimension
        # if not (isinstance(self.A, np.matrix) and isinstance(self.b, np.matrix) \
        #     and isinstance(self.G, np.matrix) and isinstance(self.g, np.matrix)):
        #     print('The type of input parameters is wrong, "Aeq", "Gub", "beq" and "gub" should be matrix.')
        #     return False
        if self.b.shape[0] != self.A.shape[0] or self.b.shape[1] != 1 or \
            self.g.shape[0] != self.G.shape[0] or self.g.shape[1] != 1 or \
            self.G.shape[1] > self.A.shape[1]:
            raise ValueError('the dimension of some input parameters are wrong.')

        # print('Input parameters seems to be correct, using "CuttingHyperplane" or "DirectApproach"', \
        #     'for the optimality analysis.')


    def CuttingHyperplane(self) -> True or False:
        ###########################################################################
        """
        Do the initialization and build the constraints and vertices. If failed with 
        unboundedness, return False. Otherwise return True.
        
        1. Return self.G_c and self.g_c are the constraint coefficients satisfying 
        G_c * c <= g_c.
        2. Return self.V is the vertex set matrix with columns being the point 
        vectors.
        """
        if not self._Initialization(): 
            # raise AssertionError('The Inexact Space is not bounded')
            return False, np.matrix([0]), 'The Inexact Space is not bounded'

        ###########################################################################
        """
        Find the optimality assurance cone. If failed, return False. Otherwise, 
        return True.

        1. Return self.M as the matrix of optimality assurance cone.
        2. Return self.x as the candidate optimal solution.
        """
        if not self._OptimalityAssuranceCone():
            # raise AssertionError('Cannot compute the optimality assurance cone')
            return False, np.matrix([0]), 'Cannot compute the optimality assurance cone'
        
        ###########################################################################
        """
        Do the iteration with self.G, self.g, self.G_c, self.g_c, self.V and self.M
        """
        while True:
            # First check if the solution is already necessarily optimal. If not, 
            # find the vertex from self.V.
            tmp = self.M @ self.V >= 0
            if np.all(tmp):
                # print('\nThe solution', self.x, 'is necessarily optimal.')
                return True, self.x, 'success with necessarily optimal'
            else:
                tmp = np.all(tmp, axis=0)
                i_d = np.where(tmp==False)[1][0]

            # Then check if the vertex is in the inexactness area. If not, find the 
            # constraints from self.A and self.b to cut
            tmp = self.G @ self.V[:, i_d] <= self.g
            if np.all(tmp):
                # print('\nThe solution', self.x, 'is not necessarily optimal.')
                return False, self.x, 'success without necessarily optimal'
            else:
                j_d = np.where(tmp==False)[0][0]

            # Separate the vertices by the j_d constraint with V_minus and V_plus
            tmp = self.G[j_d, :] @ self.V <=  self.g[j_d] @ np.ones([1,self.V.shape[1]])
            i_V_minus = np.where(tmp==True)[1].tolist()
            i_V_plus = [i for i in range(self.V.shape[1]) if i not in i_V_minus]
            V_minus, V_plus = self.V[:, i_V_minus], self.V[:, i_V_plus]

            # Update the new vertices by the j_d constraint
            if V_minus.shape[1] <= V_plus.shape[1]: # Choose the vertex set with less entries
                V_tmp = V_minus
            else: 
                V_tmp = V_plus

            # Initialize the new generated vertex set
            V_new = np.zeros([self.V.shape[0],1])
            for i in range(V_tmp.shape[1]):
                # Find the constraints passing the ith vertex
                tmp = abs(self.G_c @ V_tmp[:,i] - self.g_c) <= threshold
                tmp_i = np.where(tmp==True)[0].tolist()
                if len(tmp_i) != V_tmp.shape[0]:
                    # When there exists degeneracy on the ith vertex, choose the first p rows
                    print('There exists degeneracy')
                    tmp_i = tmp_i[:V_tmp.shape[0]]
                tmp_G, tmp_g = self.G_c[tmp_i,:], self.g_c[tmp_i,:]
                # Construct the Tabular for pivoting
                tmp_G = np.r_[tmp_G, self.G[j_d, :]]
                tmp_g = np.r_[tmp_g, self.g[j_d]]
                tabular = np.c_[tmp_G, np.eye(tmp_G.shape[0]), tmp_g]

                for j in range(tmp_G.shape[1],tmp_G.shape[1]+tmp_G.shape[0]):
                    # Calculate the pivoted tabular recursively
                    tabular = np.linalg.inv(tabular[:, list(range(tmp_G.shape[1]))+[j]]) @ tabular
                    # If condition is satisfied, add the vertex to the V_new set by columns
                    if tabular[-1,-1] >= 0:
                        V_new = np.c_[V_new, tabular[:-1,-1]]
            V_new = V_new[:,1:]
                
            # Update the vertex set and remove the repeated vertices 
            tmp = set()
            for i in range(V_minus.shape[1]):
                for j in range(V_new.shape[1]):
                    if np.all(abs(V_new[:,j] - V_minus[:,i]) <= threshold):
                        tmp.add(j)
            tmp = [x for x in range(V_new.shape[1]) if x not in tmp]
            self.V = np.c_[V_minus, V_new[:, tmp]]

            """
            Remove the redundant constraints
            """
            # Add the new constraint to G_c and g_c
            self.G_c, self.g_c = np.r_[self.G_c, self.G[j_d,:]], np.r_[self.g_c, self.g[j_d,:]]

            # Find the valid constaints on each vertex
            tmp = self.G_c @ self.V - self.g_c @ np.ones([1,self.V.shape[1]]) >= -threshold
            tmp = np.where(tmp==True)[0].tolist()
            tmp = Counter(tmp)

            i_nonvalid = []
            for i in range(self.G_c.shape[0]):
                if tmp[i] < self.V.shape[0]:
                    i_nonvalid.append(i)
            i_valid = [x for x in range(self.G_c.shape[0]) if x not in i_nonvalid]
            self.G_c, self.g_c = self.G_c[i_valid, :], self.g_c[i_valid, :]

    
    def DirectApproach(self) -> True or False:
        if not self._Initialization(): 
            # raise AssertionError('The Inexact Space is not bounded')
            return False, np.matrix([0]), 'The Inexact Space is not bounded'
        else:
            Comb = combinations(range(self.G.shape[0]), self.G.shape[1])

        self.V = np.zeros([self.G.shape[1],1])
        for rows in Comb:
            try:
                tmp = np.linalg.inv(self.G[rows,:]) @ self.g[rows,:]
                others = [x for x in range(self.G.shape[0]) if x not in rows]
                if np.all(self.G[others,:] @ tmp <= self.g[others,:]): 
                    self.V = np.c_[self.V, tmp]
            except Exception:
                pass
        self.V = self.V[:, 1:]

        if not self._OptimalityAssuranceCone():
            # raise AssertionError('Cannot compute the optimality assurance cone')
            return False, np.matrix([0]), 'Cannot compute the optimality assurance cone'

        if np.all(self.M @ self.V >= 0): 
            return True, self.x, 'success with necessarily optimal'
        else: 
            return False, self.x, 'success without necessarily optimal'


    def _Initialization(self) -> True or False:
        """
        The initialization process to construct the triangular space containing inexactness.
        If the inexactness is bounded, return "True" and give the constraints:
            self.G_c * c <= self.g_c, 
        and  and vectice set:
            self.V 
        Else return "False" to indicate the inexactness is unbounded.
        """
        
        # Initialize parameters
        n  = self.G.shape[1]
        Bounds = [(None,None)]*n

        # Triangular Constraint
        c = -np.ones(n)
        res_t = LinearProgrammingSolver(c, A_ub=self.G, b_ub=self.g, bounds=Bounds)
        if not res_t.success:
            print('The triangular constraint cannot be accomplished', res_t.status)
            return False
        else:
            self.G_c, self.g_c = np.matrix(-c), np.matrix([-res_t.fun])

        # Paralleled Constraints
        v_tmp = np.zeros(n)
        for i in range(n):
            c = np.zeros(n); c[i] = 1
            res_t = LinearProgrammingSolver(c, A_ub=self.G, b_ub=self.g, bounds=Bounds)
            if not res_t.success:
                print('The', i+1, 'th varibles is not bounded with error code', res_t.status) 
                return False
            else:
                self.G_c = np.append(self.G_c, np.matrix(-c), axis=0)
                self.g_c = np.append(self.g_c, np.matrix([-res_t.fun]), axis=0)
                v_tmp[i] = res_t.fun

        # Calculate the vertices in a matrix where the column represent the vertex vector.
        self.V = np.matrix(np.ones(n+1)).T @ np.matrix(v_tmp)
        tmp = np.sum(v_tmp)
        for i in range(n):
            self.V[i+1,i] += self.g_c[0] - tmp
        self.V = self.V.T

        return True
        

    def _OptimalityAssuranceCone(self) -> True or False:
        """
        The process to calculate the optimality assurance cone.
        If succeed, return "True" with self.M as the matrix and self.x as the solution.
        Else return "False".
        """

        # Reform the dimension of the objective function vector and find a reasonable
        # one for obtaining a feasible solution
        m, n = self.A.shape
        c = np.sum(self.V, axis=1) / self.V.shape[1]
        tmp = n - c.shape[0]
        if tmp > 0: c = np.r_[c, np.zeros([tmp,1])]
        
        # Solve the linear programming problem
        if self.mode == 'max': c = -c
        res = LinearProgrammingSolver(c, A_eq=self.A, b_eq=self.b, method='simplex')
        if not res.success:
            print('The LP problem has no solution because of', res.status)
            return False
        else:
            self.x = res.x
        
        # If the procedure succeed, Find the basic index set 
        Basis = np.where(res.x >= threshold)[0].tolist()
        if len(Basis) != m:
            # If the cardinality of the index set is not m, there exists degeneracy
            print('There exists degeneracy!')
            return False

        # Define a function to solve the subtraction of array
        def SubArray(id: list, L: int) -> list:
            return list(filter(lambda i: i not in id, range(L)))

        A_B = self.A[np.ix_(range(m), Basis)]
        A_N = self.A[np.ix_(range(m), SubArray(Basis, n))]
        tmp = np.concatenate(((-np.linalg.inv(A_B)*A_N).T, np.identity(n-m)), axis=1)
        Index_u = Basis + SubArray(Basis, n)
        idx = np.empty_like(Index_u)
        idx[Index_u] = np.arange(n)
        result = tmp[:, idx]
        if self.mode == 'max': result = -result

        self.M = result[:, :self.V.shape[0]]

        return True


if __name__ == '__main__':
    """
    
    The robust optimality analysis for a linear programming problem containing 
    uncertain coefficients in the objective function, where it is assumed the
    uncertainties are enclosed by a convex and bounded polytope.

    The linear programming problem should be in the standard form as

       max(min) c@x, s.t. Aeq@x = beq, x >= 0,

    where c is enclosed by a series of constraints with

        Gub@c <= gub,

    where Gub and gub are the matrix and vector that construct the convex and bounded 
    polytope.

    The relative paper has not published yet but is already finished and waiting for
    the judgement.

    
    The function is used by

        Object = OptimalityAnalysis('min' or 'max', 
        Aeq = None, beq = None, Gub = None, gub = None)

    where the input data should obey the rules below:
    1. the mode of the LP problem in objective function, 'min' or 'max'
    2. the coefficients in constraints, which should be equality ones.
        Aeq @ x = beq
    3. the inexact coefficients in the objective function, should be inequality.
    4. Aeq, beq, Gub and gub should all be numpy.matrix type
    5. the dimension of Gub can be less than Aeq due to the all-zero objective 
        coefficient.
    6. there should not exist all-zero objective coefficient in Gub and gub.
    """

    """
    A Numerical Example
    """
    A = np.matrix([
        [3,4,1,0,0],
        [3,1,0,1,0],
        [0,1,0,0,1]
    ])
    b = np.matrix([42, 24, 9]).T
    G = np.matrix([
        [-3, -4],
        [1, 0],
        [0, 1]
    ])
    g = np.matrix([-23, 5, 3.5]).T
    tmp = OptimalityAnalysis('max', Aeq = A, beq = b, Gub=G, gub=g)
    succ, solv, mess = tmp.CuttingHyperplane()
    print(succ, solv, mess)

    """
    Simulation Program
    """
    # TIME = datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
    # FILE = 'data'+TIME+'.txt'

    # with open(FILE, 'w') as f:
    #     f.write('')

    # # i --> the number of constraints in feasible set
    # m_l, m_u = 5, 10
    # for i in range(m_l, m_u):

    #     # j --> the dimension of state variables
    #     n_l, n_u = round(0.8 * i), round(1.2 * i)
    #     for j in range(n_l, n_u):

    #         # Construct the feasible set
    #         A = np.random.uniform(10,100,size=(i, j))
    #         A = np.c_[A, np.eye(A.shape[0])]
    #         b = np.random.uniform(100,1000,size=(i, 1))

    #         # k --> the uncertain variables in uncertainty polytope
    #         n_g_l, n_g_u = round(0.2 * j), round(0.8 * j)
    #         for k in range(n_g_l, n_g_u):

    #             # l --> the number of constraints in uncertainty polytope
    #             m_g_l, m_g_u = round(0.8 * k), round(1.2 * k)
    #             for l in range(m_g_l, m_g_u):

    #                 # Construct the uncertainty polytope
    #                 G = np.random.uniform(0,30,size=(l, k))
    #                 G = np.r_[G, -np.eye(G.shape[1])]
    #                 g = np.random.uniform(0,100,size=(l, 1))
    #                 g = np.r_[g, np.zeros([G.shape[1],1])]

    #                 # Converting to matrices
    #                 A, b, G, g = map(lambda x: np.matrix(x), [A, b, G, g])

    #                 #################################################
    #                 # with open(FILE, 'a') as f:
    #                 # f.write(str(A.shape[0]) + '\t' + str(A.shape[1]) + '\t' + \
    #                 #     str(G.shape[0]) + '\t' + str(G.shape[1]) + '\t')

    #                 tmp = OptimalityAnalysis('max', Aeq = A, beq = b, Gub=G, gub=g)
    #                 t0 = time.time()
    #                 succ, solv, mess = tmp.CuttingHyperplane()
    #                 t1 = time.time()
    #                 with open(FILE, 'a') as f:
    #                     # res = str(succ) + '\t' + str(t1-t0) + '\t'
    #                     f.write(f'{succ}\t{t1-t0:.4f}\t')
    #                 succ, solv, mess = tmp.DirectApproach()
    #                 t2 = time.time()
    #                 with open(FILE, 'a') as f:
    #                     # res = str(succ) + '\t' + str(t2-t1) + '\n'
    #                     f.write(f'{succ}\t{t2-t1:.4f}\n')
    #                 #################################################
