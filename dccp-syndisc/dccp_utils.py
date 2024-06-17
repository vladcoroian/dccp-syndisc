import cvxpy as cp
import dit
import numpy as np
from cvxopt import solvers, matrix, spdiag, log

# from syndisc import disclosure_channel, build_constraint_matrix
from numpy.linalg import svd
from syndisc.solver import extreme_points, lp_sol

# from syndisc.syndisc import build_constraint_matrix
def ext_pts(P, Px, SVDmethod='standard'):
    # this is for numerical stability (avoid singular matrices and so)
    Px = Px + 10 ** -40
    Px = Px / Px.sum()
    Px = np.array([Px]).T  # transform vector input into a nx1 array
    # Find the extremes of the channel polytope
    ext = extreme_points(P, Px, SVDmethod)
    return ext

def build_constraint_matrix(cons, d):
    """
    Build constraint matrix.

    The constraint matrix is a matrix P that is the vertical stack
    of all the preserved marginals.

    Parameters
    ----------
    cons : iter of iter
        List of variable indices to preserve.
    d : dit.Distribution
        Distribution for which to design the constraints

    Returns
    -------
    P : np.ndarray
        Constraint matrix

    """
    # Initialise a uniform distribution to make sure it has full support
    u = dit.distconst.uniform_like(d)
    n = len(u.rvs)
    l = u.rvs
    u = u.coalesce(l + l)

    # Generate one set of rows of P per constraint
    P_list = []
    for c in cons:
        pX123, pX1gX123 = u.condition_on(crvs=range(n, 2*n), rvs=c)

        pX123.make_dense()
        for p in pX1gX123:
          p.make_dense()

        P_list.append(np.hstack([p.pmf[:,np.newaxis] for p in pX1gX123]))

    # Stack rows and return
    P = np.vstack(P_list)

    return P

def compute_matrix_A(P, Px):
    U, S, Vh = svd(P)

    # Extract reduced A matrix using a small threshold to avoid numerical
    # problems
    rankP = np.sum(S > 1e-6)
    A = Vh[:rankP, :]

    b = np.matmul(A, Px)

    # Turn np.matrix into np.array for polytope computations
    A = np.array(A)
    b = np.array(b).flatten()

    return A, b, Vh, rankP

def pick_point_inside_polytope(ext):
    # Pick a linear combination of the extreme points
    coeffs = np.random.rand(len(ext))
    coeffs /= coeffs.sum()
    return np.matmul(ext.T, coeffs)


def flatten(l):
    return [item for sublist in l for item in sublist]


def quantize(array, pres=0.01):
    return ((pres ** (-1)) * array).round() * pres


def convex_check():
    a = cp.Variable(2)
    constraints = [a >= 0, cp.sum(a) == 1]

    p1 = cp.Constant([0, 1])
    p2 = cp.Constant([1, 0])

    obj = cp.Maximize(cp.sum(
        cp.entr(a @ cp.vstack((p1, p2)).T)
    ))

    prob = cp.Problem(obj, constraints)
    prob.solve()

    print(a.value)


def components(dist, cons=None):
    output = dist.rvs[-1]
    inputs = dist.rvs[:-1]
    if cons is None:
        cons = dist.rvs[:-1]

    pX, pWgX = dist.condition_on(flatten(inputs))

    # Make all distributions dense before extracting the PMFs
    pX.make_dense()
    for p in pWgX:
        p.make_dense()

    output_alphabet = len(dist.alphabet[output[0]])
    input_alphabet = np.prod([len(dist.alphabet[i[0]]) for i in inputs])
    pWgX_ndarray = np.zeros((output_alphabet, input_alphabet))

    count = 0
    for i, (_, p) in enumerate(pX.zipped()):
        if p > 0:
            pWgX_ndarray[:, i] = pWgX[count].pmf
            count = count + 1

    P = build_constraint_matrix(cons, dist.coalesce(inputs))
    return P, pX.pmf, pWgX_ndarray
