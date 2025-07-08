from sage.all import matrix, log, line, show
import fpylll
from fpylll import IntegerMatrix
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.fplll.bkz_param import BKZParam


def gen_random_basis(rank):
	B = IntegerMatrix(rank, rank)
	B.randomize("qary", k=rank//2, q=127)
	# B.randomize("uniform", bits=20) # a different basis distribution!
	basis = matrix(rank, rank)
	B.to_matrix(basis)
	return basis


# You can use gen_random_basis from lab 1, 
# or you could try building square LWE embedding matrices 
# such as those in https://eprint.iacr.org/2017/815
def bkz(basis, beta):
    Basis = IntegerMatrix.from_matrix(basis)

    Basis_GSO = fpylll.GSO.Mat(Basis, float_type="d")
    Basis_GSO.update_gso()

    # run LLL on the basis to preprocess it
    lll = fpylll.LLL.Reduction(Basis_GSO)
    lll()

    # set up BKZ
    params_fplll = BKZParam(
        block_size=beta,
        strategies=fpylll.BKZ.DEFAULT_STRATEGY,
        flags=0
        | fpylll.BKZ.VERBOSE
        | fpylll.BKZ.AUTO_ABORT
        | fpylll.BKZ.GH_BND
        | fpylll.BKZ.MAX_LOOPS,
        max_loops=20)
    bkz = BKZReduction(Basis_GSO)
    bkz(params_fplll)

    Basis_GSO.update_gso()
    profile = [log(Basis_GSO.get_r(i, i), 2)/2 for i in range(Basis_GSO.d)]

    reduced_basis = matrix([[0 for _ in range(Basis_GSO.B.ncols)] for __ in range(Basis_GSO.B.nrows)])
    Basis_GSO.B.to_matrix(reduced_basis)

    return profile, reduced_basis


rank = 30
beta = 20
basis = gen_random_basis(rank)
profile, reduced_basis = bkz(basis, beta)
g = line([(i, profile[i]) for i in range(len(profile))])
show(g)


# if you are running on your computer and want to save the plot,
# use the following:
from sage.all import save
save(g, "test.png", dpi=150)