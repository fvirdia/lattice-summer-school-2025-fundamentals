# Verifying the Geometric Series Assumption

Prerequisite: Lab 1

In this lab, we extend our predictions to the GSA.

To do this, we will need to extract the basis profile.
Generating Gram-Schmidt orthogonalisations can be a bit annoying, however fplll keeps track of those internally, which is quite convenient.

You can use the following routine to perform BKZ-beta reduction of a basis.

```python
import fpylll
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.fplll.bkz_param import BKZParam


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
```

On a Sage notebook, you can generate a plot of the profile by using the `line` and `show` functions.

Does the GSA seem to hold at various dimensions?
What happens if you choose smaller `q` values? For example, running BKZ-2, can you ever see the profile not being a straight line?

You can find more kinds of supported basis on https://github.com/fplll/fpylll/blob/master/docs/tutorial.rst
