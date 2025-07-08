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


def bkz(beta, n):
    Basis = IntegerMatrix(n, n)
    Basis.randomize("qary", k=n//2, q=127)

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

    reduced_basis = matrix([[0 for _ in range(bkz.A.ncols)]
                     for __ in range(bkz.A.nrows)])
    bkz.A.to_matrix(reduced_basis)  # type: ignore #noqa

    return profile, reduced_basis
```

On a Sage notebook, you can generate a plot of the profile by using the `line` function.

```python
profile, basis = bkz(20, 40)
g = line([(i, profile[i]) for i in range(len(profile))])
show(g)
```

It would be interesting to check how the GSA behaves as the kind of basis being generated changes.
You can find more kinds of supported basis on https://github.com/fplll/fpylll/blob/master/docs/tutorial.rst
