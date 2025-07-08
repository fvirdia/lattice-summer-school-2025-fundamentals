# Verifying the Gaussian Heuristic

I this exercise we will verify the Gaussian Heuristic for the first minimum of the primal and dual lattices of a matrix $\textbf{A} \in \mathbb{Z}_q^{m \times n}$.

## Generating $\bf A$

First, we write a function to generate a random matrix $\textbf{A} \in \mathbb{Z}_q^{m \times n}$.
To do this, you can use Sage's [matrix](https://doc.sagemath.org/html/en/reference/matrices/sage/matrix/constructor.html) function.
To produce random elements in $\mathbb{Z}_q$ there's a few options: you could use the functions in the [prandom](https://doc.sagemath.org/html/en/reference/misc/sage/misc/prandom.html) module, or you could use the `random_element` method of the `Zmod` class.

## Generating an integer basis for $\Lambda_q(\textbf{A})$

Now, we write a function that given $\bf A$, returning the basis matrix we constructed in class.
This can be achieved by using the `augment` and `stack` methods of the `matrix` class.
If you need identity matrices, you could use the `identity_matrix` function.
For zero-matrices, you can call `matrix` and pass it dimensions but not row/column entries.

## Compute the prediction given by the Gaussian Heuristic

Given a basis $\bf B$,  we can estimate the predicted norm of the shortest vector using the Gaussian Heuristic.
Note, when computing the volume of a lattice, make sure to lift the matrix to the integers, by using the `change_ring(ZZ)` method.
Note that $\bf B B^T$ is not the same as $\bf B^T B$.

## Find the shortest vector in $\Lambda(\textbf{B})$

Armed with a basis for our lattice, we want to find the shortest vector therein, to measure its norm.

To do so, we will use the [fpylll](https://github.com/fplll/fpylll) library, which comes packaged with Sage.
Fpylll implements various lattice reduction algorithms, including SVP solvers, LLL, and BKZ variants.

From fplll, in this exercise we will use the LLL and Enumeration functionality. In particular, for enumeration to return in a reasonable time, a lattice basis should be somewhat *preprocessed*, for example using LLL.
Enumeration systematically looks for one or more vectors contained in the input lattice, that are also contained within a ball of a given radius, the *enumeration radius*, centered at the origin.

Since calling fplll can be tricky, you can find an `SVP_oracle` function below, that given a basis and a radius, it looks for the shortest non-zero vector within that radius, and returns `((sqr_norm, x), B)` where `B` is the basis output by calling LLL over your input basis, and `x` is a coefficient vector such that `y = vector(x) * B` is the shortest vector in the lattice spanned by the input basis. You can verify that `y.norm()**2 == sqr_norm`.
NOTE: fplll returns squared norms.

To test the Gaussian Heuristic, we will use it to get a predicted norm of $\lambda_1$ of the shortest vector. We will then relax this radius, and call the SVP oracle, looking for the shortest vector within a ball of radius $1.5 * \lambda_1$ around the origin, and measure the norm of the returned vector (if any). You should be able to go for bases where $n \approx m \approx 30$, and somewhat larger too.

While a relative error of 50% may seem somewhat large, this is "less than one bit off", not too bad for such an aggressive heuristic.
In the next exercise, we'll check what happens when we use the Gaussian heuristic as "originally intended".

```python
def SVP_oracle(basis, radius: float):
	from fpylll import IntegerMatrix, GSO, LLL, Enumeration
	B = IntegerMatrix.from_matrix(basis)
	GS = GSO.Mat(B)
	GS.update_gso()
	L = LLL.Reduction(GS)
	L()

	R_sqrd = radius**2
	E = Enumeration(GS, nr_solutions=1)
	enumres = E.enumerate(
		0, n, R_sqrd, 0
	)
	reduced_basis = matrix([[0 for _ in range(GS.B.ncols)] for __ in range(GS.B.nrows)])
	GS.B.to_matrix(reduced_basis)
	return enumres, reduced_basis
```

## General Gaussian Heuristic: points in volume of ball

We can also extend the exercise to verify the general Gaussian heuristic, estimating the number of points in a ball.
To do so, you will want to
1. generate a basis and compute its volume V and rank K
2. choose some multiple of the lattice volume, say N * V
3. compute the radius R of a rank-K ball of volume V
4. ask fplll to enumerate a lattice, looking for points of norm <= R
5. check whether the number of points is close to the Gaussian heuristic prediction, that would be N

For this example, I'll provide a basis generator that you can use to test the heuristic on different lattice distributions.

In this example, we generate the basis differently, by using fpylll's internal generator, and allow the enumeration routine to return more than one point.
I'll also provide an example of how to call the Enumeration function within SVP_oracle so that it returns more than one lattice point.
Since we are expecting N many points, I would suggest to give it slack and allow up to, say, 10*N many solutions.
NOTE: Enumeration will return only "+v" whenever it encounters both "+/- v" within the ball around the origin. Account for this when computing the precision of the Gaussian heuristic.

Does the Gaussian heuristic hold at different ranks k? What about different bases distributions?
What about higher moments of the distribution of number of points?
The Gaussian heuristic says that the expectation of the number of points should be volume(ball)/volume(lattice).
What about the variance? Maybe Corollary 8 in https://espitau.github.io/bin/random_lattice.pdf helps (it's for random real unimodular lattices!)

```python
def gen_random_basis(rank):
	from fpylll import IntegerMatrix
	B = IntegerMatrix(n, n)
	B.randomize("qary", k=n//2, q=127)
	# B.randomize("uniform", bits=20) # a different basis distribution!
	basis = matrix(n, n)
	B.to_matrix(basis)
	return basis

# how to call Enumeration to return 10*N many points in a lattice of rank k
# GS and R_sqrt are as in the SVP_oracle example
import fpylll
E = Enumeration(GS, nr_solutions=10*N)
enumres = E.enumerate(
	0, n, R_sqrd, 0,
	pruning=fpylll.fplll.pruner.PruningParams.LinearPruningParams(k, int(k * .4)).coefficients
)
```
