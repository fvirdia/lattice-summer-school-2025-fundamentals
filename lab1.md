# Verifying the Gaussian Heuristic

I this exercise we will verify the Gaussian Heuristic for the first minimum of the primal and dual lattices of a matrix $\textbf{A} \in \mathbb{Z}_q^{m \times n}$.

## Generating $\bf A$

First, we write a function to generate a random matrix $\textbf{A} \in \mathbb{Z}_q^{m \times n}$.
To do this, you can use Sage's [matrix](https://doc.sagemath.org/html/en/reference/matrices/sage/matrix/constructor.html) function.
To produce random elements in $\mathbb{Z}_q$ there's a few options: you could use the functions in the [prandom](https://doc.sagemath.org/html/en/reference/misc/sage/misc/prandom.html) module, or you could use the `random_element` method of the `Zmod` class.

```python
def gen_A(n: int, m: int, q: int):
    ZZq = Zmod(q)
    A = matrix(ZZq, [
        [ ZZq.random_element() for col in range(n) ]
        for row in range(m)
    ])
    return A
```

## Generating an integer basis for $\Lambda_q(\textbf{A})$

Now, we write a function that given $\bf A$, returning the basis matrix we constructed in class.
This can be achieved by using the `augment` and `stack` methods of the `matrix` class.
If you need identity matrices, you could use the `identity_matrix` function.
For zero-matrices, you can call `matrix` and pass it dimensions but not row/column entries.

```python
def gen_basis(n: int, m: int, q: int):
	A = gen_A(n, m, q)
	B = q * identity_matrix(m)
	B = B.stack(A.transpose().change_ring(ZZ))
	return B.transpose()
```

## Compute the prediction given by the Gaussian Heuristic

Given a basis $\bf B$,  we can estimate the predicted norm of the shortest vector using the Gaussian Heuristic.
Note, when computing the volume of a lattice, make sure to lift the matrix to the integers, by using the `change_ring(ZZ)` method.

```python
def volume(basis):
	B = basis.change_ring(ZZ)
	return (B * B.transpose()).determinant().sqrt()

def gaussian_heuristic(basis):
	import math
	v = volume(basis)
	n = basis.rank()
	return ((math.pi * n)**(1./(2.*n))) * math.sqrt(n/(2 * math.pi * math.e)) * (v ** (1./n))
```

## Find the shortest vector in $\Lambda(\textbf{B})$

Armed with a basis for our lattice, we want to find the shortest vector therein, to measure its norm.

To do so, we will use the [fpylll](https://github.com/fplll/fpylll) library, which comes packaged with Sage.
Fpylll implements various lattice reduction algorithms, including SVP solvers, LLL, and BKZ variants.

From fplll, in this exercise we will use the LLL and Enumeration functionality. In particular, for enumeration to return in a reasonable time, a lattice basis should be somewhat *preprocessed*, for example using LLL.
Enumeration systematically looks for one or more vectors contained in the input lattice, that are also contained within a ball of a given radius, the *enumeration radius*, centered at the origin.

To test the Gaussian Heuristic, we will use it to get a predicted norm of $\lambda_1$ of the shortest vector. We will then relax this radius, looking for the shortest vector within a ball of radius $2 * \lambda_1$, and measure the norm of the returned vector (if any).

While a relative error of 50% may seem somewhat large, this is "less than one bit off", not too bad for such an aggressive heuristic.

In the next exercise, we'll check what happens when we use the Gaussian heuristic as "originally intended".

```python
from fpylll import IntegerMatrix, GSO, LLL, Enumeration

def SVP_oracle(basis):
	B = IntegerMatrix.from_matrix(basis)
	GS = GSO.Mat(B)
	GS.update_gso()
	L = LLL.Reduction(GS)
	L()

	gh = gaussian_heuristic(basis)
	R_sqrd = (1.5 * gh)**2

	E = Enumeration(GS, nr_solutions=1)
	enumres = E.enumerate(
		0, n, R_sqrd, 0
	)
	svpnorm = sqrt(min([sqr_norm for sqr_norm, _ in enumres]))
	return svpnorm, gh

print ("n", "relative error")
q = next_prime(2**7)
for n in range(30, 60, 2):
	m = n
	basis = gen_basis(n, m, q)
	svpnorm, gh = SVP_oracle(basis)
	rel_error = abs(svpnorm - gh)/svpnorm
	print(n, "%.2f%%" % (rel_error*100))
```

## General Gaussian Heuristic: points in volume of ball

We can also extend the exercise to verify the general Gaussian heuristic, estimating the number of points in a ball.

In this example, we generate the basis differently, by using fpylll's internal generator, and allow the enumeration routine to return more than one point.

```python
from fpylll import IntegerMatrix, GSO, LLL, Enumeration
import fpylll

def vol_n_ball(n, R):
	return float(RR(R)**n * RR(math.pi)**(n/2) / RR(math.gamma(n/2+1)))

def GH_test(n):
	B = IntegerMatrix(n, n)
	B.randomize("qary", k=n//2, q=127)
	# B.randomize("uniform", bits=20)
	GS = GSO.Mat(B)
	GS.update_gso()
	L = LLL.Reduction(GS)
	L()

	M = matrix(n)
	B.to_matrix(M)
	v = abs(M.determinant())
	# gh = gaussian_heuristic(basis)
	gh = ((math.pi * n)**(1./(2.*n))) * math.sqrt(n/(2 * math.pi * math.e)) * (v ** (1./n))
	R_sqrd = (1.1 * gh)**2
	vol_ball = vol_n_ball(n, sqrt(R_sqrd))
	# vol_B = volume(basis).n()

	E = Enumeration(GS, nr_solutions=10*float(vol_ball / v))
	enumres = E.enumerate(
		0, n, R_sqrd, 0,
		pruning=fpylll.fplll.pruner.PruningParams.LinearPruningParams(n, int(n * .4)).coefficients
	)
	nsols = len(enumres)
	svpnorm = sqrt(min([sqr_norm for sqr_norm, _ in enumres]))
	return nsols, float(vol_ball / v), svpnorm, gh


print ("n", "relative error points", "relative error lambda_1")
q = next_prime(2**6)
for n in range(30, 60, 2):
	points, gh_points, svpnorm, gh = GH_test(n)
	rel_error_points = abs(svpnorm - gh)/svpnorm
	rel_error = abs(points - gh_points)/points
	print(n, "%.2f%%" % (rel_error_points*100), "%.2f%%" % (rel_error*100))
```