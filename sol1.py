from sage.all import Zmod, matrix, identity_matrix, ZZ, RR, next_prime, gamma, pi, vector, sqrt
from fpylll import IntegerMatrix, GSO, LLL, Enumeration
import fpylll


def gen_A(n: int, m: int, q: int):
    ZZq = Zmod(q)
    A = matrix(ZZq, [
        [ ZZq.random_element() for col in range(n) ]
        for row in range(m)
    ])
    return A


def gen_basis(n: int, m: int, q: int):
	A = gen_A(n, m, q)
	B = q * identity_matrix(m)
	B = B.stack(A.transpose().change_ring(ZZ))
	return B.transpose()


def volume(basis):
	B = basis.change_ring(ZZ)
	return (B * B.transpose()).determinant().sqrt()


def gaussian_heuristic(vol, rank):
	import math
	v = vol
	n = rank
	return ((math.pi * n)**(1./(2.*n))) * math.sqrt(n/(2 * math.pi * math.e)) * (v ** (1./n))


def SVP_oracle(basis, radius: float):
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


print("Gaussian heuristic for lambda_1")
print ("n", "relative error")
q = next_prime(2**7)
for n in range(30, 60, 2):
	m = n
	basis = gen_basis(n, m, q)
	vol = volume(basis)
	rank = basis.rank()
	gh = gaussian_heuristic(vol, rank)
	enumres, reduced_basis = SVP_oracle(basis, 1.5 * gh)
	svp_sqr_norm = enumres[0][0]
	sv_coeff = vector(enumres[0][1])
	sv = sv_coeff * reduced_basis
	assert abs(sv.norm()**2 - svp_sqr_norm) < 0.01
	rel_error = abs(sqrt(svp_sqr_norm) - gh)/sqrt(svp_sqr_norm)
	print(n,"%.2f, %.2f, %.2f%%" % (gh, sqrt(svp_sqr_norm), rel_error*100))
print()


def radius_ball(vol_ball, rank):
	return gamma(rank/2 + 1)**(1/rank) * vol_ball**(1/rank) / sqrt(pi)


def gen_random_basis(rank):
	B = IntegerMatrix(n, n)
	B.randomize("qary", k=n//2, q=127)
	# B.randomize("uniform", bits=20) # a different basis distribution!
	basis = matrix(n, n)
	B.to_matrix(basis)
	return basis


def GH_test(basis, N):
	B = IntegerMatrix.from_matrix(basis)
	GS = GSO.Mat(B)
	GS.update_gso()
	L = LLL.Reduction(GS)
	L()

	vol_lattice = volume(basis)
	rank = basis.rank()
	vol_ball = N * vol_lattice
	rad_ball = radius_ball(vol_ball, rank)
	R_sqrd = rad_ball**2

	E = Enumeration(GS, nr_solutions=10 * N)
	enumres = E.enumerate(
		0, n, R_sqrd, 0,
		pruning=fpylll.fplll.pruner.PruningParams.LinearPruningParams(rank, int(rank * .4)).coefficients
	)
	nsols = len(enumres)
	return nsols


print("Gaussian heuristic for N points")
print ("n", "relative error points", "relative error lambda_1")
q = next_prime(2**6)
for n in range(30, 60, 2):
	basis = gen_random_basis(n)
	N = 1000
	points = GH_test(basis, N)
	points *= 2 # account for +/- v in the ball
	rel_error = abs(points - N)/points
	print(n, N, points, "%.2f%%" % (rel_error*100))
