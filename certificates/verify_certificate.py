"""Check exact certificates of spherical codes in exact arithmetic.

A certificate (<N>x<D>.json in this folder, format in README.md) claims that N unit vectors exist in R^D whose
largest pairwise cosine is exactly the algebraic number t. This script proves the claim from the file alone.
It does not prove that the code is optimal.

Proved, with rational numbers only (python-flint for the number field, fractions for the intervals):
  1  The field polynomial P is irreducible over Q and has exactly one root in a rational interval [a, b] of
     width 2e-100. The field is K = Q[x]/(P), embedded in R by that root. The largest cosine t is the stated element
     of K; it is a root of the stated irreducible polynomial, the only one in the interval t takes on [a, b].
  2  The Gram matrix A of the D basis points is positive definite: LDL^T in K, every pivot positive on [a, b].
  3  With B the inner products of the basis points with all points that have contacts, G = B^T A^-1 B holds
     exactly in K for every pair, diagonal included. So the Gram matrix G has rank D, is positive semidefinite
     and has unit diagonal: the points with contacts exist in R^D.
  4  Every inner product other than t is below t on [a, b], t occurs, 0 < t < 1, and every point in the list
     of points with contacts has a contact.
  5  Every point without contacts is a normalized rational combination of the basis points, and its cosine
     with every other point is below t (compared without square roots, on [a, b]).

Usage
    python certificates/verify_certificate.py certificates/*.json

Prints one line per certificate. Exit status 1 if any certificate fails.
Requires python-flint.
"""
import json, sys
from fractions import Fraction as F

from flint import fmpq, fmpq_poly, fmpz_poly

ZERO, ONE = fmpq_poly([0]), fmpq_poly([1])
WIDTH = F(1, 10 ** 100)                                      # half width of the interval [a, b]


def irreducible(coef_hi):
    content, factors = fmpz_poly(coef_hi[::-1]).factor()
    return abs(int(content)) == 1 and len(factors) == 1 and factors[0][1] == 1


def horner(coef_low, lo, hi):
    """Rational bounds of sum c_k y^k for y in [lo, hi]."""
    a = b = F(0)
    for c in reversed(coef_low):
        cand = (a * lo, a * hi, b * lo, b * hi)
        a, b = min(cand) + c, max(cand) + c
    return a, b


def one_root(coef_hi, lo, hi):
    """Exactly one root in [lo, hi]: a sign change at the ends and a derivative of one sign."""
    low = [F(c) for c in coef_hi[::-1]]
    if lo == hi:                                             # a rational number: it is the root or it is not
        return horner(low, lo, lo)[0] == 0
    ends = horner(low, lo, lo)[0] * horner(low, hi, hi)[0]
    dlo, dhi = horner([k * low[k] for k in range(1, len(low))], lo, hi)
    return ends < 0 and (dlo > 0 or dhi < 0)


def bracket(coef_hi, approx):
    """A rational interval around the root nearest to approx: Newton steps in rationals, then +- WIDTH."""
    low = [F(c) for c in coef_hi[::-1]]
    if len(low) == 2:
        return -low[0] / low[1], -low[0] / low[1]
    der = [k * low[k] for k in range(1, len(low))]
    x = F(approx)
    for _ in range(5):
        x -= horner(low, x, x)[0] / horner(der, x, x)[0]
        x = F(round(x * 10 ** 130), 10 ** 130)
    return x - WIDTH, x + WIDTH


def check(path):
    cert = json.loads(open(path).read())
    n, d = cert["n"], cert["d"]
    with_contacts, without_contacts = cert["points_with_contacts"], cert["points_without_contacts"]
    labels, S = cert["gram_labels"], cert["basis"]
    coef = [int(c) for c in cert["field_poly"]]              # P, highest degree first
    tp = [int(c) for c in cert["cosine_poly"]]
    deg, m = len(coef) - 1, len(with_contacts)
    P = fmpq_poly(coef[::-1])
    ok = {}

    def elem(strings):
        return fmpq_poly([fmpq(F(c).numerator, F(c).denominator) for c in strings]) % P

    def interval(e, lo, hi):
        return horner([F(int(e[k].numer()), int(e[k].denom())) for k in range(deg)], lo, hi)

    def inv(e):
        gcd, s, _ = e.xgcd(P)
        return s * fmpq_poly([1 / gcd[0]]) % P

    def dot(u, v):
        return sum((x * y for x, y in zip(u, v)), ZERO) % P

    values = [elem(v) for v in cert["gram_values"]]
    t = elem(cert["cosine_in_field"])
    contact = [i for i, v in enumerate(values) if v == t]
    others = [v - t for i, v in enumerate(values) if i not in contact]

    ok["format"] = (sorted(with_contacts + without_contacts) == list(range(n))
                    and len(S) == d == len(set(S)) and all(0 <= s < m for s in S)
                    and len(labels) == m and all(len(r) == m for r in labels)
                    and all(labels[i][i] == -1 for i in range(m))
                    and all(0 <= labels[i][j] < len(values) and labels[i][j] == labels[j][i] for i in range(m) for j in range(i))
                    and len(cert["positions_without_contacts"]) == len(without_contacts))
    if not ok["format"]:
        print(f"{n} x {d}: NOT VERIFIED (the file does not have the stated format)")
        return False

    # 1. P irreducible with one root in [a, b]; t is the root of its polynomial
    a, b = bracket(coef, cert["field_root"])
    ok["field polynomial"] = irreducible(coef) and one_root(coef, a, b)
    tlo, thi = interval(t, a, b)
    ok["cosine polynomial"] = (irreducible(tp) and one_root(tp, tlo, thi)
                               and sum((fmpq_poly([c]) * t ** k for k, c in enumerate(tp[::-1])), ZERO) % P == ZERO)

    def g(i, j):
        return ONE if i == j else values[labels[i][j]]

    # 2. LDL^T of A = G[S, S] in K, pivots positive on [a, b]
    A = [[g(i, j) for j in S] for i in S]
    L = [[ZERO] * d for _ in range(d)]
    piv = []
    for k in range(d):
        piv.append((A[k][k] - sum((L[k][s] * L[k][s] % P * piv[s] for s in range(k)), ZERO)) % P)
        if piv[k] == ZERO:
            break
        di = inv(piv[k])
        for i in range(k + 1, d):
            L[i][k] = (A[i][k] - sum((L[i][s] * L[k][s] % P * piv[s] for s in range(k)), ZERO)) * di % P
    ok["basis positive definite"] = len(piv) == d and all(interval(p, a, b)[0] > 0 for p in piv)
    if not ok["basis positive definite"]:
        print(f"{n} x {d}: NOT VERIFIED (the basis points do not have a positive definite Gram matrix)")
        return False

    # A^-1 by Gauss-Jordan in K
    M = [A[i][:] + [ONE if i == j else ZERO for j in range(d)] for i in range(d)]
    for k in range(d):
        p = next(r for r in range(k, d) if M[r][k] != ZERO)
        M[k], M[p] = M[p], M[k]
        ik = inv(M[k][k])
        M[k] = [v * ik % P for v in M[k]]
        for r in range(d):
            if r != k and M[r][k] != ZERO:
                f = M[r][k]
                M[r] = [(v - f * w) % P for v, w in zip(M[r], M[k])]
    Ainv = [row[d:] for row in M]
    ok["inverse of the basis Gram matrix"] = all(dot(A[i], [Ainv[k][j] for k in range(d)]) == (ONE if i == j else ZERO)
                                                 for i in range(d) for j in range(d))

    # 3. G = B^T A^-1 B on all pairs of points with contacts
    B = [[g(s, j) for s in S] for j in range(m)]             # column j of B
    W = [[dot(Ainv[r], B[j]) for r in range(d)] for j in range(m)]
    ok["rank D, positive semidefinite, unit diagonal"] = all(dot(B[i], W[j]) == g(i, j) for i in range(m) for j in range(i, m))

    # 4. the other inner products are below t, and t is a cosine of distinct points
    pairs = sum(labels[i][j] in contact for i in range(m) for j in range(i))
    ok["other inner products below t"] = (len(contact) == 1 and pairs > 0 and tlo > 0 and thi < 1
                                          and all(interval(e, a, b)[1] < 0 for e in others))
    used = {labels[i][j] for i in range(m) for j in range(i)}
    ok["every point has a contact"] = all(any(labels[i][j] in contact for j in range(m) if j != i) for i in range(m))

    # 5. points without contacts
    pos = [[F(c) for c in r] for r in cert["positions_without_contacts"]]
    good = all(len(c) == d for c in pos)
    if pos and good:
        def lin(c, vec):                                     # sum_k c_k vec_k in K
            return sum((fmpq_poly([fmpq(ck.numerator, ck.denominator)]) * v for ck, v in zip(c, vec)), ZERO) % P

        def below_t(num, den2):                              # num / sqrt(den2) < t on [a, b], given den2 > 0
            hi = interval(num, a, b)[1]
            return hi <= 0 or hi * hi < tlo * tlo * interval(den2, a, b)[0]

        Ac = [[lin(c, A[r]) for r in range(d)] for c in pos]                 # A c
        q = [lin(c, Ac[i]) for i, c in enumerate(pos)]                       # c^T A c, the squared length
        good = all(interval(x, a, b)[0] > 0 for x in q)
        for i, c in enumerate(pos):
            good = good and all(below_t(lin(c, B[j]), q[i]) for j in range(m))               # with the points with contacts
            good = good and all(below_t(lin(c, Ac[j]), q[i] * q[j] % P) for j in range(i))   # with each other
    ok["points without contacts below t"] = good

    fails = [k for k, v in ok.items() if not v]
    print(f"{n} x {d}: largest cosine {float(tlo):.15f}, polynomial degree {len(tp) - 1}, field degree {deg}, "
          f"{m} points with contacts + {len(pos)} without, {pairs} contact pairs, {len(used)} distinct inner products:  "
          + ("VERIFIED" if not fails else f"NOT VERIFIED {fails}"), flush=True)
    return not fails


if __name__ == "__main__":
    results = [check(p) for p in sys.argv[1:]]
    print(f"{sum(results)} of {len(results)} certificates verified")
    sys.exit(0 if all(results) else 1)
