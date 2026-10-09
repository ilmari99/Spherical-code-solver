# Certificates of the exact codes

This folder holds an exact certificate for each of the 24 codes in the paper's table of exact codes (`<N>x<D>.json`), and `verify_certificate.py`, a script that proves a certificate from the file alone.

A certificate proves that a code with exactly the stated largest cosine exists. It does not prove that the code is optimal.

A *contact* is a pair of points whose cosine equals the largest cosine $t$ of the code. A point without contacts (a rattler in the paper) can move without changing $t$, so its position is not determined; the certificate places it at an explicit position. Removing points without contacts leaves $t$ unchanged, so a certificate for $N$ points with $k$ points without contacts also gives codes with the same largest cosine for $N-1, \ldots, N-k$ points.

## Checking the certificates

    pip install python-flint
    python certificates/verify_certificate.py certificates/*.json

The script prints one line per certificate and takes about 12 seconds for all 24. It uses rational arithmetic only and proves, for each file:

1. The field polynomial $P$ is irreducible and has exactly one root in a rational interval of width $2\cdot10^{-100}$. The number field is $K = \mathbb{Q}[x]/(P)$, embedded in the real numbers by that root. The largest cosine $t$ is the stated element of $K$ and a root of the stated irreducible polynomial.
2. The Gram matrix $A$ of the $D$ basis points is positive definite.
3. With $B$ the inner products of the basis points with all points that have contacts, the Gram matrix of those points equals $B^\top A^{-1} B$ exactly, diagonal included. It therefore has rank $D$, is positive semidefinite and has unit diagonal, so the points exist in $\mathbb{R}^D$.
4. Every inner product other than $t$ is below $t$, $t$ occurs, $0 < t < 1$, and every point in the first list has a contact.
5. Every point without contacts has a cosine below $t$ with every other point.

## File format

| key | content |
|---|---|
| `n`, `d` | number of points $N$ and dimension $D$ |
| `cosine_poly` | minimal polynomial of the largest cosine $t$: integer coefficients, highest degree first |
| `field_poly` | the polynomial $P$ of the number field $K = \mathbb{Q}[x]/(P)$, in the same convention |
| `field_root` | 40 digits of the real root of $P$ that embeds $K$ in the real numbers |
| `cosine_in_field` | $t$ as an element of $K$ |
| `points_with_contacts` | indices of the $m$ points that have a contact |
| `points_without_contacts` | indices of the other points; the two lists together are $0, \ldots, N-1$ |
| `gram_values` | the distinct inner products between points with contacts, as elements of $K$ |
| `gram_labels` | $m \times m$ matrix; entry $(i, j)$ is the index in `gram_values` of the inner product of the $i$-th and the $j$-th point with contacts, and $-1$ on the diagonal |
| `basis` | $D$ positions in `points_with_contacts` whose Gram matrix is positive definite |
| `positions_without_contacts` | for each point without contacts, $D$ rational numbers $c_k$: the point is $\sum_k c_k b_k$ divided by its length, with $b_k$ the basis points |

An element of $K$ is a list of $\deg P$ rational numbers, written as strings: the coefficients of $1, x, x^2, \ldots$

In 21 of the 24 certificates every inner product lies in $\mathbb{Q}(t)$, and `field_poly` equals `cosine_poly`. In the other three (53 × 24, 56 × 25 and 64 × 25) $K$ is a quadratic extension of $\mathbb{Q}(t)$.

The point indices are those of the code the certificate was computed from. The certificate determines the code up to an orthogonal transformation.
