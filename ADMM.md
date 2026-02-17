Great question — **section 4 is the heart of ADMM**. If this part clicks, *everything* else becomes mechanical. I’ll go slowly, cleanly, and stay close to the math you’re actually using.

We start from the **wavelet-regularized CT problem**:

\[
\min_x \frac{1}{2}\|Ax - y\|_2^2 + \lambda \|W x\|_1
\]

---

## 0. Why ADMM is needed (one sentence)

You can’t minimize this directly because:

* \( \|Ax - y\|_2^2 \) wants **least squares**
* \( \|Wx\|_1 \) wants **soft thresholding**

ADMM lets each term do what it’s good at.

---

## 1. Variable splitting (the key trick)

Introduce an auxiliary variable:
\[
z = W x
\]

Rewritten problem:

\[
\min_{x,z}
\frac{1}{2}\|Ax - y\|_2^2
+
\lambda \|z\|_1
\quad \text{s.t. } \quad
W x - z = 0
\]

Now:

* \(x\) appears only in smooth quadratic terms
* \(z\) appears only in the \( \ell_1 \) norm

---

## 2. Augmented Lagrangian (scaled form)

ADMM minimizes this instead:

\[
\mathcal{L}(x,z,u) = \frac{1}{2}\|Ax - y\|_2^2 + \lambda \|z\|_1 + \frac{\rho}{2}\|W x - z + u\|_2^2
\]

* \(u\) = scaled dual variable
* \( \rho > 0 \) = penalty parameter

This last term:

* Encourages \(Wx \approx z\)
* Makes the problem strictly convex in \(x\)

---

## 4. Core ADMM math (step-by-step)

ADMM alternates between **three minimizations**.

---

## 4.1 x-update (quadratic minimization)

We fix \(z^k, u^k\) and solve:

\[
x^{k+1} = \arg\min_x \frac{1}{2}\|Ax - y\|_2^2 + \frac{\rho}{2}\|W x - z^k + u^k\|_2^2
\]

### Expand both terms

First term:
\[
\frac{1}{2}(Ax-y)^T(Ax-y)
\]

Second term:
\[
\frac{\rho}{2}(Wx - z^k + u^k)^T(Wx - z^k + u^k)
\]

---

### Take gradient and set to zero

\[
\nabla_x = A^T(Ax - y) + \rho W^T(Wx - z^k + u^k) = 0
\]

Rearrange:

\[
\boxed{(A^T A + \rho W^T W)x = A^T y + \rho W^T(z^k - u^k)}
\]

This is a **linear system**.

---

### Why this is solvable

* \(A^T A\) is symmetric positive semi-definite
* \(W^T W\) is positive definite
* Their sum is well-conditioned

You solve it with **conjugate gradient**, not inversion.

---

## 4.2 z-update (proximal operator)

Now fix \(x^{k+1}, u^k\):

\[
z^{k+1} = \arg\min_z \lambda \|z\|_1 + \frac{\rho}{2}\|z - (Wx^{k+1} + u^k)\|_2^2
\]

This is the **proximal operator of the \( \ell_1 \) norm**.

### Closed-form solution

\[
\boxed{z^{k+1} = \mathcal{S}_{\lambda/\rho}(Wx^{k+1} + u^k)}
\]

Where soft thresholding is:

\[
\mathcal{S}_\tau(t) = \text{sign}(t)\max(|t|-\tau, 0)
\]

---

### Intuition

* Small wavelet coefficients → killed
* Large coefficients → shrunk
* Sparsity is enforced **here**, not in the x-step

---

## 4.3 u-update (dual ascent)

Finally:

\[
\boxed{u^{k+1} = u^k + (W x^{k+1} - z^{k+1})}
\]

This tracks **constraint violation**.

* If \(Wx = z\), \(u\) stops changing
* If not, \(u\) increases the penalty

---

## 5. Special simplification: orthonormal wavelets

For Haar / db1:

\[
W^T W = I
\]

So x-update becomes:

\[
\boxed{(A^T A + \rho I)x = A^T y + \rho W^T(z - u)}
\]

This is **huge**:

* No wavelet operator inside CG loop
* Much faster

---

## 6. What each step is *really doing*

| Step     | Meaning                                |
| -------- | -------------------------------------- |
| x-update | Fit CT data + respect current sparsity |
| z-update | Enforce wavelet sparsity               |
| u-update | Enforce consistency (Wx = z)           |

They **take turns pulling the solution** toward their own objective.

---

## 7. Why ADMM converges

ADMM works because:

* Each subproblem is convex
* Constraint violation is penalized
* Dual variables coordinate agreement

Under mild conditions:
\[
x^k \rightarrow x^\star
\]

---

## 8. One-line mental model

> **ADMM alternates between physics (Ax≈y), sparsity (Wx sparse), and negotiation (u).**

---

If you want next, I can:

* Derive this using **KKT conditions**
* Show convergence diagnostics (primal/dual residuals)
* Compare this with ISTA mathematically