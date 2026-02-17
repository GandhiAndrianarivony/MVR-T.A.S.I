# Biorthogonal Wavelets in ADMM: What Changes with bior4.4?

Excellent question — now we’re touching something subtle and important.

Using **Biorthogonal 4.4 (bior4.4)** changes the math slightly compared to Haar/db1.

---

## 1. What changes with bior4.4?

The key difference:

> **bior4.4 is NOT orthonormal.**

This means:

$$
W^T W \neq I
$$

So we must adjust the ADMM x-update carefully.

---

## 2. Wavelet Family Comparison

| Type         | Example         | Property       |
| ------------ | --------------- | -------------- |
| Orthonormal  | db1 (Haar), db2 | $W^T W = I$    |
| Biorthogonal | bior4.4         | $W^T W \neq I$ |

Biorthogonal wavelets use:
- One set of filters for decomposition
- A different dual set for reconstruction

So:
$$
W^{-1} \neq W^T
$$

---

## 3. What remains valid?

Your operator is still correct:

```python
W = LinearOperator(
    matvec=forward_transform,    # Decomposition
    rmatvec=inverse_transform    # Reconstruction
)
```

Even though reconstruction is not literally the transpose of the filter bank, it is the **adjoint operator in the discrete implementation**. Your optimization remains mathematically valid.

---

## 4. What changes in the ADMM math?

Recall the ADMM x-update:

$$
(A^T A + \rho W^T W)x = A^T y + \rho W^T(z-u)
$$

**For Haar (orthonormal):**
$$
W^T W = I
$$
So the system simplifies to:
$$
(A^T A + \rho I)x = A^T y + \rho(z-u)
$$

**For bior4.4 (biorthogonal):**
$$
W^T W \neq I
$$
So you must use the full operator:
$$
W^T(Wx)
$$

---

## 5. Computational implementation

**Instead of:**
```python
A.rmatvec(A.matvec(v)) + rho * v
```

**You must do:**
```python
A.rmatvec(A.matvec(v)) + rho * W.rmatvec(W.matvec(v))
```

---

## 6. Why is this still fine?

Even though $W^T W \neq I$:

- It is **still symmetric positive definite**
- The system remains solvable by CG
- Convergence is still guaranteed
- It just costs **one extra wavelet transform per CG iteration**

---

## 7. Updated ADMM x-step (bior4.4 case)

```python
def normal_eq_matvec(v):
    """Matrix-vector product for the normal equations."""
    return (
        A.rmatvec(A.matvec(v)) 
        + rho * W.rmatvec(W.matvec(v))
    )

# Right-hand side
rhs = A.rmatvec(y) + rho * W.rmatvec(z - u)

# Conjugate gradient solve
x, _ = cg(
    A=LinearOperator(H, matvec=normal_eq_matvec),
    b=rhs,
    x0=x,
    maxiter=50,
    tol=1e-6
)
```

No other changes are needed.

---

## 8. Why use bior4.4 at all?

Biorthogonal 4.4 offers significant advantages:

| Property | Benefit |
|---------|---------|
| **Linear phase** | Preserves edge positions, no phase distortion |
| **Symmetry** | Symmetric filters reduce boundary artifacts |
| **Less ringing** | Fewer Gibbs phenomena around sharp edges |
| **Better sparsity** | More efficient representation of piecewise smooth images |

**In CT reconstruction specifically:**

- Sharper edges with fewer artifacts
- Better perceptual quality
- More stable multilevel decomposition
- Often preferred in medical imaging

---

## 9. Mathematical intuition

**Orthonormal case:**
$$
\|Wx\|_2^2 = \|x\|_2^2
$$
The penalty is isotropic in the original domain.

**Biorthogonal case:**
$$
\|Wx\|_2^2 = x^T (W^T W) x
$$
ADMM now penalizes a **slightly weighted quadratic form**. This can actually improve conditioning.

---

## 10. Does soft thresholding change?

**No.** The z-step remains identical:

```python
z = soft_threshold(Wx + u, lambd/rho)
```

Sparsity is enforced directly on the wavelet coefficients — the transform itself doesn't change this operation.

---

## 11. Practical impact in CT reconstruction

| Aspect | db1 (Haar) | bior4.4 |
|--------|-----------|---------|
| **Speed** | Slightly faster | Slightly slower (extra transforms) |
| **Memory** | Lower | Similar |
| **Edge quality** | Blocky artifacts | Smooth, natural |
| **Ringing** | Less | Minimal |
| **Artifacts** | Checkerboard patterns | Cleaner |
| **Clinical preference** | Rare | Common |

---

## 12. Complete comparison table

| Property              | db1 (Haar)      | bior4.4         |
| --------------------- | --------------- | --------------- |
| Orthonormal           | ✅              | ❌              |
| $W^T W = I$           | ✅              | ❌              |
| Linear phase          | ❌              | ✅              |
| Symmetric filters     | ❌              | ✅              |
| Edge quality          | Moderate        | Better          |
| ADMM speed            | Slightly faster | Slightly slower |
| CG iterations         | Fewer           | Slightly more   |
| Code complexity       | Simple          | Same interface  |
| Mathematical simplicity| Higher         | Lower           |

---

## 13. Bottom line

You can simply use:

```python
wavelet = 'bior4.4'
```

**No structural code changes required** — just ensure your x-update uses the full $W^T(Wx)$ operator rather than assuming $W^T W = I$.

The extra wavelet transform per CG iteration is a small price for significantly better reconstruction quality.

---

## Next steps

If you want, I can:

1. **Show numerical conditioning differences** — condition number of $W^T W$ for bior4.4 vs Haar
2. **Compare reconstruction visually** — Haar vs bior4.4 on the same CT phantom
3. **Explain biorthogonal wavelets mathematically** — dual bases, perfect reconstruction, and the lifting scheme
4. **Discuss tight frames vs orthonormal bases** — and where biorthogonal wavelets fit in
5. **Optimize the CG solve** — preconditioning for $W^T W \neq I$ case

Which direction would be most helpful?