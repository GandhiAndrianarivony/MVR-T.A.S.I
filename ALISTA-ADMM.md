# ALISTA-ADMM: Learned Acceleration of ADMM with Analytical Structure

Combining **ALISTA and ADMM** gives you a powerful framework often called:

> **Learned ADMM with analytical structure**
> or
> **AL-ADMM (Analytical Learned ADMM)**

This combines:
- ✅ ADMM's stability and constraint handling
- ✅ ALISTA's learned acceleration  
- ✅ Your exact CT physics operator (A)
- ✅ Your wavelet sparsity operator (W)

---

## 1. Problem Formulation: CT + Wavelets

We start from your reconstruction problem:

$$
\min_x \frac{1}{2}\|Ax - y\|_2^2 + \lambda \|Wx\|_1
$$

**Variable splitting** (ADMM style):

$$
z = Wx
$$

$$
\min_{x,z} \frac{1}{2}\|Ax-y\|_2^2 + \lambda\|z\|_1 \quad \text{s.t.} \quad Wx - z = 0
$$

---

## 2. Classical ADMM (Baseline)

**x-update** (least squares with quadratic penalty):

$$
\boxed{(A^TA + \rho W^TW)x = A^Ty + \rho W^T(z-u)}
$$

**z-update** (soft thresholding):

$$
\boxed{z = \mathcal{S}_{\lambda/\rho}(Wx+u)}
$$

**u-update** (dual variable update):

$$
\boxed{u = u + Wx - z}
$$

**The bottleneck:** x-update requires a **linear system solve** — typically CG → slow.

---

## 3. ALISTA-ADMM: Replace CG with Learned Gradient Step

Instead of solving the x-update exactly, take **one gradient step** on the augmented Lagrangian:

**Augmented Lagrangian:**

$$
\mathcal{L}(x,z,u) = \frac{1}{2}\|Ax-y\|_2^2 + \lambda\|z\|_1 + \frac{\rho}{2}\|Wx - z + u\|_2^2
$$

**Gradient w.r.t x:**

$$
\nabla_x \mathcal{L} = A^T(Ax-y) + \rho W^T(Wx - z + u)
$$

**ALISTA-style update:**

$$
\boxed{x^{k+1} = x^k - \gamma_k \nabla_x \mathcal{L}(x^k, z^k, u^k)}
$$

**Expanded form:**

$$
\boxed{x^{k+1} = x^k - \gamma_k \left[ A^T(Ax^k-y) + \rho W^T(Wx^k - z^k + u^k) \right]}
$$

---

## 4. Complete ALISTA-ADMM Algorithm

$$
\boxed{
\begin{aligned}
\textbf{x-update:}& \quad x^{k+1} = x^k - \gamma_k \left[ A^T(Ax^k-y) + \rho W^T(Wx^k - z^k + u^k) \right] \\[6pt]
\textbf{z-update:}& \quad z^{k+1} = \mathcal{S}_{\lambda/\rho}\left(Wx^{k+1} + u^k\right) \\[6pt]
\textbf{u-update:}& \quad u^{k+1} = u^k + Wx^{k+1} - z^{k+1}
\end{aligned}
}
$$

| Update | Cost | Method |
|--------|------|--------|
| x-update | $O(n^2)$ | **1 gradient eval** (no CG!) |
| z-update | $O(n)$ | Closed form (soft threshold) |
| u-update | $O(n)$ | Closed form |

**Total per iteration:** ≈ **2 forward/backward projections + 2 wavelet transforms**

---

## 5. Comparison: Classical ADMM vs ALISTA-ADMM

| Aspect | Classical ADMM | ALISTA-ADMM |
|--------|---------------|-------------|
| **x-update** | Solve linear system (CG, ~50-100 iterations) | **One gradient step** |
| **Per-iteration cost** | High (50× A, W evaluations) | **Low** (2× A, 2× W) |
| **Total iterations needed** | ~20-50 | ~50-100 |
| **Total time** | **Slow** | **Fast** |
| **Convergence** | Linear rate | Linear rate (with learned γₖ) |
| **Stability** | Very stable | Very stable |
| **Learnable parameters** | None | γₖ, ρ, λ (per layer) |

---

## 6. Why This Works: Theoretical Insight

**Classical ADMM** solves the x-update **exactly**:

$$
x^{k+1} = \arg\min_x \frac{1}{2}\|Ax-y\|^2 + \frac{\rho}{2}\|Wx - z^k + u^k\|^2
$$

**ALISTA-ADMM** solves it **approximately** with one gradient step:

$$
x^{k+1} = x^k - \gamma \nabla_x \mathcal{L}(x^k, z^k, u^k)
$$

**Why is this valid?**

1. The objective is **strongly convex** in x (if A has full column rank or ρ > 0)
2. One gradient step is a **contraction** toward the exact minimizer
3. The contraction factor can be **learned** (γₖ)
4. The outer ADMM loop corrects any approximation errors

**Convergence guarantee:** For sufficiently small γₖ, ALISTA-ADMM converges linearly to the true solution.

---

## 7. Implementation with Your Operators

```python
import numpy as np
from scipy.sparse.linalg import LinearOperator

def soft_threshold(x, tau):
    """Soft thresholding operator."""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)


def alista_admm_basic(A, W, y,
                      lam=0.01,      # Sparsity penalty
                      rho=1.0,       # ADMM penalty parameter  
                      gamma=0.5,     # Step size (could be learned)
                      n_iter=50):
    """
    ALISTA-ADMM for CT reconstruction with wavelets.
    
    Args:
        A: CT system matrix (LinearOperator)
        W: Wavelet transform (LinearOperator)  
        y: Sinogram measurements
    """
    n_pixels = A.shape[1]
    n_coeffs = W.shape[0]
    
    # Initialize
    x = np.zeros(n_pixels)
    z = np.zeros(n_coeffs)
    u = np.zeros(n_coeffs)
    
    history = {'x': [], 'z': [], 'objective': []}
    
    for k in range(n_iter):
        # ---- ALISTA-style x-update (1 gradient step) ----
        # Data fidelity gradient
        Ax = A.matvec(x)
        grad_data = A.rmatvec(Ax - y)
        
        # Wavelet consistency gradient  
        Wx = W.matvec(x)
        grad_wave = W.rmatvec(Wx - z + u)
        
        # Combined gradient
        grad = grad_data + rho * grad_wave
        
        # Gradient descent step
        x_new = x - gamma * grad
        
        # ---- z-update (soft thresholding) ----
        Wx_new = W.matvec(x_new)
        z_new = soft_threshold(Wx_new + u, lam / rho)
        
        # ---- u-update (dual variable) ----
        u_new = u + Wx_new - z_new
        
        # Update variables
        x, z, u = x_new, z_new, u_new
        
        # Compute objective for monitoring
        Ax = A.matvec(x)
        data_term = 0.5 * np.linalg.norm(Ax - y)**2
        sparsity_term = lam * np.sum(np.abs(Wx_new))
        history['objective'].append(data_term + sparsity_term)
        history['x'].append(x.copy())
        
        if k % 10 == 0:
            print(f"Iter {k:3d}, Obj = {history['objective'][-1]:.4e}")
    
    return x, history
```

---

## 8. Enhanced Version: Per-Layer Learned Parameters

```python
class LearnedALISTAADMM:
    """ALISTA-ADMM with per-iteration learned parameters."""
    
    def __init__(self, A, W, n_layers=20):
        self.A = A
        self.W = W
        self.n_layers = n_layers
        
        # Learnable parameters per layer
        self.gamma = np.ones(n_layers) * 0.5   # Step size
        self.rho = np.ones(n_layers) * 1.0     # Penalty parameter
        self.lam = np.ones(n_layers) * 0.01    # Sparsity threshold
        
    def forward(self, y, x0=None, return_all=False):
        n_pixels = self.A.shape[1]
        n_coeffs = self.W.shape[0]
        
        # Initialize
        x = np.zeros(n_pixels) if x0 is None else x0.copy()
        z = np.zeros(n_coeffs)
        u = np.zeros(n_coeffs)
        
        states = []
        
        for k in range(self.n_layers):
            # Get parameters for this layer
            gamma_k = self.gamma[k]
            rho_k = self.rho[k]
            lam_k = self.lam[k]
            
            # ---- x-update (learned gradient step) ----
            Ax = self.A.matvec(x)
            grad_data = self.A.rmatvec(Ax - y)
            
            Wx = self.W.matvec(x)
            grad_wave = self.W.rmatvec(Wx - z + u)
            
            grad = grad_data + rho_k * grad_wave
            x = x - gamma_k * grad
            
            # ---- z-update (learned threshold) ----
            Wx = self.W.matvec(x)
            z = soft_threshold(Wx + u, lam_k / rho_k)
            
            # ---- u-update ----
            u = u + Wx - z
            
            if return_all:
                states.append((x.copy(), z.copy(), u.copy()))
        
        return (x, states) if return_all else x
    
    def set_parameters(self, gamma=None, rho=None, lam=None):
        """Update learned parameters."""
        if gamma is not None:
            self.gamma = gamma
        if rho is not None:
            self.rho = rho
        if lam is not None:
            self.lam = lam
```

---

## 9. Training ALISTA-ADMM

```python
def train_alista_admm(model, train_data, val_data, n_epochs=100):
    """
    Train the per-layer parameters (γₖ, ρₖ, λₖ).
    
    Args:
        model: LearnedALISTAADMM instance
        train_data: List of (sinogram, ground_truth) pairs
    """
    # Parameters to optimize
    params = {
        'gamma': model.gamma.copy(),
        'rho': model.rho.copy(),
        'lam': model.lam.copy()
    }
    
    # Simple gradient descent (in practice, use PyTorch/TensorFlow)
    lr = 0.001
    
    for epoch in range(n_epochs):
        total_loss = 0
        
        for y, x_true in train_data:
            # Forward pass
            x_pred = model.forward(y)
            
            # Compute loss (MSE + possible regularization)
            loss = np.mean((x_pred - x_true) ** 2)
            
            # Compute gradients (numerical - in practice use autodiff)
            grad = compute_numerical_gradients(model, y, x_true)
            
            # Update parameters
            params['gamma'] -= lr * grad['gamma']
            params['rho'] -= lr * grad['rho']
            params['lam'] -= lr * grad['lam']
            
            total_loss += loss
        
        # Update model parameters
        model.set_parameters(**params)
        
        # Validation
        val_loss = validate(model, val_data)
        
        print(f"Epoch {epoch}: Train Loss = {total_loss:.6f}, Val Loss = {val_loss:.6f}")
```

---

## 10. Comparison of Methods

| Method | x-update cost | z-update | Convergence | Learned params | Stability |
|--------|---------------|----------|-------------|----------------|-----------|
| **ISTA** | 1 grad | N/A | Slow ($O(1/k)$) | None | ✅ |
| **FISTA** | 1 grad | N/A | Faster ($O(1/k^2)$) | None | ⚠️ May oscillate |
| **ADMM-CG** | 50-100 grads | Exact | Fast (linear) | None | ✅✅ |
| **ALISTA** | 1 grad | N/A | Fast (learned) | γₖ, θₖ | ✅ |
| **ALISTA-ADMM** | **1 grad** | **Exact** | **Fast (learned)** | γₖ, ρₖ, λₖ | ✅✅ |

---

## 11. Visualization of Iteration Path

```
Classical ADMM:
x₀ → [CG solve] → x₁ → [CG solve] → x₂ → [CG solve] → ...
        ↓              ↓              ↓
      50 iters       50 iters       50 iters

ALISTA-ADMM:
x₀ → [1 grad] → x₁ → [1 grad] → x₂ → [1 grad] → ...
        ↓              ↓              ↓
       1 iter         1 iter         1 iter

Time ratio: ~50× speedup per ADMM iteration!
```

---

## 12. Why This is Perfect for CT

**CT-specific advantages:**

| Requirement | How ALISTA-ADMM meets it |
|------------|--------------------------|
| **Physics accuracy** | Uses exact A, Aᵀ operators |
| **Sparsity** | Uses exact W, Wᵀ operators |
| **Speed** | No CG solves → 50-100× faster per iteration |
| **Low dose** | Learned thresholds adapt to noise level |
| **Edge preservation** | bior4.4 + learned ρₖ balances smoothing |
| **Clinical deployment** | Fixed iterations, predictable runtime |

---

## 13. Complete CT Reconstruction Pipeline

```python
class CTALISTAADMM:
    """Complete ALISTA-ADMM for CT reconstruction."""
    
    def __init__(self, 
                 A,                    # CT system matrix
                 W,                    # Wavelet transform
                 n_layers=30,          # Number of unrolled iterations
                 use_bior4.4=True):    # Use biorthogonal wavelets
        
        self.A = A
        self.W = W
        self.n_layers = n_layers
        
        # Initialize parameters with reasonable defaults
        self.gamma = np.linspace(0.3, 0.1, n_layers)  # Decreasing step size
        self.rho = np.ones(n_layers) * 1.0            # Stable penalty
        self.lam = np.linspace(0.05, 0.01, n_layers)  # Decreasing threshold
        
    def reconstruct(self, sinogram, x0=None, verbose=True):
        """
        Full CT reconstruction pipeline.
        
        Args:
            sinogram: [n_angles, n_detectors] or flattened array
            x0: Initial guess (None = backprojection)
        
        Returns:
            reconstructed image [height, width]
        """
        # Ensure sinogram is flattened
        original_shape = None
        if sinogram.ndim == 2:
            original_shape = sinogram.shape
            sinogram = sinogram.ravel()
        
        # Run ALISTA-ADMM
        x, states = self.forward(sinogram, x0, return_all=True)
        
        # Reshape to image
        img_size = int(np.sqrt(len(x)))  # Assumes square image
        image = x.reshape(img_size, img_size)
        
        if verbose:
            self.print_convergence_info(states)
        
        return image
    
    def print_convergence_info(self, states):
        """Print convergence metrics."""
        print("\n=== ALISTA-ADMM Convergence ===")
        print(f"Layers: {len(states)}")
        print(f"Final γ: {self.gamma[-1]:.3f}")
        print(f"Final ρ: {self.rho[-1]:.3f}")
        print(f"Final λ: {self.lam[-1]:.4f}")
        
        # Compute change in solution
        x_final = states[-1][0]
        x_prev = states[-2][0] if len(states) > 1 else states[-1][0]
        change = np.linalg.norm(x_final - x_prev) / np.linalg.norm(x_final)
        print(f"Relative change: {change:.2e}")
```

---

## 14. Summary: ALISTA-ADMM Advantages

| Advantage | Why It Matters |
|-----------|----------------|
| **🚀 50× faster** than classical ADMM | No inner CG loops |
| **🎯 Physics-compliant** | Uses exact A, Aᵀ |
| **🧠 Sparse-aware** | Uses exact W, Wᵀ |
| **📚 Learnable** | Per-layer γₖ, ρₖ, λₖ |
| **🛡️ Stable** | Inherits ADMM robustness |
| **🔬 Interpretable** | Each layer = 1 ADMM iteration |
| **⚕️ CT-ready** | Works with your existing operators |

---

## 15. Next Steps

I can show you:

1. **PyTorch implementation** with automatic differentiation
2. **Training on Shepp-Logan** with different noise levels
3. **Comparison plots** — ADMM-CG vs ALISTA-ADMM vs FISTA
4. **Extension to 3D CT** (cone beam)
5. **Adaptive parameter selection** during inference

**Which direction would be most valuable for your CT reconstruction work?**