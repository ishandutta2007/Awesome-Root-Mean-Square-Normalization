# The Numerical Underflow Hazard (&epsilon; Placement)

In deep neural networks, activations can occasionally shrink to near-zero values. In Root Mean Square Normalization (RMSNorm), this creates a significant risk of division-by-zero or numerical underflow, leading to NaN (Not-a-Number) values that can ruin model training.

---

## 1. The Division-by-Zero Risk

The normalization step in RMSNorm is defined as:

$$\bar{x}_i = \frac{x_i}{\text{RMS}(x)}$$

If $x = \vec{0}$ or if all activations are extremely small, $\text{RMS}(x)$ approaches $0$. Dividing by $0$ yields infinity or NaN, which propagates through the network, destroying the weights.

---

## 2. Epsilon (&epsilon;) as a Stabilizer

To prevent this collapse, a tiny stabilization constant $\epsilon$ is added to the variance calculation. The placement of $\epsilon$ is critical for numerical precision:

1. **Inside the Square Root (Correct/Standard):**
   $$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$$
   Adding $\epsilon$ inside the square root guarantees that the denominator never falls below $\sqrt{\epsilon}$ (e.g., $3.16\text{e-}3$ for $\epsilon = 1\text{e-}5$).

2. **Outside the Square Root (Prone to Precision Issues):**
   $$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2} + \epsilon$$
   If the sum of squares is $0$, the square root is $0$, and we add $\epsilon$ (e.g., $1\text{e-}5$). While this avoids absolute division-by-zero, it can lead to precision loss and overflow/underflow issues in half-precision training (like Float16 or BFloat16).

---

## 3. Stability Check Flow

```mermaid
graph TD
    Input[Input Vector: x] --> SumSq[Compute Sum of Squares]
    SumSq --> AddEps[Add Epsilon inside sqrt]
    AddEps --> Sqrt[Compute Square Root: Denominator]
    Sqrt --> Check{Is Denominator >= sqrt(epsilon)?}
    Check -- Yes --> Normalize[Scale Activations Safely]
    Check -- No --> Fail[NaN / Numerical Underflow Prevented]
    Normalize --> Output[Stable Output]
```

---

[← Back to README](../README.md)
