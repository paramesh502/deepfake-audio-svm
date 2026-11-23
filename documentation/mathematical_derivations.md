# Mathematical Derivations: SVM for Deepfake Audio Detection

## 1. Problem Formulation

Given N labeled speech recordings with features x_i in R^d and labels y_i in {-1, +1}
(where +1 = FAKE, -1 = REAL), find a hyperplane  f(x) = w^T x + b  that separates
the two classes with maximum margin.

The decision rule is:  y_hat = sign(w^T x + b)

## 2. Linear SVM Primal (Hard Margin)

For linearly separable data:

    min_{w, b}   (1/2) ||w||^2
    subject to:  y_i (w^T x_i + b) >= 1   for all i = 1,...,n

The geometric margin is 2 / ||w||. Maximising the margin minimises ||w||^2.

## 3. Soft-Margin SVM (C-SVM)

Real data is not linearly separable. Introduce slack variables xi_i >= 0:

    min_{w, b, xi}   (1/2)||w||^2  +  C/n * SUM_i xi_i
    subject to:      y_i (w^T x_i + b) >= 1 - xi_i,   xi_i >= 0

Symbol meanings:
  w       : weight vector (normal to the decision hyperplane)
  b       : bias (intercept)
  xi_i    : slack variable for sample i (xi_i > 0 allows margin violation)
  C       : regularisation parameter (larger C -> harder margin)
  n       : number of training samples

## 4. Hinge Loss Formulation

The slack-variable problem is equivalent to:

    min_{w, b}   (1/2)||w||^2  +  (C/n) * SUM_i L_hinge(y_i, f(x_i))

where the hinge loss is:

    L_hinge(y, f) = max(0, 1 - y * f(x))

Derivation of equivalence:
  At the optimal xi_i satisfies  xi_i = max(0, 1 - y_i f(x_i)),
  since we minimise xi_i subject to xi_i >= max(0, 1 - y_i f(x_i)).

## 5. Subgradient Derivation

The hinge loss is convex but non-differentiable at y_i f(x_i) = 1.
We use the subgradient (generalised gradient):

For sample i, let  m_i = y_i (w^T x_i + b):

Case 1 — correctly classified (m_i >= 1):
    Partial derivative of L_hinge w.r.t. w = 0
    Total subgradient:   g_w = w

Case 2 — margin violation (m_i < 1):
    Partial derivative of L_hinge w.r.t. w = -(C/n) * y_i * x_i
    Total subgradient:   g_w = w - (C/n) * y_i * x_i

Subgradient w.r.t. bias b:
    Case 1 (m_i >= 1): g_b = 0
    Case 2 (m_i <  1): g_b = -(C/n) * y_i

## 6. Pegasos Learning-Rate Schedule

The Pegasos algorithm (Shalev-Shwartz et al., 2007) uses:

    eta_t = 1 / (lambda * t)   where lambda = 1/C

This schedule converges in O(1/lambda * eps) iterations to eps-accuracy
in the objective. The rate decreases like 1/t, ensuring convergence while
allowing large initial steps.

## 7. Dual Formulation (for Kernel SVM)

The dual problem uses Lagrange multipliers alpha_i >= 0:

    max_{alpha}   SUM_i alpha_i  -  (1/2) SUM_{i,j} alpha_i alpha_j y_i y_j K(x_i, x_j)
    subject to:   0 <= alpha_i <= C,   SUM_i alpha_i y_i = 0

The decision function in the dual is:
    f(x) = SUM_i alpha_i y_i K(x_i, x)  +  b

## 8. Kernel Trick

The kernel trick replaces the dot product x_i^T x_j with K(x_i, x_j),
implicitly mapping data to a higher-dimensional feature space.

Linear kernel:   K(x_i, x_j) = x_i^T x_j
RBF kernel:      K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)

The RBF kernel creates a nonlinear decision boundary in the original
space, potentially capturing complex acoustic feature relationships.
gamma controls the width: large gamma -> narrow Gaussians -> more complex boundary.

## 9. Support Vectors

Support vectors are training samples with alpha_i > 0, i.e., samples that
lie on or inside the margin. They are the ONLY samples that influence the
decision boundary. For a soft-margin SVM:

  alpha_i = 0:        x_i is correctly classified outside the margin
  0 < alpha_i < C:    x_i is exactly on the margin (support vector)
  alpha_i = C:        x_i is a margin violator (xi_i > 0)

## 10. Platt Scaling (Probability Calibration)

Raw decision scores f(x) are not probabilities. Platt (1999) fits:

    P(y=1 | f) = 1 / (1 + exp(A*f + B))

Parameters A and B are fitted on a held-out calibration set by
minimising the negative log-likelihood (cross-entropy loss).
Label smoothing is applied to prevent overconfidence:

    t_i = (n_pos + 1)/(n_pos + 2)   if y_i = 1
    t_i = 1 / (n_neg + 2)           if y_i = 0

The Expected Calibration Error (ECE) measures calibration quality:

    ECE = SUM_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

where B_b is the set of samples in the b-th probability bin.
