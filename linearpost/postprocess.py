"""Post-processing for fair classification."""
from typing import Any, Optional, Literal
from itertools import product

import numpy as np
import cvxpy as cp

ConstraintType = list[tuple[int, list[int]]]


class LinearPost:

  # constraints[c] is a tuple (y_c, list of groups), specifiying the
  # requirement that:
  #   | Pr[ h(X) = y_c | Z_k = 1 ] - Pr[ h(X) = y_c | Z_k' = 1 ] | <= alpha / 2
  # for all k, k' in the list of groups

  def __init__(self,
               fairness_constraints: Optional[ConstraintType] = None,
               alpha: Optional[float] = None,
               noise: float = 1e-4,
               seed: Optional[int] = None) -> None:
    self.fairness_constraints = fairness_constraints
    self.alpha = alpha
    self.noise = noise
    self.seed = seed
    self.rng = np.random.default_rng(seed)

  def perturb_risk(self, risk: np.ndarray) -> np.ndarray:
    return risk + self.rng.uniform(-self.noise, self.noise, size=risk.shape)

  # TODO: sample weight
  def fit(self,
          risk,
          probas_group,
          solver: Optional[str] = None,
          solve_kwargs: Optional[dict[str, Any]] = None,
          solve_primal: bool = True) -> 'LinearPost':
    solve_kwargs = solve_kwargs or {}

    if self.alpha is None or self.alpha == float('inf'):
      return self

    # risk.shape = [n_examples, n_classes]
    # probas_group.shape = [n_examples, n_groups]
    self.n_classes = risk.shape[1]
    self.n_groups = probas_group.shape[1]

    # Default fairness_constraints is SP
    if self.fairness_constraints is None:
      self.fairness_constraints = [
          (g, list(range(self.n_classes))) for g in range(self.n_groups)
      ]
    self.n_psi = sum(len(I) for _, I in self.fairness_constraints)

    # Perturb risk to circumvent colinearity
    self.risk_mean_ = np.mean(np.max(risk, axis=1))
    self.noise = self.noise * self.risk_mean_
    risk = self.perturb_risk(risk)

    marginal_probas_group = probas_group.mean(axis=0)  # shape = (n_groups,)
    gamma = probas_group / marginal_probas_group[None, ...]

    # TODO: catch situations where the solver fails (i.e., numerical issues)
    if solve_primal:
      problem = self.linprog_primal_(risk, gamma, self.alpha)
      problem.solve(solver=solver, **solve_kwargs)
      self.psi_ = (np.array([
          c.dual_value for c in problem.constraints[-2 * self.n_psi::2]
      ]) - np.array(
          [c.dual_value for c in problem.constraints[-2 * self.n_psi + 1::2]]))
      self.phi_ = -problem.constraints[0].dual_value
      self.pi_ = problem.var_dict['pi'].value
    else:
      problem = self.linprog_dual_(risk, gamma, self.alpha)
      problem.solve(solver=solver, **solve_kwargs)
      self.psi_ = problem.var_dict['psi_pos'].value - problem.var_dict[
          'psi_neg'].value
      self.phi_ = problem.var_dict['phi'].value

    self.w_ = np.zeros((self.n_classes, self.n_groups))
    for i, (y_c, k) in enumerate(
        self.flatten_constraints(self.fairness_constraints)):
      self.w_[y_c, k] -= self.psi_[i] / marginal_probas_group[k]

    self.score_ = problem.value
    self.risk_ = risk  # for debugging
    self.gamma_ = gamma
    return self

  def predict_score(self, risk: np.ndarray,
                    probas_group: np.ndarray) -> np.ndarray:
    if self.alpha is None or self.alpha == float('inf'):
      return risk
    risk = self.perturb_risk(risk)  # perturb risk to circumvent colinearity
    fair_risk = (probas_group[:, None, :] * self.w_).sum(axis=-1)
    return risk + fair_risk

  def predict(self, risk: np.ndarray, probas_group: np.ndarray) -> np.ndarray:
    fair_risk = self.predict_score(risk, probas_group)
    return np.argmin(fair_risk, axis=1)

  def linprog_primal_(self, risk: np.ndarray, gamma: np.ndarray,
                      alpha: float) -> cp.Problem:
    n_examples = risk.shape[0]
    n_constraints = len(self.fairness_constraints)

    alpha = cp.Parameter(value=alpha, name="alpha")
    pi = cp.Variable((n_examples, self.n_classes), name="pi", nonneg=True)
    q = cp.Variable(n_constraints, name="q", nonneg=True)

    # Get constraints
    constraints = []

    # sum_y pi(x, y) = 1, for all x
    constraints.append(cp.sum(pi, axis=1) == 1)

    # | sum_x gamma(x, k) * pi(x, y_c) * p(x) - q_c | <= alpha / 2, for all c, k
    for i, (y_c, I) in enumerate(self.fairness_constraints):
      for k in I:
        t = cp.sum(cp.multiply(gamma[:, k], pi[:, y_c]))
        constraints.append(-alpha * n_examples / 2 <= t - q[i] * n_examples)
        constraints.append(t - q[i] * n_examples <= alpha * n_examples / 2)

    return cp.Problem(cp.Minimize(cp.sum(cp.multiply(pi, risk))), constraints)

  def linprog_dual_(self, risk: np.ndarray, gamma: np.ndarray,
                    alpha: float) -> cp.Problem:
    n_examples = risk.shape[0]

    alpha = cp.Parameter(value=alpha, name="alpha")
    phi = cp.Variable(risk.shape[0], name="phi")
    psi_pos = cp.Variable(self.n_psi, name="psi_pos", nonneg=True)
    psi_neg = cp.Variable(self.n_psi, name="psi_neg", nonneg=True)

    # Get constraints
    constraints = []

    # (*) sum_{k in I_c} psi_pos_{c,k} - psi_neg_{c,k} = 0, for all c
    i = 0
    for _, I in self.fairness_constraints:
      t = 0
      for k in I:
        t += psi_pos[i] - psi_neg[i]
        i += 1
      constraints.append(t == 0)

    # risk(x,y) >= phi(x)
    #       + sum_{c,k} 1[y_c=y] * gamma(x,k) * (psi_pos_{c,k} - psi_neg_{c,k}),
    # for all x, y
    t = [0 for _ in range(self.n_classes)]
    for i, (y_c, k) in enumerate(
        self.flatten_constraints(self.fairness_constraints)):
      t[y_c] += cp.multiply(gamma[:, k], psi_pos[i] - psi_neg[i])
    for y, s in enumerate(t):
      constraints.append(phi + s <= risk[:, y])

    # A factor of `1/2` is omitted, because constraint (*) above already gives
    # sum_k psi_pos_{c, k} = sum_k psi_neg_{c, k}, for all c
    return cp.Problem(
        cp.Maximize(cp.sum(phi) - alpha * cp.sum(psi_pos) * n_examples),
        constraints)

  @staticmethod
  def flatten_constraints(
      fairness_constraints: ConstraintType) -> list[tuple[int, int]]:
    return [
        pair for y_c, I in fairness_constraints for pair in product([y_c], I)
    ]


class LinearPostSimple:

  def __init__(self,
               n_classes: int,
               n_groups: int,
               fairness_criterion: Literal['sp', 'tpr', 'fpr', 'eo'] = 'sp',
               remove_unused: bool = False,
               class_weight: Optional[list[float]] = None,
               alpha: Optional[float] = None,
               noise: float = 1e-4,
               seed: Optional[int] = None) -> None:

    self.cls_loss_fn = 1 - np.eye(n_classes)
    if class_weight is not None:
      self.cls_loss_fn *= np.array(class_weight)[:, None]

    self.n_classes = n_classes
    self.n_groups = n_groups
    # n_groups refers to sensitive attributes, not the same as the semantics
    # of groups in LinearPost

    self.fairness_criterion = fairness_criterion
    self.remove_unused = remove_unused
    if fairness_criterion == 'sp':
      # groups passed to LinearPost are sensitive attributes A
      fairness_constraints = [
          (y_c, np.arange(n_groups)) for y_c in range(n_classes)
      ]
    elif fairness_criterion == 'fpr' and n_classes == 2:
      # groups passed to LinearPost are joint (A, Y)
      if not remove_unused:
        fairness_constraints = [(1, 2 * np.arange(n_groups))]
      else:
        fairness_constraints = [(1, np.arange(n_groups))]
    else:
      if n_classes == 2 and fairness_criterion == 'tpr':
        # groups passed to LinearPost are joint (A, Y)
        if not remove_unused:
          fairness_constraints = [(1, 2 * np.arange(n_groups) + 1)]
        else:
          fairness_constraints = [(1, np.arange(n_groups))]
      else:
        fairness_constraints = []
        for y_c in range(n_classes):
          for y in range(n_classes):
            if fairness_criterion == 'tpr' and y != y_c:
              continue
            fairness_constraints.append(
                (y_c, n_classes * np.arange(n_groups) + y))

    self.postprocessor = LinearPost(fairness_constraints=fairness_constraints,
                                    alpha=alpha,
                                    noise=noise,
                                    seed=seed)

  def get_risk_and_group_probas_(
      self,
      p_a_x: Optional[np.ndarray] = None,
      p_y_x: Optional[np.ndarray] = None,
      p_ay_x: Optional[np.ndarray] = None,
  ):
    if self.fairness_criterion == 'sp':
      if p_ay_x is None and (p_a_x is None or p_y_x is None):
        raise ValueError(
            'p_ay_x or (p_a_x and p_y_x) must be provided for `sp` criterion')
      if p_a_x is None:
        p_a_x = p_ay_x.reshape(-1, self.n_groups, self.n_classes).sum(axis=2)
      if p_y_x is None:
        p_y_x = p_ay_x.reshape(-1, self.n_groups, self.n_classes).sum(axis=1)
      p_g_x = p_a_x.reshape(-1, self.n_groups)
    if self.fairness_criterion in ['tpr', 'fpr', 'eo']:
      if p_ay_x is None:
        raise ValueError('p_ay_x must be provided for `eopp` or `eo` criterion')
      if p_y_x is None:
        p_y_x = p_ay_x.reshape(-1, self.n_groups, self.n_classes).sum(axis=1)
      p_g_x = p_ay_x.reshape(-1, self.n_groups * self.n_classes)
    risk = np.sum(p_y_x[..., None] * self.cls_loss_fn[None, :], axis=1)
    if self.remove_unused and self.n_groups == 2:
      if self.fairness_criterion == 'tpr':
        p_g_x = p_g_x.reshape(-1, self.n_groups, self.n_classes)[:, :, 1]
      elif self.fairness_criterion == 'fpr':
        p_g_x = p_g_x.reshape(-1, self.n_groups, self.n_classes)[:, :, 0]
    return risk, p_g_x

  def fit(self,
          p_a_x: Optional[np.ndarray] = None,
          p_y_x: Optional[np.ndarray] = None,
          p_ay_x: Optional[np.ndarray] = None,
          solver: Optional[str] = None,
          solve_kwargs: Optional[dict[str, Any]] = None,
          solve_primal: bool = True) -> 'LinearPostBasic':
    self.postprocessor.fit(
        *self.get_risk_and_group_probas_(p_a_x, p_y_x, p_ay_x),
        solver=solver,
        solve_kwargs=solve_kwargs,
        solve_primal=solve_primal,
    )
    return self

  def predict_score(self,
                    p_a_x: Optional[np.ndarray] = None,
                    p_y_x: Optional[np.ndarray] = None,
                    p_ay_x: Optional[np.ndarray] = None) -> np.ndarray:
    return self.postprocessor.predict_score(
        *self.get_risk_and_group_probas_(p_a_x, p_y_x, p_ay_x))

  def predict(self,
              p_a_x: Optional[np.ndarray] = None,
              p_y_x: Optional[np.ndarray] = None,
              p_ay_x: Optional[np.ndarray] = None) -> np.ndarray:
    return self.postprocessor.predict(
        *self.get_risk_and_group_probas_(p_a_x, p_y_x, p_ay_x))
