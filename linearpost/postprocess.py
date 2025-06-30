from typing import Any, Optional, Literal
from itertools import combinations, product

import numpy as np
import cvxpy as cp

from .models import DecisionCalibrator

ConstraintType = list[tuple[int, list[int]]]


class LinearPost:
  # constraints[c] is a tuple (y_c, list of groups), specifiying the
  # requirement that:
  #   | Pr[ h(X) = y_c | Z_k = 1 ] - Pr[ h(X) = y_c | Z_k' = 1 ] | <= alpha / 2
  # for all k, k' in the list of groups

  def __init__(self,
               fairness_constraints: Optional[ConstraintType] = None,
               alpha: Optional[float] = None,
               max_iters_dcal: int = 10,
               noise_scale: Optional[float] = None,
               seed: Optional[int] = None) -> None:
    self.fairness_constraints = fairness_constraints
    self.alpha = alpha
    self.max_iters_dcal = max_iters_dcal
    self.noise_scale = noise_scale
    self.seed = seed
    self.rng = np.random.default_rng(seed)
    self.dcalib: Optional[DecisionCalibratorForLinearPost] = None

  def fit(self,
          risk: np.ndarray,
          probas_group: np.ndarray,
          groups: Optional[np.ndarray] = None,
          sample_weight: Optional[np.ndarray] = None,
          idx_post: Optional[np.ndarray] = None,
          idx_dcal: Optional[np.ndarray] = None,
          solver: Optional[str] = None,
          solve_kwargs: Optional[dict[str, Any]] = None,
          solve_primal: bool = True) -> 'LinearPost':
    solve_kwargs = solve_kwargs or {}

    if sample_weight is None:
      sample_weight = np.ones(risk.shape[0])
    if idx_post is None:
      idx_post = np.arange(risk.shape[0])
    if idx_dcal is None:
      idx_dcal = np.arange(risk.shape[0])

    if self.alpha is None or self.alpha == float('inf'):
      return self

    # risk.shape = [n_examples, n_classes]
    # probas_group.shape = [n_examples, n_groups]
    self.n_classes = risk.shape[1]
    self.n_groups = probas_group.shape[1]

    # Default fairness_constraints to SP
    if self.fairness_constraints is None:
      self.fairness_constraints = [
          (g, list(range(self.n_classes))) for g in range(self.n_groups)
      ]

    # Perturb risk to circumvent colinearity
    if self.noise_scale is None:
      self.risk_mean_ = np.mean(np.max(risk, axis=1))
      self.noise_scale = 1e-4 * self.risk_mean_
    risk_orig = risk.copy()
    risk = self.perturb_risk(risk_orig)

    # If group labels are given, perform decision calibration before fitting
    if groups is not None:
      self.dcalib = DecisionCalibratorForLinearPost(
          fairness_constraints=self.fairness_constraints,
          alpha=self.alpha,
          max_iters_dcal=self.max_iters_dcal,
          seed=self.seed).fit(
              self.perturb_risk(risk_orig[idx_dcal]),
              probas_group[idx_dcal],
              groups[idx_dcal],
              # sample_weight=sample_weight[idx_dcal],  # TODO
              solver=solver,
              solve_kwargs=solve_kwargs,
              solve_primal=solve_primal)
      probas_group = self.dcalib.transform(risk, probas_group)

    self.fit_(risk[idx_post],
              probas_group[idx_post],
              sample_weight=sample_weight[idx_post],
              solver=solver,
              solve_kwargs=solve_kwargs,
              solve_primal=solve_primal)
    return self

  def fit_(self,
           risk: np.ndarray,
           probas_group: np.ndarray,
           sample_weight: np.ndarray,
           solver: Optional[str] = None,
           solve_kwargs: Optional[dict[str, Any]] = None,
           solve_primal: bool = True) -> None:
    self.n_psi = sum(len(I) for _, I in self.fairness_constraints)

    if solve_primal:
      problem = self.linprog_primal_(risk, probas_group, self.alpha,
                                     sample_weight)
      problem.solve(solver=solver, **solve_kwargs)
      if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise cp.SolverError(
            f"Solver failed: {problem.status}, {problem.solver_stats}")
      self.psi_ = (np.array([
          c.dual_value for c in problem.constraints[-2 * self.n_psi::2]
      ]) - np.array(
          [c.dual_value for c in problem.constraints[-2 * self.n_psi + 1::2]]))
      self.phi_ = -problem.constraints[0].dual_value
      self.pi_ = problem.var_dict['pi'].value
      self.q_ = problem.var_dict['q'].value
    else:
      problem = self.linprog_dual_(risk, probas_group, self.alpha,
                                   sample_weight)
      problem.solve(solver=solver, **solve_kwargs)
      if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise cp.SolverError(
            f"Solver failed: {problem.status}, {problem.solver_stats}")
      self.psi_ = problem.var_dict['psi_pos'].value - problem.var_dict[
          'psi_neg'].value
      self.phi_ = problem.var_dict['phi'].value

    marginals_group_inv = self.get_marginal_probas(probas_group,
                                                   sample_weight=sample_weight,
                                                   return_inv=True)
    self.w_ = np.zeros((self.n_classes, self.n_groups))
    for i, (y_c, k) in enumerate(
        self.flatten_constraints(self.fairness_constraints)):
      self.w_[y_c, k] -= self.psi_[i] * marginals_group_inv[k]

    self.score_ = problem.value
    self.risk_ = risk  # for debugging
    self.probas_group_ = probas_group  # for debugging

  def predict_score(self, risk: np.ndarray,
                    probas_group: np.ndarray) -> np.ndarray:
    if self.alpha is None or self.alpha == float('inf'):
      return risk
    risk = self.perturb_risk(risk)  # perturb risk to circumvent colinearity
    if self.dcalib is not None:
      probas_group = self.dcalib.transform(risk, probas_group)
    fair_risk = (probas_group[:, None, :] * self.w_).sum(axis=-1)
    return risk + fair_risk

  def predict(self, risk: np.ndarray, probas_group: np.ndarray) -> np.ndarray:
    fair_risk = self.predict_score(risk, probas_group)
    return np.argmin(fair_risk, axis=1)

  def linprog_primal_(self, risk: np.ndarray, probas_group: np.ndarray,
                      alpha: float, sample_weight: np.ndarray) -> cp.Problem:
    n_constraints = len(self.fairness_constraints)
    gamma = probas_group * self.get_marginal_probas(
        probas_group, sample_weight=sample_weight, return_inv=True)
    w = sample_weight.sum()

    alpha = cp.Parameter(value=alpha, name="alpha")
    pi = cp.Variable((risk.shape[0], self.n_classes), name="pi", nonneg=True)
    q = cp.Variable(n_constraints, name="q", nonneg=True)

    # Get constraints
    constraints = []

    # sum_y pi(x, y) = 1, for all x
    constraints.append(cp.sum(pi, axis=1) == 1)

    # | sum_x gamma(x, k) * pi(x, y_c) * p(x) - q_c | <= alpha / 2, for all c, k
    for c, (y_c, I) in enumerate(self.fairness_constraints):
      for k in I:
        t = cp.sum(cp.multiply(gamma[:, k] * sample_weight, pi[:, y_c]))
        constraints.append(-alpha * w / 2 <= t - q[c] * w)
        constraints.append(t - q[c] * w <= alpha * w / 2)

    return cp.Problem(
        cp.Minimize(cp.sum(cp.multiply(pi, risk * sample_weight[:, None]))),
        constraints)

  def linprog_dual_(self, risk: np.ndarray, probas_group: np.ndarray,
                    alpha: float, sample_weight: np.ndarray) -> cp.Problem:
    gamma = probas_group * self.get_marginal_probas(
        probas_group, sample_weight=sample_weight, return_inv=True)
    w = sample_weight.sum()

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
        cp.Maximize(
            cp.sum(cp.multiply(phi, sample_weight)) -
            alpha * cp.sum(psi_pos) * w), constraints)

  def perturb_risk(self, risk: np.ndarray) -> np.ndarray:
    return risk + self.rng.uniform(
        -self.noise_scale, self.noise_scale, size=risk.shape)

  @staticmethod
  def flatten_constraints(
      fairness_constraints: ConstraintType) -> list[tuple[int, int]]:
    return [
        pair for y_c, I in fairness_constraints for pair in product([y_c], I)
    ]

  @staticmethod
  def get_marginal_probas(probas: np.ndarray,
                          sample_weight: Optional[np.ndarray] = None,
                          return_inv: bool = False) -> np.ndarray:
    if sample_weight is None:
      sample_weight = np.ones(probas.shape[0])
    marginal_probas = (probas * sample_weight[:, None]).sum(
        axis=0) / sample_weight.sum()
    if return_inv:
      # Avoid division by 0
      mask = marginal_probas == 0
      marginal_probas[~mask] = 1 / marginal_probas[~mask]
    return marginal_probas


class DecisionCalibratorForLinearPost:

  def __init__(self,
               fairness_constraints: Optional[ConstraintType] = None,
               alpha: Optional[float] = None,
               max_iters_dcal: int = 10,
               seed: Optional[int] = None) -> None:
    self.fairness_constraints = fairness_constraints
    self.alpha = alpha
    self.max_iters_dcal = max_iters_dcal
    self.seed = seed

  def fit(self,
          risk: np.ndarray,
          probas_group: np.ndarray,
          groups: np.ndarray,
          solver: Optional[str] = None,
          solve_kwargs: Optional[dict[str, Any]] = None,
          solve_primal: bool = True) -> 'DecisionCalibratorForLinearPost':
    self.n_classes = risk.shape[1]
    self.n_groups = probas_group.shape[1]
    self.multilabel = (groups.ndim == 2) and (groups.shape[1]
                                              == probas_group.shape[1])
    n_group_labels = groups.shape[1] if self.multilabel else 1

    self.dcalibs = [
        DecisionCalibrator(
            n_actions=self.n_groups if not self.multilabel else 2)
        for _ in range(n_group_labels)
    ]

    for _ in range(self.max_iters_dcal):
      postprocessor = LinearPost(fairness_constraints=self.fairness_constraints,
                                 alpha=self.alpha,
                                 noise_scale=0,
                                 seed=self.seed).fit(risk,
                                                     probas_group,
                                                     solver=solver,
                                                     solve_kwargs=solve_kwargs,
                                                     solve_primal=solve_primal)
      # probas_action = postprocessor.pi_ if solve_primal else None

      probas_group_expanded, groups_expanded = self.expand_probas_group_for_dcal(
          probas_group, groups, multilabel=self.multilabel)

      for i in range(n_group_labels):

        def predict_fn(_, risk, probas_group, postprocessor=postprocessor):
          return np.eye(self.n_classes)[postprocessor.predict(
              risk, probas_group)]

        # Step dcalib and transform/calibrate probas_group
        probas_group_one = self.dcalibs[i].add_(
            probas_group_expanded[:, i],
            groups_expanded[:, i],
            predict_fn,
            # probas_action=probas_action,
            risk=risk,
            probas_group=probas_group)

        # One-step calibrated probas_group
        probas_group_expanded[:, i] = probas_group_one
      probas_group = self.squeeze_probas_group_expanded_for_dcal(
          probas_group_expanded, multilabel=self.multilabel)

    return self

  def transform(self, risk, probas_group):
    probas_group = probas_group.copy()
    # for t in range(self.max_iters_dcal):
    for t in range(len(self.dcalibs[0].adjustments_)):
      probas_group_expanded = self.expand_probas_group_for_dcal(
          probas_group, multilabel=self.multilabel)
      for i, dcalib in enumerate(self.dcalibs):
        probas_group_expanded[:, i] = dcalib.apply_(
            probas_group_expanded[:, i], dcalib.adjustments_[t],
            dcalib.predict_fns_[t](probas_group_expanded[:, i],
                                   risk=risk,
                                   probas_group=probas_group))
      probas_group = self.squeeze_probas_group_expanded_for_dcal(
          probas_group_expanded, multilabel=self.multilabel)
    return probas_group

  @staticmethod
  def expand_probas_group_for_dcal(probas_group, groups=None, multilabel=False):
    if multilabel:
      probas_group_expanded = np.stack([1 - probas_group, probas_group],
                                       axis=-1)
      # probas_group_expanded.shape = (n_examples, n_groups, 2)
      # groups.shape = (n_examples, n_groups)
    else:
      probas_group_expanded = probas_group[:, None, :]
      # probas_group_expanded.shape = (n_examples, 1, n_groups)
      if groups is not None:
        groups = groups[:, None]  # shape = (n_examples, 1)
    if groups is not None:
      return probas_group_expanded, groups
    else:
      return probas_group_expanded

  @staticmethod
  def squeeze_probas_group_expanded_for_dcal(probas_group_expanded,
                                             multilabel=False):
    return (probas_group_expanded[:, :, 1]
            if multilabel else probas_group_expanded[:, 0])


# Instantiations


class LinearPostSimple:
  # Wrapper around LinearPost for simple non-overlapping groups
  # and standard group fairness criteria

  def __init__(self,
               n_classes: int,
               n_groups: int,
               fairness_criterion: Literal['sp', 'tpr', 'fpr', 'eo'] = 'sp',
               remove_unused: bool = False,
               class_weight: Optional[list[float]] = None,
               alpha: Optional[float] = None,
               max_iters_dcal: int = 10,
               noise_scale: Optional[float] = None,
               seed: Optional[int] = None) -> None:

    # TODO: allow customizing the loss function, default here is 0/1 loss
    self.cls_loss_fn = 1 - np.eye(n_classes)
    if class_weight is not None:
      self.cls_loss_fn *= np.array(class_weight)[:, None]

    # n_groups refers to sensitive attributes, not the same as the semantics
    # of groups in LinearPost
    self.n_groups = n_groups
    self.n_classes = n_classes

    self.fairness_criterion = fairness_criterion
    self.remove_unused = remove_unused

    self.postprocessor = LinearPost(
        fairness_constraints=self.get_fairness_constraints_(
            n_classes,
            n_groups,
            fairness_criterion=fairness_criterion,
            remove_unused=remove_unused),
        alpha=alpha,
        max_iters_dcal=max_iters_dcal,
        noise_scale=noise_scale,
        seed=seed,
    )

  def fit(self,
          p_a_x: Optional[np.ndarray] = None,
          p_y_x: Optional[np.ndarray] = None,
          p_ay_x: Optional[np.ndarray] = None,
          groups: Optional[np.ndarray] = None,
          labels_ay: Optional[np.ndarray] = None,
          sample_weight: Optional[np.ndarray] = None,
          idx_post: Optional[np.ndarray] = None,
          idx_dcal: Optional[np.ndarray] = None,
          solver: Optional[str] = None,
          solve_kwargs: Optional[dict[str, Any]] = None,
          solve_primal: bool = True) -> 'LinearPostSimple':
    self.postprocessor.fit(
        **self.get_risk_and_group_probas_(p_a_x=p_a_x,
                                          p_y_x=p_y_x,
                                          p_ay_x=p_ay_x,
                                          groups=groups,
                                          labels_ay=labels_ay),
        sample_weight=sample_weight,
        idx_post=idx_post,
        idx_dcal=idx_dcal,
        solver=solver,
        solve_kwargs=solve_kwargs,
        solve_primal=solve_primal,
    )
    return self

  def predict_score(self,
                    p_a_x: Optional[np.ndarray] = None,
                    p_y_x: Optional[np.ndarray] = None,
                    p_ay_x: Optional[np.ndarray] = None) -> np.ndarray:
    return self.postprocessor.predict_score(**self.get_risk_and_group_probas_(
        p_a_x=p_a_x, p_y_x=p_y_x, p_ay_x=p_ay_x))

  def predict(self,
              p_a_x: Optional[np.ndarray] = None,
              p_y_x: Optional[np.ndarray] = None,
              p_ay_x: Optional[np.ndarray] = None) -> np.ndarray:
    return self.predict_score(p_a_x=p_a_x, p_y_x=p_y_x,
                              p_ay_x=p_ay_x).argmin(axis=1)

  @staticmethod
  def get_fairness_constraints_(
      n_classes: int,
      n_groups: int,
      fairness_criterion: Literal['sp', 'tpr', 'fpr', 'eo'] = 'sp',
      remove_unused: bool = False,
  ):
    if fairness_criterion == 'sp':
      # groups passed to LinearPost are sensitive attributes A
      fairness_constraints = [
          (y_c, np.arange(n_groups)) for y_c in range(n_classes)
      ]
    elif fairness_criterion == 'fpr':
      assert n_classes == 2
      # groups passed to LinearPost are joint (A, Y)
      if not remove_unused:
        fairness_constraints = [(1, n_classes * np.arange(n_groups) + 0)]
      else:
        fairness_constraints = [(1, np.arange(n_groups))]
    else:
      if fairness_criterion == 'tpr' and n_classes == 2:
        # groups passed to LinearPost are joint (A, Y)
        if not remove_unused:
          fairness_constraints = [(1, n_classes * np.arange(n_groups) + 1)]
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
    return fairness_constraints

  def get_risk_and_group_probas_(self,
                                 p_a_x: Optional[np.ndarray] = None,
                                 p_y_x: Optional[np.ndarray] = None,
                                 p_ay_x: Optional[np.ndarray] = None,
                                 groups: Optional[np.ndarray] = None,
                                 labels_ay: Optional[np.ndarray] = None):
    # groups_ is the group labels to be passed to LinearPost (not necessarily
    # the same as the sensitive attributes A passed in here as groups)
    groups_ = None

    if self.fairness_criterion == 'sp':
      if p_ay_x is None and (p_a_x is None or p_y_x is None):
        raise ValueError(
            'p_ay_x or (p_a_x and p_y_x) must be provided for `sp` criterion')
      if p_a_x is None:
        p_a_x = p_ay_x.reshape(-1, self.n_groups, self.n_classes).sum(axis=2)
      if p_y_x is None:
        p_y_x = p_ay_x.reshape(-1, self.n_groups, self.n_classes).sum(axis=1)
      p_g_x = p_a_x.reshape(-1, self.n_groups)
      if groups is not None:
        groups_ = groups
    else:
      if p_ay_x is None:
        raise ValueError('p_ay_x must be provided for `eopp` or `eo` criterion')
      if p_y_x is None:
        p_y_x = p_ay_x.reshape(-1, self.n_groups, self.n_classes).sum(axis=1)
      p_g_x = p_ay_x.reshape(-1, self.n_groups * self.n_classes)
      if labels_ay is not None:
        groups_ = labels_ay

    if self.remove_unused and self.n_classes == 2:
      if self.fairness_criterion == 'tpr':
        p_g_x = p_g_x.reshape(-1, self.n_groups, self.n_classes)[:, :, 1]
        # p_g_x.shape = (n_examples, n_groups)
        # multilabel
        if labels_ay is not None:
          groups_ = ((self.n_classes * np.arange(self.n_groups) +
                      1) == labels_ay[:, None]).astype(int)
      elif self.fairness_criterion == 'fpr':
        p_g_x = p_g_x.reshape(-1, self.n_groups, self.n_classes)[:, :, 0]
        if labels_ay is not None:
          groups_ = ((self.n_classes * np.arange(self.n_groups) +
                      0) == labels_ay[:, None]).astype(int)

    risk = np.sum(p_y_x[..., None] * self.cls_loss_fn[None, :], axis=1)

    res = {'probas_group': p_g_x, 'risk': risk}
    if groups_ is not None:
      res['groups'] = groups_
    return res


class LinearPostOverlapping(LinearPostSimple):
  # p_s_x.shape = [n_examples, 2**n_groups_overlap]
  # p_sy_x.shape = [n_examples, 2**n_groups_overlap, n_classes]

  # For example, if n_groups_overlap is 5 (overlapping), then
  # s = 0b11101
  # means that the example belongs to groups 0, 1, 2, and 4 (0-indexed)

  # The group s = 0 is not protected and ignored

  def __init__(
      self,
      n_classes: int,
      n_groups: int,  # number of overlapping groups
      ways: list[int] | Literal['all'] = None,
      fairness_criterion: Literal['sp', 'tpr', 'fpr', 'eo'] = 'sp',
      remove_unused: bool = False,
      class_weight: Optional[list[float]] = None,
      alpha: Optional[float] = None,
      noise_scale: Optional[float] = None,
      seed: Optional[int] = None) -> None:
    self.n_groups_overlap = n_groups_overlap = n_groups
    if ways is None:
      ways = [1]
    self.ways = list(range(1, n_groups_overlap + 1)) if ways == 'all' else ways
    super().__init__(
        n_classes=n_classes,
        n_groups=self.get_n_subgroups_(n_groups_overlap, self.ways),
        fairness_criterion=fairness_criterion,
        remove_unused=remove_unused,
        class_weight=class_weight,
        alpha=alpha,
        noise_scale=noise_scale,
        seed=seed,
    )

  @staticmethod
  def get_n_subgroups_(n_groups: int, ways: list[int]) -> int:
    return sum(1 for k in ways for _ in combinations(range(1, n_groups), k))

  @staticmethod
  def get_idx_containing_groups_(groups: list[int],
                                 n_groups_overlap: int) -> list[int]:
    subgroup_enc = sum((1 << (s - 1)) for s in groups)
    contains_s = lambda i: (i & subgroup_enc) == subgroup_enc
    return [i for i in range(2**n_groups_overlap) if contains_s(i)]

  def get_risk_and_group_probas_(self,
                                 p_a_x: Optional[np.ndarray] = None,
                                 p_y_x: Optional[np.ndarray] = None,
                                 p_ay_x: Optional[np.ndarray] = None,
                                 groups: Optional[np.ndarray] = None,
                                 labels_ay: Optional[np.ndarray] = None):
    # TODO: doesn't work with decision calibration yet
    if p_y_x is None:
      assert p_ay_x is not None, 'p_y_x or p_ay_x must be provided'
      p_y_x = p_ay_x.reshape(-1, 2**self.n_groups_overlap,
                             self.n_classes).sum(axis=1)
    if p_a_x is not None:
      p_s_x = []
      for g in (g for k in self.ways
                for g in combinations(range(1, self.n_groups_overlap), k)):
        i = self.get_idx_containing_groups_(g, self.n_groups_overlap)
        p_s_x.append(p_a_x[:, i].sum(axis=1))
      p_a_x = np.stack(p_s_x, axis=1)  # overwrite p_a_x
    if p_ay_x is not None:
      p_ay_x = p_ay_x.reshape(-1, 2**self.n_groups_overlap, self.n_classes)
      p_sy_x = []
      for g in (g for k in self.ways
                for g in combinations(range(1, self.n_groups_overlap), k)):
        i = self.get_idx_containing_groups_(g, self.n_groups_overlap)
        p_sy_x.append(p_ay_x[:, i].sum(axis=1))
      p_ay_x = np.stack(p_sy_x, axis=1)  # overwrite p_ay_x
    return super().get_risk_and_group_probas_(p_a_x=p_a_x,
                                              p_y_x=p_y_x,
                                              p_ay_x=p_ay_x)
