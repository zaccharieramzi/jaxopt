# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implicit differentiation of logistic regression.
=============================================
"""

from absl import app
import jax
import jax.numpy as jnp
from jaxopt import implicit_diff
from jaxopt import OptaxSolver
from jaxopt import LBFGS
from jaxopt import loss
from jaxopt.objective import l2_multiclass_logreg_with_intercept
import optax
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing

_logloss_vmap = _logloss_vmap = jax.vmap(loss.multiclass_logistic_loss)


@implicit_diff.custom_root(
  jax.grad(l2_multiclass_logreg_with_intercept),
  has_aux=True,
)
def log_reg_solver(init_params, l2reg, data):
  """Solve logisitic regression by LBFGS."""
  solver = LBFGS(
    fun=l2_multiclass_logreg_with_intercept,
    tol=1e-5,
    history_size=30,
  )

  return solver.run(init_params=init_params, l2reg=l2reg, data=data)


# Perhaps confusingly, theta is a parameter of the outer objective,
# but l2reg = jnp.exp(theta) is an hyper-parameter of the inner objective.
def outer_objective(theta, init_inner, data):
  """Validation loss."""
  X_tr, X_val, y_tr, y_val = data
  # We use the bijective mapping l2reg = jnp.exp(theta)
  # both to optimize in log-space and to ensure positivity.
  l2reg = jnp.exp(theta)
  w_fit, b_fit = log_reg_solver(init_inner, l2reg, (X_tr, y_tr)).params
  y_pred = jnp.dot(X_val, w_fit) + b_fit
  loss_value = jnp.mean(_logloss_vmap(y_val, y_pred))
  # We return w_fit as auxiliary data.
  # Auxiliary data is stored in the optimizer state (see below).
  return loss_value, (w_fit, b_fit)


def main(argv):
  del argv

  # Prepare data.
  X, y = datasets.load_digits(return_X_y=True)
  X = preprocessing.normalize(X)
  n_classes = len(set(y))
  # data = (X_tr, X_val, y_tr, y_val)
  data = model_selection.train_test_split(X, y, test_size=0.33, random_state=0)

  # Initialize solver.
  solver = OptaxSolver(opt=optax.sgd(5e-1), fun=outer_objective, has_aux=True)
  theta = -3.0
  init_w = jnp.ones((X.shape[1], n_classes))
  init_b = jnp.zeros((n_classes,))
  init_inner = (init_w, init_b)
  state = solver.init_state(theta, init_inner=init_inner, data=data)

  # Run outer loop.
  for _ in range(50):
    theta, state = solver.update(params=theta, state=state, init_inner=init_inner,
                                 data=data)
    # The auxiliary data returned by the outer loss is stored in the state.
    init_inner = state.aux
    print(init_inner[1])
    print(f"[Step {state.iter_num}] Validation loss: {state.value:.3f}, Reg param: {jnp.exp(theta):.3e}.")

if __name__ == "__main__":
  app.run(main)
