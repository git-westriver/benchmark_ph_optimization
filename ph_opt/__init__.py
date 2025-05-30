from .ph_compute.ph_computation_library import RipsPH, Bar

from .ph_loss.algorithms import PHOptimization, GradientDescent, Diffeo, BigStep, Continuation
from .ph_loss.persistence_based_loss import get_rph, PersistenceBasedLoss, WassersteinLoss, ExpandLoss
from .ph_loss.regularization import Regularization, RectangleRegularization

from .ph_grad.wasserstein import powered_wasserstein_distance_one_sided
from .ph_grad.singleton import singleton_loss_from_bar_to_target

from .utils.visualization import is_persistence_diagram, get_max_death_of_pds, plot_pd_with_specified_lim, get_animation
from .utils.ph_trainer import PHTrainerConfig, PHTrainer