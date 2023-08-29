"""A hierarchical model for learning visual concepts and planning with them."""

import flax
import flax.linen as nn
import flax.struct as struct
from flax.training import train_state

@struct.dataclass
class VisualConceptPlanner:
    """A hierarchical model for learning visual concepts and planning with them."""
    executor_model: train_state.TrainState
    planner_model: train_state.TrainState

