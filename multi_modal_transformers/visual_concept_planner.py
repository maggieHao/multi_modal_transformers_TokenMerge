"""A hierarchical model for learning visual concepts and planning with them."""

import flax
import flax.linen as nn
import flax.struct as struct


@struct.dataclass
class VisualConceptPlanner:
    """A hierarchical model for learning visual concepts and planning with them."""
    executor: nn.Module
    planner: nn.Module

