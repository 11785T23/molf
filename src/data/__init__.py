from .fact import *
from .sql import *
from .medmcqa import *

ft_dataset_builder_map = {
    'fact': CounterfactDatasetBuilder,
    'sql': SQLDatasetBuilder,
    'med': MedMCQADatasetBuilder,
}
