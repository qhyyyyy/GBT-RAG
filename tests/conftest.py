from pathlib import Path
import pytest
from experiments.run import _load_config


@pytest.fixture
def cfg():
    config = _load_config()
    config.retrieval.scope = "candidate_debug"
    config.reranker.enabled = False
    config.gbtrag.enable_dimension_discovery = False
    config.gbtrag.enable_dead_leaf_pruning = False
    config.gbtrag.convergence_epsilon = 0.00001
    config.gbtrag.stagnation_delta = 0.000001
    config.experiment.cache_dir = None
    return config
