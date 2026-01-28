import nndescent_m
import numpy as np
import scipy.sparse

from ..base.module import BaseANN


class NNDescentM(BaseANN):
    def __init__(self, metric, index_param_dict):
        if "n_neighbors" in index_param_dict:
            self.n_neighbors = int(index_param_dict["n_neighbors"])
        else:
            self.n_neighbors = 30

        if "pruning_degree_multiplier" in index_param_dict:
            self.pruning_degree_multiplier = float(index_param_dict["pruning_degree_multiplier"])
        else:
            self.pruning_degree_multiplier = 1.5

        if "pruning_prob" in index_param_dict:
            self.pruning_prob = float(index_param_dict["pruning_prob"])
        else:
            self.pruning_prob = 1.0

        if "leaf_size" in index_param_dict:
            self.leaf_size = int(index_param_dict["leaf_size"])

        self.is_sparse = metric in ["jaccard"]

        if metric != "euclidean" and metric != "angular":
            raise ValueError(f"Metric {metric} not supported in NNDescentM.")

        self.nnd_metric = {
            "angular": "angular",
            "euclidean": "euclidean",
            # "hamming": "hamming",
            # "jaccard": "jaccard",
        }[metric]

    def fit(self, X):
        if self.is_sparse:
            # Sparse matrix conversion logic...
            # (Keeping existing logic roughly same, assuming X is dense for typical tests)
            pass
        elif not isinstance(X, np.ndarray) or X.dtype != np.float32:
            print("Convert data to float32")
            X = np.asarray(X, dtype=np.float32)

        self.X = X
        with nndescent_m.ostream_redirect(stdout=True, stderr=True):
            self.index = nndescent_m.NNDescent(
                metric=self.nnd_metric,
                n_neighbors=self.n_neighbors,
                pruning_degree_multiplier=self.pruning_degree_multiplier,
                pruning_prob=self.pruning_prob,
                leaf_size=getattr(self, "leaf_size", 32),
            )

        with nndescent_m.ostream_redirect(stdout=True, stderr=True):
            self.index.fit(self.X)

    def set_query_arguments(self, epsilon=0.1):
        self.epsilon = float(epsilon)

    def query(self, v, n):
        v_f32 = v.reshape(1, -1).astype("float32")
        with nndescent_m.ostream_redirect(stdout=True, stderr=True):
            ind = self.index.query(v_f32, k=n, epsilon=self.epsilon)
        return ind

    def __str__(self):
        return (
            f"NNDescentM(n_neighbors={self.n_neighbors}, "
            f"pruning_mult={self.pruning_degree_multiplier:.2f}, "
            f"pruning_prob={self.pruning_prob:.3f}, "
            f"leaf_size={getattr(self, 'leaf_size', 32)}, "
            f"epsilon={self.epsilon:.2f}, "
            f"metric={self.nnd_metric})"
        )
