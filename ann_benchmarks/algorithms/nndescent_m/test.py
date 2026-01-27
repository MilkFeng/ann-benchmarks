# cd .\source\ ; pip install . ; cd .. ; python ./test.py

import os
import sys
import time

import h5py
import nndescent_m
import numpy as np
import scipy.sparse
from sklearn import neighbors


class NNDescentM:
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
        # 转换查询向量为 float32
        v_f32 = v.reshape(1, -1).astype("float32")

        # 修正：C++ 只返回了 indices (std::vector<uint32_t>)
        # 不需要接收 dist
        with nndescent_m.ostream_redirect(stdout=True, stderr=True):
            ind = self.index.query(v_f32, k=n, epsilon=self.epsilon)
        return ind  # BaseANN 期望返回索引列表

    def __str__(self):
        return (
            f"NNDescentM(n_neighbors={self.n_neighbors}, "
            f"pruning_mult={self.pruning_degree_multiplier:.2f}, "
            f"pruning_prob={self.pruning_prob:.3f}, "
            f"leaf_size={getattr(self, 'leaf_size', 32)}, "
            f"metric={self.nnd_metric})"
        )


def test(X_train, X_test, ground_truth, distance_metric):
    # 初始化算法
    params = {"n_neighbors": 20, "pruning_degree_multiplier": 0.5, "pruning_prob": 1.0, "leaf_size": 20}

    print("\n[1/3] 初始化与构建索引 (Fit)...")
    algo = NNDescentM(distance_metric, params)

    start_time = time.time()
    algo.fit(X_train)
    print(f"构建耗时: {time.time() - start_time:.4f} 秒")

    print(f"构建结果：\n{algo}")

    print("\n[2/3] 执行查询 (Query)...")
    algo.set_query_arguments(epsilon=0.2)

    k = 10
    hits = 0
    total = 0

    total_queries = len(X_test)
    query_start_time = time.time()

    for i in range(len(X_test)):
        # 你的 C++ query 接口
        query_vec = X_test[i]
        predicted_ids = algo.query(query_vec, k)

        # 简单的 Recall@K 计算
        true_ids = set(ground_truth[i][:k])
        # 注意：C++ 返回的是 list，直接用
        predicted_set = set(predicted_ids)

        intersection = true_ids.intersection(predicted_set)
        hits += len(intersection)
        total += k

        if i == 0:
            print("\n  [示例 Query #0]")
            print(f"  预测: {predicted_ids}")
            print(f"  真值: {true_ids}")
            print(f"  重合: {intersection}/{k}")

    print(f"\n[3/3] 结果验证...")

    recall = hits / total

    query_total_time = time.time() - query_start_time
    qps = total_queries / query_total_time

    print("-" * 50)
    print("🏆 最终结果 (Swiss Roll Dataset):")
    print(f"   Recall@{k} : {recall:.4f}")
    print(f"   QPS        : {qps:.2f} queries/s")
    print("-" * 50)

    if recall > 0.5:
        print("\n✅ 测试通过！你的 NNDescentM 已经可以工作了。")
        print("下一步：尝试运行 python run.py --algorithm nndescent_m --dataset glove-100-angular")
    else:
        print("\n⚠️ 召回率较低，请检查 C++ 的距离计算或构建参数。")


def load_hdf5_and_test(filename):
    print(f"正在读取 {filename} ...")
    with h5py.File(filename, "r") as f:
        X_train = np.array(f["train"])
        X_test = np.array(f["test"])
        ground_truth = np.array(f["neighbors"])
        distance_metric = f.attrs["distance"]

    print(f"数据加载完毕: Train={X_train.shape}, Test={X_test.shape}, Metric={distance_metric}")

    test(X_train, X_test, ground_truth, distance_metric)


def generate_and_test():
    from sklearn.datasets import make_swiss_roll
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestNeighbors

    print("生成 Swiss Roll 数据集...")
    X, _ = make_swiss_roll(n_samples=10000, noise=0.1)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    # 计算真实邻居
    nbrs = NearestNeighbors(n_neighbors=10, algorithm="brute", metric="cosine").fit(X_train)
    distances, indices = nbrs.kneighbors(X_test)

    print("数据集参数：")
    print(f"  训练集大小: {X_train.shape}")
    print(f"  测试集大小: {X_test.shape}")
    print(f"  距离度量: angular")

    test(X_train, X_test, indices, distance_metric="angular")


if __name__ == "__main__":
    # test_file = "C:\\Users\\31070\\Desktop\\PROGRAMS\\db\\ann-benchmarks\\data\\mnist-784-euclidean.hdf5"
    test_file = "C:\\Users\\31070\\Desktop\\PROGRAMS\\db\\ann-benchmarks\\data\\glove-100-angular.hdf5"
    load_hdf5_and_test(test_file)

    # generate_and_test()
