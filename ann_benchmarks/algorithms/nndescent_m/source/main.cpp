#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <memory>
#include <numeric>
#include <print>
#include <queue>
#include <random>
#include <ranges>
#include <utility>
#include <vector>

namespace py = pybind11;

using py_float_array_t = py::array_t<float, py::array::c_style | py::array::forcecast>;

struct Neighbor {
  uint32_t id = {};
  float dist = {};

  bool is_new = true;
};

struct Candidate {
  uint32_t id = {};
  float dist = {};

  bool operator>(const Candidate& rhs) const {
    if (dist == rhs.dist) {
      return id > rhs.id;
    }
    return dist > rhs.dist;
  }
  bool operator<(const Candidate& rhs) const {
    if (dist == rhs.dist) {
      return id < rhs.id;
    }
    return dist < rhs.dist;
  }
};

struct RPTreeNode {
  bool is_leaf = {};

  uint32_t idx_a = {};
  uint32_t idx_b = {};

  std::unique_ptr<RPTreeNode> left = {};
  std::unique_ptr<RPTreeNode> right = {};

  std::vector<uint32_t> leaf_indices = {};
};

class NNDescent {
 public:
  NNDescent(std::string metric, const size_t n_neighbors, const float pruning_degree_multiplier,
            const float pruning_prob, const size_t leaf_size)
      : metric_(std::move(metric)),
        n_neighbors_(n_neighbors),
        pruning_degree_multiplier_(pruning_degree_multiplier),
        pruning_prob_(pruning_prob),
        leaf_size_(leaf_size),
        rng_(std::time(nullptr)) {
    max_candidates_ = std::min(60uz, n_neighbors_);

    if (metric_ == "euclidean") {
      normalize_ = false;
    } else if (metric_ == "angular" || metric_ == "cosine") {
      normalize_ = true;
    } else {
      throw std::runtime_error("Only 'euclidean' and 'angular'('cosine') metrics are supported.");
    }
  }

  void fit(const py_float_array_t input) {
    auto buf = input.request();
    assert(buf.ndim == 2);

    num_elements_ = buf.shape[0];
    dim_ = buf.shape[1];
    const float* raw_data_ptr = static_cast<const float*>(buf.ptr);

    data_.resize(num_elements_ * dim_);
    std::memcpy(data_.data(), raw_data_ptr, sizeof(float) * num_elements_ * dim_);

    if (normalize_) {
      std::println("Normalizing dataset for angular metric...");
      for (const auto i : std::views::iota(0u, num_elements_)) {
        normalize_vec(get_vec(i));
      }
    }

    graph_.resize(num_elements_);

    init_build_sketch();

    std::println("Initializing graph with RP Trees...");
    initialize_graph();

    std::println("Starting NNDescent iterations...");
    build_graph();

    std::println("Adding reverse edges...");
    add_reverse_edges();

    std::println("Pruning graph with RNG rules...");
    prune_graph();

    clear_build_sketch();

    visited_tags_.resize(num_elements_, 0);
    current_query_tag_ = 0;

    std::println("Graph Index build complete.");
  }

  std::vector<uint32_t> query(const py_float_array_t query_vec, const size_t k, const float epsilon = 0.1f) {
    ++current_query_tag_;
    if (current_query_tag_ == 0) {
      std::ranges::fill(visited_tags_, 0);
      current_query_tag_ = 1;
    }

    const auto buf = query_vec.request();
    assert(buf.ndim == 2);

    const uint32_t num_query = buf.shape[0];
    assert(num_query == 1);

    const uint32_t dim = buf.shape[1];
    assert(dim == dim_);

    const float* query_raw_ptr = static_cast<const float*>(buf.ptr);
    std::vector<float> query_storage = {query_raw_ptr, query_raw_ptr + dim_};

    if (normalize_) {
      normalize_vec(query_storage.data());
    }

    const float* query_ptr = query_storage.data();

    const size_t ef = std::max(k, static_cast<size_t>(k * (1.0f + epsilon)));

    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> candidates = {};
    std::priority_queue<Candidate> top_candidates = {};

    auto init_ids = query_rp_trees(query_ptr);
    if (init_ids.empty()) {
      init_ids.push_back(0);
    }

    for (const auto id : init_ids) {
      mark_visited(id);
      const auto dist = compute_dist(query_ptr, get_vec(id));

      candidates.push({id, dist});
      top_candidates.push({id, dist});
    }

    while (!candidates.empty()) {
      auto [id, dist] = candidates.top();
      candidates.pop();

      // if top_candidates is full and the new distance is greater than the worst distance in top_candidates, skip
      if (top_candidates.size() >= ef && dist > top_candidates.top().dist) {
        break;
      }

      for (auto [neighbor_id, dist, _] : graph_[id]) {
        if (is_visited(neighbor_id)) {
          continue;
        }
        mark_visited(neighbor_id);

        const auto new_dist = compute_dist(query_ptr, get_vec(neighbor_id));

        if (top_candidates.size() < ef || new_dist < top_candidates.top().dist) {
          candidates.push({neighbor_id, new_dist});
          top_candidates.push({neighbor_id, new_dist});

          // if top_candidates is full, pop the worst candidate
          if (top_candidates.size() > ef) {
            top_candidates.pop();
          }
        }
      }
    }

    // get top k candidates
    std::vector<uint32_t> top_k_candidates = {};
    top_k_candidates.reserve(top_candidates.size());
    while (!top_candidates.empty()) {
      auto [id, dist] = top_candidates.top();
      top_candidates.pop();

      top_k_candidates.push_back(id);
    }
    if (top_k_candidates.size() > k) {
      const auto start_it = top_k_candidates.end() - k;
      std::vector<uint32_t> final_results = {start_it, top_k_candidates.end()};
      std::ranges::reverse(final_results);
      return final_results;
    } else {
      std::ranges::reverse(top_k_candidates);
      return top_k_candidates;
    }
  }

 private:
  const float* get_vec(const uint32_t id) const { return &data_[id * dim_]; }

  float* get_vec(const uint32_t id) { return &data_[id * dim_]; }

  void normalize_vec(float* vec) const {
    float sum_sq = 0.0f;
    for (const auto i : std::views::iota(0u, dim_)) {
      sum_sq += vec[i] * vec[i];
    }

    if (sum_sq > 1e-12f) {
      float norm_inv = 1.0f / std::sqrt(sum_sq);
      for (const auto i : std::views::iota(0u, dim_)) {
        vec[i] *= norm_inv;
      }
    }
  }

  float compute_dist(const float* a, const float* b) const {
    float dist = 0.0f;
    for (const auto i : std::views::iota(0u, dim_)) {
      float diff = a[i] - b[i];
      dist += diff * diff;
    }
    return dist;
  }

  float compute_dist(const uint32_t i, const uint32_t j) const { return compute_dist(get_vec(i), get_vec(j)); }

  void sort_and_unique(std::vector<Neighbor>& neighbors) {
    std::ranges::sort(neighbors, std::less{}, &Neighbor::id);
    const auto unique_range = std::ranges::unique(neighbors, {}, &Neighbor::id);
    neighbors.erase(unique_range.begin(), unique_range.end());

    std::ranges::sort(neighbors, std::less{}, &Neighbor::dist);
  }

  void sort_and_unique(std::vector<uint32_t>& candidates) {
    std::ranges::sort(candidates);
    const auto unique_range = std::ranges::unique(candidates);
    candidates.erase(unique_range.begin(), unique_range.end());
  }

  template <typename T>
  void insert_with_order(std::vector<T>& items, const T& new_item) {
    const auto insert_it = std::ranges::lower_bound(items, new_item.dist, std::less{}, &T::dist);
    items.insert(insert_it, new_item);
  }

  void init_build_sketch() {
    new_candidates_sketch_.resize(num_elements_);
    old_candidates_sketch_.resize(num_elements_);

    for (auto& candidates : new_candidates_sketch_) {
      candidates.reserve(max_candidates_);
    }
    for (auto& candidates : old_candidates_sketch_) {
      candidates.reserve(max_candidates_);
    }
  }

  void clear_build_sketch() {
    new_candidates_sketch_.clear();
    new_candidates_sketch_.shrink_to_fit();

    old_candidates_sketch_.clear();
    old_candidates_sketch_.shrink_to_fit();
  }

  bool is_visited(const uint32_t id) const { return visited_tags_[id] == current_query_tag_; }

  void mark_visited(const uint32_t id) { visited_tags_[id] = current_query_tag_; }

  void initialize_graph() {
    build_rp_trees();

    for (auto& neighbors : graph_) {
      sort_and_unique(neighbors);
    }

    randomly_fill_graph();
  }

  void build_rp_trees() {
    roots_.clear();

    std::vector<uint32_t> indices(num_elements_);
    std::ranges::iota(indices, 0);

    const size_t dynamic_trees = 5 + std::round(std::sqrt(num_elements_) / 20.0);
    const size_t n_trees = std::min(64uz, std::max(5uz, dynamic_trees));

    std::println("Building {} RP Trees...", n_trees);

    for (const auto tree_iter : std::views::iota(0u, n_trees)) {
      std::ranges::shuffle(indices, rng_);
      auto root = build_rp_tree(indices);
      roots_.emplace_back(std::move(root));
    }
  }

  void randomly_fill_graph() {
    std::uniform_int_distribution<uint32_t> dist_gen{0u, static_cast<uint32_t>(num_elements_ - 1)};

    const size_t target_size = std::min(static_cast<size_t>(n_neighbors_), num_elements_ - 1);
    for (const auto& [id, neighbors] : std::views::enumerate(graph_)) {
      while (neighbors.size() < target_size) {
        auto rand_id = dist_gen(rng_);
        if (rand_id == id) {
          continue;
        }

        const bool exists = std::ranges::contains(neighbors, rand_id, &Neighbor::id);
        if (!exists) {
          insert_with_order(neighbors, {rand_id, compute_dist(id, rand_id), true});
        }
      }

      if (neighbors.size() > n_neighbors_) {
        neighbors.resize(n_neighbors_);
      }
    }
  }

  std::unique_ptr<RPTreeNode> build_rp_tree(const std::span<uint32_t> indices) {
    auto node = std::make_unique<RPTreeNode>();

    if (indices.size() <= leaf_size_) {
      // leaf node

      node->is_leaf = true;
      node->leaf_indices.assign(indices.begin(), indices.end());

      for (const auto i : std::views::iota(0u, indices.size())) {
        uint32_t u = indices[i];
        for (const auto j : std::views::iota(i + 1u, indices.size())) {
          uint32_t v = indices[j];
          float d = compute_dist(u, v);

          graph_[u].emplace_back(v, d, true);
          graph_[v].emplace_back(u, d, true);
        }
      }
      return node;
    }

    // split into two partitions:
    // left partition: closer to idx_a
    // right partition: closer to idx_b
    std::uniform_int_distribution<size_t> idx_dist{0, indices.size() - 1};
    uint32_t idx_a = indices[idx_dist(rng_)];
    uint32_t idx_b = indices[idx_dist(rng_)];

    uint32_t attempts = 0;
    while (idx_a == idx_b && attempts < 10) {
      idx_b = indices[idx_dist(rng_)];
      ++attempts;
    }

    node->idx_a = idx_a;
    node->idx_b = idx_b;

    const auto part_result = std::ranges::partition(indices, [&](uint32_t idx) {
      float dist_a = compute_dist(idx, idx_a);
      float dist_b = compute_dist(idx, idx_b);
      return dist_a < dist_b;
    });

    const auto right_count = part_result.size();
    auto left_count = indices.size() - right_count;

    if (left_count == 0 || right_count == 0) {
      std::ranges::shuffle(indices, rng_);
      left_count = indices.size() / 2;
    }

    node->left = build_rp_tree(indices.first(left_count));
    node->right = build_rp_tree(indices.subspan(left_count));

    return node;
  }

  std::vector<uint32_t> query_rp_trees(const float* query_vec) {
    std::vector<uint32_t> candidate_indices = {};

    for (const auto& root : roots_) {
      RPTreeNode* current_node = root.get();

      while (!current_node->is_leaf) {
        float dist_a = compute_dist(query_vec, get_vec(current_node->idx_a));
        float dist_b = compute_dist(query_vec, get_vec(current_node->idx_b));

        if (dist_a < dist_b) {
          current_node = current_node->left.get();
        } else {
          current_node = current_node->right.get();
        }
      }

      candidate_indices.insert(candidate_indices.end(), current_node->leaf_indices.begin(),
                               current_node->leaf_indices.end());
    }

    // remove duplicates
    sort_and_unique(candidate_indices);
    return candidate_indices;
  }

  void build_graph() {
    constexpr size_t max_iters = 30;
    const size_t min_updates_threshold = std::max<size_t>(100uz, num_elements_ * n_neighbors_ * 0.001);

    const auto try_connect_mono = [&](const uint32_t u, const uint32_t v, const float dist) {
      auto& neighbors = graph_[u];
      if (neighbors.size() < n_neighbors_ || dist < neighbors.back().dist) {
        insert_with_order(neighbors, {v, dist, true});
        if (neighbors.size() > n_neighbors_) {
          neighbors.pop_back();
        }
        return true;
      }
      return false;
    };

    const auto try_connect = [&](const uint32_t u, const uint32_t v) {
      if (u == v) {
        return false;
      }

      const auto u_has_v = std::ranges::contains(graph_[u], v, &Neighbor::id);
      const auto v_has_u = std::ranges::contains(graph_[v], u, &Neighbor::id);
      if (u_has_v && v_has_u) {
        return false;
      }

      const auto dist = compute_dist(u, v);
      bool changed = false;

      if (!u_has_v) {
        changed |= try_connect_mono(u, v, dist);
      }

      if (!v_has_u) {
        changed |= try_connect_mono(v, u, dist);
      }

      return changed;
    };

    for (const auto iter : std::views::iota(0u, max_iters)) {
      std::println("Starting iteration {}/{}...", iter + 1, max_iters);

      sample_candidates();

      size_t updates = 0;

      for (const auto u : std::views::iota(0u, num_elements_)) {
        const auto& new_candidates = new_candidates_sketch_[u];
        const auto& old_candidates = old_candidates_sketch_[u];

        // new-new
        for (const auto i : std::views::iota(0u, new_candidates.size())) {
          for (const auto j : std::views::iota(i + 1u, new_candidates.size())) {
            if (try_connect(new_candidates[i].id, new_candidates[j].id)) {
              ++updates;
            }
          }
        }

        // new-old
        for (const auto& new_candidate : new_candidates) {
          for (const auto& old_candidate : old_candidates) {
            if (try_connect(new_candidate.id, old_candidate.id)) {
              ++updates;
            }
          }
        }
      }

      std::println("Iteration {}/{}: {} updates.", iter + 1, max_iters, updates);
      if (updates < min_updates_threshold) {
        std::println("Converged.");
        break;
      }
    }
  }

  void sample_candidates() {
    for (auto& candidates : new_candidates_sketch_) {
      candidates.clear();
    }
    for (auto& candidates : old_candidates_sketch_) {
      candidates.clear();
    }

    const auto checked_push = [&](std::vector<Candidate>& candidates, const uint32_t id, const float priority) {
      if (candidates.size() >= max_candidates_ && priority >= candidates.back().dist) {
        return false;
      }

      const bool exists = std::ranges::contains(candidates, id, &Candidate::id);
      if (exists) {
        return false;
      }

      insert_with_order(candidates, {id, priority});

      if (candidates.size() > max_candidates_) {
        candidates.pop_back();
      }

      return true;
    };

    std::uniform_real_distribution<float> priority_dist{0.0f, 1.0f};

    for (const auto& [u, neighbors] : std::views::enumerate(graph_)) {
      for (const auto [v, dist, is_new] : neighbors) {
        const auto priority = priority_dist(rng_);
        if (is_new) {
          checked_push(new_candidates_sketch_[u], v, priority);
          checked_push(new_candidates_sketch_[v], u, priority);
        } else {
          checked_push(old_candidates_sketch_[u], v, priority);
          checked_push(old_candidates_sketch_[v], u, priority);
        }
      }
    }

    // mark sampled candidates as old
    for (const auto& [u, neighbors] : std::views::enumerate(graph_)) {
      for (auto& [v, _, is_new] : neighbors) {
        if (!is_new) {
          continue;
        }

        if (std::ranges::contains(new_candidates_sketch_[u], v, &Candidate::id)) {
          is_new = false;
        }
      }
    }
  }

  void add_reverse_edges() {
    std::vector<std::vector<Neighbor>> reverse_graph(num_elements_);

    for (const auto& [id, neighbors] : std::views::enumerate(graph_)) {
      for (auto& neighbor : neighbors) {
        reverse_graph[neighbor.id].emplace_back(static_cast<uint32_t>(id), neighbor.dist);
      }
    }

    for (const auto& [id, new_edges] : std::views::enumerate(reverse_graph)) {
      auto& neighbors = graph_[id];
      neighbors.insert(neighbors.end(), new_edges.begin(), new_edges.end());

      sort_and_unique(neighbors);
      if (neighbors.size() > n_neighbors_) {
        neighbors.resize(n_neighbors_);
      }
    }
  }

  void prune_graph() {
    // RNG rule:
    // dis(current, existing) < dis(current, candidate) && dis(candidate, existing) < dis(current, candidate) ->
    // candidate is shadowed by existing neighbor, keep it with probability

    const size_t max_degree = n_neighbors_ * pruning_degree_multiplier_;

    std::uniform_real_distribution<float> prob_dist{0.0f, 1.0f};

    for (auto& neighbors : graph_) {
      std::vector<Neighbor> pruned_neighbors = {};
      pruned_neighbors.reserve(max_degree);

      // enumerate candidates from closest to furthest
      for (const auto& candidate : neighbors) {
        if (pruned_neighbors.size() >= max_degree) {
          break;
        }

        // check if candidate is shadowed by existing neighbors
        bool keep = true;
        for (const auto& existing : pruned_neighbors) {
          float dist = compute_dist(existing.id, candidate.id);
          if (dist < candidate.dist) {
            // keep it with probability
            if (prob_dist(rng_) < pruning_prob_) {
              keep = false;
              break;
            }
          }
        }

        if (keep) {
          pruned_neighbors.push_back(candidate);
        }
      }

      neighbors = std::move(pruned_neighbors);
    }
  }

 private:
  std::string metric_ = {};
  bool normalize_ = {};

  size_t n_neighbors_ = {};
  float pruning_degree_multiplier_ = {};
  float pruning_prob_ = {};
  size_t leaf_size_ = {};

  size_t max_candidates_ = {};

  size_t num_elements_ = {};
  size_t dim_ = {};

  std::vector<float> data_ = {};

  std::vector<std::unique_ptr<RPTreeNode>> roots_ = {};

  // build sketch
  std::vector<std::vector<Candidate>> new_candidates_sketch_ = {};
  std::vector<std::vector<Candidate>> old_candidates_sketch_ = {};

  std::vector<std::vector<Neighbor>> graph_ = {};

  // do not use std::vector<bool> to record visit status
  std::vector<uint32_t> visited_tags_ = {};
  uint32_t current_query_tag_ = {};

  std::mt19937 rng_ = {};
};

PYBIND11_MODULE(nndescent_m, m) {
  py::class_<NNDescent>(m, "NNDescent")
      .def(py::init<std::string, size_t, float, float, size_t>(), py::arg("metric"), py::arg("n_neighbors"),
           py::arg("pruning_degree_multiplier"), py::arg("pruning_prob"), py::arg("leaf_size"))
      .def("fit", &NNDescent::fit, py::arg("input"))
      .def("query", &NNDescent::query, py::arg("query_vec"), py::arg("k"), py::arg("epsilon") = 0.1f);

  py::add_ostream_redirect(m, "ostream_redirect");
}
