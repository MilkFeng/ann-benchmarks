#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <print>
#include <queue>
#include <random>
#include <ranges>
#include <utility>
#include <vector>

namespace py = pybind11;

struct Neighbor {
  std::uint32_t id;
  std::float_t dist;
};

class NNDescent {
 public:
  NNDescent(std::string metric, const std::int32_t n_neighbors, const std::float_t pruning_degree_multiplier,
                 const std::float_t pruning_prob, const std::int32_t leaf_size)
      : metric_(std::move(metric)),
        n_neighbors_(n_neighbors),
        pruning_degree_multiplier_(pruning_degree_multiplier),
        pruning_prob_(pruning_prob),
        leaf_size_(leaf_size) {
    if (metric_ == "euclidean") {
      normalize_ = false;
    } else if (metric_ == "angular" || metric_ == "cosine") {
      normalize_ = true;
    } else {
      throw std::runtime_error("Only 'euclidean' and 'angular'('cosine') metrics are supported.");
    }
  }

  void fit(const py::array_t<std::float_t, py::array::c_style | py::array::forcecast>& input) {
    const auto buf = input.request();
    assert(buf.ndim == 2);

    num_elements_ = buf.shape[0];
    dim_ = buf.shape[1];

    data_.resize(num_elements_ * dim_);
    std::memcpy(data_.data(), buf.ptr, sizeof(std::float_t) * num_elements_ * dim_);

    if (normalize_) {
      std::println("Normalizing dataset for angular metric...");
      for (const auto i : std::views::iota(0u, num_elements_)) {
        normalize_vector(get_vector(i));
      }
    }

    graph_.resize(num_elements_);

    std::println("Initializing random graph...");
    initialize_graph();

    std::println("Starting NNDescent iterations...");
    build_graph();

    std::println("Adding reverse edges...");
    add_reverse_edges();

    std::println("Pruning graph with RNG rules...");
    prune_graph();

    visited_tags_.resize(num_elements_, 0);
    current_query_tag_ = 0;

    std::println("Graph Index build complete.");
  }

  std::vector<std::uint32_t> query(py::array_t<std::float_t, py::array::c_style | py::array::forcecast> query_vec,
                                   const std::uint32_t k, const std::float_t epsilon = 0.1f) {
    ++current_query_tag_;

    const auto buf = query_vec.request();
    assert(buf.ndim == 2);

    const std::size_t num_query = buf.shape[0];
    assert(num_query == 1);

    const std::size_t dim = buf.shape[1];
    assert(dim == dim_);

    const std::float_t* query_raw_ptr = static_cast<const std::float_t*>(buf.ptr);
    const std::float_t* query_ptr = query_raw_ptr;

    std::vector<std::float_t> query_storage;
    if (normalize_) {
      query_storage = {query_raw_ptr, query_raw_ptr + dim};
      normalize_vector(query_storage.data());
      query_ptr = query_storage.data();
    }

    const std::size_t ef = std::max(k, static_cast<std::uint32_t>(k * (1.0f + epsilon)));
    const std::uint32_t start_node = 0;

    using Candidate = std::pair<std::float_t, std::uint32_t>;
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> candidates = {};
    std::priority_queue<Candidate> top_candidates = {};

    float dist_start = compute_dist(query_ptr, get_vector(start_node));
    candidates.push({dist_start, start_node});
    top_candidates.push({dist_start, start_node});

    while (!candidates.empty()) {
      auto [dist, id] = candidates.top();
      candidates.pop();

      // if top_candidates is full and the new distance is greater than the worst distance in top_candidates, skip
      if (top_candidates.size() >= ef && dist > top_candidates.top().first) {
        break;
      }

      for (auto [neighbor_id, dist] : graph_[id]) {
        if (is_visited(neighbor_id)) {
          continue;
        }
        mark_visited(neighbor_id);

        float new_dist = compute_dist(query_ptr, get_vector(neighbor_id));

        if (top_candidates.size() < ef || new_dist < top_candidates.top().first) {
          candidates.push({new_dist, neighbor_id});
          top_candidates.push({new_dist, neighbor_id});

          // if top_candidates is full, pop the worst candidate
          if (top_candidates.size() > ef) {
            top_candidates.pop();
          }
        }
      }
    }

    // get top k candidates
    std::vector<std::uint32_t> top_k_candidates;
    while (!top_candidates.empty()) {
      auto [dist, id] = top_candidates.top();
      top_candidates.pop();

      bool is_duplicate = std::ranges::contains(top_k_candidates, id);
      if (!is_duplicate) {
        top_k_candidates.push_back(id);
      }
    }
    if (top_k_candidates.size() > k) {
      auto start_it = top_k_candidates.end() - k;
      std::vector<std::uint32_t> final_results(start_it, top_k_candidates.end());
      std::ranges::reverse(final_results);
      return final_results;
    } else {
      std::ranges::reverse(top_k_candidates);
      return top_k_candidates;
    }
  }

 private:
  const std::float_t* get_vector(std::uint32_t id) const { return &data_[id * dim_]; }

  std::float_t* get_vector(std::uint32_t id) { return &data_[id * dim_]; }

  void normalize_vector(std::float_t* vec) const {
    std::float_t sum_sq = 0.0f;
    for (const auto i : std::views::iota(0u, dim_)) {
      sum_sq += vec[i] * vec[i];
    }

    if (sum_sq > 1e-12f) {
      std::float_t norm_inv = 1.0f / std::sqrt(sum_sq);
      for (const auto i : std::views::iota(0u, dim_)) {
        vec[i] *= norm_inv;
      }
    }
  }

  std::float_t compute_dist(const std::float_t* a, const std::float_t* b) const {
    std::float_t dist = 0.0f;
    for (const auto i : std::views::iota(0u, dim_)) {
      std::float_t diff = a[i] - b[i];
      dist += diff * diff;
    }
    return dist;
  }

  std::float_t compute_dist(const std::uint32_t i, const std::uint32_t j) const {
    return compute_dist(get_vector(i), get_vector(j));
  }

  void unique_and_sort(std::vector<Neighbor>& neighbors) {
    std::ranges::sort(neighbors, std::less{}, &Neighbor::id);
    const auto unique_range = std::ranges::unique(neighbors, {}, &Neighbor::id);
    neighbors.erase(unique_range.begin(), unique_range.end());

    std::ranges::sort(neighbors, std::less{}, &Neighbor::dist);
  }

  bool is_visited(const std::uint32_t id) const { return visited_tags_[id] == current_query_tag_; }

  void mark_visited(const std::uint32_t id) { visited_tags_[id] = current_query_tag_; }

  void initialize_graph() {
    std::mt19937 rng(std::time(nullptr));
    std::uniform_int_distribution<std::uint32_t> dist(0, num_elements_ - 1);

    for (const auto& [id, neighbors] : std::views::enumerate(graph_)) {
      neighbors.reserve(n_neighbors_ * 2);

      for (const auto k : std::views::iota(0u, n_neighbors_)) {
        auto rand_id = dist(rng);
        while (rand_id == id) {
          rand_id = dist(rng);
        }
        std::float_t dist = compute_dist(id, rand_id);
        neighbors.emplace_back(rand_id, dist);
      }

      unique_and_sort(neighbors);
    }
  }

  void build_graph() {
    constexpr std::size_t max_iters = 30;
    for (const auto iter : std::views::iota(0u, max_iters)) {
      std::println("Starting iteration {}/{}...", iter + 1, max_iters);

      std::size_t updates = 0;
      for (const auto& [id, neighbors] : std::views::enumerate(graph_)) {
        std::vector<std::uint32_t> candidates = {};
        candidates.reserve(n_neighbors_ * n_neighbors_);

        // collect candidates
        for (const auto& n : neighbors) {
          candidates.push_back(n.id);
          for (const auto& nn : graph_[n.id]) {
            if (nn.id != id) {
              candidates.push_back(nn.id);
            }
          }
        }

        // sort and remove duplicates
        std::ranges::sort(candidates);
        const auto unique_range = std::ranges::unique(candidates);
        candidates.erase(unique_range.begin(), unique_range.end());

        bool changed = false;
        for (const auto candidate_id : candidates) {
          std::float_t dist = compute_dist(id, candidate_id);
          if (neighbors.size() < n_neighbors_ || dist < neighbors.back().dist) {
            const bool exists = std::ranges::contains(neighbors, candidate_id, &Neighbor::id);

            if (!exists) {
              // insert the new neighbor at the correct position to maintain sorted order
              auto insert_it = std::ranges::lower_bound(neighbors, dist, std::less{}, &Neighbor::dist);
              neighbors.insert(insert_it, {candidate_id, dist});

              // truncate the neighbors vector if it exceeds the maximum number of neighbors
              if (neighbors.size() > n_neighbors_) {
                neighbors.pop_back();
              }

              updates++;
            }
          }
        }
      }

      std::println("Iteration {}/{}: {} updates.", iter + 1, max_iters, updates);
      if (updates < num_elements_ * 0.01) {
        break;
      }
    }
  }

  void add_reverse_edges() {
    for (const auto& [id, neighbors] : std::views::enumerate(graph_)) {
      for (auto& neighbor : neighbors) {
        graph_[neighbor.id].emplace_back(static_cast<std::uint32_t>(id), neighbor.dist);
      }
    }

    for (auto& neighbors : graph_) {
      unique_and_sort(neighbors);
    }
  }

  void prune_graph() {
    // RNG rule:
    // dis(current, existing) < dis(current, candidate) && dis(candidate, existing) < dis(current, candidate) ->
    // candidate is shadowed by existing neighbor, keep it with probability

    const std::size_t max_degree = n_neighbors_ * pruning_degree_multiplier_;

    std::mt19937 rng(std::time(nullptr));
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

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
          std::float_t dist = compute_dist(existing.id, candidate.id);
          if (dist < candidate.dist) {
            // keep it with probability
            if (prob_dist(rng) < pruning_prob_) {
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

  std::uint32_t n_neighbors_ = {};
  std::float_t pruning_degree_multiplier_ = {};
  std::float_t pruning_prob_ = {};
  std::uint32_t leaf_size_ = {};

  std::size_t num_elements_ = {};
  std::size_t dim_ = {};

  std::vector<std::float_t> data_ = {};
  std::vector<std::vector<Neighbor>> graph_ = {};

  // do not use std::vector<bool> to record visit status
  std::vector<std::size_t> visited_tags_ = {};
  std::size_t current_query_tag_ = {};
};

PYBIND11_MODULE(nndescent_m, m) {
  py::class_<NNDescent>(m, "NNDescent")
      .def(py::init<std::string, std::int32_t, std::float_t, std::float_t, std::int32_t>(), py::arg("metric"),
           py::arg("n_neighbors"), py::arg("pruning_degree_multiplier"), py::arg("pruning_prob"), py::arg("leaf_size"))
      .def("fit", &NNDescent::fit, py::arg("input"))
      .def("query", &NNDescent::query, py::arg("query_vec"), py::arg("k"), py::arg("epsilon") = 0.1f);

  py::add_ostream_redirect(m, "ostream_redirect");
}
