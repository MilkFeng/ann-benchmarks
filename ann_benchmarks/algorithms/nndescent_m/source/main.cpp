#include <immintrin.h>
#include <omp.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <print>
#include <queue>
#include <random>
#include <ranges>
#include <utility>
#include <vector>

namespace py = pybind11;

using py_float_array_t = py::array_t<float, py::array::c_style | py::array::forcecast>;

template <typename T, const size_t Alignment = 32>
struct AlignedAllocator {
  using value_type = T;

  AlignedAllocator() noexcept = default;

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  T* allocate(size_t n) {
    if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::bad_alloc();
    }

    void* ptr = nullptr;

#if defined(_MSC_VER) || defined(__MINGW32__)
    ptr = _aligned_malloc(sizeof(T) * n, Alignment);
#else
    if (posix_memalign(&ptr, Alignment, sizeof(T) * n) != 0) {
      ptr = nullptr;
    }
#endif
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p, size_t) noexcept {
#if defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(p);
#else
    free(p);
#endif
  }
};

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
        leaf_size_(leaf_size) {
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
    dim_padded_ = (dim_ + 7) & ~7;
    dim_aligned_ = dim_ - (dim_ & 7);
    const float* raw_data_ptr = static_cast<const float*>(buf.ptr);

    data_.resize(num_elements_ * dim_padded_);
    for (size_t i = 0; i < num_elements_; ++i) {
      float* row_ptr = &data_[i * dim_padded_];
      std::memcpy(row_ptr, raw_data_ptr + i * dim_, sizeof(float) * dim_);
    }

    if (normalize_) {
      std::println("Normalizing dataset for angular metric...");
      for (const auto i : std::views::iota(0u, num_elements_)) {
        normalize_vec(get_vec(i));
      }
    }

    graph_.resize(num_elements_);
    node_locks_ = std::vector<std::mutex>(num_locks_);
    node_locks_2_ = std::vector<std::mutex>(num_locks_);

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
    std::vector<float, AlignedAllocator<float>> query_storage = {query_raw_ptr, query_raw_ptr + dim_};

    if (normalize_) {
      normalize_vec(query_storage.data());
    }

    const float* query_ptr = query_storage.data();

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

      if (top_candidates.size() > k) {
        top_candidates.pop();
      }
    }

    while (!candidates.empty()) {
      auto [curr_id, curr_dist] = candidates.top();
      candidates.pop();

      float search_radius = top_candidates.top().dist * (1.0f + epsilon);
      if (top_candidates.size() >= k && curr_dist > search_radius) {
        break;
      }

      for (auto [id, _1, _2] : graph_[curr_id]) {
        if (is_visited(id)) {
          continue;
        }
        mark_visited(id);

        const auto dist = compute_dist(query_ptr, get_vec(id));

        if (top_candidates.size() < k || dist < top_candidates.top().dist) {
          top_candidates.push({id, dist});

          if (top_candidates.size() > k) {
            top_candidates.pop();
          }

          search_radius = top_candidates.top().dist * (1.0f + epsilon);
        }

        if (dist < search_radius) {
          candidates.push({id, dist});
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
  const float* get_vec(const uint32_t id) const { return &data_[id * dim_padded_]; }

  float* get_vec(const uint32_t id) { return &data_[id * dim_padded_]; }

  void normalize_vec(float* vec) const {
    float sum_sq = compute_squared_norm(vec);

    if (sum_sq > 1e-12f) {
      float norm_inv = 1.0f / std::sqrt(sum_sq);

      if (dim_aligned_ > 0) {
        __m256 inv_vec = _mm256_set1_ps(norm_inv);

        for (size_t i = 0; i < dim_aligned_; i += 8) {
          __m256 v = _mm256_load_ps(&vec[i]);
          v = _mm256_mul_ps(v, inv_vec);
          _mm256_store_ps(&vec[i], v);
        }
      }

      for (const auto i : std::views::iota(dim_aligned_, dim_)) {
        vec[i] *= norm_inv;
      }
    }
  }

  float compute_squared_norm(const float* vec) const {
    float sum_sq = 0.0f;

    if (dim_aligned_ > 0) {
      __m256 sum_vec = _mm256_setzero_ps();
      for (size_t i = 0; i < dim_aligned_; i += 8) {
        __m256 v = _mm256_load_ps(&vec[i]);
        sum_vec = _mm256_fmadd_ps(v, v, sum_vec);
      }

      __m128 vlow = _mm256_castps256_ps128(sum_vec);
      __m128 vhigh = _mm256_extractf128_ps(sum_vec, 1);
      __m128 vsum = _mm_add_ps(vlow, vhigh);

      vsum = _mm_hadd_ps(vsum, vsum);
      vsum = _mm_hadd_ps(vsum, vsum);

      sum_sq += _mm_cvtss_f32(vsum);
    }

    for (const auto i : std::views::iota(dim_aligned_, dim_)) {
      sum_sq += vec[i] * vec[i];
    }

    return sum_sq;
  }

  float compute_dist(const float* a, const float* b) const {
    float dist = 0.0f;

    if (dim_aligned_ > 0) {
      __m256 sum_vec = _mm256_setzero_ps();
      for (size_t i = 0; i < dim_aligned_; i += 8) {
        __m256 vec_a = _mm256_load_ps(&a[i]);
        __m256 vec_b = _mm256_load_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(vec_a, vec_b);
        sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
      }

      __m128 vlow = _mm256_castps256_ps128(sum_vec);
      __m128 vhigh = _mm256_extractf128_ps(sum_vec, 1);
      __m128 vsum = _mm_add_ps(vlow, vhigh);

      vsum = _mm_hadd_ps(vsum, vsum);
      vsum = _mm_hadd_ps(vsum, vsum);

      dist += _mm_cvtss_f32(vsum);
    }

    for (const auto i : std::views::iota(dim_aligned_, dim_)) {
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
  static void insert_with_order(std::vector<T>& items, const T& new_item) {
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

  size_t get_lock_index(const uint32_t id) const { return id & (node_locks_.size() - 1); }

  std::lock_guard<std::mutex> lock_mutex(const uint32_t id, const uint32_t batch = 0) {
    assert(batch == 0 || batch == 1);
    if (batch == 0) {
      return std::lock_guard<std::mutex>(node_locks_[get_lock_index(id)]);
    } else {
      return std::lock_guard<std::mutex>(node_locks_2_[get_lock_index(id)]);
    }
  }

  void connect_safe(std::vector<std::vector<Neighbor>>& graph, const uint32_t u, const uint32_t v, const float dist) {
    std::lock_guard<std::mutex> lock = lock_mutex(u);
    graph[u].emplace_back(v, dist, true);
  }

  void connect_safe(const uint32_t u, const uint32_t v, const float dist) {
    std::lock_guard<std::mutex> lock = lock_mutex(u);
    graph_[u].emplace_back(v, dist, true);
  }

  bool contains_safe(const uint32_t u, const uint32_t v) {
    std::lock_guard<std::mutex> lock = lock_mutex(u);
    return std::ranges::contains(graph_[u], v, &Neighbor::id);
  }

  bool contains_safe(const std::vector<std::vector<Neighbor>>& graph, const uint32_t u, const uint32_t v) {
    std::lock_guard<std::mutex> lock = lock_mutex(u);
    return std::ranges::contains(graph[u], v, &Neighbor::id);
  }

  bool contains_safe(const std::vector<std::vector<Candidate>>& candidates, const uint32_t u, const uint32_t v,
                     const uint32_t batch = 0) {
    std::lock_guard<std::mutex> lock = lock_mutex(u);
    return std::ranges::contains(candidates[u], v, &Candidate::id);
  }

  void initialize_graph() {
    build_rp_trees();

#pragma omp parallel for schedule(static)
    for (int32_t i = 0; i < num_elements_; ++i) {
      sort_and_unique(graph_[i]);
    }

    randomly_fill_graph();
  }

  void build_rp_trees() {
    const size_t dynamic_trees = 5 + std::round(std::sqrt(num_elements_) / 20.0);
    const size_t n_trees = std::min(64uz, std::max(5uz, dynamic_trees));

    roots_.clear();
    roots_.resize(n_trees);

    std::println("Building {} RP Trees...", n_trees);

    const size_t seed = std::time(nullptr);
#pragma omp parallel
    {
      std::mt19937 rng(seed + omp_get_thread_num());

      std::vector<uint32_t> indices(num_elements_);
      std::ranges::iota(indices, 0u);

#pragma omp for schedule(dynamic)
      for (int32_t i = 0; i < n_trees; ++i) {
        std::ranges::shuffle(indices, rng);
        roots_[i] = build_rp_tree(indices, rng);
      }
    }
  }

  void randomly_fill_graph() {
    const size_t seed = std::time(nullptr);
    std::uniform_int_distribution<uint32_t> dist_gen{0u, static_cast<uint32_t>(num_elements_ - 1)};

    const size_t target_size = std::min(static_cast<size_t>(n_neighbors_), num_elements_ - 1);

#pragma omp parallel
    {
      std::mt19937 rng(seed + omp_get_thread_num());

#pragma omp for schedule(dynamic, 128)
      for (int32_t id = 0; id < num_elements_; ++id) {
        auto& neighbors = graph_[id];

        if (neighbors.capacity() < target_size) {
          neighbors.reserve(target_size);
        }

        while (neighbors.size() < target_size) {
          auto rand_id = dist_gen(rng);
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
  }

  std::unique_ptr<RPTreeNode> build_rp_tree(const std::span<uint32_t> indices, std::mt19937 rng) {
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

          connect_safe(u, v, d);
          connect_safe(v, u, d);
        }
      }
      return node;
    }

    // split into two partitions:
    // left partition: closer to idx_a
    // right partition: closer to idx_b
    std::uniform_int_distribution<size_t> idx_dist{0, indices.size() - 1};
    uint32_t idx_a = indices[idx_dist(rng)];
    uint32_t idx_b = indices[idx_dist(rng)];

    uint32_t attempts = 0;
    while (idx_a == idx_b && attempts < 10) {
      idx_b = indices[idx_dist(rng)];
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
      std::ranges::shuffle(indices, rng);
      left_count = indices.size() / 2;
    }

    node->left = build_rp_tree(indices.first(left_count), rng);
    node->right = build_rp_tree(indices.subspan(left_count), rng);

    return node;
  }

  std::vector<uint32_t> query_rp_trees(const float* query_vec) {
    std::vector<uint32_t> candidate_indices = {};
#pragma omp parallel
    {
      std::vector<uint32_t> local_candidates = {};

#pragma omp for schedule(static)
      for (int32_t i = 0; i < roots_.size(); ++i) {
        RPTreeNode* current_node = roots_[i].get();

        while (!current_node->is_leaf) {
          float dist_a = compute_dist(query_vec, get_vec(current_node->idx_a));
          float dist_b = compute_dist(query_vec, get_vec(current_node->idx_b));

          if (dist_a < dist_b) {
            current_node = current_node->left.get();
          } else {
            current_node = current_node->right.get();
          }
        }

        local_candidates.insert(local_candidates.end(), current_node->leaf_indices.begin(),
                                current_node->leaf_indices.end());
      }

      if (!local_candidates.empty()) {
#pragma omp critical
        {
          candidate_indices.insert(candidate_indices.end(), local_candidates.begin(), local_candidates.end());
        }
      }
    }

    // remove duplicates
    sort_and_unique(candidate_indices);
    return candidate_indices;
  }

  void build_graph() {
    constexpr size_t max_iters = 30;
    const size_t min_updates_threshold = std::max<size_t>(100uz, num_elements_ * n_neighbors_ * 0.001);

    const auto try_connect_mono_safe = [&](const uint32_t u, const uint32_t v, const float dist) {
      std::lock_guard<std::mutex> lock_guard = lock_mutex(u);

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

    const auto try_connect_safe = [&](const uint32_t u, const uint32_t v) {
      if (u == v) {
        return false;
      }

      const auto u_has_v = contains_safe(u, v);
      const auto v_has_u = contains_safe(v, u);
      if (u_has_v && v_has_u) {
        return false;
      }

      const auto dist = compute_dist(u, v);
      bool changed = false;

      if (!u_has_v) {
        changed |= try_connect_mono_safe(u, v, dist);
      }

      if (!v_has_u) {
        changed |= try_connect_mono_safe(v, u, dist);
      }

      return changed;
    };

    for (const auto iter : std::views::iota(0u, max_iters)) {
      std::println("Starting iteration {}/{}...", iter + 1, max_iters);

      sample_candidates();

      size_t updates = 0;

#pragma omp parallel for schedule(dynamic) reduction(+ : updates)
      for (int32_t u = 0; u < num_elements_; ++u) {
        const auto& new_candidates = new_candidates_sketch_[u];
        const auto& old_candidates = old_candidates_sketch_[u];

        // new-new
        for (const auto i : std::views::iota(0u, new_candidates.size())) {
          for (const auto j : std::views::iota(i + 1u, new_candidates.size())) {
            if (try_connect_safe(new_candidates[i].id, new_candidates[j].id)) {
              ++updates;
            }
          }
        }

        // new-old
        for (const auto& new_candidate : new_candidates) {
          for (const auto& old_candidate : old_candidates) {
            if (try_connect_safe(new_candidate.id, old_candidate.id)) {
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
#pragma omp parallel for schedule(static, 256)
    for (int32_t i = 0; i < num_elements_; ++i) {
      new_candidates_sketch_[i].clear();
      old_candidates_sketch_[i].clear();
    }

    const auto checked_push = [&](std::vector<std::vector<Candidate>>& candidates, const uint32_t u, const uint32_t v,
                                  const float priority, const uint32_t batch) -> bool {
      std::lock_guard<std::mutex> lock_guard = lock_mutex(u, batch);

      if (candidates[u].size() >= max_candidates_ && priority >= candidates[u].back().dist) {
        return false;
      }

      const bool exists = std::ranges::contains(candidates[u], v, &Candidate::id);
      if (exists) {
        return false;
      }

      insert_with_order(candidates[u], {v, priority});

      if (candidates[u].size() > max_candidates_) {
        candidates[u].pop_back();
      }

      return true;
    };

    const size_t seed = std::time(nullptr);
    std::uniform_real_distribution<float> priority_dist{0.0f, 1.0f};
#pragma omp parallel
    {
      std::mt19937 rng(seed + omp_get_thread_num());

#pragma omp for schedule(dynamic)
      for (int32_t u = 0; u < num_elements_; ++u) {
        auto& neighbors = graph_[u];
        for (const auto [v, dist, is_new] : neighbors) {
          const auto priority = priority_dist(rng);
          if (is_new) {
            checked_push(new_candidates_sketch_, u, v, priority, 0);
            checked_push(new_candidates_sketch_, v, u, priority, 0);
          } else {
            checked_push(old_candidates_sketch_, u, v, priority, 1);
            checked_push(old_candidates_sketch_, v, u, priority, 1);
          }
        }
      }
    }

    // mark sampled candidates as old

#pragma omp parallel for schedule(dynamic)
    for (int32_t u = 0; u < num_elements_; ++u) {
      auto& neighbors = graph_[u];
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
#pragma omp parallel for schedule(dynamic)
    for (int32_t u = 0; u < num_elements_; ++u) {
      for (const auto& neighbor : graph_[u]) {
        connect_safe(reverse_graph, neighbor.id, u, neighbor.dist);
      }
    }

#pragma omp parallel for
    for (int32_t id = 0; id < num_elements_; ++id) {
      auto& neighbors = graph_[id];
      auto& new_neighbors = reverse_graph[id];

      neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());

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

    const size_t seed = std::time(nullptr);
    std::uniform_real_distribution<float> prob_dist{0.0f, 1.0f};

#pragma omp parallel
    {
      std::mt19937 rng(seed + omp_get_thread_num());

#pragma omp for schedule(dynamic)
      for (int32_t i = 0; i < num_elements_; ++i) {
        auto& neighbors = graph_[i];

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
  size_t dim_padded_ = {};
  size_t dim_aligned_ = {};

  std::vector<float, AlignedAllocator<float>> data_ = {};

  std::vector<std::unique_ptr<RPTreeNode>> roots_ = {};

  // build sketch
  std::vector<std::vector<Candidate>> new_candidates_sketch_ = {};
  std::vector<std::vector<Candidate>> old_candidates_sketch_ = {};

  std::vector<std::vector<Neighbor>> graph_ = {};
  std::vector<std::mutex> node_locks_ = {};
  std::vector<std::mutex> node_locks_2_ = {};

  // do not use std::vector<bool> to record visit status
  std::vector<uint32_t> visited_tags_ = {};
  uint32_t current_query_tag_ = {};

 private:
  static constexpr size_t num_locks_ = 4096;
};

PYBIND11_MODULE(nndescent_m, m) {
  py::class_<NNDescent>(m, "NNDescent")
      .def(py::init<std::string, size_t, float, float, size_t>(), py::arg("metric"), py::arg("n_neighbors"),
           py::arg("pruning_degree_multiplier"), py::arg("pruning_prob"), py::arg("leaf_size"))
      .def("fit", &NNDescent::fit, py::arg("input"))
      .def("query", &NNDescent::query, py::arg("query_vec"), py::arg("k"), py::arg("epsilon") = 0.1f);

  py::add_ostream_redirect(m, "ostream_redirect");
}
