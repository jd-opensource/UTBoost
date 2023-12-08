/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#include "serial_tree_learner.h"

namespace UTBoost {

void SerialTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  train_data_ = train_data;
  num_data_ = train_data_->GetNumSamples();
  num_features_ = train_data_->GetNumFeatures();
  num_treats_ = train_data->GetDistinctTreatNum();
  // Get the max size of pool
  int max_cache_size = config_->num_leaves;
  // at least need 2 leaves
  max_cache_size = std::max(2, max_cache_size);
  max_cache_size = std::min(max_cache_size, config_->num_leaves);
  // push split information for all leaves
  best_split_per_leaf_.resize(max_cache_size);
  // initialize splits for leaf
  smaller_leaf_splits_.reset(new LeafSplit(train_data_->GetNumSamples(), train_data->GetDistinctTreatNum()));
  larger_leaf_splits_.reset(new LeafSplit(train_data_->GetNumSamples(), train_data->GetDistinctTreatNum()));
  // initialize data partition
  data_partition_.reset(new DataPartition(num_data_, config_->num_leaves));
  col_sampler_.SetTrainingData(train_data_);
  // is_feature_used_.resize(num_features_);
  histogram_pool_.DynamicChangeSize(train_data_, max_cache_size, num_treats_,config_);
  Log::Info("Number of data points in the train set: %d, number of used features: %d", num_data_, num_features_);
}

Tree *SerialTreeLearner::Train(const score_t *gradients, const score_t *hessians, bool is_first_tree, const SplitCriteria* split_criteria) {
  gradients_ = gradients;
  hessians_ = hessians;
  int num_threads = OMP_GET_NUM_THREADS();

  // some initial works before training
  BeforeTrain();

  auto tree = std::unique_ptr<Tree>(new Tree(config_->num_leaves, num_treats_));
  auto tree_ptr = tree.get();

  // root leaf
  int left_leaf = 0;
  int cur_depth = 1;
  // only root leaf can be splitted on first time
  int right_leaf = -1;

  for (int split = 0; split < config_->num_leaves - 1; ++split) {
    // some initial works before finding best split
    if (BeforeFindBestSplit(tree_ptr, left_leaf, right_leaf)) {
      // find best threshold for every feature
      FindBestSplits(tree_ptr, nullptr, split_criteria);
      // FindBestSplits(tree_ptr);
    }
    // Get a leaf with max split gain
    int best_leaf = static_cast<int>(ArgMax(best_split_per_leaf_));
    // Get split information for best leaf
    const SplitInfo& best_leaf_SplitInfo = best_split_per_leaf_[best_leaf];
    // cannot split, quit
    if (best_leaf_SplitInfo.gain <= 0.0) {
      break;
    }
    // split tree with best leaf
    Split(tree_ptr, best_leaf, &left_leaf, &right_leaf);
    cur_depth = std::max(cur_depth, tree->leaf_depth(left_leaf));
  }

  Log::Debug("Trained a tree with leaves = %d and depth = %d", tree->num_leaves(), cur_depth);
  return tree.release();
}


void SerialTreeLearner::Split(Tree* tree, int best_leaf, int* left_leaf, int* right_leaf) {

  const SplitInfo& best_split_info = best_split_per_leaf_[best_leaf];
  const int feature_index = best_split_info.feature;

  *left_leaf = best_leaf;

  auto threshold_double = train_data_->RealThreshold(feature_index, best_split_info.threshold);
  *right_leaf = tree->Split(best_leaf,
                            feature_index,
                            best_split_info.threshold, threshold_double,
                            best_split_info.left_output.data(),
                            best_split_info.right_output.data(),
                            best_split_info.left_count,
                            best_split_info.right_count,
                            0.0, 0.0,
                            static_cast<float>(best_split_info.gain),
                            !train_data_->GetFMapper(feature_index)->use_missing(),
                            best_split_info.default_left);
  data_partition_->Split(best_leaf, train_data_, feature_index,
                         best_split_info.threshold, best_split_info.default_left, *right_leaf);

  if ((best_split_info.left_count) < (best_split_info.right_count)) {
    smaller_leaf_splits_->Init(
        *left_leaf, data_partition_.get(),
        best_split_info.left_wgradients_sum.data(), best_split_info.left_whessians_sum.data(),
        best_split_info.left_num_data.data(), best_split_info.left_label_sum.data()
        );
    larger_leaf_splits_->Init(
        *right_leaf, data_partition_.get(),
        best_split_info.right_wgradients_sum.data(), best_split_info.right_whessians_sum.data(),
        best_split_info.right_num_data.data(), best_split_info.right_label_sum.data()
    );
  }
  else {
    larger_leaf_splits_->Init(
        *left_leaf, data_partition_.get(),
        best_split_info.left_wgradients_sum.data(), best_split_info.left_whessians_sum.data(),
        best_split_info.left_num_data.data(), best_split_info.left_label_sum.data()
    );
    smaller_leaf_splits_->Init(
        *right_leaf, data_partition_.get(),
        best_split_info.right_wgradients_sum.data(), best_split_info.right_whessians_sum.data(),
        best_split_info.right_num_data.data(), best_split_info.right_label_sum.data()
    );

  }

}

void SerialTreeLearner::BeforeTrain() {
  col_sampler_.ResetByTree();
  is_feature_used_ = col_sampler_.is_feature_used_bytree();
  // initialize data partition
  data_partition_->Init();
  // reset the splits for leaves
  for (int i = 0; i < config_->num_leaves; ++i) {
    best_split_per_leaf_[i].Reset();
  }
  // Sumup for root
  if (data_partition_->leaf_count(0) == num_data_) {
    // use all data
    smaller_leaf_splits_->Init(gradients_, hessians_, train_data_->GetMetaInfo().GetLabel(), train_data_->GetMetaInfo().GetTreatment());
  } else {
    // use bagging, only use part of data
    smaller_leaf_splits_->Init(0, data_partition_.get(), gradients_, hessians_, train_data_->GetMetaInfo().GetLabel(), train_data_->GetMetaInfo().GetTreatment());
  }
  larger_leaf_splits_->Init();
}

bool SerialTreeLearner::BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) {
  // check depth of current leaf
  if (config_->max_depth > 0) {
    // only need to check left leaf, since right leaf is in same level of left leaf
    if (tree->leaf_depth(left_leaf) >= config_->max_depth) {
      best_split_per_leaf_[left_leaf].gain = kMinScore;
      if (right_leaf >= 0) {
        best_split_per_leaf_[right_leaf].gain = kMinScore;
      }
      return false;
    }
  }
  data_size_t num_data_in_left_child = GetGlobalDataCountInLeaf(left_leaf);
  data_size_t num_data_in_right_child = GetGlobalDataCountInLeaf(right_leaf);

  if (num_data_in_right_child < static_cast<data_size_t>(config_->min_data_in_leaf * 2)
      && num_data_in_left_child < static_cast<data_size_t>(config_->min_data_in_leaf * 2)) {
    best_split_per_leaf_[left_leaf].gain = kMinScore;
    if (right_leaf >= 0) {
      best_split_per_leaf_[right_leaf].gain = kMinScore;
    }
    return false;
  }
  parent_leaf_histogram_array_ = nullptr;
  // only have root
  if (right_leaf < 0) {
    histogram_pool_.Get(left_leaf, &smaller_leaf_histogram_array_);
    larger_leaf_histogram_array_ = nullptr;
  } else if (num_data_in_left_child < num_data_in_right_child) {
    if (histogram_pool_.Get(left_leaf, &larger_leaf_histogram_array_)) { parent_leaf_histogram_array_ = larger_leaf_histogram_array_; }
    histogram_pool_.Move(left_leaf, right_leaf);
    histogram_pool_.Get(left_leaf, &smaller_leaf_histogram_array_);
  } else {
    if (histogram_pool_.Get(left_leaf, &larger_leaf_histogram_array_)) { parent_leaf_histogram_array_ = larger_leaf_histogram_array_; }
    histogram_pool_.Get(right_leaf, &smaller_leaf_histogram_array_);
  }
  return true;
}

void SerialTreeLearner::FindBestSplits(const Tree *tree, const std::set<int> *force_features, const SplitCriteria* split_criteria) {
  std::vector<int8_t> is_feature_used(num_features_, 0);
#pragma omp parallel for schedule(static, 256) if (num_features_ >= 512)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!col_sampler_.is_feature_used_bytree()[feature_index]) continue;
    if (parent_leaf_histogram_array_ != nullptr
        && !parent_leaf_histogram_array_[feature_index].is_splittable()) {
      smaller_leaf_histogram_array_[feature_index].set_is_splittable(false);
      continue;
    }
    is_feature_used[feature_index] = 1;
  }
  bool use_subtract = parent_leaf_histogram_array_ != nullptr;

  ConstructHistograms(is_feature_used, use_subtract);
  FindBestSplitsFromHistograms(is_feature_used, use_subtract, split_criteria, tree);
}

void SerialTreeLearner::ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) {
  // construct smaller leaf
  // The data of each feature histogram is already pointing to vector.data in HistogramPool during the initialization of DynamicChangeSize.
  // Therefore, here we only need to take the first element and assign it continuously.
  BinEntry* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData();
  train_data_->ConstructHistograms(
      is_feature_used, smaller_leaf_splits_->data_indices(),
      smaller_leaf_splits_->num_data_in_leaf(), smaller_leaf_splits_->leaf_index(),gradients_, hessians_
      , ptr_smaller_leaf_hist_data);
}

void SerialTreeLearner::FindBestSplitsFromHistograms(const std::vector<int8_t> &is_feature_used,
                                                     bool use_subtract,
                                                     const SplitCriteria* split_criteria,
                                                     const Tree *) {
  int num_threads = OMP_GET_NUM_THREADS();
  std::vector<SplitInfo> smaller_best(num_threads);
  std::vector<SplitInfo> larger_best(num_threads);
  std::vector<int8_t> smaller_node_used_features = col_sampler_.GetByNode(0);
  std::vector<int8_t> larger_node_used_features;
  if (larger_leaf_splits_->leaf_index() >= 0) {
    larger_node_used_features = col_sampler_.GetByNode(0);
  }
// find splits
#pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!is_feature_used[feature_index]) {
      continue;
    }
    const int tid = OMP_GET_THREAD_NUM();
    SplitInfo smaller_split;
    smaller_leaf_histogram_array_[feature_index].FindBestThreshold(num_treats_,
        smaller_leaf_splits_->sum_gradients(), smaller_leaf_splits_->sum_hessians(),
        smaller_leaf_splits_->sum_wgradients(), smaller_leaf_splits_->sum_whessians(),
        smaller_leaf_splits_->num_data(),
        smaller_leaf_splits_->sum_labels(),
        split_criteria,
        &smaller_split
        );
    smaller_split.feature = feature_index;
    if (smaller_split > smaller_best[tid]) {
      smaller_best[tid] = smaller_split;
    }

    // only has root leaf
    if (larger_leaf_splits_ == nullptr || larger_leaf_splits_->leaf_index() < 0) {
      continue;
    }

    if (use_subtract) {
      larger_leaf_histogram_array_[feature_index].Subtract(smaller_leaf_histogram_array_[feature_index]);
    }

    SplitInfo larger_split;
    larger_leaf_histogram_array_[feature_index].FindBestThreshold(num_treats_,
        larger_leaf_splits_->sum_gradients(), larger_leaf_splits_->sum_hessians(),
        larger_leaf_splits_->sum_wgradients(), larger_leaf_splits_->sum_whessians(),
        larger_leaf_splits_->num_data(),
        larger_leaf_splits_->sum_labels(),
        split_criteria,
        &larger_split
    );
    larger_split.feature = feature_index;

    if (larger_split > larger_best[tid]) {
      larger_best[tid] = larger_split;
    }

  }
  auto smaller_best_idx = ArgMax(smaller_best);
  int leaf = smaller_leaf_splits_->leaf_index();
  best_split_per_leaf_[leaf] = smaller_best[smaller_best_idx];

  if (larger_leaf_splits_ != nullptr && larger_leaf_splits_->leaf_index() >= 0) {
    leaf = larger_leaf_splits_->leaf_index();
    auto larger_best_idx = ArgMax(larger_best);
    best_split_per_leaf_[leaf] = larger_best[larger_best_idx];
  }
}

void SerialTreeLearner::RenewTreeOutputByIndices(Tree* tree, const SplitCriteria* split_criteria, const data_size_t* bag_indices, data_size_t bag_cnt, const score_t* gradients, const score_t* hessians) const {
  if (bag_cnt < 1) return;
  std::vector<int> leaf_idx(bag_cnt, 0);
  const treatment_t* treats = train_data_->GetMetaInfo().GetTreatment();
  const label_t* labels = train_data_->GetMetaInfo().GetLabel();
  tree->GetLeafIndex(train_data_, bag_indices, bag_cnt, leaf_idx.data());
  int num_threads = OMP_GET_NUM_THREADS();
  std::vector<BinEntry> buffers(num_threads * tree->num_leaves(), BinEntry(train_data_->GetDistinctTreatNum()));
  int block = static_cast<int>(bag_cnt / num_threads);
  if (hessians == nullptr) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_threads; ++i) {
      int end = (i + 1) * block > bag_cnt ? bag_cnt : (i + 1) * block;
      BinEntry* buffer = buffers.data() + i * tree->num_leaves();
      for (int j = i * block; j < end; ++j) {
        data_size_t idx = bag_indices[j];
        buffer[leaf_idx[j]].PushData(labels[idx], treats[idx], gradients[idx], 1);
      }
    }
  } else {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_threads; ++i) {
      int end = (i + 1) * block > bag_cnt ? bag_cnt : (i + 1) * block;
      BinEntry* buffer = buffers.data() + i * tree->num_leaves();
      for (int j = i * block; j < end; ++j) {
        data_size_t idx = bag_indices[j];
        buffer[leaf_idx[j]].PushData(labels[idx], treats[idx], gradients[idx], hessians[idx]);
      }
    }
  }

  for (int i = 0; i < tree->num_leaves(); ++i) {
    BinEntry* entry = buffers.data() + i;
    for (int j = 1; j < num_threads; ++j) {  // reduce
      entry->Add(*(entry + tree->num_leaves() * j));
    }
    if ((entry->num_total_data_) < config_->min_data_in_leaf) {
      continue;
    }
    tree->SetLeafOutput(i, split_criteria->CalculateLeafOutput(entry).data(), num_treats_);
  }

}

SerialTreeLearner::SerialTreeLearner(const Config* config)
    : config_(config), col_sampler_(config) {
}

SerialTreeLearner::~SerialTreeLearner() {}

}  // namespace UTBoost
