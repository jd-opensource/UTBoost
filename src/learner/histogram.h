/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#ifndef UTBOOST_INCLUDE_UTBOOST_HIST_H_
#define UTBOOST_INCLUDE_UTBOOST_HIST_H_

#include "UTBoost/dataset.h"
#include "UTBoost/bin.h"
#include "UTBoost/config.h"
#include "UTBoost/split_criteria.h"
#include "split_info.h"

#include <memory>

namespace UTBoost {

struct FeatureMeta {
  int num_bin;
  int use_missing;
  const Config* config;
};

class FeatureHistogram {
 public:
  void Init(BinEntry* data, const FeatureMeta* meta) {
    data_ = data;
    meta_ = meta;
  }

  BinEntry* RawData() {
    return data_;
  }

  void Subtract(const FeatureHistogram& other) {
    for (int i = 0; i < meta_->num_bin; ++i) {
      data_[i].Subtract(other.data_[i]);
    }
  }

  void FindBestThreshold(int num_treat, double sum_gradient, double sum_hessian, const double* sum_wgradients, const double* sum_whessians, const double* num_datas, const double* sum_labels, const SplitCriteria* split_criteria, SplitInfo* output) {
    output->default_left = true;
    output->gain = -std::numeric_limits<double>::infinity();
    splittable_ = false;
#define ARGUMENTS num_treat, sum_gradient, sum_hessian, sum_wgradients, sum_whessians, num_datas, sum_labels, split_criteria, output
    if (meta_->use_missing && meta_->num_bin > 2) {
      FindBestThresholdInner<true, true>(ARGUMENTS);
      FindBestThresholdInner<false, true>(ARGUMENTS);
    } else {
      FindBestThresholdInner<true, false>(ARGUMENTS);
      if (meta_->use_missing)
        output->default_left = false;
    }
  }

  template <bool REVERSE, bool NA_AS_MISSING>
  void FindBestThresholdInner(int num_treat, double sum_gradient, double sum_hessian, const double* sum_wgradients, const double* sum_whessians, const double* num_datas, const double* sum_labels, const SplitCriteria* split_criteria, SplitInfo* output) {

    BinEntry best_left(num_treat);
    BinEntry best_right(num_treat, sum_wgradients, sum_whessians, sum_labels, num_datas);
    double best_gain = -std::numeric_limits<double>::infinity();

    double root_gain = split_criteria->GetSplitGains(&best_right);

    int best_threshold = meta_->num_bin;

    if (REVERSE) {  // from right to left
      BinEntry left(num_treat, sum_wgradients, sum_whessians, sum_labels, num_datas);
      BinEntry right(num_treat);

      int t = meta_->num_bin - 1 - NA_AS_MISSING;
      int t_end = 1;

      for (; t >= t_end; --t) {
        right.Add(data_[t]);
        left.Subtract(data_[t]);
        if ( right.num_total_data_ < meta_->config->min_data_in_leaf) continue;
        if ( left.num_total_data_ < meta_->config->min_data_in_leaf) break;
        double current_gain = split_criteria->SplitScore(&left, &right, &best_right);

        // gain with split is worse than without split
        if (current_gain <= root_gain) {
          continue;
        }

        splittable_ = true;
        if (current_gain > best_gain) {
          best_left = left;
          // left is <= threshold, right is > threshold. so this is t-1
          best_threshold = static_cast<int>(t - 1);
          best_gain = current_gain;
        }
      }
    } else {  // from left to right
      BinEntry right(num_treat, sum_wgradients, sum_whessians, sum_labels, num_datas);
      BinEntry left(num_treat);

      int t = 0;
      int t_end = meta_->num_bin - 2;

      for (; t <= t_end; ++t) {
        right.Subtract(data_[t]);
        left.Add(data_[t]);
        if ( left.num_total_data_ < meta_->config->min_data_in_leaf) continue;
        if ( right.num_total_data_ < meta_->config->min_data_in_leaf) break;

        double current_gain = split_criteria->SplitScore(&left, &right, &best_right);

        // gain with split is worse than without split
        if (current_gain <= root_gain) {
          continue;
        }

        splittable_ = true;
        if (current_gain > best_gain) {
          best_left = left;
          best_threshold = static_cast<int>(t);
          best_gain = current_gain;
        }
      }
    }

    if (splittable_ && best_gain > output->gain) {
      best_right.Subtract(best_left);
      output->CopyFromEntry(&best_left, true);
      output->CopyFromEntry(&best_right, false);
      output->threshold = best_threshold;
      output->left_output = split_criteria->CalculateLeafOutput(&best_left);
      output->right_output = split_criteria->CalculateLeafOutput(&best_right);
      output->gain = best_gain - root_gain;
      output->default_left = REVERSE;
    }

  }
  /*! \brief True if this histogram can be splitted */
  bool is_splittable() const { return splittable_; }
  /*! \brief Set splittable to this histogram */
  void set_is_splittable(bool val) { splittable_ = val; }
 private:
  bool splittable_;
  const FeatureMeta* meta_;
  BinEntry* data_;
};

class HistogramPool {
 public:
  HistogramPool() {};

  void DynamicChangeSize(const Dataset* train_data, int cache_size, int num_treat, const Config* config) {
    if (feature_metas_.empty()) {
      int num_feature = train_data->GetNumFeatures();
      feature_metas_.resize(num_feature);
      for (int i = 0; i < num_feature; ++i) {
        feature_metas_[i].num_bin = train_data->GetFMapperNum(i);
        feature_metas_[i].use_missing = train_data->GetFMapper(i)->use_missing();
        feature_metas_[i].config = config;
      }
    }
    int num_total_bin = train_data->GetNumTotalBins();
    int old_cache_size = static_cast<int>(leaf_pool_.size());

    if (cache_size > old_cache_size) {
      leaf_pool_.resize(cache_size);
      data_.resize(cache_size);
    }

    for (int i = old_cache_size; i < cache_size; ++i) {
      leaf_pool_[i].reset(new FeatureHistogram[train_data->GetNumFeatures()]);
      data_[i].resize(num_total_bin, BinEntry(num_treat));
      int offset = 0;
      for (int j = 0; j < train_data->GetNumFeatures(); ++j) {
        leaf_pool_[i][j].Init(data_[i].data() + offset, &feature_metas_[j]);
        auto num_bin = train_data->GetFMapperNum(j);
        offset += static_cast<int>(num_bin);
      }
    }
  }

  bool Get(int idx, FeatureHistogram** out) {
    *out = leaf_pool_[idx].get();
    return true;
  }

  void Move(int src_idx, int dst_idx) {
    std::swap(leaf_pool_[src_idx], leaf_pool_[dst_idx]);
  }

 private:
  std::vector<FeatureMeta> feature_metas_;
  std::vector<std::unique_ptr<FeatureHistogram[]>> leaf_pool_;
  std::vector<std::vector<BinEntry>> data_;
};

} // namespace UTBoost

#endif //UTBOOST_INCLUDE_UTBOOST_HIST_H_
