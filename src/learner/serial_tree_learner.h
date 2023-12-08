/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#ifndef UTBOOST_SRC_LEARNER_SERIAL_TREE_LEARNER_H_
#define UTBOOST_SRC_LEARNER_SERIAL_TREE_LEARNER_H_

#include "UTBoost/tree_learner.h"
#include "UTBoost/utils/log_wrapper.h"
#include "data_partition.h"
#include "leaf_split.h"
#include "split_info.h"
#include "col_sampler.h"
#include "histogram.h"


namespace UTBoost {

class SerialTreeLearner: public TreeLearner {
 public:
  explicit SerialTreeLearner(const Config* config);
  ~SerialTreeLearner();
  void Init(const Dataset* train_data, bool is_constant_hessian) override;

  void ResetTrainingDataInner(const Dataset* train_data, bool is_constant_hessian, bool reset_multi_val_bin) {
    train_data_ = train_data;
    num_data_ = train_data_->GetNumSamples();
    ASSERT_EQ(num_features_, train_data_->GetNumFeatures());
    // initialize splits for leaf
    smaller_leaf_splits_->ResetNumData(num_data_);
    larger_leaf_splits_->ResetNumData(num_data_);
    // initialize data partition
    data_partition_->ResetNumData(num_data_);
  }

  Tree* Train(const score_t* gradients, const score_t *hessians, bool is_first_tree, const SplitCriteria* split_criteria) override;

  virtual void BeforeTrain();

  virtual bool BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf);

  virtual void ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract);

  virtual void FindBestSplitsFromHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract, const SplitCriteria* split_criteria, const Tree*);

  virtual void Split(Tree* tree, int best_leaf, int* left_leaf, int* right_leaf);

  void RenewTreeOutputByIndices(Tree* tree, const SplitCriteria* split_criteria, const data_size_t* bag_indices, data_size_t bag_cnt, const score_t* gradients, const score_t* hessians) const override;

  void SetBaggingData(const Dataset* subset, const data_size_t* used_indices, data_size_t num_data) override {
    if (subset == nullptr) {
      data_partition_->SetUsedDataIndices(used_indices, num_data);
    } else {
      ResetTrainingDataInner(subset, true, false);
      bagging_use_indices = used_indices;
      bagging_indices_cnt = num_data;
    }
  }

  inline data_size_t GetGlobalDataCountInLeaf(int leaf_idx) const {
    if (leaf_idx >= 0) {
      return data_partition_->leaf_count(leaf_idx);
    } else {
      return 0;
    }
  }

  virtual void FindBestSplits(const Tree* tree, const std::set<int>* force_features, const SplitCriteria* split_criteria);

  void AddPredictionToScore(const Tree* tree, double* out_score) const override {
    // y0 is in the front, and ite is at the back.
    if (tree->num_leaves() <= 1) { return; }
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < tree->num_leaves(); ++i) {
      const double* output = tree->LeafOutput(i);
      int cnt_leaf_data = 0;
      auto tmp_idx = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
      for (int j = 0; j < cnt_leaf_data; ++j) {
        for (int k = 0; k < num_treats_; ++k) {
          out_score[tmp_idx[j] + k * num_data_] += output[k];
        }
      }
    }
  }

 protected:
  /*! \brief number of data */
  data_size_t num_data_;
  /*! \brief number of features */
  int num_features_;
  int num_treats_;
  /*! \brief training data */
  const Dataset* train_data_;
  /*! \brief gradients of current iteration */
  const score_t* gradients_;
  /*! \brief hessians of current iteration */
  const score_t* hessians_;
  /*! \brief training data partition on leaves */
  std::unique_ptr<DataPartition> data_partition_;
  /*! \brief pointer to histograms array of parent of current leaves */
  FeatureHistogram* parent_leaf_histogram_array_;
  /*! \brief pointer to histograms array of smaller leaf */
  FeatureHistogram* smaller_leaf_histogram_array_;
  /*! \brief pointer to histograms array of larger leaf */
  FeatureHistogram* larger_leaf_histogram_array_;
  /*! \brief store best split points for all leaves */
  std::vector<SplitInfo> best_split_per_leaf_;
  /*! \brief store best split per feature for all leaves */
  std::vector<SplitInfo> splits_per_leaf_;
  /*! \brief stores best thresholds for all feature for smaller leaf */
  std::unique_ptr<LeafSplit> smaller_leaf_splits_;
  /*! \brief stores best thresholds for all feature for larger leaf */
  std::unique_ptr<LeafSplit> larger_leaf_splits_;
  /*! \brief used to cache historical histogram to speed up*/
  HistogramPool histogram_pool_;
  /*! \brief config of tree learner*/
  const Config* config_;
  ColSampler col_sampler_;
  // std::vector<int8_t> is_feature_used_;
  const data_size_t* bagging_use_indices;
  data_size_t bagging_indices_cnt;
  std::vector<int8_t> is_feature_used_;
};

}  // namespace UTBoost

#endif //UTBOOST_SRC_LEARNER_SERIAL_TREE_LEARNER_H_
