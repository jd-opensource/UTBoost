/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef UTBOOST_INCLUDE_UTBOOST_SAMPLE_STRATEGY_H_
#define UTBOOST_INCLUDE_UTBOOST_SAMPLE_STRATEGY_H_

#include <memory>
#include <vector>

#include "UTBoost/definition.h"
#include "UTBoost/dataset.h"
#include "UTBoost/tree_learner.h"
#include "UTBoost/config.h"
#include "UTBoost/objective_function.h"
#include "UTBoost/utils/random.h"
#include "UTBoost/utils/threading.h"


namespace UTBoost {

class SampleStrategy {
 public:
  SampleStrategy() : bagging_runner_(0, bagging_rand_block_) {}

  virtual ~SampleStrategy() {}

  static SampleStrategy* CreateSampleStrategy(const Config* config, const Dataset* train_data, const ObjectiveFunction* objective_function);

  virtual void Bagging(int iter, TreeLearner* tree_learner, score_t* gradients, score_t* hessians) = 0;

  virtual void ResetSampleConfig(const Config* config, bool is_change_dataset) = 0;

  bool is_use_subset() const { return is_use_subset_; }

  data_size_t bag_data_cnt() const { return bag_data_cnt_; }

  std::vector<data_size_t>& bag_data_indices() { return bag_data_indices_; }

 protected:
  const Config* config_;
  const Dataset* train_data_;
  const ObjectiveFunction* objective_function_;
  std::vector<data_size_t> bag_data_indices_;
  data_size_t bag_data_cnt_;
  data_size_t num_data_;
  std::unique_ptr<Dataset> tmp_subset_;
  bool is_use_subset_;
  const int bagging_rand_block_ = 1024;
  std::vector<Random> bagging_rands_;
  ParallelPartitionRunner<data_size_t, false> bagging_runner_;
};

}  // namespace UTBoost


#endif //UTBOOST_INCLUDE_UTBOOST_SAMPLE_STRATEGY_H_
