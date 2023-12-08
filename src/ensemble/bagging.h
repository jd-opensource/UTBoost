/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 * Created by Junjie Gao on 2023/3/1.
 */

#ifndef UTBOOST_SRC_ENSEMBLE_BAGGING_H_
#define UTBOOST_SRC_ENSEMBLE_BAGGING_H_

#include "UTBoost/sample_strategy.h"

namespace UTBoost {

class BaggingSampleStrategy : public SampleStrategy {
 public:
  BaggingSampleStrategy(const Config* config, const Dataset* train_data, const ObjectiveFunction* objective_function)
      : need_re_bagging_(false) {
    config_ = config;
    train_data_ = train_data;
    num_data_ = train_data->GetNumSamples();
    objective_function_ = objective_function;
  }

  ~BaggingSampleStrategy() {}

  void Bagging(int iter, TreeLearner* tree_learner, score_t* /*gradients*/, score_t* /*hessians*/) override {
    // if need bagging
    if ((bag_data_cnt_ < num_data_ && iter % config_->bagging_freq == 0) ||
        need_re_bagging_) {
      need_re_bagging_ = false;
      auto left_cnt = bagging_runner_.Run<true>(
          num_data_,
          [=](int, data_size_t cur_start, data_size_t cur_cnt, data_size_t* left,
              data_size_t*) {
            data_size_t cur_left_count = 0;
            cur_left_count = BaggingHelper(cur_start, cur_cnt, left);
            return cur_left_count;
          },
          bag_data_indices_.data());
      bag_data_cnt_ = left_cnt;
      Log::Debug("Re-bagging, using %d data to train", bag_data_cnt_);
      // set bagging data to tree learner
      if (!is_use_subset_) {
        tree_learner->SetBaggingData(nullptr, bag_data_indices_.data(), bag_data_cnt_);
      } else {
        // get subset
        tmp_subset_->ReSize(bag_data_cnt_);
        tmp_subset_->CopySubrow(train_data_, bag_data_indices_.data(),
                                bag_data_cnt_, false);
        tree_learner->SetBaggingData(tmp_subset_.get(), bag_data_indices_.data(),
                                     bag_data_cnt_);
      }
    }
  }

  void ResetSampleConfig(const Config* config, bool is_change_dataset) override {
    // if need bagging, create buffer
    data_size_t num_pos_data = 0;
    if (objective_function_ != nullptr) {
      num_pos_data = objective_function_->NumPositiveData();
    }

    if ((config->bagging_fraction < 1.0 ) && config->bagging_freq > 0) {
      need_re_bagging_ = false;

      if (!is_change_dataset &&
          config_ != nullptr && config_->bagging_fraction == config->bagging_fraction && config_->bagging_freq == config->bagging_freq) {
        config_ = config;
        return;
      }
      config_ = config;
      bag_data_cnt_ = static_cast<data_size_t>(config_->bagging_fraction * num_data_);

      bag_data_indices_.resize(num_data_);
      bagging_runner_.ReSize(num_data_);
      bagging_rands_.clear();
      for (int i = 0;
           i < (num_data_ + bagging_rand_block_ - 1) / bagging_rand_block_; ++i) {
        bagging_rands_.emplace_back(config_->bagging_seed + i);
      }

      is_use_subset_ = false;
      need_re_bagging_ = true;
    } else {
      bag_data_cnt_ = num_data_;
      bag_data_indices_.clear();
      bagging_runner_.ReSize(0);
      is_use_subset_ = false;
    }
  }

 private:
  data_size_t BaggingHelper(data_size_t start, data_size_t cnt, data_size_t* buffer) {
    if (cnt <= 0) {
      return 0;
    }
    data_size_t cur_left_cnt = 0;
    data_size_t cur_right_pos = cnt;
    // random bagging, minimal unit is one record
    for (data_size_t i = 0; i < cnt; ++i) {
      auto cur_idx = start + i;
      if (bagging_rands_[cur_idx / bagging_rand_block_].NextFloat() < config_->bagging_fraction) {
        buffer[cur_left_cnt++] = cur_idx;
      } else {
        buffer[--cur_right_pos] = cur_idx;
      }
    }
    return cur_left_cnt;
  }

  /*! \brief whether need restart bagging in continued training */
  bool need_re_bagging_;
};

}  // namespace UTBoost

#endif //UTBOOST_SRC_ENSEMBLE_BAGGING_H_
