/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 * Created by Junjie Gao on 2023/4/26.
 */

#ifndef UTBOOST_SRC_ENSEMBLE_TREE_BAGGING_H_
#define UTBOOST_SRC_ENSEMBLE_TREE_BAGGING_H_

#include "causal_gbm.h"

namespace UTBoost {

/*!
 * \brief Random Forest implementation
 */
class CausalRF : public CausalGBM {
 public:
  CausalRF() : CausalGBM() {
    average_output_ = true;
  }

  ~CausalRF() {}

  void Init(const Config* config, const Dataset* train_data, const ObjectiveFunction* objective_function,
            const std::vector<const Metric*>& training_metrics) override {

    ASSERT((config->bagging_freq > 0 && config->bagging_fraction < 1.0f && config->bagging_fraction > 0.0f) ||
        (config->feature_fraction < 1.0f && config->feature_fraction > 0.0f));
    CausalGBM::Init(config, train_data, objective_function, training_metrics);
    // not shrinkage rate for the RF
    shrinkage_rate_ = 1.0f;
    // only boosting one time
    Boosting();
  }

  void Boosting() override {
    if (objective_function_ == nullptr) {
      Log::Error("RF mode do not support custom objective function, please use built-in objectives.");
    }
    init_scores_.resize(num_treat_, 0.0);
    for (int cur_treat = 0; cur_treat < num_treat_; ++cur_treat) {
      init_scores_[cur_treat] = BoostFromAverage(cur_treat, false);
    }
    size_t total_size = static_cast<size_t>(num_data_) * num_treat_;
    std::vector<double> tmp_scores(total_size, 0.0f);
#pragma omp parallel for schedule(static)
    for (int j = 0; j < num_treat_; ++j) {
      size_t offset = static_cast<size_t>(j)* num_data_;
      for (data_size_t i = 0; i < num_data_; ++i) {
        tmp_scores[offset + i] = init_scores_[j];
      }
    }
    objective_function_->
        GetGradients(tmp_scores.data(), gradients_.data(), hessians_.data());
  }

  bool TrainOneIter(const score_t* gradients, const score_t* hessians) override {
    // bagging logic
    data_sample_strategy_ ->Bagging(iter_, tree_learner_.get(), gradients_.data(), hessians_.data());
    const data_size_t bag_data_cnt = data_sample_strategy_->bag_data_cnt();
    const std::vector<data_size_t>& bag_data_indices = data_sample_strategy_->bag_data_indices();

    gradients = gradients_.data();
    hessians = hessians_.data();

    std::unique_ptr<Tree> new_tree(new Tree(2, num_treat_));
    if (train_data_->GetNumFeatures() > 0) {
      bool is_first_tree = models_.empty();
      new_tree.reset(tree_learner_->Train(gradients, hessians, is_first_tree, split_criteria_.get()));
    }

    if (new_tree->num_leaves() > 1) {
      if (config_->use_honesty && num_data_ - bag_data_cnt > 0) {  // honesty
        tree_learner_->RenewTreeOutputByIndices(new_tree.get(), split_criteria_.get(),
                                                data_sample_strategy_->bag_data_indices().data() + bag_data_cnt,
                                                num_data_ - bag_data_cnt, gradients, hessians);
      }
      for (int cur_treat = 0; cur_treat < num_treat_; ++cur_treat) {
        std::fabs(init_scores_[cur_treat]) > kEpsilon;
        new_tree->AddBias(init_scores_[cur_treat], cur_treat);
      }
      // update score
      MultiplyScore(iter_);
      UpdateScore(new_tree.get(), 0);
      MultiplyScore(1.0 / (iter_ + 1));
    } else {
      // only add default score one-time
      if (models_.empty()) {
        double score;
        for (int cur_treat = 0; cur_treat < num_treat_; ++cur_treat) {
          // updates scores
          score = objective_function_->BoostFromScore(cur_treat);
          new_tree->AsConstantTree(score, cur_treat);
        }      // update score
        MultiplyScore(iter_);
        UpdateScore(new_tree.get(), 0);
        MultiplyScore(1.0 / (iter_ + 1));
      }
    }
    // add model
    models_.push_back(std::move(new_tree));

    ++iter_;
    return false;
  }

  void RollbackOneIter() override {
    if (iter_ <= 0) { return; }
    int cur_iter = iter_ - 1;
    // reset score
    auto curr_tree = cur_iter;
    models_[curr_tree]->Shrinkage(-1.0);
    MultiplyScore(iter_);
    train_score_updater_->AddScore(models_[curr_tree].get(), 0);
    for (auto& score_updater : valid_score_updater_) {
      score_updater->AddScore(models_[curr_tree].get(), 0);
    }
    MultiplyScore(1.0f / (iter_ - 1.0));
    // remove model
    models_.pop_back();
    --iter_;
  }

  void MultiplyScore(double val) {
    train_score_updater_->MultiplyScore(val);
    for (auto& score_updater : valid_score_updater_) {
      score_updater->MultiplyScore(val);
    }
  }

  void AddValidDataset(const Dataset* valid_data,
                       const std::vector<const Metric*>& valid_metrics) override {
    CausalGBM::AddValidDataset(valid_data, valid_metrics);
    if (iter_ > 0) {
      valid_score_updater_.back()->MultiplyScore(1.0f / static_cast<float>(iter_));
    }
  }

 private:
  std::vector<double> init_scores_;
};

}  // namespace UTBoost

#endif //UTBOOST_SRC_ENSEMBLE_TREE_BAGGING_H_
