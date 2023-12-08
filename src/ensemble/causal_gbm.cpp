/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 * Created by Junjie Gao on 2023/3/1.
 */

#include "causal_gbm.h"

namespace UTBoost {

CausalGBM::CausalGBM() {
  iter_ = 0;
  train_data_ = nullptr;
  config_ = nullptr;
  objective_function_ = nullptr;
  max_feature_idx_ = 0;
  shrinkage_rate_ = 0.1f;
  average_output_ = false;
  tree_learner_ = nullptr;
  data_sample_strategy_.reset(nullptr);
  gradients_pointer_ = nullptr;
  hessians_pointer_ = nullptr;
  num_treat_ = 0;
  num_treat_ = 0;
  num_data_ = 0;
  num_iteration_for_pred_ = 0;
  start_iteration_for_pred_ = 0;
}

void CausalGBM::Init(const Config *config,
                     const Dataset *train_data,
                     const ObjectiveFunction* objective_function,
                     const std::vector<const Metric *> &training_metrics) {
  train_data_ = train_data;
  iter_ = 0;
  num_iteration_for_pred_ = 0;
  max_feature_idx_ = 0;
  config_ = std::unique_ptr<Config>(new Config(*config));
  shrinkage_rate_ = config_->learning_rate;
  num_treat_ = train_data->GetDistinctTreatNum();
  num_data_ = train_data_->GetNumSamples();
  max_feature_idx_ = train_data->GetNumFeatures() - 1;

  objective_function_ = objective_function;
  split_criteria_.reset(SplitCriteria::Create(config->split_criteria, config));
  data_sample_strategy_.reset(SampleStrategy::CreateSampleStrategy(config_.get(), train_data_, objective_function_));
  data_sample_strategy_->ResetSampleConfig(config_.get(), true);
  ResetGradientBuffers();

  tree_learner_ = std::unique_ptr<TreeLearner>(TreeLearner::CreateTreeLearner(config));
  tree_learner_->Init(train_data_, true);

  training_metrics_.clear();
  for (const auto& metric : training_metrics) {
    training_metrics_.push_back(metric);
  }
  training_metrics_.shrink_to_fit();

  train_score_updater_.reset(new ScoreUpdater(train_data_, num_treat_));

}

void CausalGBM::AddValidDataset(const Dataset *valid_data, const std::vector<const Metric *> &valid_metrics) {
  if (valid_data->GetDistinctTreatNum() > num_treat_) {
    Log::Error("The number of distinct treatments in the training set (%d) "
               "is smaller than that in the validation set (%d).", num_treat_, valid_data->GetDistinctTreatNum());
  }
  // for a validation dataset, we need its score and metric
  auto new_score_updater = std::unique_ptr<ScoreUpdater>(new ScoreUpdater(valid_data, num_treat_));

  valid_score_updater_.push_back(std::move(new_score_updater));
  valid_metrics_.emplace_back();
  for (const auto& metric : valid_metrics) {
    valid_metrics_.back().push_back(metric);
  }
  valid_metrics_.back().shrink_to_fit();

}

double ObtainAutomaticInitialScore(const ObjectiveFunction* fobj, treatment_t treatment_id) {
  double init_score = 0.0;
  if (fobj != nullptr) {
    init_score = fobj->BoostFromScore(treatment_id);
  }
  return init_score;
}

bool CausalGBM::TrainOneIter(const score_t *gradients, const score_t *hessians) {
  std::vector<double> init_scores(num_treat_, 0.0);
  // boosting first
  if (gradients == nullptr || hessians == nullptr) {
    for (int cur_treat = 0; cur_treat < num_treat_; ++cur_treat) {
      init_scores[cur_treat] = BoostFromAverage(cur_treat, true);
    }
    Boosting();
    gradients = gradients_pointer_;
    hessians = hessians_pointer_;
  }

  // bagging logic
  data_sample_strategy_->Bagging(iter_, tree_learner_.get(), gradients_.data(), hessians_.data());
  const data_size_t bag_data_cnt = data_sample_strategy_->bag_data_cnt();
  const std::vector<data_size_t>& bag_data_indices = data_sample_strategy_->bag_data_indices();

  bool should_continue = false;

  std::unique_ptr<Tree> new_tree(new Tree(2, num_treat_));
  if (train_data_->GetNumFeatures() > 0) {
    bool is_first_tree = models_.empty();
    new_tree.reset(tree_learner_->Train(gradients, hessians, is_first_tree, split_criteria_.get()));
  }

  if (new_tree->num_leaves() > 1) {
    should_continue = true;
    if (config_->use_honesty && num_data_ - bag_data_cnt > 0) {  // honesty
      tree_learner_->RenewTreeOutputByIndices(new_tree.get(), split_criteria_.get(),
                                              data_sample_strategy_->bag_data_indices().data() + bag_data_cnt,
                                              num_data_ - bag_data_cnt, gradients, hessians);
    }
    // shrinkage by learning rate
    new_tree->Shrinkage(shrinkage_rate_);
    // update score
    UpdateScore(new_tree.get(), 0);
    // update score
    for (int cur_treat = 0; cur_treat < num_treat_; ++cur_treat) {
      std::fabs(init_scores[cur_treat]) > kEpsilon;
      new_tree->AddBias(init_scores[cur_treat], cur_treat);
    }
  } else {
   // only add default score one-time
   if (models_.empty()) {
     if (objective_function_ != nullptr) {
       for (int cur_treat = 0; cur_treat < num_treat_; ++cur_treat) {
         // updates scores
         init_scores[cur_treat] = ObtainAutomaticInitialScore(objective_function_, cur_treat);
         train_score_updater_->AddScore(init_scores[cur_treat], cur_treat);
         for (auto& score_updater : valid_score_updater_) {
           score_updater->AddScore(init_scores[cur_treat], cur_treat);
         }
         new_tree->AsConstantTree(init_scores[cur_treat], cur_treat);
       }
     }
   }
  }
  // add model
  models_.push_back(std::move(new_tree));

  if (!should_continue) {
    Log::Warn("Stopped training because there are no more leaves that meet the split requirements");
    if (models_.size() > static_cast<size_t>(1)) {
      models_.pop_back();
    }
    return true;
  }

  ++iter_;
  return false;
}

void CausalGBM::Boosting() {
  // objective function will calculate gradients and hessians
  int64_t num_score = 0;
  objective_function_->
      GetGradients(GetTrainingScore(&num_score), gradients_pointer_, hessians_pointer_);
}

const double *CausalGBM::GetTrainingScore(int64_t *out_len) {
  *out_len = static_cast<int64_t>(train_score_updater_->num_data()) * num_treat_;
  return train_score_updater_->score();
}

double CausalGBM::BoostFromAverage(treatment_t treatment_id, bool update_scorer) {
  // boosting from average label; or customized "average" if implemented for the current objective
  if (models_.empty() && objective_function_ != nullptr) {
    if (config_->boost_from_average || (train_data_ != nullptr && train_data_->GetNumFeatures() == 0)) {
      double init_score = ObtainAutomaticInitialScore(objective_function_, treatment_id);
      if (std::fabs(init_score) > kEpsilon) {
        if (update_scorer) {
          train_score_updater_->AddScore(init_score, treatment_id);
          for (auto& score_updater : valid_score_updater_) {
            score_updater->AddScore(init_score, treatment_id);
          }
        }
        if (treatment_id == 0) {
          Log::Info("Control starts training from score %lf", init_score);
        } else {
          Log::Info("Treatment id: %d starts training from score %lf", treatment_id, init_score);
        }
        return init_score;
      }
    }
  }
  return 0.0f;
}

void CausalGBM::RollbackOneIter() {
  if (iter_ <= 0) { return; }
  // reset score
  auto curr_tree = models_.size() - 1;
  models_[curr_tree]->Shrinkage(-1.0);
  train_score_updater_->AddScore(models_[curr_tree].get(), 0);
  for (auto& score_updater : valid_score_updater_) {
    score_updater->AddScore(models_[curr_tree].get(), 0);
  }
  // remove model
  models_.pop_back();
  --iter_;
}


void CausalGBM::ResetGradientBuffers() {
  const auto total_size = static_cast<size_t>(num_data_);
  if (gradients_.size() < total_size) {
    gradients_.resize(total_size);
    hessians_.resize(total_size);
  }
  gradients_pointer_ = gradients_.data();
  hessians_pointer_ = hessians_.data();
}


void CausalGBM::UpdateScore(const Tree *tree, const int cur_tree_id) {
  // update training score
  if (!data_sample_strategy_->is_use_subset()) {
    train_score_updater_->AddScore(tree_learner_.get(), tree, cur_tree_id);

    const data_size_t bag_data_cnt = data_sample_strategy_->bag_data_cnt();
    // we need to predict out-of-bag scores of data for boosting
    if (num_data_ - bag_data_cnt > 0) {
      train_score_updater_->AddScore(tree, data_sample_strategy_->bag_data_indices().data() + bag_data_cnt, num_data_ - bag_data_cnt, cur_tree_id);
    }
  } else {
    train_score_updater_->AddScore(tree, cur_tree_id);
  }

  // update validation score
  for (auto& score_updater : valid_score_updater_) {
    score_updater->AddScore(tree, cur_tree_id);
  }
}


std::vector<double> CausalGBM::GetEvalAt(int data_idx) const {
  ASSERT(data_idx >= 0 && data_idx <= static_cast<int>(valid_score_updater_.size()))
  std::vector<double> ret;
  if (data_idx == 0) {
    for (auto& sub_metric : training_metrics_) {
      auto scores = EvalOneMetric(sub_metric,
                                  train_score_updater_->score(),
                                  train_score_updater_->num_data());
      for (auto score : scores) {
        ret.push_back(score);
      }
    }
  } else {
    auto used_idx = data_idx - 1;
    for (size_t j = 0; j < valid_metrics_[used_idx].size(); ++j) {
      auto test_scores = EvalOneMetric(valid_metrics_[used_idx][j],
                                       valid_score_updater_[used_idx]->score(),
                                       valid_score_updater_[used_idx]->num_data());
      for (auto score : test_scores) {
        ret.push_back(score);
      }
    }
  }
  return ret;
}


std::vector<double> CausalGBM::EvalOneMetric(const Metric *metric,
                                             const double *score,
                                             const data_size_t num_data) const {
  return metric->Eval(score, objective_function_);
}


void CausalGBM::PredictRaw(const double *features, double *output) const {
  // set zero
  std::memset(output, 0, sizeof(double) * num_treat_);
  const int end_iteration_for_pred = start_iteration_for_pred_ + num_iteration_for_pred_;
  for (int i = start_iteration_for_pred_; i < end_iteration_for_pred; ++i) {
    const double* preds = models_[i]->Predict(features);
    for (int j = 0; j < num_treat_; ++j) {
      output[j] += preds[j];
    }
  }
  // add baseline
  for (int j = 1; j < num_treat_; ++j) {
    output[j] += output[0];
  }
}

void CausalGBM::Predict(const double* features, double* output) const {
  PredictRaw(features, output);
  if (average_output_) {
    for (int j = 0; j < num_treat_; ++j) {
      output[j] /= num_iteration_for_pred_;
    }
  }
  if (objective_function_ != nullptr) {
    objective_function_->ConvertOutput(output, output, num_treat_);
  }
}

}  // namespace UTBoost
