/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#ifndef UTBOOST_SRC_OBJECTIVE_BINARY_OBJECTIVE_H_
#define UTBOOST_SRC_OBJECTIVE_BINARY_OBJECTIVE_H_

#include "UTBoost/objective_function.h"

#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace UTBoost {

/*!
 * \brief Objective function for binary classification
 */
class BinaryLogloss: public ObjectiveFunction {
 public:
  explicit BinaryLogloss(const Config& config) {
    is_unbalance_ = config.auto_balance;
    is_pos_ = [](label_t label) -> bool { return label > kZeroThreshold; };
    is_treat_ = [](treatment_t treat) -> bool { return treat > 0; };
    treat_weights_ = config.scale_treat_weight;
  }

  ~BinaryLogloss() {}

  void Init(const MetaInfo& meta, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = meta.GetLabel();
    weights_ = meta.GetWeight();
    treatment_ = meta.GetTreatment();
    num_treat_ = meta.GetNumDistinctTreat();
    if (treat_weights_.empty()) {
      treat_weights_ = std::vector<double> (num_treat_, 1.0);
    } else if (treat_weights_.size() != num_treat_){
      Log::Error("The number of treat weights %d and the number of treatments %d of training data are not equal", treat_weights_.size(), num_treat_);
    }
    data_size_t cnt_positive = 0;
    data_size_t cnt_negative = 0;
    // count for positive and negative samples
#pragma omp parallel for schedule(static) reduction(+:cnt_negative, cnt_positive)
    for (data_size_t i = 0; i < num_data_; ++i) {
      if (is_pos_(label_[i])) {
        ++cnt_positive;
      } else {
        ++cnt_negative;
      }
    }
    num_pos_data_ = cnt_positive;

    control_label_avg_ = AverageScore(0);

    need_train_ = true;
    if (num_pos_data_ == 0 || (num_data - num_pos_data_) == 0) {
      Log::Warn("Contains only one class");
      // not need to boost.
      need_train_ = false;
    }
    // use -1 for negative class, and 1 for positive class
    label_val_[0] = -1;
    label_val_[1] = 1;
    // weight for label
    label_weights_[0] = 1.0f;
    label_weights_[1] = 1.0f;
    if (is_unbalance_ && cnt_positive > 0 && cnt_negative > 0) {
      if (cnt_positive > cnt_negative) {
        label_weights_[0] = static_cast<double>(cnt_positive) / cnt_negative;
      } else {
        label_weights_[1] = static_cast<double>(cnt_negative) / cnt_positive;
      }
    }
  }

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    if (!need_train_) {
      return;
    }
    if (weights_ == nullptr) {
#pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        // get label and label weights
        const int is_pos = is_pos_(label_[i]);
        const int label = label_val_[is_pos];
        const double label_weight = label_weights_[is_pos];
        const double treat_weight = treat_weights_[treatment_[i]];
        // calculate gradients and hessians
        const double learner_score = is_treat_(treatment_[i])? (score[i] + score[i + num_data_ * treatment_[i]]) : score[i];
        const double response = -label / (1.0f + std::exp(label * learner_score));
        const double abs_response = fabs(response);
        gradients[i] = static_cast<score_t>(response * label_weight * treat_weight);
        hessians[i] = static_cast<score_t>(abs_response * (1.0 - abs_response) * label_weight * treat_weight);
      }
    } else {
#pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        // get label and label weights
        const int is_pos = is_pos_(label_[i]);
        const int label = label_val_[is_pos];
        const double label_weight = label_weights_[is_pos];
        const double treat_weight = treat_weights_[treatment_[i]];
        // calculate gradients and hessians
        const double learner_score = is_treat_(treatment_[i])? (score[i] + score[i + num_data_ * treatment_[i]]) : score[i];
        const double response = -label / (1.0f + std::exp(label * learner_score));
        const double abs_response = fabs(response);
        gradients[i] = static_cast<score_t>(response * label_weight * weights_[i] * treat_weight);
        hessians[i] = static_cast<score_t>(abs_response * (1.0 - abs_response) * label_weight * weights_[i] * treat_weight);
      }
    }
  }

  double AverageScore(treatment_t treatment_id) const {
    double suml = 0.0f;
    double sumw = 0.0f;
    if (weights_ != nullptr) {
#pragma omp parallel for schedule(static) reduction(+:suml, sumw)
      for (data_size_t i = 0; i < num_data_; ++i) {
        if (treatment_[i] == treatment_id) {
          suml += is_pos_(label_[i]) * weights_[i];
          sumw += weights_[i];
        }
      }
    } else {
#pragma omp parallel for schedule(static) reduction(+:suml, sumw)
      for (data_size_t i = 0; i < num_data_; ++i) {
        if (treatment_[i] == treatment_id) {
          suml += is_pos_(label_[i]);
          sumw += 1.0;
        }
      }
    }

    double pavg = suml / (sumw + kEpsilon);
    pavg = std::min(pavg, 1.0 - kEpsilon);
    pavg = std::max<double>(pavg, kEpsilon);
    return pavg;
  }

  // implement custom average to boost from (if enabled among options)
  double BoostFromScore(treatment_t treatment_id) const override {
    double control_init = std::log(control_label_avg_ / (1.0f - control_label_avg_));
    if (treatment_id == 0) {  // control
      Log::Info("[%s:%s]: control pavg=%f -> initscore=%f", GetName(), __func__, control_label_avg_, control_init);
      return control_init;
    } else {  // treated
      double pavg = AverageScore(treatment_id);
      double initscore = std::log(pavg / (1.0f - pavg)) - control_init;
      Log::Info("[%s:%s]: treatment=%d p_avg=%f -> initscore=%f", GetName(), __func__, treatment_id, pavg, initscore);
      return initscore;
    }
  }

  const char* GetName() const override {
    return "logloss";
  }

  void ConvertOutput(const double* input, double* output, int len) const override {
    for (int i = 0; i < len; ++i) {
      output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
  }

  void ConvertOutput(const double input, double *output) const override {
    *output = 1.0f / (1.0f + std::exp(-input));
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  data_size_t NumPositiveData() const override { return num_pos_data_; }

 protected:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Number of positive samples */
  data_size_t num_pos_data_;
  int num_treat_;
  /*! \brief Pointer of label */
  const label_t* label_;
  const treatment_t* treatment_;
  /*! \brief True if using unbalance training */
  bool is_unbalance_;
  /*! \brief Sigmoid parameter */
  double control_label_avg_;
  /*! \brief Values for positive and negative labels */
  int label_val_[2];
  /*! \brief Weights for positive and negative labels */
  double label_weights_[2];
  std::vector<double> treat_weights_;
  /*! \brief Weights for data */
  const label_t* weights_;
  std::function<bool(label_t)> is_pos_;
  std::function<bool(treatment_t)> is_treat_;
  bool need_train_;
};

}  // namespace UTBoost


#endif //UTBOOST_SRC_OBJECTIVE_BINARY_OBJECTIVE_H_
