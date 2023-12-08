/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#ifndef UTBOOST_SRC_OBJECTIVE_REGRESSION_OBJECTIVE_H_
#define UTBOOST_SRC_OBJECTIVE_REGRESSION_OBJECTIVE_H_

#include "UTBoost/objective_function.h"

#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace UTBoost {

/*!
 * \brief Objective function for regression
 */
class RegressionL2loss : public ObjectiveFunction {
 public:
  explicit RegressionL2loss(const Config &config) {
    is_treat_ = [](treatment_t treat) { return treat > 0; };
    treat_weights_ = config.scale_treat_weight;
  }

  ~RegressionL2loss() {
  }

  void Init(const MetaInfo& meta, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = meta.GetLabel();
    weights_ = meta.GetWeight();
    treatment_ = meta.GetTreatment();
    num_treat_ = meta.GetNumDistinctTreat();
    control_label_avg_ = AverageScore(0);
    if (treat_weights_.empty()) {
      treat_weights_ = std::vector<double> (num_treat_, 1.0);
    } else if (treat_weights_.size() != num_treat_){
      Log::Error("The number of treat weights %d and the number of treatments %d of training data are not equal", treat_weights_.size(), num_treat_);
    }
  }

  void GetGradients(const double *score, score_t *gradients, score_t *hessians) const override {
    if (weights_ == nullptr) {
#pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const score_t treat_weight = static_cast<score_t>(treat_weights_[treatment_[i]]);
        const double learner_score = is_treat_(treatment_[i])? (score[i] + score[i + num_data_ * treatment_[i]]) : score[i];
        gradients[i] = static_cast<score_t>(learner_score - label_[i]) * treat_weight;
        hessians[i] = treat_weight;
      }
    } else {
#pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const score_t treat_weight = static_cast<score_t>(treat_weights_[treatment_[i]]);
        const double learner_score = is_treat_(treatment_[i])? (score[i] + score[i + num_data_ * treatment_[i]]) : score[i];
        gradients[i] = static_cast<score_t>(static_cast<score_t>((learner_score - label_[i])) * weights_[i] * treat_weight);
        hessians[i] = static_cast<score_t>(weights_[i] * treat_weight);
      }
    }
  }

  const char *GetName() const override {
    return "mse";
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  double AverageScore(treatment_t treatment_id) const {
    double suml = 0.0f;
    double sumw = 0.0f;
    if (weights_ != nullptr) {
#pragma omp parallel for schedule(static) reduction(+:suml, sumw)
      for (data_size_t i = 0; i < num_data_; ++i) {
        if (treatment_[i] == treatment_id) {
          suml += static_cast<double>(label_[i]) * weights_[i];
          sumw += weights_[i];
        }
      }
    } else {
#pragma omp parallel for schedule(static) reduction(+:suml, sumw)
      for (data_size_t i = 0; i < num_data_; ++i) {
        if (treatment_[i] == treatment_id) {
          suml += label_[i];
          sumw += 1.0;
        }
      }
    }
    return suml / (sumw + kEpsilon);
  }

  double BoostFromScore(treatment_t treatment_id) const override {
    if (treatment_id == 0) {
      return control_label_avg_;
    } else {
      return AverageScore(treatment_id) - control_label_avg_;
    }
  }

 protected:
  /*! \brief Number of data */
  data_size_t num_data_;
  int num_treat_;
  /*! \brief Pointer of label */
  const label_t *label_;
  const treatment_t* treatment_;
  /*! \brief Pointer of weights */
  const label_t *weights_;
  std::vector<label_t> trans_label_;
  std::vector<double> treat_weights_;
  std::function<bool(treatment_t)> is_treat_;
  double control_label_avg_;
};

}  // namespace UTBoost

#endif //UTBOOST_SRC_OBJECTIVE_REGRESSION_OBJECTIVE_H_
