/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#ifndef UTBOOST_SRC_OBJECTIVE_DEFAULT_OBJECTIVE_H_
#define UTBOOST_SRC_OBJECTIVE_DEFAULT_OBJECTIVE_H_

#include "UTBoost/objective_function.h"

#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "regression_objective.h"

namespace UTBoost {

/*!
 * \brief Objective function for default task
 */
class DefaultLoss: public RegressionL2loss {
 public:
  explicit DefaultLoss(const Config &config): RegressionL2loss(config) {};

  ~DefaultLoss() override {}

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    if (weights_ == nullptr) {
#pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double learner_score = is_treat_(treatment_[i])? score[i + num_data_ * treatment_[i]] : score[i];
        gradients[i] = static_cast<score_t>(label_[i] - learner_score);
        hessians[i] = 1.0;
      }
    } else {
#pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double learner_score = is_treat_(treatment_[i])? score[i + num_data_ * treatment_[i]] : score[i];
        gradients[i] = static_cast<score_t>(static_cast<score_t>(label_[i] - learner_score) * weights_[i]);
        hessians[i] = static_cast<score_t>(weights_[i]);
      }
    }
  }

  const char* GetName() const override {
    return "default";
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }
};

}  // namespace UTBoost

#endif //UTBOOST_SRC_OBJECTIVE_DEFAULT_OBJECTIVE_H_
