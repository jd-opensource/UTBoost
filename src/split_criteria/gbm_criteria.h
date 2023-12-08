/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#ifndef UTBOOST_SRC_SPLIT_CRITERIA_GBM_CRITERIA_H_
#define UTBOOST_SRC_SPLIT_CRITERIA_GBM_CRITERIA_H_


#include "UTBoost/split_criteria.h"

#include <cmath>

namespace UTBoost {

class GBM : public SplitCriteria {
 public:
  explicit GBM(const Config* config) {
    gbm_gain_type_ = config->gbm_gain_type_;
    constrains_ = config->effect_constrains;
  }

  double SplitScore(const BinEntry* left, const BinEntry* right, const BinEntry* parent) const override {
    for (int i = 0; i < left->num_treat_; ++i) {
      if (left->num_data_[i] < 1 || right->num_data_[i] < 1) {
        return 0.0f;
      }
    }
    if (!constrains_.empty()) {  // no constrains
      double control_avg_l = left->label_sum_[0] / left->num_data_[0];
      double control_avg_r = right->label_sum_[0] / right->num_data_[0];
      for (int i = 1; i < std::min(left->num_treat_, static_cast<int>(constrains_.size() + 1)); ++i) {
        int constrain = constrains_[i - 1];
        if ( (left->label_sum_[i] / left->num_data_[i] - control_avg_l) * constrain < 0 || (right->label_sum_[i] / right->num_data_[i] - control_avg_r) * constrain < 0) {
          return 0.0f;
        }
      }
    }
    return GetSplitGains(left) + GetSplitGains(right);
  }

  double GetSplitGains(const BinEntry* entry) const override {
    double value = -1.0 * entry->wgradients_sum_[0] / (entry->whessians_sum_[0] + kEpsilon);
    double optimal = 0.0f;

    if (gbm_gain_type_ == 0) {
      optimal += entry->gradients_sum_ * value + 0.5 * entry->hessians_sum_ * Square(value);
    }

    for (int i = 1; i < entry->num_treat_; ++i) {
      optimal -= Square(entry->wgradients_sum_[i] + entry->whessians_sum_[i] * value)  / (2.0 * entry->whessians_sum_[i] + kEpsilon);
      if (gbm_gain_type_ == 1) {
        optimal += entry->wgradients_sum_[i] * value + 0.5 * entry->whessians_sum_[i] * Square(value);
      }
    }
    return -optimal;
  }

  std::vector<double> CalculateLeafOutput(const BinEntry* entry) const override {
    std::vector<double> ret(entry->num_treat_, 0.0);
    if (entry->num_total_data_ == 0) return ret;
    ret[0] = -1.0 * entry->wgradients_sum_[0]/ (entry->whessians_sum_[0] + kEpsilon);
    for (int i = 1; i < entry->num_treat_; ++i) {
      ret[i] = -1.0 * (entry->wgradients_sum_[i] + entry->whessians_sum_[i] * ret[0]) / (entry->whessians_sum_[i] + kEpsilon);
    }
    return ret;
  }

  std::string ToString() const override { return std::string("gbm"); };

 protected:
  int gbm_gain_type_;
  std::vector<int32_t> constrains_;
};

}  // namespace UTBoost

#endif //UTBOOST_SRC_SPLIT_CRITERIA_GBM_CRITERIA_H_
