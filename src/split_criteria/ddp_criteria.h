/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#ifndef UTBOOST_SRC_SPLIT_CRITERIA_DDP_CRITERIA_H_
#define UTBOOST_SRC_SPLIT_CRITERIA_DDP_CRITERIA_H_

#include "UTBoost/split_criteria.h"

#include <cmath>

namespace UTBoost {

class DDP : public SplitCriteria {
 public:
  explicit DDP(const Config* config) {
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
    return Square(DeltaSum(left) - DeltaSum(right)) * left->num_data_[0] * right->num_data_[0] / Square(parent->num_data_[0]);
  }

  double DeltaSum(const BinEntry* entry) const {
    if (entry->num_data_[0] == 0) return 0.0f;
    double control_avg = entry->label_sum_[0] / (entry->num_data_[0] + kEpsilon);
    double delta_sum = 0.0f;
    for (int i = 1; i < entry->num_treat_; ++i) {
      if (entry->num_data_[i] == 0) continue;
      delta_sum += entry->wgradients_sum_[i] / (entry->num_data_[i] + kEpsilon) - control_avg;
    }
    return delta_sum;
  }

  double GetSplitGains(const BinEntry* entry) const override {
    return 0.0f;
  }

  std::vector<double> CalculateLeafOutput(const BinEntry* entry) const override {
    std::vector<double> ret(entry->num_treat_, 0.0);
    if (entry->num_data_[0] == 0) return ret;
    double control_avg = (entry->label_sum_[0]) / (entry->num_data_[0] + kEpsilon);
    ret[0] = (entry->wgradients_sum_[0]) / (entry->num_data_[0] + kEpsilon);
    for (int i = 1; i < entry->num_treat_; ++i) {
      ret[i] = entry->wgradients_sum_[i] / (entry->num_data_[i] + kEpsilon) - control_avg;
    }
    return ret;
  }

  std::string ToString() const override { return std::string("ddp"); };

 protected:
  std::vector<int32_t> constrains_;
};

}  // namespace UTBoost


#endif //UTBOOST_SRC_SPLIT_CRITERIA_DDP_CRITERIA_H_
