/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#ifndef UTBOOST_SRC_SPLIT_CRITERIA_DIVERGENCE_CRITERIA_H_
#define UTBOOST_SRC_SPLIT_CRITERIA_DIVERGENCE_CRITERIA_H_


#include "UTBoost/split_criteria.h"

#include <cmath>

namespace UTBoost {

class CriteriaDivergence : public SplitCriteria {
 public:
  explicit CriteriaDivergence(const Config* config) {
    constrains_ = config->effect_constrains;
  }

  double SplitScore(const BinEntry* left, const BinEntry* right, const BinEntry* parent) const override {
    for (int i = 0; i < left->num_treat_; ++i) {
      if (left->num_data_[i] < 1 || right->num_data_[i] < 1) {
        return 0.0f;
      }
    }

    if (!constrains_.empty()) {  // constrains
      double control_avg_l = left->label_sum_[0] / left->num_data_[0];
      double control_avg_r = right->label_sum_[0] / right->num_data_[0];
      for (int i = 1; i < std::min(left->num_treat_, static_cast<int>(constrains_.size() + 1)); ++i) {
        int constrain = constrains_[i - 1];
        if ( (left->label_sum_[i] / left->num_data_[i] - control_avg_l) * constrain < 0 || (right->label_sum_[i] / right->num_data_[i] - control_avg_r) * constrain < 0) {
          return 0.0f;
        }
      }
    }
    return (left->num_data_[0] * Divergence(left) + right->num_data_[0] * Divergence(right)) / parent->num_data_[0];
  }

  std::vector<double> CalculateLeafOutput(const BinEntry* entry) const override {
    std::vector<double> ret(entry->num_treat_, 0.0);
    if (entry->num_data_[0] == 0) return ret;
    double control_avg = (entry->label_sum_[0]) / (entry->num_data_[0] + kEpsilon);
    ret[0] = control_avg;
    for (int i = 1; i < entry->num_treat_; ++i) {
      ret[i] = entry->label_sum_[i] / (entry->num_data_[i] + kEpsilon) - control_avg;
    }
    return ret;
  }

  double GetSplitGains(const BinEntry* entry) const override {
    return Divergence(entry);
  }

  virtual double Divergence(const BinEntry* entry) const = 0;

 protected:
  std::vector<int32_t> constrains_;
};


class ED : public CriteriaDivergence {
 public:
  explicit ED(const Config* config): CriteriaDivergence(config) {}

  double Divergence(const BinEntry* entry) const override {
    if (entry->num_data_[0] == 0) return 0.0f;
    double pc = entry->label_sum_[0] / (entry->num_data_[0] + kEpsilon);
    double divergence = 0.0f, pt;
    for (int i = 1; i < entry->num_treat_; ++i) {
      if (entry->num_data_[i] == 0) continue;
      pt = entry->label_sum_[i] / (entry->num_data_[i] + kEpsilon);
      divergence += Square(pt - pc);
    }
    return divergence;
  }

  std::string ToString() const override { return std::string("ed"); };
};


class KL : public CriteriaDivergence {
 public:
  explicit KL(const Config* config): CriteriaDivergence(config) {}

  double Divergence(const BinEntry* entry) const override {
    if (entry->num_data_[0] == 0) return 0.0f;
    double pc = entry->label_sum_[0] / (entry->num_data_[0] + kEpsilon);
    ASSERT(pc >= 0.0f && pc <= 1.0)
    Clip(pc, kEpsilon, 1 - kEpsilon);
    double divergence = 0.0f, pt;
    for (int i = 1; i < entry->num_treat_; ++i) {
      if (entry->num_data_[i] == 0) continue;
      pt = entry->label_sum_[i] / (entry->num_data_[i] + kEpsilon);
      ASSERT(pt >= 0.0f && pt <= 1.0)
      if (pt <= kEpsilon)
        divergence -= std::log(1 - pt);
      else if (pt >= 1 - kEpsilon)
        divergence -= std::log(pt);
      else
        divergence += pt * std::log(pt / pc) + (1 - pt) * std::log((1 - pt) / (1 - pc));
    }
    return divergence;
  }

  std::string ToString() const override { return std::string("kl"); };
};


class Chi : public CriteriaDivergence {
 public:
  explicit Chi(const Config* config): CriteriaDivergence(config) {}

  double Divergence(const BinEntry* entry) const override {
    if (entry->num_data_[0] == 0) return 0.0f;
    double pc = entry->label_sum_[0] / (entry->num_data_[0] + kEpsilon);
    double divergence = 0.0f, pt;
    for (int i = 1; i < entry->num_treat_; ++i) {
      if (entry->num_data_[i] == 0) continue;
      pt = entry->label_sum_[i] / (entry->num_data_[i] + kEpsilon);
      divergence += Square(pc - pt) / (pc + kEpsilon);
    }
    return divergence;
  }

  std::string ToString() const override { return std::string("chi"); };
};

}  // namespace UTBoost


#endif //UTBOOST_SRC_SPLIT_CRITERIA_DIVERGENCE_CRITERIA_H_
