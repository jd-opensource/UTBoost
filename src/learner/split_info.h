/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#ifndef UTBOOST_SRC_LEARNER_SPLIT_INFO_H_
#define UTBOOST_SRC_LEARNER_SPLIT_INFO_H_

#include <cstdint>
#include <limits>
#include <vector>
#include <cmath>

#include "UTBoost/bin.h"


namespace UTBoost {

struct SplitInfo {
 public:
  /*! \brief Feature index */
  int feature = -1;
  /*! \brief Split threshold */
  uint32_t threshold = 0;
  /*! \brief Left number of data after split */
  data_size_t left_count = 0;
  std::vector<double> left_num_data;
  /*! \brief Right number of data after split */
  data_size_t right_count = 0;
  std::vector<double> right_num_data;
  /*! \brief Left output after split */
  std::vector<double> left_output;
  /*! \brief Right output after split */
  std::vector<double> right_output;
  /*! \brief Left sum gradient after split */
  double left_sum_gradient = 0;
  std::vector<double> left_label_sum;
  std::vector<double> left_wgradients_sum;
  /*! \brief Left sum hessian after split */
  double left_sum_hessian = 0;
  std::vector<double> left_whessians_sum;
  /*! \brief Right sum gradient after split */
  double right_sum_gradient = 0;
  std::vector<double> right_label_sum;
  std::vector<double> right_wgradients_sum;
  /*! \brief Right sum hessian after split */
  double right_sum_hessian = 0;
  std::vector<double> right_whessians_sum;
  /*! \brief Split gain */
  double gain = -std::numeric_limits<double>::infinity();
  /*! \brief True if default split is left */
  bool default_left = true;
  int monotone_type = 0;

  void CopyFromEntry(const BinEntry* entry, bool is_left) {
    if (is_left) {
      left_sum_gradient = entry->gradients_sum_;
      left_wgradients_sum = entry->wgradients_sum_;
      left_sum_hessian = entry->hessians_sum_;
      left_whessians_sum = entry->whessians_sum_;
      left_num_data = entry->num_data_;
      left_label_sum = entry->label_sum_;
      left_count = static_cast<data_size_t>(entry->num_total_data_);
    } else {
      right_sum_gradient = entry->gradients_sum_;
      right_wgradients_sum = entry->wgradients_sum_;
      right_sum_hessian = entry->hessians_sum_;
      right_whessians_sum = entry->whessians_sum_;
      right_num_data = entry->num_data_;
      right_label_sum = entry->label_sum_;
      right_count = static_cast<data_size_t>(entry->num_total_data_);
    }
  }

  inline void Reset() {
    feature = -1;
    gain = -std::numeric_limits<double>::infinity();
  }

  inline bool operator > (const SplitInfo& si) const {
    double local_gain = this->gain;
    double other_gain = si.gain;
    // replace nan with -inf
    if (local_gain == NAN) {
      local_gain = -std::numeric_limits<double>::infinity();
    }
    // replace nan with -inf
    if (other_gain == NAN) {
      other_gain = -std::numeric_limits<double>::infinity();
    }
    int local_feature = this->feature;
    int other_feature = si.feature;
    // replace -1 with max int
    if (local_feature == -1) {
      local_feature = INT32_MAX;
    }
    // replace -1 with max int
    if (other_feature == -1) {
      other_feature = INT32_MAX;
    }
    if (local_gain != other_gain) {
      return local_gain > other_gain;
    }
    else {
      // if same gain, use smaller feature
      return local_feature < other_feature;
    }
  }

  inline bool operator == (const SplitInfo& si) const {
    double local_gain = this->gain;
    double other_gain = si.gain;
    // replace nan with -inf
    if (local_gain == NAN) {
      local_gain = -std::numeric_limits<double>::infinity();
    }
    // replace nan with -inf
    if (other_gain == NAN) {
      other_gain = -std::numeric_limits<double>::infinity();
    }
    int local_feature = this->feature;
    int other_feature = si.feature;
    // replace -1 with max int
    if (local_feature == -1) {
      local_feature = INT32_MAX;
    }
    // replace -1 with max int
    if (other_feature == -1) {
      other_feature = INT32_MAX;
    }
    if (local_gain != other_gain) {
      return local_gain == other_gain;
    }
    else {
      // if same gain, use smaller feature
      return local_feature == other_feature;
    }
  }
};

}  // namespace UTBoost


#endif //UTBOOST_SRC_LEARNER_SPLIT_INFO_H_
