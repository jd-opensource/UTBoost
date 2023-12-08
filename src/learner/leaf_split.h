/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#ifndef UTBOOST_SRC_LEARNER_LEAF_SPLIT_H_
#define UTBOOST_SRC_LEARNER_LEAF_SPLIT_H_

#include "data_partition.h"
#include "UTBoost/config.h"

namespace UTBoost {

/*!
 * \brief used to find split candidates for a leaf
 */
class LeafSplit {
 public:
  LeafSplit(int num_data, int num_treat) :num_data_in_leaf_(num_data), num_data_(num_data) ,data_indices_(nullptr), num_treat_(num_treat){}

  void ResetNumData(data_size_t num_data) {
    num_data_ = num_data;
    num_data_in_leaf_ = num_data;
  }

  void Init() {
    leaf_index_ = -1;
    data_indices_ = nullptr;
    num_data_in_leaf_ = 0;
    sum_gradients_ = 0;
    sum_hessians_ = 0;
    sum_wgradients_ = std::vector<double>(num_treat_, 0);
    sum_whessians_ = std::vector<double>(num_treat_, 0);
    sum_labels_ = std::vector<double>(num_treat_, 0);
    num_datas_ = std::vector<double>(num_treat_, 0);
  }

  /*!
   * \brief Init splits on the current leaf, it will traverse all data to sum up the results
   * \param gradients
   * \param hessians
   */
  void Init(const score_t* gradients, const score_t* hessians, const label_t* labels, const treatment_t* treats) {
    Init();
    num_data_in_leaf_ = num_data_;
    leaf_index_ = 0;
    data_indices_ = nullptr;
    for (data_size_t i = 0; i < num_data_in_leaf_; ++i) {
      const treatment_t &t = treats[i];
      sum_wgradients_[t] += gradients[i];
      sum_whessians_[t] += hessians[i];
      sum_labels_[t] += labels[i];
      num_datas_[t] += 1.0;
    }

    for (int i = 0; i < num_treat_; ++i) {
      sum_gradients_ += sum_wgradients_[i];
      sum_hessians_ += sum_whessians_[i];
    }
  }

  void Init(int leaf, const DataPartition* data_partition, const float* gradients, const float* hessians, const label_t* labels, const treatment_t* treats) {
    Init();
    leaf_index_ = leaf;
    data_indices_ = data_partition->GetIndexOnLeaf(leaf, &num_data_in_leaf_);
    for (int i = 0; i < num_data_in_leaf_; ++i) {
      const data_size_t idx = data_indices_[i];
      const treatment_t &t = treats[idx];
      sum_wgradients_[t] += gradients[idx];
      sum_whessians_[t] += hessians[idx];
      sum_labels_[t] += labels[idx];
      num_datas_[t] += 1.0;
    }
    for (int i = 0; i < num_treat_; ++i) {
      sum_gradients_ += sum_wgradients_[i];
      sum_hessians_ += sum_whessians_[i];
    }

  }

  void Init(int leaf, const DataPartition* data_partition, const double* sum_wgradients, const double* sum_whessians, const double* num_data, const double* sum_label) {
    leaf_index_ = leaf;
    data_indices_ = data_partition->GetIndexOnLeaf(leaf, &num_data_in_leaf_);
    sum_gradients_ = 0.0;
    sum_hessians_ = 0.0;
    for (int i = 0; i < num_treat_; ++i) {
      sum_wgradients_[i] = sum_wgradients[i];
      sum_whessians_[i] = sum_whessians[i];
      sum_labels_[i] = sum_label[i];
      num_datas_[i] = num_data[i];
      sum_gradients_ += sum_wgradients[i];
      sum_hessians_ += sum_whessians[i];
    }
  }

  const int* data_indices() const { return data_indices_; }
  int num_data_in_leaf() const { return num_data_in_leaf_; }
  int leaf_index() const { return leaf_index_; }
  double sum_gradients() const { return sum_gradients_; }
  double sum_hessians() const { return sum_hessians_; }
  const double* sum_wgradients() const { return sum_wgradients_.data(); }
  const double* sum_whessians() const { return sum_whessians_.data(); }
  const double* sum_labels() const { return sum_labels_.data(); }
  const double* num_data() const { return num_datas_.data(); }

 private:
  int num_data_in_leaf_;
  int num_data_;
  int num_treat_;
  int leaf_index_;
  const int* data_indices_;
  double sum_gradients_;
  double sum_hessians_;
  std::vector<double> sum_wgradients_;
  std::vector<double> sum_whessians_;
  std::vector<double> sum_labels_;
  std::vector<double> num_datas_;
};

}  // namespace UTBoost

#endif //UTBOOST_SRC_LEARNER_LEAF_SPLIT_H_
