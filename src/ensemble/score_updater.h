/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#ifndef UTBOOST_SRC_ENSEMBLE_SCORE_UPDATER_H_
#define UTBOOST_SRC_ENSEMBLE_SCORE_UPDATER_H_

#include "UTBoost/dataset.h"
#include "UTBoost/definition.h"
#include "UTBoost/tree_learner.h"

namespace UTBoost {

class ScoreUpdater {
 public:
  ScoreUpdater(const Dataset* data, int num_treat) :data_(data), num_treat_(num_treat) {
    num_data_ = data->GetNumSamples();
    int total_size = static_cast<int>(num_data_) * num_treat_;
    score_.resize(total_size);
    // default start score is zero
    std::memset(score_.data(), 0, total_size * sizeof(double));
  }

  inline const double* score() const { return score_.data(); }
  inline data_size_t num_data() const { return num_data_; }

  inline void AddScore(const TreeLearner* tree_learner, const Tree* tree, int cur_tree_id) {
    const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id * num_treat_;
    tree_learner->AddPredictionToScore(tree, score_.data());
  }

  inline void AddScore(double val, treatment_t treatment_id) {
    if (treatment_id == 0) {  // used for control base predict
#pragma omp parallel for schedule(static, 512) if (num_data_ >= 1024)
      for (int i = 0; i < num_data_; ++i) {
        score_[i] += val;
      }
    } else {
#pragma omp parallel for schedule(static, 512) if (num_data_ >= 1024)
      for (int i = 0; i < num_data_; ++i) {
        score_[i + treatment_id * num_data_] += val;
      }
    }
  }

  /*!
   * \brief Using tree model to get prediction number, then adding to scores for all data
   *        Note: this function generally will be used on validation data too.
   * \param tree Trained tree model
   * \param cur_tree_id Current tree for multiclass training
   */
  virtual inline void AddScore(const Tree* tree, int cur_tree_id) {
    const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id * num_treat_;
    tree->AddPredictionToScore(data_, num_data_, score_.data());
  }

  /*!
   * \brief Using tree model to get prediction number, then adding to scores for parts of data
   *        Used for prediction of training out-of-bag data
   * \param tree Trained tree model
   * \param data_indices Indices of data that will be processed
   * \param data_cnt Number of data that will be processed
   * \param cur_tree_id Current tree for multiclass training
   */
  virtual inline void AddScore(const Tree* tree, const data_size_t* data_indices, data_size_t data_cnt, int cur_tree_id) {
    const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id * num_treat_;
    tree->AddPredictionToScore(data_, data_indices, data_cnt, score_.data() + offset);
  }

  virtual inline void MultiplyScore(double val) {
#pragma omp parallel for schedule(static, 512) if (num_data_ * num_treat_ >= 1024)
    for (int i = 0; i < num_data_ * num_treat_; ++i) {
      score_[i] *= val;
    }
  }

 private:
  int num_data_;
  int num_treat_;
  const Dataset* data_;
  std::vector<double> score_;
};

}  // namespace UTBoost


#endif //UTBOOST_SRC_ENSEMBLE_SCORE_UPDATER_H_
