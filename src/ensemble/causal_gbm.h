/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 * Created by Junjie Gao on 2023/4/26.
 */

#ifndef UTBOOST_SRC_ENSEMBLE_TREE_BOOSTING_H_
#define UTBOOST_SRC_ENSEMBLE_TREE_BOOSTING_H_

#include "UTBoost/ensemble_model.h"
#include "UTBoost/tree_learner.h"
#include "UTBoost/sample_strategy.h"
#include "score_updater.h"

namespace UTBoost {

class CausalGBM : public EnsembleModel {

 public:

  CausalGBM();
  /*!
   * \brief Initialization logic
   * \param gbdt_config Config for boosting
   * \param train_data Training data
   * \param objective_function Training objective function
   * \param training_metrics Training metrics
   */
  void Init(const Config* config, const Dataset* train_data, const ObjectiveFunction* objective_function,
            const std::vector<const Metric*>& training_metrics) override;

  bool TrainOneIter(const score_t* gradients, const score_t* hessians) override;

  /*!
   * \brief Rollback one iteration
   */
  void RollbackOneIter() override;
  double BoostFromAverage(treatment_t treatment_id, bool update_scorer);

  /*!
   * \brief updating score after tree was trained
   * \param tree Trained tree of this iteration
   * \param cur_tree_id Current tree for multiclass training
   */
  virtual void UpdateScore(const Tree* tree, const int cur_tree_id);

  /*!
   * \brief Adding a validation dataset
   * \param valid_data Validation dataset
   * \param valid_metrics Metrics for validation dataset
   */
  void AddValidDataset(const Dataset* valid_data, const std::vector<const Metric*>& valid_metrics) override;

  inline void InitPredict(int start_iteration, int num_iteration) override {
    num_iteration_for_pred_ = static_cast<int>(models_.size()) ;
    start_iteration = std::max(start_iteration, 0);
    start_iteration = std::min(start_iteration, num_iteration_for_pred_);
    if (num_iteration > 0) {
      num_iteration_for_pred_ = std::min(num_iteration, num_iteration_for_pred_ - start_iteration);
    } else {
      num_iteration_for_pred_ = num_iteration_for_pred_ - start_iteration;
    }
    start_iteration_for_pred_ = start_iteration;
  }

  /*!
   * \brief eval results for one metric
   */
  virtual std::vector<double> EvalOneMetric(const Metric* metric, const double* score, const data_size_t num_data) const;

  /*!
   * \brief Get evaluation result at data_idx data
   * \param data_idx 0: training data, 1: 1st validation data
   * \return evaluation result
   */
  std::vector<double> GetEvalAt(int data_idx) const override;

  /*!
   * \brief Save model to file
   * \param start_iteration The model will be saved start from
   * \param num_iterations Number of model that want to save, -1 means save all
   * \param feature_importance_type Type of feature importance, 0: split, 1: gain
   * \param filename Filename that want to save to
   * \return is_finish Is training finished or not
   */
  bool SaveModelToFile(int start_iteration, int num_iterations,
                       int feature_importance_type,
                       const char* filename) const override;

  /*!
   * \brief Dump model to json format string
   * \param start_iteration The model will be saved start from
   * \param num_iteration Number of iterations that want to dump, -1 means dump all
   * \param feature_importance_type Type of feature importance, 0: split, 1: gain
   * \return Json format string of model
   */
  std::string DumpModel(int start_iteration, int num_iteration) const override;

  /*!
   * \brief Dump model to json format string
   * \param start_iteration The model will be saved start from
   * \param num_iteration Number of iterations that want to dump, -1 means dump all
   * \param feature_importance_type Type of feature importance, 0: split, 1: gain
   * \return Json format string of model
   */
  bool DumpModelToFile(int start_iteration, int num_iteration, const char* filename) const override;

  /*!
   * \brief Get Type name of this boosting object
   */
  const char* SubModelName() const override { return "BoostingTree"; }

  /*!
   * \brief Save model to string
   * \param start_iteration The model will be saved start from
   * \param num_iterations Number of model that want to save, -1 means save all
   * \param feature_importance_type Type of feature importance, 0: split, 1: gain
   * \return Non-empty string if succeeded
   */
  std::string SaveModelToString(int start_iteration, int num_iterations, int feature_importance_type) const override;

  /*!
   * \brief Restore from a serialized buffer
   */
  bool LoadModelFromString(const char* buffer, size_t len) override;

  /*!
   * \brief Calculate feature importances
   * \param num_iteration Number of model that want to use for feature importance, -1 means use all
   * \param importance_type: 0 for split, 1 for gain
   * \return vector of feature_importance
   */
  std::vector<double> FeatureImportance(int num_iteration, int importance_type) const override;


  /*!
   * \brief Get number of prediction for one data
   * \param start_iteration Start index of the iteration to predict
   * \param num_iteration number of used iterations
   * \param is_pred_leaf True if predicting leaf index
   * \param is_pred_contrib True if predicting feature contribution
   * \return number of prediction
   */
  inline int NumPredictOneRow(int start_iteration, int num_iteration, bool is_pred_leaf) const override {
    int num_pred_in_one_row = num_treat_;
    if (is_pred_leaf) {
      int max_iteration = static_cast<int>(models_.size());
      start_iteration = std::max(start_iteration, 0);
      start_iteration = std::min(start_iteration, max_iteration);
      if (num_iteration > 0) {
        num_pred_in_one_row *= static_cast<int>(std::min(max_iteration - start_iteration, num_iteration));
      } else {
        num_pred_in_one_row *= (max_iteration - start_iteration);
      }
    }
    return num_pred_in_one_row;
  }

  int GetNumTreatment() const override {
    return num_treat_;
  }

  /*!
   * \brief Get max feature index of this model
   * \return Max feature index of this model
   */
  inline int MaxFeatureIdx() const override { return max_feature_idx_; }

  void PredictRaw(const double* features, double* output) const override;

  void Predict(const double* features, double* output) const override;

  /*!
   * \brief Get current iteration
   */
  int GetCurrentIteration() const override { return static_cast<int>(models_.size()); }

 protected:
  /*!
   * \brief calculate the objective function
   */
  virtual void Boosting();
  /*!
   * \brief Get current training score
   * \param out_len length of returned score
   * \return training score
   */
  const double* GetTrainingScore(int64_t* out_len) override;

  /*!
   * \brief Reset gradient buffers, must be called after sample strategy is reset
   */
  void ResetGradientBuffers();

  int iter_;
  /*! \brief Pointer to training data */
  const Dataset* train_data_;
  /*! \brief Config of gbdt */
  std::unique_ptr<Config> config_;
  /*! \brief Tree learner, will use this class to learn trees */
  std::unique_ptr<TreeLearner> tree_learner_;
  /*! \brief Objective function */
  const ObjectiveFunction* objective_function_;
  std::unique_ptr<SplitCriteria> split_criteria_;
  /*! \brief Store and update training data's score */
  std::unique_ptr<ScoreUpdater> train_score_updater_;
  /*! \brief Metrics for training data */
  std::vector<const Metric*> training_metrics_;
  /*! \brief Store and update validation data's scores */
  std::vector<std::unique_ptr<ScoreUpdater>> valid_score_updater_;
  /*! \brief Metric for validation data */
  std::vector<std::vector<const Metric*>> valid_metrics_;
  /*! \brief Trained models(trees) */
  std::vector<std::unique_ptr<Tree>> models_;
  /*! \brief Max feature index of training data*/
  int max_feature_idx_;
  /*! \brief First order derivative of training data */
  std::vector<score_t> gradients_;
  /*! \brief Second order derivative of training data */
  std::vector<score_t> hessians_;
  /*! \brief Pointer to gradient vector, can be on CPU or GPU */
  score_t* gradients_pointer_;
  /*! \brief Pointer to hessian vector, can be on CPU or GPU */
  score_t* hessians_pointer_;
  /*! \brief Number of training data */
  data_size_t num_data_;
  /*! \brief Number of class */
  int num_treat_;
  /*! \brief number of used model */
  int num_iteration_for_pred_;
  /*! \brief Start iteration of used model */
  int start_iteration_for_pred_;
  /*! \brief Shrinkage rate for one iteration */
  double shrinkage_rate_;
  std::unique_ptr<ObjectiveFunction> loaded_objective_;
  bool average_output_;
  std::unique_ptr<SampleStrategy> data_sample_strategy_;
};

}  // namespace UTBoost

#endif //UTBOOST_SRC_ENSEMBLE_TREE_BOOSTING_H_
