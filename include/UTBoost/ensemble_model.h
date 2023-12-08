/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 * Created by Junjie Gao on 2023/3/1.
 */

#ifndef UTBOOST_INCLUDE_UTBOOST_ENSEMBLE_H_
#define UTBOOST_INCLUDE_UTBOOST_ENSEMBLE_H_

#include "UTBoost/dataset.h"
#include "UTBoost/tree_learner.h"
#include "UTBoost/definition.h"
#include "UTBoost/metric.h"
#include "UTBoost/split_criteria.h"
#include "UTBoost/config.h"
#include "UTBoost/tree.h"
#include "UTBoost/objective_function.h"
#include "UTBoost/definition.h"

namespace UTBoost {

/*! \brief The interface for ensemble model */
class UTBOOST_EXPORT EnsembleModel {
 public:
  EnsembleModel() = default;
  /*! \brief virtual destructor */
  virtual ~EnsembleModel() {};

  /*!
   * \brief Create boosting object
   * \param ensemble_method Type of ensemble model
   * \param filename name of model file, if existing will continue to train from this model
   * \return The boosting object
   */
  static EnsembleModel* CreateEnsembleModel(const std::string& ensemble_method, const char* filename);

  /*!
   * \brief Initialize model
   * \param config Configs for boosting
   * \param train_data Training data
   * \param objective_function Training objective function
   * \param training_metrics Training metric
   */
  virtual void Init(const Config* config, const Dataset* train_data, const ObjectiveFunction* objective_function,
                    const std::vector<const Metric*>& training_metrics) = 0;
  /*!
   * \brief Training logic
   * \param gradients nullptr for using default objective, otherwise use self-defined boosting
   * \param hessians nullptr for using default objective, otherwise use self-defined boosting
   * \return True if cannot train anymore
   */
  virtual bool TrainOneIter(const score_t* gradients, const score_t* hessians) = 0;

  /*!
   * \brief Rollback one iteration
   */
  virtual void RollbackOneIter() = 0;

  /*!
   * \brief Get current training score
   * \param out_len length of returned score
   * \return training score
   */
  virtual const double* GetTrainingScore(int64_t* out_len) = 0;

  virtual int GetNumTreatment() const = 0;

  /*!
   * \brief Get evaluation result at data_idx data
   * \param data_idx 0: training data, 1: 1st validation data
   * \return evaluation result
   */
  virtual std::vector<double> GetEvalAt(int data_idx) const = 0;

  virtual int NumPredictOneRow(int start_iteration, int num_iteration, bool is_pred_leaf) const = 0;

  /*!
   * \brief Get max feature index of this model
   * \return Max feature index of this model
   */
  virtual int MaxFeatureIdx() const = 0;

  /*! \brief Name of submodel */
  virtual const char* SubModelName() const = 0;

  /*!
   * \brief Initial work for the prediction
   * \param start_iteration Start index of the iteration to predict
   * \param num_iteration number of used iteration
   */
  virtual void InitPredict(int start_iteration, int num_iteration) = 0;

  /*!
   * \brief Prediction for one record, sigmoid transformation will be used if needed
   * \param feature_values Feature value on this record
   * \param output Prediction result for this record
   */
  virtual void Predict(const double* features, double* output) const = 0;

  /*!
   * \brief Prediction for one record, not sigmoid transform
   * \param feature_values Feature value on this record
   * \param output Prediction result for this record
   */
  virtual void PredictRaw(const double* features, double* output) const =0;

  /*!
   * \brief Add a validation data
   * \param valid_data Validation data
   * \param valid_metrics Metric for validation data
   */
  virtual void AddValidDataset(const Dataset* valid_data,
                               const std::vector<const Metric*>& valid_metrics) = 0;

  /*!
   * \brief Save model to file
   * \param start_iteration The model will be saved start from
   * \param num_iterations Number of model that want to save, -1 means save all
   * \param feature_importance_type Type of feature importance, 0: split, 1: gain
   * \param filename Filename that want to save to
   * \return true if succeeded
   */
  virtual bool SaveModelToFile(int start_iteration, int num_iterations, int feature_importance_type, const char* filename) const = 0;

  /*!
   * \brief Dump model to json format string
   * \param start_iteration The model will be saved start from
   * \param num_iteration Number of iterations that want to dump, -1 means dump all
   * \return Json format string of model
   */
  virtual std::string DumpModel(int start_iteration, int num_iteration) const = 0;

  /*!
   * \brief Dump model to file
   * \param start_iteration The model will be saved start from
   * \param num_iterations Number of model that want to save, -1 means save all
   * \param filename Filename that want to save to
   * \return true if succeeded
   */
  virtual bool DumpModelToFile(int start_iteration, int num_iteration, const char* filename) const = 0;

  /*!
   * \brief Save model to string
   * \param start_iteration The model will be saved start from
   * \param num_iterations Number of model that want to save, -1 means save all
   * \param feature_importance_type Type of feature importance, 0: split, 1: gain
   * \return Non-empty string if succeeded
   */
  virtual std::string SaveModelToString(int start_iteration, int num_iterations, int feature_importance_type) const = 0;

  /*!
   * \brief Restore from a serialized string
   * \param buffer The content of model
   * \param len The length of buffer
   * \return true if succeeded
   */
  virtual bool LoadModelFromString(const char* buffer, size_t len) = 0;

  /*!
   * \brief Calculate feature importances
   * \param num_iteration Number of model that want to use for feature importance, -1 means use all
   * \param importance_type: 0 for split, 1 for gain
   * \return vector of feature_importance
   */
  virtual std::vector<double> FeatureImportance(int num_iteration, int importance_type) const = 0;
  static bool LoadFileToBoosting(EnsembleModel* boosting, const char* filename);

  /*! \brief return current iteration */
  virtual int GetCurrentIteration() const = 0;
};

}  // namespace UTBoost


#endif //UTBOOST_INCLUDE_UTBOOST_ENSEMBLE_H_
