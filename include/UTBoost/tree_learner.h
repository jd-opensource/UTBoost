/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef UTBOOST_INCLUDE_UTBOOST_LEARNER_H_
#define UTBOOST_INCLUDE_UTBOOST_LEARNER_H_

#include "UTBoost/dataset.h"
#include "UTBoost/definition.h"
#include "UTBoost/tree.h"
#include "UTBoost/config.h"
#include "objective_function.h"
#include "UTBoost/split_criteria.h"

namespace UTBoost {

/*! \brief Interface for tree learner */
class TreeLearner {
 public:
  /*! \brief virtual destructor */
  virtual ~TreeLearner() = default;

  /*!
   * \brief Initialize tree learner with training dataset
   * \param train_data The used training data
   * \param is_constant_hessian True if all hessians share the same value
   */
  virtual void Init(const Dataset* train_data, bool is_constant_hessian) = 0;

  /*!
   * \brief training tree model on dataset
   * \param gradients The first order gradients
   * \param hessians The second order gradients
   * \param is_first_tree If linear tree learning is enabled, first tree needs to be handled differently
   * \return A trained tree
   */
  virtual Tree* Train(const score_t* gradients, const score_t* hessians, bool is_first_tree,
                      const SplitCriteria* split_criteria) = 0;

  /*!
   * \brief Set bagging data
   * \param subset subset of bagging
   * \param used_indices Used data indices
   * \param num_data Number of used data
   */
  virtual void SetBaggingData(const Dataset* subset, const data_size_t* used_indices, data_size_t num_data) = 0;

  /*!
   * \brief Using last trained tree to predict score then adding to out_score;
   * \param out_score output score
   */
  virtual void AddPredictionToScore(const Tree* tree, double* out_score) const = 0;

  /*! \brief Renew tree output (value) by other data, used for honesty estimation. */
  virtual void RenewTreeOutputByIndices(Tree* tree, const SplitCriteria* split_criteria,
                                        const data_size_t* bag_indices, data_size_t bag_cnt,
                                        const score_t* gradients, const score_t* hessians) const = 0;

  /*! \brief default constructor */
  TreeLearner() = default;
  /*! \brief Disable copy */
  TreeLearner& operator=(const TreeLearner&) = delete;
  /*! \brief Disable copy */
  TreeLearner(const TreeLearner&) = delete;

  /*!
   * \brief Create object of tree learner
   * \param config config of tree
   */
  static TreeLearner* CreateTreeLearner(const Config* config);

};

}  // namespace UTBoost

#endif //UTBOOST_INCLUDE_UTBOOST_LEARNER_H_
