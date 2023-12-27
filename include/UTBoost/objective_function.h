/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 * Created by Junjie Gao on 2023/3/1.
 */

#ifndef UTBOOST_INCLUDE_UTBOOST_OBJECTIVE_FUNCTION_H_
#define UTBOOST_INCLUDE_UTBOOST_OBJECTIVE_FUNCTION_H_

#include "UTBoost/config.h"
#include "UTBoost/dataset.h"
#include "UTBoost/definition.h"
#include "UTBoost/utils/log_wrapper.h"

#include <string>
#include <functional>

namespace UTBoost {

/*! \brief The interface of Objective Function. */
class UTBOOST_EXPORT ObjectiveFunction {
 public:
  /*! \brief virtual destructor */
  virtual ~ObjectiveFunction() {}

  /*!
   * \brief Initialize
   * \param metadata Label data
   * \param num_data Number of data
   */
  virtual void Init(const MetaInfo &meta, data_size_t num_data) = 0;

  /*!
   * \brief calculating first order derivative of loss function
   * \param score prediction score in this round
   * \gradients Output gradients
   * \hessians Output hessians
   */
  virtual void GetGradients(const double *score, score_t *gradients, score_t *hessians) const = 0;

  virtual const char *GetName() const = 0;

  virtual double BoostFromScore(treatment_t treatment_id) const { return 0.0; }

  /*! \brief Return the number of positive samples. Return 0 if no binary classification tasks.*/
  virtual data_size_t NumPositiveData() const { return 0; }

  virtual void ConvertOutput(const double *input, double *output, int len) const {
    for (int i = 0; i < len; ++i) {
      output[i] = input[i];
    }
  }

  virtual void ConvertOutput(const double input, double *output) const {
    *output = input;
  }

  virtual std::string ToString() const = 0;

  ObjectiveFunction() = default;
  /*! \brief Disable copy */
  ObjectiveFunction &operator=(const ObjectiveFunction &) = delete;
  /*! \brief Disable copy */
  ObjectiveFunction(const ObjectiveFunction &) = delete;

  /*!
   * \brief Create object of objective function
   * \param type Specific type of objective function
   * \param config Config for objective function
   */
  static ObjectiveFunction *CreateObjectiveFunction(const std::string &type, const Config &config);
};

}  // namespace UTBoost

#endif //UTBOOST_INCLUDE_UTBOOST_OBJECTIVE_FUNCTION_H_
