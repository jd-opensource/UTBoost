/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 * Created by Junjie Gao on 2023/3/1.
 */

#ifndef UTBOOST_INCLUDE_UTBOOST_METRIC_H_
#define UTBOOST_INCLUDE_UTBOOST_METRIC_H_

#include "UTBoost/dataset.h"
#include "UTBoost/definition.h"
#include "UTBoost/objective_function.h"
#include "UTBoost/utils/log_wrapper.h"
#include "UTBoost/config.h"

#include "vector"
#include "string"

namespace UTBoost {

/*! \brief The interface of metric. */
class UTBOOST_EXPORT Metric {
 public:
  /*! \brief virtual destructor */
  virtual ~Metric() {}

  /*!
   * \brief Initialize
   * \param test_name Specific name for this metric, will output on log
   * \param meta Label data
   * \param num_data Number of data
   */
  virtual void Init(const MetaInfo& meta, data_size_t num_data) = 0;

  virtual const std::vector<std::string>& GetName() const = 0;

  virtual double factor_to_bigger_better() const = 0;
  /*!
   * \brief Calculating and printing metric result
   * \param score Current prediction score
   */
  virtual std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const = 0;

  Metric() = default;
  /*! \brief Disable copy */
  Metric& operator=(const Metric&) = delete;
  /*! \brief Disable copy */
  Metric(const Metric&) = delete;

  /*!
   * \brief Create object of metrics
   * \param type Specific type of metric
   * \param config Config for metric
   */
  static Metric* CreateMetric(const std::string& type, const Config& config);
};


}

#endif //UTBOOST_INCLUDE_UTBOOST_METRIC_H_
