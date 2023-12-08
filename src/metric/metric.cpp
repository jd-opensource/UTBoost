/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#include "UTBoost/metric.h"

#include "binary_metric.h"
#include "qini_metric.h"
#include "regression_metric.h"

namespace UTBoost {

Metric *Metric::CreateMetric(const std::string &type, const Config &config) {
  if (type == std::string("logloss")) {
    return new BinaryLoglossMetric(config);
  } else if (type == std::string("auc")) {
    return new AUCMetric(config);
  } else if (type == std::string("qini_area")) {
    return new QiniMetric(config);
  } else if (type == std::string("qini_coff")) {
    return new NormalQiniMetric(config);
  } else if (type == std::string("l2")) {
    return new L2Metric(config);
  } else if (type == std::string("rmse")) {
    return new RMSEMetric(config);
  }
  return nullptr;
}

}  // namespace UTBoost

