/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#include "UTBoost/objective_function.h"
#include "binary_objective.h"
#include "regression_objective.h"
#include "default_objective.h"

namespace UTBoost {

ObjectiveFunction* ObjectiveFunction::CreateObjectiveFunction(const std::string& type, const Config& config) {
  if (type == std::string("mse")) {
    return new RegressionL2loss(config);
  } else if (type == std::string("logloss")) {
    return new BinaryLogloss(config);
  } else if (type == std::string("default")) {
    return new DefaultLoss(config);
  }
  Log::Error("Unknown objective function: %s", type.c_str());
  return nullptr;
}

}  // namespace UTBoost

