/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#include "UTBoost/sample_strategy.h"
#include "bagging.h"

namespace UTBoost {

SampleStrategy* SampleStrategy::CreateSampleStrategy(
    const Config* config,
    const Dataset* train_data,
    const ObjectiveFunction* objective_function) {
  return new BaggingSampleStrategy(config, train_data, objective_function);
}

}  // namespace UTBoost
