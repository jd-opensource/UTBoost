/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#include "UTBoost/tree_learner.h"
#include "serial_tree_learner.h"

namespace UTBoost {

TreeLearner *TreeLearner::CreateTreeLearner(const Config *config) {
  return new SerialTreeLearner(config);
}

}  // namespace UTBoost
