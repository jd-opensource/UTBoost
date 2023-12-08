/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#include "UTBoost/split_criteria.h"
#include "ddp_criteria.h"
#include "gbm_criteria.h"
#include "divergence_criteria.h"

namespace UTBoost {

SplitCriteria *SplitCriteria::Create(const std::string &type, const Config* config) {
  if (type == std::string("gbm")) {
    return new GBM(config);
  } else if (type == std::string("ddp")) {
    return new DDP(config);
  } else if (type == std::string("ed")) {
    if (config->ensemble != std::string("rf"))
      Log::Error("split criterion %s is available only when the ensemble method is rf", type.c_str());
    return new ED(config);
  } else if (type == std::string("kl")) {
    if (config->ensemble != std::string("rf"))
      Log::Error("split criterion %s is available only when the ensemble method is rf", type.c_str());
    return new KL(config);
  } else if (type == std::string("chi")) {
    if (config->ensemble != std::string("rf"))
      Log::Error("split criterion %s is available only when the ensemble method is rf", type.c_str());
    return new Chi(config);
  }
  Log::Error("Unknown split criterion: %s", type.c_str());
  return nullptr;
}

}  // namespace UTBoost