/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 * Created by Junjie Gao on 2023/4/22.
 */

#include "UTBoost/utils/text_reader.h"
#include "UTBoost/ensemble_model.h"
#include "causal_gbm.h"
#include "causal_rf.h"

namespace UTBoost {

bool EnsembleModel::LoadFileToBoosting(EnsembleModel* boosting, const char* filename) {
  auto start_time = std::chrono::steady_clock::now();
  if (boosting != nullptr) {
    TextReader<size_t> model_reader(filename, true);
    size_t buffer_len = 0;
    auto buffer = model_reader.ReadContent(&buffer_len);
    if (!boosting->LoadModelFromString(buffer.data(), buffer_len)) {
      return false;
    }
  }
  std::chrono::duration<double, std::milli> delta = (std::chrono::steady_clock::now() - start_time);
  Log::Debug("Time for loading model: %f seconds", 1e-3*delta);
  return true;
}

EnsembleModel *EnsembleModel::CreateEnsembleModel(const std::string &ensemble_method, const char* filename) {
  if (filename == nullptr || filename[0] == '\0') {
    if (ensemble_method == std::string("boost")) {
      return new CausalGBM();
    } else if (ensemble_method == std::string("rf")) {
      return new CausalRF();
    } else {
      Log::Error("Unknown ensemble method: %s", ensemble_method.c_str());
    }
  } else {
    std::unique_ptr<EnsembleModel> ret;
    if (ensemble_method == std::string("boost")) {
      ret.reset(new CausalGBM());
    } else if (ensemble_method == std::string("rf")) {
      ret.reset(new CausalRF());
    } else {
      Log::Error("Unknown ensemble method: %s", ensemble_method.c_str());
    }
    ret.reset(new CausalGBM());
    LoadFileToBoosting(ret.get(), filename);
    return ret.release();
  }
  return nullptr;
}

}  // namespace UTBoost
