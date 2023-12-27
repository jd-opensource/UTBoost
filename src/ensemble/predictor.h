/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#ifndef UTBOOST_SRC_ENSEMBLE_PREDICTOR_H_
#define UTBOOST_SRC_ENSEMBLE_PREDICTOR_H_

#include <UTBoost/ensemble_model.h>
#include <UTBoost/dataset.h>
#include <UTBoost/definition.h>
#include <UTBoost/utils/common.h>
#include <UTBoost/utils/omp_wrapper.h>

#include <string>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace UTBoost {

/*!
 * \brief Used to predict data with input model
 */
class Predictor {
 public:
  /*!
   * \brief Constructor
   * \param boosting Input boosting model
   * \param start_iteration Start index of the iteration to predict
   * \param num_iteration Number of boosting round
   */
  Predictor(EnsembleModel* boosting, int start_iteration, int num_iteration) {
    boosting->InitPredict(start_iteration, num_iteration);
    boosting_ = boosting;
    num_pred_one_row_ = boosting_->NumPredictOneRow(start_iteration, num_iteration, false);
    num_feature_ = boosting_->MaxFeatureIdx() + 1;
    predict_buf_.resize(
        OMP_GET_NUM_THREADS(),
        std::vector<double>(num_feature_, 0.0f)
            );
    predict_fun_ = [=](const std::vector<std::pair<int, double>>& features, double* output) {
      int tid = OMP_GET_THREAD_NUM();
      CopyToPredictBuffer(predict_buf_[tid].data(), features);
      boosting_->Predict(predict_buf_[tid].data(), output);
      ClearPredictBuffer(predict_buf_[tid].data(),
                         predict_buf_[tid].size(), features);
    };
  }

  /*! \brief Destructor */
  ~Predictor() {
  }

  inline const PredictFunction& GetPredictFunction() const {
    return predict_fun_;
  }

 private:
  void CopyToPredictBuffer(double* pred_buf, const std::vector<std::pair<int, double>>& features) {
    for (const auto &feature : features) {
      if (feature.first < num_feature_) {
        pred_buf[feature.first] = feature.second;
      }
    }
  }

  void ClearPredictBuffer(double* pred_buf, size_t buf_size, const std::vector<std::pair<int, double>>& features) {
    if (features.size() > static_cast<size_t>(buf_size / 2)) {
      std::memset(pred_buf, 0, sizeof(double)*(buf_size));
    } else {
      for (const auto &feature : features) {
        if (feature.first < num_feature_) {
          pred_buf[feature.first] = 0.0f;
        }
      }
    }
  }

  /*! \brief Boosting model */
  const EnsembleModel* boosting_;
  /*! \brief function for prediction */
  PredictFunction predict_fun_;
  int num_feature_;
  int num_pred_one_row_;
  std::vector<std::vector<double>> predict_buf_;
};

}  // namespace UTBoost

#endif //UTBOOST_SRC_ENSEMBLE_PREDICTOR_H_
