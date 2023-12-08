/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#ifndef UTBOOST_SRC_METRIC_REGRESSION_METRIC_H_
#define UTBOOST_SRC_METRIC_REGRESSION_METRIC_H_

#include <UTBoost/metric.h>
#include <UTBoost/utils/common.h>
#include <UTBoost/utils/log_wrapper.h>

#include <string>
#include <algorithm>
#include <sstream>
#include <vector>

namespace UTBoost {

/*!
 * \brief Metric for regression task.
 * Use static class "PointWiseLossCalculator" to calculate loss point-wise
 */
template<typename PointWiseLossCalculator>
class RegressionMetric: public Metric {
 public:
  explicit RegressionMetric(const Config& config) :config_(config) {
  }

  virtual ~RegressionMetric() {
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return -1.0f;
  }

  void Init(const MetaInfo& metadata, data_size_t num_data) override {
    name_.emplace_back(PointWiseLossCalculator::Name());
    num_data_ = num_data;
    // get label
    label_ = metadata.GetLabel();
    // get weights
    weights_ = metadata.GetWeight();
    treats_ = metadata.GetTreatment();
    if (weights_ == nullptr) {
      sum_weights_ = static_cast<double>(num_data_);
    } else {
      sum_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_data_; ++i) {
        sum_weights_ += weights_[i];
      }
    }
    for (data_size_t i = 0; i < num_data_; ++i) {
      PointWiseLossCalculator::CheckLabel(label_[i]);
    }

    real_score_ = [this](data_size_t i, const double* score) {
      return treats_[i] > 0 ? score[i] + score[i + num_data_ * treats_[i]] : score[i];
    };
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override {
    double sum_loss = 0.0f;
    if (objective == nullptr) {
      if (weights_ == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], real_score_(i, score), config_);
        }
      } else {
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], real_score_(i, score), config_) * weights_[i];
        }
      }
    } else {
      if (weights_ == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          // add loss
          double t = 0;
          objective->ConvertOutput(real_score_(i, score), &t);
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], t, config_);
        }
      } else {
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          // add loss
          double t = 0;
          objective->ConvertOutput(real_score_(i, score), &t);
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], t, config_) * weights_[i];
        }
      }
    }
    double loss = PointWiseLossCalculator::AverageLoss(sum_loss, sum_weights_);
    return std::vector<double>(1, loss);
  }

  inline static double AverageLoss(double sum_loss, double sum_weights) {
    return sum_loss / sum_weights;
  }

  inline static void CheckLabel(label_t) {
  }

 protected:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Pointer of weighs */
  const label_t* weights_;
  const treatment_t* treats_;
  /*! \brief Sum weights */
  double sum_weights_;
  /*! \brief Name of this test set */
  Config config_;
  std::vector<std::string> name_;
  std::function<double(data_size_t, const double*)> real_score_;
};

/*! \brief RMSE loss for regression task */
class RMSEMetric: public RegressionMetric<RMSEMetric> {
 public:
  explicit RMSEMetric(const Config& config) :RegressionMetric<RMSEMetric>(config) {}

  inline static double LossOnPoint(label_t label, double score, const Config&) {
    return (score - label)*(score - label);
  }

  inline static double AverageLoss(double sum_loss, double sum_weights) {
    // need sqrt the result for RMSE loss
    return std::sqrt(sum_loss / sum_weights);
  }

  inline static const char* Name() {
    return "rmse";
  }
};

/*! \brief L2 loss for regression task */
class L2Metric: public RegressionMetric<L2Metric> {
 public:
  explicit L2Metric(const Config& config) :RegressionMetric<L2Metric>(config) {}

  inline static double LossOnPoint(label_t label, double score, const Config&) {
    return (score - label)*(score - label);
  }

  inline static const char* Name() {
    return "l2";
  }
};

}

#endif //UTBOOST_SRC_METRIC_REGRESSION_METRIC_H_
