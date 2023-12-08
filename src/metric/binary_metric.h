/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#ifndef UTBOOST_SRC_METRIC_BINARY_METRIC_H_
#define UTBOOST_SRC_METRIC_BINARY_METRIC_H_

#include <UTBoost/metric.h>
#include <UTBoost/utils/common.h>
#include <UTBoost/utils/log_wrapper.h>

#include <string>
#include <algorithm>
#include <sstream>
#include <vector>

namespace UTBoost {

/*!
 * \brief Metric for binary classification task.
 * Use static class "PointWiseLossCalculator" to calculate loss point-wise
 */
template<typename PointWiseLossCalculator>
class BinaryMetric: public Metric {
 public:
  explicit BinaryMetric(const Config&) {
  }

  virtual ~BinaryMetric() {
  }

  void Init(const MetaInfo& metadata, data_size_t num_data) override {
    name_.emplace_back(PointWiseLossCalculator::Name());
    num_data_ = num_data;
    // get label
    label_ = metadata.GetLabel();
    // get weights
    weights_ = metadata.GetWeight();
    treats_ = metadata.GetTreatment();
    real_score_ = [this](data_size_t i, const double* score) {
      return treats_[i] > 0 ? score[i] + score[i + num_data_ * treats_[i]] : score[i];
    };
    if (weights_ == nullptr) {
      sum_weights_ = static_cast<double>(num_data_);
    } else {
      sum_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_data; ++i) {
        sum_weights_ += weights_[i];
      }
    }
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return -1.0f;
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override {
    double sum_loss = 0.0f;
    if (objective == nullptr) {
      if (weights_ == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], real_score_(i, score));
        }
      } else {
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], real_score_(i, score)) * weights_[i];
        }
      }
    } else {
      if (weights_ == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          double prob = 0;
          objective->ConvertOutput(real_score_(i, score), &prob);
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], prob);
        }
      } else {
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          double prob = 0;
          objective->ConvertOutput(real_score_(i, score), &prob);
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], prob) * weights_[i];
        }
      }
    }
    double loss = sum_loss / sum_weights_;
    return std::vector<double>(1, loss);
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
  std::function<double(data_size_t, const double*)> real_score_;
  /*! \brief Name of test set */
  std::vector<std::string> name_;
};

/*!
 * \brief Log loss metric for binary classification task.
 */
class BinaryLoglossMetric: public BinaryMetric<BinaryLoglossMetric> {
 public:
  explicit BinaryLoglossMetric(const Config& config) :BinaryMetric<BinaryLoglossMetric>(config) {}

  inline static double LossOnPoint(label_t label, double prob) {
    if (label <= 0) {
      if (1.0f - prob > kEpsilon) {
        return -std::log(1.0f - prob);
      }
    } else {
      if (prob > kEpsilon) {
        return -std::log(prob);
      }
    }
    return -std::log(kEpsilon);
  }

  inline static const char* Name() {
    return "binary_logloss";
  }
};

/*! \brief Auc Metric for binary classification task. */
class AUCMetric: public Metric {
 public:
  explicit AUCMetric(const Config&) {
  }

  virtual ~AUCMetric() {
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return 1.0f;
  }

  void Init(const MetaInfo& metadata, data_size_t num_data) override {
    name_.emplace_back("auc");
    num_data_ = num_data;
    // get label
    label_ = metadata.GetLabel();
    // get weights
    weights_ = metadata.GetWeight();
    treats_ = metadata.GetTreatment();
    real_score_ = [this](data_size_t i, const double* score) {
      return treats_[i] > 0 ? score[i] + score[i + num_data_ * treats_[i]] : score[i];
    };

    if (weights_ == nullptr) {
      sum_weights_ = static_cast<double>(num_data_);
    } else {
      sum_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_data; ++i) {
        sum_weights_ += weights_[i];
      }
    }
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction*) const override {
    // get indices sorted by score, descent order
    std::vector<data_size_t> sorted_idx;
    sorted_idx.reserve(num_data_);
    std::vector<double> real_score;
    real_score.reserve(num_data_);
    for (data_size_t i = 0; i < num_data_; ++i) {
      sorted_idx.emplace_back(i);
      real_score.emplace_back(real_score_(i, score));
    }
    score = real_score.data();
    ParallelSort(sorted_idx.begin(), sorted_idx.end(), [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // temp sum of positive label
    double cur_pos = 0.0f;
    // total sum of positive label
    double sum_pos = 0.0f;
    // accumulate of AUC
    double accum = 0.0f;
    // temp sum of negative label
    double cur_neg = 0.0f;
    double threshold = score[sorted_idx[0]];
    if (weights_ == nullptr) {  // no weights
      for (data_size_t i = 0; i < num_data_; ++i) {
        const label_t cur_label = label_[sorted_idx[i]];
        const double cur_score = score[sorted_idx[i]];
        // new threshold
        if (cur_score != threshold) {
          threshold = cur_score;
          // accumulate
          accum += cur_neg*(cur_pos * 0.5f + sum_pos);
          sum_pos += cur_pos;
          // reset
          cur_neg = cur_pos = 0.0f;
        }
        cur_neg += (cur_label <= 0);
        cur_pos += (cur_label > 0);
      }
    } else {  // has weights
      for (data_size_t i = 0; i < num_data_; ++i) {
        const label_t cur_label = label_[sorted_idx[i]];
        const double cur_score = score[sorted_idx[i]];
        const label_t cur_weight = weights_[sorted_idx[i]];
        // new threshold
        if (cur_score != threshold) {
          threshold = cur_score;
          // accumulate
          accum += cur_neg*(cur_pos * 0.5f + sum_pos);
          sum_pos += cur_pos;
          // reset
          cur_neg = cur_pos = 0.0f;
        }
        cur_neg += (cur_label <= 0)*cur_weight;
        cur_pos += (cur_label > 0)*cur_weight;
      }
    }
    accum += cur_neg*(cur_pos * 0.5f + sum_pos);
    sum_pos += cur_pos;
    double auc = 1.0f;
    if (sum_pos > 0.0f && sum_pos != sum_weights_) {
      auc = accum / (sum_pos *(sum_weights_ - sum_pos));
    }
    return std::vector<double>(1, auc);
  }

 private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Pointer of weighs */
  const label_t* weights_;
  const treatment_t* treats_;
  /*! \brief Sum weights */
  double sum_weights_;
  std::function<double(data_size_t, const double*)> real_score_;
  /*! \brief Name of test set */
  std::vector<std::string> name_;
};

}  // namespace UTBoost


#endif //UTBOOST_SRC_METRIC_BINARY_METRIC_H_
