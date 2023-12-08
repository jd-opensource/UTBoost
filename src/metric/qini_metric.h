/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#ifndef UTBOOST_SRC_METRIC_QINI_METRIC_H_
#define UTBOOST_SRC_METRIC_QINI_METRIC_H_

#include <UTBoost/metric.h>
#include <UTBoost/utils/common.h>
#include <UTBoost/utils/log_wrapper.h>

#include <string>
#include <algorithm>
#include <sstream>
#include <vector>

namespace UTBoost {

/*! \brief Qini Metric */
class QiniMetric: public Metric {
 public:
  explicit QiniMetric(const Config&) {}

  virtual ~QiniMetric() {}

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return 1.0f;
  }

  void Init(const MetaInfo& metadata, data_size_t num_data) override {
    name_.emplace_back("qini");
    num_data_ = num_data;
    // get label
    label_ = metadata.GetLabel();
    treats_ = metadata.GetTreatment();
    num_treat_ = metadata.GetNumDistinctTreat();
  }

  void QiniCurve(const double* score, const label_t* label, const treatment_t* treatment, int len,
                 std::vector<std::pair<double, double>> &curve,
                 double *out_num_ctrl, double *out_num_trmnt, double *out_y_ctrl, double *out_y_trmnt) const {
    std::vector<data_size_t> sorted_idx;
    sorted_idx.reserve(len);
    for (data_size_t i = 0; i < len; ++i) {
      sorted_idx.emplace_back(i);
    }
    // sorted idx by uplift score
    ParallelSort(sorted_idx.begin(), sorted_idx.end(), [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    double threshold = score[sorted_idx[0]];
    double num_trmnt = 0.0f, num_ctrl = 0.0f, y_trmnt = 0.0f, y_ctrl = 0.0f;
    curve.clear();
    curve.reserve(len);
    curve.emplace_back(0.0f, 0.0f);
    for (int i = 0; i < len; ++i) {
      data_size_t idx = sorted_idx[i];
      const double cur_score = score[idx];
      if (treatment[idx] <= 0) {
        num_ctrl += 1.0f;
        y_ctrl += label[idx];
      } else {
        num_trmnt += 1.0f;
        y_trmnt += label[idx];
      }
      // new threshold
      if (cur_score != threshold) {
        threshold = cur_score;
        // accumulate
        if ((num_trmnt > 0.0) && (num_ctrl > 0.0)) {
          curve.emplace_back(num_trmnt + num_ctrl, y_trmnt - y_ctrl * num_trmnt / num_ctrl);
        }
      }
    }
    if (curve.back().first != (num_trmnt + num_ctrl)) {
      if ((num_trmnt > 0.0) && (num_ctrl > 0.0)) {
        curve.emplace_back(num_trmnt + num_ctrl, y_trmnt - y_ctrl * num_trmnt / num_ctrl);
      }
    }

    *out_num_ctrl = num_ctrl;
    *out_num_trmnt = num_trmnt;
    *out_y_ctrl = y_ctrl;
    *out_y_trmnt = y_trmnt;
  }

  virtual double QiniArea(const double* score, const label_t* label, const treatment_t* treatment, int len) const {
    if (len < 2) return 0.0f;
    std::vector<std::pair<double, double>> curve;
    double num_ctrl, num_trmnt, y_ctrl, y_trmnt;
    QiniCurve(score, label, treatment, len, curve, &num_ctrl, &num_trmnt, &y_ctrl, &y_trmnt);
    if (curve.size() < 3) return 0.0f;
    double baseline = curve.back().first * curve.back().second / 2.0f;
    return Trapz(curve) - baseline;
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction* obj) const override {

    if (num_treat_ < 2) {
      Log::Error("Qini metric is not available when the number of treatment is less than 2");
      return std::vector<double>(1, 0);
    }

    double tmp;
    obj->ConvertOutput(score[0], &tmp);
    // std::vector<double> trans_score(num_data_ * num_treat_, 0.0f);
    std::vector<double> trans_score;
    if (score[0] != tmp) {
      trans_score.reserve(num_data_ * (num_treat_ - 1));
      for (int i = 1; i < num_treat_; ++i) {
        for (int j = 0; j < num_data_; ++j) {
          double pc, pt;
          obj->ConvertOutput(score[j], &pc);
          obj->ConvertOutput(score[j] + score[i * num_data_ + j], &pt);
          // trans_score[i * num_data_ + j] = pt - pc;
          trans_score.emplace_back(pt - pc);
        }
      }
      score = trans_score.data();
    } else {
      score = score + num_data_;
    }

    // get indices sorted by score, descent order
    if (num_treat_ == 2) {
      return std::vector<double>(1, QiniArea(score, label_, treats_, num_data_));
    } else if (num_treat_ > 2) {
      std::vector<double> real_score; real_score.reserve(num_data_);
      std::vector<label_t> real_label; real_label.reserve(num_data_);
      std::vector<treatment_t> real_treat; real_treat.reserve(num_data_);
      for (data_size_t i = 0; i < num_data_; ++i) {
        int max_idx = 0;
        double max_score = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < num_treat_ - 1; ++j) {
          double lift_score = score[i + j * num_data_];
          if (lift_score > max_score) {
            max_idx = j;
            max_score = lift_score;
          }
        }
        if (treats_[i] == 0) {
          real_score.push_back(max_score);
          real_treat.emplace_back(0);
          real_label.push_back(label_[i]);
        } else if (treats_[i] == max_idx) {
          real_score.push_back(max_score);
          real_treat.emplace_back(1);
          real_label.push_back(label_[i]);
        }
      }
      return std::vector<double>(1, QiniArea(real_score.data(),
                                             real_label.data(),
                                             real_treat.data(),
                                             static_cast<int>(real_score.size())
      ));
    } else {
      Log::Error("Qini metric is not available when the number of treatment is less than 2");
    }
  }

 protected:
  // Number of data
  data_size_t num_data_;
  // Number of treatment
  int num_treat_;
  // Pointer of label
  const label_t* label_;
  // Pointer of treatment
  const treatment_t* treats_;
  // Name of test set
  std::vector<std::string> name_;
};

/*!
 * \brief Normal Qini Metric for binary classification task.
 */
class NormalQiniMetric: public QiniMetric {
 public:
  explicit NormalQiniMetric(const Config& config): QiniMetric(config) {}

  virtual ~NormalQiniMetric() {}

  double QiniArea(const double* score, const label_t* label, const treatment_t* treatment, int len) const override {
    if (len < 2) return 0.0f;
    std::vector<std::pair<double, double>> curve;
    double num_ctrl, num_trmnt, y_ctrl, y_trmnt;
    QiniCurve(score, label, treatment, len, curve, &num_ctrl, &num_trmnt, &y_ctrl, &y_trmnt);

    if (curve.size() < 3 || num_ctrl == 0.0) return 0.0f;

    std::vector<std::pair<double, double>> perfect_curve;
    perfect_curve.emplace_back(0.0, 0.0);
    perfect_curve.emplace_back(y_trmnt, y_trmnt);
    perfect_curve.emplace_back(num_trmnt + num_ctrl - y_ctrl, y_trmnt);
    perfect_curve.emplace_back(num_trmnt + num_ctrl, y_trmnt - y_ctrl / num_ctrl * num_trmnt);

    double baseline = curve.back().first * curve.back().second / 2.0f;
    double real = Trapz(curve) - baseline;
    double perfect = Trapz(perfect_curve) - baseline;

    return real / perfect;
  }
};

}  // namespace UTBoost

#endif //UTBOOST_SRC_METRIC_QINI_METRIC_H_
