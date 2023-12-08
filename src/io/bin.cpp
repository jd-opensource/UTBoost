/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#include "UTBoost/bin.h"

namespace UTBoost {

BinMapper::BinMapper() {
  boundaries_.clear();
  boundaries_.push_back(std::numeric_limits<double>::infinity());
  num_bin_ = 1;
  use_missing_ = false;
  is_trivial_ = true;
}

BinMapper::BinMapper(const BinMapper& other) {
  num_bin_ = other.num_bin_;
  is_trivial_ = other.is_trivial_;
  use_missing_ = other.use_missing_;
  boundaries_ = other.boundaries_;
}

std::vector<double> GreedyFindBoundary(const double* distinct_values, const int* counts, int num_distinct_values,
                                       int max_bin, int total_cnt, int min_data_in_bin) {

  std::vector<double> bin_upper_bound;

  if (num_distinct_values <= max_bin) {
    bin_upper_bound.clear();
    int cur_cnt_inbin = 0;
    for (int i = 0; i < num_distinct_values - 1; ++i) {
      cur_cnt_inbin += counts[i];
      if (cur_cnt_inbin > min_data_in_bin) {
        auto val = std::nextafter(((distinct_values[i] + distinct_values[i + 1]) / 2.0), INFINITY);
        if (bin_upper_bound.empty() || val > std::nextafter(bin_upper_bound.back(), INFINITY)) {
          bin_upper_bound.push_back(val);
          cur_cnt_inbin = 0;
        }
      }
    }
    bin_upper_bound.push_back(std::numeric_limits<double>::infinity());
  } else {
    if (min_data_in_bin > 0) {
      max_bin = std::min(max_bin, static_cast<int>(total_cnt / min_data_in_bin));
      max_bin = std::max(max_bin, 1);
    }

    double mean_bin_size = static_cast<double>(total_cnt) / max_bin;
    int rest_bin_cnt = max_bin;
    int rest_sample_cnt = static_cast<int>(total_cnt);
    std::vector<bool> is_big_count_value(num_distinct_values, false);
    for (int i = 0; i < num_distinct_values; ++i) {
      if (counts[i] > mean_bin_size) {
        is_big_count_value[i] = true;
        --rest_bin_cnt;
        rest_sample_cnt -= counts[i];
      }
    }
    mean_bin_size = static_cast<double>(rest_sample_cnt) / rest_bin_cnt;
    std::vector<double> upper_bounds(max_bin, std::numeric_limits<double>::infinity());
    std::vector<double> lower_bounds(max_bin, std::numeric_limits<double>::infinity());

    int bin_cnt = 0;
    lower_bounds[bin_cnt] = distinct_values[0];

    int cur_cnt_inbin = 0;
    for (int i = 0; i < num_distinct_values - 1; ++i) {
      if (!is_big_count_value[i]) {
        rest_sample_cnt -= counts[i];
      }
      cur_cnt_inbin += counts[i];
      if (is_big_count_value[i] || cur_cnt_inbin >= mean_bin_size ||
          (is_big_count_value[i+1] && cur_cnt_inbin >= std::max(1.0 , mean_bin_size*0.5f))) {
        upper_bounds[bin_cnt] = distinct_values[i];
        ++bin_cnt;
        lower_bounds[bin_cnt] = distinct_values[i + 1];

        if (bin_cnt >= max_bin - 1) { break; }
        cur_cnt_inbin = 0;

        if (!is_big_count_value[i]) {
          --rest_bin_cnt;
          mean_bin_size = rest_sample_cnt / static_cast<double>(rest_bin_cnt);
        }
      }
    }
    bin_upper_bound.clear();

    for (int i = 0; i < bin_cnt; ++i) {
      auto val = std::nextafter((upper_bounds[i] + lower_bounds[i + 1]) / 2.0, INFINITY);
      if (bin_upper_bound.empty() || val > std::nextafter(bin_upper_bound.back(), INFINITY)) {
        bin_upper_bound.push_back(val);
      }
    }
    bin_upper_bound.push_back(std::numeric_limits<double>::infinity());
  }
  return bin_upper_bound;
}

void BinMapper::FindBoundary(double* values, data_size_t num_row, int max_bin, int min_data_bin) {
  int normal_num_values = 0;
  for (int i = 0; i < num_row; ++i) {
    if (!std::isnan(values[i])) {
      values[normal_num_values++] = values[i];
    }
  }

  int na_cnt = static_cast<int>(num_row - normal_num_values);

  use_missing_ = na_cnt > min_data_bin;

  std::stable_sort(values, values + normal_num_values);

  std::vector<double> distinct_values;
  std::vector<int> counts;

  if (normal_num_values > 0) {
    distinct_values.push_back(values[0]);
    counts.push_back(1);
  }

  for (int i = 1; i < normal_num_values; ++i) {
    if (values[i] > std::nextafter(values[i - 1], INFINITY)) {
      distinct_values.push_back(values[i]);
      counts.push_back(1);
    }
    else {
      distinct_values.back() = values[i];
      ++counts.back();
    }
  }

  std::vector<int> cnt_in_bin;
  int num_distinct_values = static_cast<int>(distinct_values.size());

  if (use_missing_) {
    boundaries_ = GreedyFindBoundary(distinct_values.data(), counts.data(), num_distinct_values,
                                     max_bin - 1, num_row - na_cnt, min_data_bin);
    boundaries_.push_back(NAN);
  } else {
    boundaries_ = GreedyFindBoundary(distinct_values.data(), counts.data(), num_distinct_values,
                                     max_bin , num_row, min_data_bin);
  }

  num_bin_ = static_cast<int>(boundaries_.size());
  // check trivial(num_bin_ == 1) feature
  if (num_bin_ <= 1 || (num_bin_ == 2 && normal_num_values == 0) ) {
    is_trivial_ = true;
  } else {
    is_trivial_ = false;
  }
  ASSERT_LE(boundaries_.size(), static_cast<size_t>(max_bin));
}

void FeatureBin::CopySubrow(const FeatureBin *full_bin, const data_size_t *used_indices, data_size_t num_used_indices) {
  for (int i = 0; i < num_used_indices; ++i) {
    data_[i] = full_bin->data_[used_indices[i]];
  }
}

} // namespace UTBoost

