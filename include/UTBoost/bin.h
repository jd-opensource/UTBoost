/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 * Created by Junjie Gao on 2023/3/1.
 */

#ifndef UTBOOST_INCLUDE_UTBOOST_BIN_H_
#define UTBOOST_INCLUDE_UTBOOST_BIN_H_

#include <vector>
#include <unordered_set>
#include <algorithm>
#include <utility>
#include <memory>
#include <cmath>

#include "UTBoost/definition.h"
#include "UTBoost/utils/omp_wrapper.h"
#include "UTBoost/utils/common.h"
#include "UTBoost/utils/log_wrapper.h"

namespace UTBoost {

/*! \brief Used to store statistics of the left and right nodes during feature bin traversal */
struct BinEntry {
  /*!
   * \brief Initialize bin
   * \param num_treat Number of treatments
   */
  explicit BinEntry(int num_treat) {
    num_treat_ = num_treat;
    num_total_data_ = 0;
    gradients_sum_ = 0.0;
    hessians_sum_ = 0.0;
    whessians_sum_ = std::vector<double>(num_treat, 0);
    wgradients_sum_ = std::vector<double>(num_treat, 0);
    label_sum_ = std::vector<double>(num_treat, 0);
    num_data_ = std::vector<double>(num_treat, 0);
  }

  /*! \brief Reset bin entry */
  void Reset() {
    num_total_data_ = 0;
    gradients_sum_ = 0.0;
    hessians_sum_ = 0.0;
    std::fill(wgradients_sum_.begin(), wgradients_sum_.end(), 0.0);
    std::fill(whessians_sum_.begin(), whessians_sum_.end(), 0.0);
    std::fill(label_sum_.begin(), label_sum_.end(), 0.0);
    std::fill(num_data_.begin(), num_data_.end(), 0.0);
  }

  /*!
   * \brief Copy Constructor
   * \param other other entry
   */
  BinEntry(const BinEntry& other) {
    num_treat_ = other.num_treat_;
    num_total_data_ = other.num_total_data_;
    gradients_sum_ = other.gradients_sum_;
    hessians_sum_ = other.hessians_sum_;
    whessians_sum_ = other.whessians_sum_;
    wgradients_sum_ = other.wgradients_sum_;
    label_sum_ = other.label_sum_;
    num_data_ = other.num_data_;
  }

  /*!
   * \brief Initialize bin by statistical information
   * \param num_treat Number of treatments
   * \param wgradients gradients sum of each treatment
   * \param whessians hessians sum of each treatment
   * \param labels labels sum of each treatment
   * \param num_datas Samples number of each treatment
   */
  BinEntry(int num_treat, const double* wgradients, const double* whessians, const double* labels, const double* num_datas) {
    num_treat_ = num_treat;
    num_total_data_ = 0;
    gradients_sum_ = 0.0;
    hessians_sum_ = 0.0;
    whessians_sum_.resize(num_treat_);
    wgradients_sum_.resize(num_treat_);
    label_sum_.resize(num_treat_);
    num_data_.resize(num_treat_);
    for (int i = 0; i < num_treat_; ++i) {
      gradients_sum_ += wgradients[i];
      hessians_sum_ += whessians[i];
      num_total_data_ += num_datas[i];
      wgradients_sum_[i] = wgradients[i];
      whessians_sum_[i] = whessians[i];
      label_sum_[i] = labels[i];
      num_data_[i] = num_datas[i];
    }
  }

  /*! \brief Push one sample's metadata in entry */
  inline void PushData(label_t label, treatment_t treat, double gradient, double hessian) {
    gradients_sum_ += gradient;
    hessians_sum_ += hessian;
    wgradients_sum_[treat] += gradient;
    whessians_sum_[treat] += hessian;
    label_sum_[treat] += label;
    num_data_[treat] += 1.0;
    num_total_data_ += 1.0;
  }

  /*!
   * \brief Add bin entry
   * \param other other bin entry
   * \return whether treatment number equal
   */
  bool Add(const BinEntry& other) {
    if (other.num_treat_ != num_treat_) {
      return false;
    }
    gradients_sum_ += other.gradients_sum_;
    hessians_sum_ += other.hessians_sum_;
    num_total_data_ += other.num_total_data_;
    for (int i = 0; i < num_treat_; ++i) {
      wgradients_sum_[i] += other.wgradients_sum_[i];
      whessians_sum_[i] += other.whessians_sum_[i];
      label_sum_[i] += other.label_sum_[i];
      num_data_[i] += other.num_data_[i];
    }
    return true;
  }

  /*! \brief Subtract other bin entry from this one */
  bool Subtract(const BinEntry& other) {
    if (other.num_treat_ != num_treat_) {
      return false;
    }
    gradients_sum_ -= other.gradients_sum_;
    hessians_sum_ -= other.hessians_sum_;
    num_total_data_ -= other.num_total_data_;
    for (int i = 0; i < num_treat_; ++i) {
      wgradients_sum_[i] -= other.wgradients_sum_[i];
      whessians_sum_[i] -= other.whessians_sum_[i];
      label_sum_[i] -= other.label_sum_[i];
      num_data_[i] -= other.num_data_[i];
    }
    return true;
  }

  // Number of treatments
  int num_treat_;
  // Number of total samples
  double num_total_data_;
  // Gradient sum of all data
  double gradients_sum_;
  // Hessian sum of all data
  double hessians_sum_;
  // Hessian sum of different treatment samples
  std::vector<double> whessians_sum_;
  // Gradient sum of different treatment samples
  std::vector<double> wgradients_sum_;
  // Label sum of different treatment samples
  std::vector<double> label_sum_;
  // Gradient sum of different treatment samples
  std::vector<double> num_data_;
};

/*! \brief Used to find feature boundaries and convert value to bin */
class BinMapper {
 public:
  BinMapper();
  /*! \brief deep copy function */
  BinMapper(const BinMapper& other);
  /*! \brief default destructor */
  ~BinMapper() = default;

  /*!
   * \brief Find boundaries of feature values
   * \param values feature values
   * \param num_row number rows
   * \param max_bin max bin number
   * \param min_data_bin minimum number of data in one bin
   */
  void FindBoundary(double* values, data_size_t num_row, int max_bin, int min_data_bin);

  /*!
   * \brief Find the bin using binary search method, na is treated specially
   * \param value value of feature
   * \return bin index
   */
  inline bin_t GetBinIndex(double value) const {
    if (std::isnan(value)) {
      if (use_missing_) {
        return num_bin_ - 1;
      } else {  // The data set has NAN, but the number is too small and is treated as zero.
        value = 0.0;
      }
    }
    int l = 0;
    int r = num_bin_ - 1;
    if (use_missing_) {
      r -= 1;
    }
    while (l < r) {
      int m = (r + l - 1) / 2;
      if (value <= boundaries_[m]) {
        r = m;
      }
      else {
        l = m + 1;
      }
    }
    return l;
  }

  /*!
   * \brief Get the upper boundary of bin
   * \param index bin index
   * \return upper boundary value
   */
  inline double GetBinUpper(bin_t index) const { return boundaries_[index]; };

  /*!
   * \brief Get the number of bins for this feature
   * \return number of bins
   */
  inline int GetNumBin() const { return num_bin_; }

  /*! \brief Whether this feature is trivial */
  inline bool IsTrivial() const { return is_trivial_; }

  /*! \brief Whether missing is treated specially */
  inline bool use_missing() const { return use_missing_; };

 private:
  // Number of bins
  int num_bin_;
  // Store upper bound for each bin
  std::vector<double> boundaries_;
  // whether missing is treated specially
  bool use_missing_;
  // whether feature is trivial
  bool is_trivial_;
};

class FeatureBin;

/*! \brief Iterator for one bin feature */
class BinIterator {
 public:
  explicit BinIterator(const FeatureBin* bin_data) : bin_data_(bin_data) {}
  inline bin_t Get(data_size_t idx);

 private:
  const FeatureBin* bin_data_;
};

/*! \brief Bin data container, and are also used to construct histograms */
class FeatureBin {
 public:
  friend BinIterator;
  /*!
   * \brief Constructor
   * \param num_samples number of sample
   */
  explicit FeatureBin(data_size_t num_samples): data_(num_samples, 0) {}
  ~FeatureBin() {}

  /*!
   * \brief Resize container
   * \param num_data number of sample
   */
  void inline ReSize(data_size_t num_data) {
    data_.resize(num_data);
  }

  inline bin_t data(data_size_t idx) const {
    return data_[idx];
  }

  /*!
   * \brief Deep copy other FeatureBin data by indices
   * \param full_bin other feature bin container
   * \param used_indices target indices
   * \param num_used_indices number of indices
   */
  void CopySubrow(const FeatureBin* full_bin, const data_size_t* used_indices,
                  data_size_t num_used_indices);

  /*!
   * \brief Insert feature bin into container
   * \param row row index
   * \param value feature bin
   */
  void inline InsertValue(data_size_t row, bin_t value) { data_[row] = value; }

  /*!
   * \brief Given gradient information of the input samples, construct histogram of this feature and copy it to out
   * \tparam USE_INDICES
   * \param data_indices used data indices
   * \param num_data number of data
   * \param gradients Pointer to gradients
   * \param hessians Pointer to hessians
   * \param labels Pointer to labels
   * \param treats Pointer to treatments
   * \param out histogram
   */
  template <bool USE_INDICES>
  void ConstructHistogram(const int* data_indices, int num_data,
                          const float* gradients, const float* hessians,
                          const label_t* labels, const treatment_t* treats,
                          BinEntry* out) const;

  /*!
   * \brief Get bin data iterator of the feature
   * \return Bin data iterator
   */
  BinIterator* GetIterator() const {
    return new BinIterator(this);
  }

  /*!
   * \brief Given the split threshold, copy the data index to left or right indices container according to bin value
   * \tparam USE_MISSING whether missing is treated specially
   * \param max_bin maximum bin index
   * \param default_left whether missing bin is put into left
   * \param threshold split threshold
   * \param data_indices data indices on current leaf
   * \param num_data number of data
   * \param lte_indices output left indices
   * \param gt_indices output right indices
   * \return number of left data
   */
  template <bool USE_MISSING>
  data_size_t Split(uint32_t max_bin, bool default_left, uint32_t threshold,
                    const data_size_t* data_indices, int num_data,
                    data_size_t* lte_indices, data_size_t* gt_indices) {

    if (num_data <= 0) { return 0; }

    int lte_count = 0;
    int gt_count = 0;
    data_size_t* missing_default_indices = gt_indices;
    data_size_t* missing_default_count = &gt_count;

    if (USE_MISSING && default_left) {
      missing_default_indices = lte_indices;
      missing_default_count = &lte_count;
    }

    for (int i = 0; i < num_data; ++i) {
      const int idx = data_indices[i];
      const uint32_t bin = data_[idx];
      if (USE_MISSING && bin == max_bin) {
        missing_default_indices[(*missing_default_count)++] = idx;
      } else if (bin > threshold) {
        gt_indices[gt_count++] = idx;
      } else {
        lte_indices[lte_count++] = idx;
      }
    }
    return lte_count;
  };

 private:
  // store bin data of this feature
  std::vector<bin_t> data_;
};


bin_t BinIterator::Get(data_size_t idx) {
  return bin_data_->data(idx);;
}

template <bool USE_INDICES>
void FeatureBin::ConstructHistogram(const int *data_indices,
                                    int num_data,
                                    const float *gradients,
                                    const float *hessians,
                                    const label_t *labels,
                                    const treatment_t *treats,
                                    BinEntry *out) const {
  if (hessians == nullptr) {
    if (USE_INDICES) {
      for (int i = 0; i < num_data; ++i) {
        const auto idx = data_indices[i];
        out[data_[idx]].PushData(labels[idx], treats[idx], gradients[idx], 1.0);
      }
    } else {
      for (int i = 0; i < num_data; ++i) {
        out[data_[i]].PushData(labels[i], treats[i], gradients[i], 1.0);
      }
    }

  } else {
    if (USE_INDICES) {
      for (int i = 0; i < num_data; ++i) {
        const auto idx = data_indices[i];
        out[data_[idx]].PushData(labels[idx], treats[idx], gradients[idx], hessians[idx]);
      }
    } else {
      for (int i = 0; i < num_data; ++i) {
        out[data_[i]].PushData(labels[i], treats[i], gradients[i], hessians[i]);
      }
    }
  }
}

}  // namespace UTBoost

#endif //UTBOOST_INCLUDE_UTBOOST_BIN_H_
