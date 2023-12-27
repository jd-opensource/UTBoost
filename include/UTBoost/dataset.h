/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 * Created by Junjie Gao on 2023/3/1.
 */

#ifndef UTBOOST_INCLUDE_UTBOOST_DATASET_H_
#define UTBOOST_INCLUDE_UTBOOST_DATASET_H_

#include <vector>
#include <unordered_set>
#include <algorithm>
#include <utility>
#include <memory>
#include <cmath>

#include "UTBoost/definition.h"
#include "UTBoost/bin.h"
#include "UTBoost/utils/omp_wrapper.h"
#include "UTBoost/utils/common.h"
#include "UTBoost/utils/text_reader.h"

namespace UTBoost {

/*!
 * \brief This class is used to store some meta(non-feature) data for training data,
 *        e.g. labels, weights, treatments information.
 */
class UTBOOST_EXPORT MetaInfo {
 public:
  /*! \brief Constructor */
  MetaInfo() {num_samples_ = 0; num_distinct_treat_ = 0;};

  /*! \brief Initialize container */
  void Init(data_size_t num_data, bool has_weight);

  /*!
   * \brief init as subset
   * \param metadata Filename of data
   * \param used_indices indices pointer for build subset
   * \param num_used_indices number of index
   */
  void Init(const MetaInfo& fullset, const data_size_t* used_indices, data_size_t num_used_indices);

  /*!
   * \brief Setter of label
   * \param label label pointer
   * \param num_data number of data
   */
  void SetLabel(const label_t* label, data_size_t num_data);

  /*! \brief weight setter */
  void SetWeight(const weight_t* weight, data_size_t num_data);

  /*! \brief treatment setter */
  void SetTreatment(const treatment_t* treat, data_size_t num_data);

  /*! \brief Get label pointer */
  inline const label_t* GetLabel() const { return label_.data(); }

  /*! \brief Get weight pointer */
  inline const weight_t* GetWeight() const { return weight_.data(); }

  /*! \brief Get treatment pointer */
  inline const treatment_t* GetTreatment() const { return treat_.data(); }

  /*! \brief Get the number of treatments */
  inline treatment_t GetNumDistinctTreat() const { return num_distinct_treat_; }

  /*! \brief Get the number of samples */
  inline data_size_t GetSampleNum() const { return num_samples_; }

 private:
  // number of sample
  data_size_t num_samples_;
  // number of treatments
  treatment_t num_distinct_treat_;
  // label data
  std::vector<label_t> label_;
  // weight data
  std::vector<weight_t> weight_;
  // treatment data
  std::vector<treatment_t> treat_;
};

/*!
 * \brief Main container of data set,
 *        also used to provide some auxiliary operations during the training process,
 *        such as splitting the data and constructing histograms of the features.
 */
class UTBOOST_EXPORT Dataset {
 public:
  /*! \brief Constructor */
  Dataset() {};

  /*!
   * \brief Constructor
   * \param n number of samples
   */
  Dataset(data_size_t n) { num_samples_ = n; };

  /*! \brief Default Destructor */
  ~Dataset() = default;

  /*!
   * \brief Create aligned validation datasets based on references, thus ensuring the same bins boundaries.
   * \param dataset reference dataset
   */
  void CreateValid(const Dataset* dataset);

  /*!
   * \brief Build dataset from bin mappers
   * \param bin_mappers mappers with boundaries
   * \param num_row total number of samples
   */
  void Build(std::vector<std::unique_ptr<BinMapper>>& bin_mappers, data_size_t num_row);

  /*!
   * \brief Construct feature histograms
   * \param is_feature_used Whether the feature needs to construct a histogram.
   * \param data_indices Sample index used to construct histograms
   * \param num_data Number of index data
   * \param leaf_idx leaf index
   * \param gradients gradients pointer
   * \param hessians hessians pointer
   * \param hist_data out histograms
   */
  void ConstructHistograms(const std::vector<int8_t>& is_feature_used,
                           const int* data_indices, int num_data,
                           int leaf_idx,
                           const float* gradients , const float* hessians,
                           BinEntry* hist_data) const;

  /*!
   * \brief Push one row of data into the corresponding bin
   * \param row_idx row index
   * \param feature_values vector of feature values
   */
  inline void PushRow(data_size_t row_idx, const std::vector<double>& feature_values) {
    for (int i = 0; i < static_cast<int>(feature_values.size()); ++i) {
      PushValue(i, row_idx, feature_values[i]);
    }
  }

  /*!
   * \brief Push a specific feature in a row into the corresponding bin
   * \param feature_idx feature index
   * \param row_idx row index
   * \param value value of feature
   */
  inline void PushValue(int feature_idx, data_size_t row_idx, double value) {
    bin_t bin = mappers_[feature_idx]->GetBinIndex(value);
    if (bin == 0) { return; }  // Initially, all datas are in bin0
    bin_data_[feature_idx]->InsertValue(row_idx, bin);
  }

  /*!
   * \brief Given feature and threshold, placing the data indices in two different containers (lte_indices and gt_indices).
   * \param feature feature index
   * \param threshold split threshold value
   * \param default_left whether nan bin be placed to left
   * \param data_indices which data needs to be split
   * \param cnt number of data in current node
   * \param lte_indices out indices where their feature value <= threshold
   * \param gt_indices out indices where their feature value > threshold
   * \return number of data in left child
   */
  inline data_size_t Split(int feature, const uint32_t threshold,
                           bool default_left, const data_size_t* data_indices,
                           data_size_t cnt, data_size_t* lte_indices,
                           data_size_t* gt_indices) const {
    uint32_t max_bin = mappers_[feature]->GetNumBin() - 1;
    if (GetFMapper(feature)->use_missing()) {
      return bin_data_[feature]->Split<true>(max_bin, default_left, threshold, data_indices,
                                       cnt, lte_indices, gt_indices);
    } else {
      return bin_data_[feature]->Split<false>(max_bin, default_left, threshold, data_indices,
                                             cnt, lte_indices, gt_indices);
    }

  }

  /*!
   * \brief Get valid feature indices in data set
   * \return feature indices
   */
  inline std::vector<int> ValidFeatureIndices() const {
    std::vector<int> ret;
    for (int i = 0; i < num_features_; ++i) {
      if (!mappers_[i]->IsTrivial()) {
        ret.push_back(i);
      }
    }
    return ret;
  }

  /*!
   * \brief Resize data container
   * \param num_data number of data
   */
  void ReSize(data_size_t num_data);

  /*!
   * \brief Copy subset
   * \param fullset full dataset
   * \param used_indices target indices pointer
   * \param num_used_indices number of indices
   * \param need_meta_data whether copy meta
   */
  void CopySubrow(const Dataset* fullset, const data_size_t* used_indices, data_size_t num_used_indices, bool need_meta_data);

  /*!
   * \brief Get the real threshold value of bin
   * \param i feature index
   * \param threshold bin index
   * \return threshold value
   */
  inline double RealThreshold(int i, bin_t threshold) const {
    return mappers_[i]->GetBinUpper(threshold);
  }

  /*!
   * \brief Get iterator of feature bin data
   * \param i feature index
   * \return iterator of the feature data
   */
  inline BinIterator* FeatureIterator(int i) const {
    return bin_data_[i]->GetIterator();
  }

  /*!
   * \brief Set float meta, such as label and weight.
   * \param name field name
   * \param data field data pointer
   * \param num_data number of datas
   */
  void SetMetaFloat(const char* name, const float* data, data_size_t num_data);

  /*!
   * \brief Set treatment data
   * \param data treatment data pointer
   * \param num_data number of datas
   */
  void SetMetaTreatment(const treatment_t* data, data_size_t num_data) { meta_.SetTreatment(data, num_data); };

  void DumpBinMapper(const char* text_filename);

  /*! Get number of distinct treatments */
  inline treatment_t GetDistinctTreatNum() const { return meta_.GetNumDistinctTreat(); }
  /*! Get number of samples */
  inline data_size_t GetNumSamples() const { return num_samples_; }
  /*! Get number of features */
  inline int GetNumFeatures() const { return num_features_; }
  /*! Get number of bins */
  inline int GetFMapperNum(int feature_idx) const { return mappers_[feature_idx]->GetNumBin(); }
  /*! Get feature mapper */
  inline const BinMapper* GetFMapper(int feature_idx) const { return mappers_[feature_idx].get(); }
  /*! Get number of total bins */
  inline int GetNumTotalBins() const { return num_total_bins; }
  /*! Get metas */
  inline const MetaInfo& GetMetaInfo() const { return meta_; }
  /*! Check this data set is applicable for training */
  bool FinshLoad(bool with_meta);
  /*! \brief Disable copy */
  Dataset& operator=(const Dataset&) = delete;
  /*! \brief Disable copy */
  Dataset(const Dataset&) = delete;

 private:
  // Total sample number
  data_size_t num_samples_;
  // Number of features
  int num_features_;
  // Total number of bins
  int num_total_bins;
  // Store bin data of each feature
  std::vector<std::unique_ptr<FeatureBin>> bin_data_;
  // Store bin mappers of each feature
  std::vector<std::unique_ptr<BinMapper>> mappers_;
  // Store some label data
  MetaInfo meta_;
};


/*!
 * \brief The Parser class is used to parse data files,
 *        and provides the functionality to copy the parsing results to an output array.
 */
class Parser {
 public:
  /*!
   * \brief Constructor that initializes the member variable num_samples_ to 0.
   */
  Parser() { num_samples_ = 0; }

  /*!
   * \brief Parses the given file and stores the parsing results in the internal member variables.
   * \param filename The name of the file.
   * \param label_idx The index of the label.
   * \param treatment_idx The index of the treatment.
   * \param skip_first_line Whether to skip the first line of the file.
   */
  virtual void parseFile(const char* filename, int label_idx, int treatment_idx, bool skip_first_line) = 0;

  /*!
   * \brief Copies the parsing results to the output arrays.
   * \param out_feature The output array for features.
   * \param max_feature_idx The maximum feature index.
   * \param out_label The output array for labels.
   * \param out_treat The output array for treatments.
   */
  void copyTo(double* out_feature, int max_feature_idx, label_t* out_label, treatment_t* out_treat);

  /*!
   * \brief Gets the number of samples.
   * \return The number of samples.
   */
  data_size_t num_samples() { return num_samples_; }

 protected:
  // The number of samples.
  data_size_t num_samples_;
  // The feature vectors.
  std::vector<std::vector<std::pair<int, double>>> features_;
  // The label vectors.
  std::vector<label_t> labels_;
  // The treatment vectors.
  std::vector<treatment_t> treatments_;
};


/*!
 * \brief The LibsvmParser class inherits from the Parser class and is used to parse data files in the libsvm format.
 */
class LibsvmParser: public Parser {
 public:
  void parseFile(const char* filename, int label_idx, int treatment_idx, bool skip_first_line) override;
};

}

#endif //UTBOOST_INCLUDE_UTBOOST_DATASET_H_
