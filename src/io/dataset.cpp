/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#include "UTBoost/dataset.h"
#include "UTBoost/utils/log_wrapper.h"
#include "UTBoost/utils/file_io.h"

namespace UTBoost {

void MetaInfo::SetLabel(const label_t *label, data_size_t num_data) {
  ASSERT_NOTNULL(label)
  if (num_samples_ == 0) {
    num_samples_ = num_data;
  } else {
    ASSERT_EQ(num_data, num_samples_)
  }
  TIMER_START(x);
  if (label_.empty()) label_.resize(num_data);
#pragma omp parallel for schedule(static, 512) if (num_samples_ >= 1024)
  for (data_size_t i = 0; i < num_data; ++i) {
    label_[i] = label[i];
  }
  TIMER_STOP(x);
  Log::Debug("Label load complete, with %d entities, cost %f s.", num_samples_, TIMER_SEC(x));
}

void MetaInfo::SetWeight(const weight_t *weight, data_size_t num_data) {
  ASSERT_NOTNULL(weight)
  if (num_samples_ == 0) {
    num_samples_ = num_data;
  } else {
    ASSERT_EQ(num_data, num_samples_)
  }
  TIMER_START(x);
  if (weight_.empty()) weight_.resize(num_data);
#pragma omp parallel for schedule(static, 512) if (num_samples_ >= 1024)
  for (data_size_t i = 0; i < num_data; ++i) {
    weight_[i] = weight[i];
  }
  TIMER_STOP(x);
  Log::Debug("Weight load complete, with %d entities, cost %f s.", num_samples_, TIMER_SEC(x));
}

void MetaInfo::SetTreatment(const treatment_t *treat, data_size_t num_data) {
  ASSERT_NOTNULL(treat)
  if (num_samples_ == 0) {
    num_samples_ = num_data;
  } else {
    ASSERT_EQ(num_data, num_samples_)
  }
  TIMER_START(x);
  if (treat_.empty()) treat_.resize(num_data);
#pragma omp parallel for schedule(static, 512) if (num_samples_ >= 1024)
  for (data_size_t i = 0; i < num_data; ++i) {
    treat_[i] = treat[i];
  }

  std::unordered_set<treatment_t> dist_treat(treat_.begin(), treat_.end());
  num_distinct_treat_ = static_cast<treatment_t>(dist_treat.size());
  treatment_t max_treat = *std::max_element(dist_treat.begin(), dist_treat.end());
  treatment_t min_treat = *std::min_element(dist_treat.begin(), dist_treat.end());

  if ( (max_treat != (num_distinct_treat_ - 1)) || (min_treat != 0) ) {
    treat_.clear(), treat_.shrink_to_fit();
    Log::Error("There are %d distinct treatments, corresponding treatment indicator should in {0,%d}.",
               num_distinct_treat_, num_distinct_treat_ - 1,
               min_treat, max_treat);
  }
  TIMER_STOP(x);
  Log::Debug("Treatment load complete, with %d entities, %d distinct treatment, "
             "cost %f s.", num_samples_, num_distinct_treat_, TIMER_SEC(x));
}

void MetaInfo::Init(data_size_t num_data, bool has_weight) {
  num_samples_ = num_data;
  label_.resize(num_data);
  treat_.resize(num_data);
  if (has_weight) {
      weight_.resize(num_data);
  }
}

void MetaInfo::Init(const MetaInfo &fullset, const data_size_t *used_indices, data_size_t num_used_indices) {
  num_samples_ = num_used_indices;

  label_ = std::vector<label_t>(num_used_indices);
#pragma omp parallel for schedule(static, 512) if (num_used_indices >= 1024)
  for (data_size_t i = 0; i < num_used_indices; ++i) {
    label_[i] = fullset.label_[used_indices[i]];
    treat_[i] = fullset.treat_[used_indices[i]];
  }

  if (!fullset.weight_.empty()) {
    weight_ = std::vector<label_t>(num_used_indices);
#pragma omp parallel for schedule(static, 512) if (num_used_indices >= 1024)
    for (data_size_t i = 0; i < num_used_indices; ++i) {
      weight_[i] = fullset.weight_[used_indices[i]];
    }
  }
}

void Dataset::Build(std::vector<std::unique_ptr<BinMapper>> &bin_mappers, data_size_t num_row) {
  num_total_bins = 0;
  num_features_ = static_cast<int>(bin_mappers.size());
  bin_data_.reserve(num_features_);
  mappers_.reserve(num_features_);

  for (auto & bin_mapper : bin_mappers) {
    mappers_.push_back(std::unique_ptr<BinMapper>(bin_mapper.release()));  // transfer ownership
    num_total_bins += mappers_.back()->GetNumBin();
  }

  for (int i = 0; i < num_features_; ++i) {
    bin_data_.push_back(std::unique_ptr<FeatureBin>(new FeatureBin(num_row)));
  }
  num_samples_ = num_row;
}


void Dataset::SetMetaFloat(const char *name, const float *data, data_size_t num_data) {
  if (std::strcmp(name, "label") == 0) {
    meta_.SetLabel(data, num_data);
  } else if (std::strcmp(name, "weight") == 0) {
    meta_.SetWeight(data, num_data);
  } else {
    Log::Error("Unknown meta info name: {}", name);
  }
}


bool Dataset::FinshLoad(bool with_meta) {
  if (num_features_ <= 0) {
    Log::Warn("No feature in dataset");
    return false;
  }

  int trivial_num = 0;
  for (int i = 0; i < num_features_; ++i) {
    if (mappers_[i]->IsTrivial()) {
      trivial_num++;
      Log::Debug("Feature %d is trivial", i);
    }
  }

  if (trivial_num == num_features_) {
    Log::Warn("All features are trivial");
    return false;
  }

  if (with_meta) {
    if (meta_.GetNumDistinctTreat() <= 1){
      Log::Warn("Number of distinct treatment less than 2");
    }

    if (meta_.GetSampleNum() != num_samples_) {
      Log::Warn("Number of samples in meta info is not equal to number of samples in dataset");
      return false;
    }
  }

  return true;
}

void Dataset::ReSize(data_size_t num_data) {
  if (num_samples_ != num_data) {
    num_samples_ = num_data;
#pragma omp parallel for schedule(static)
    for (int fid = 0; fid < num_features_; ++fid) {
      bin_data_[fid]->ReSize(num_samples_);
    }
  }
}

void Dataset::CopySubrow(const Dataset *fullset,
                         const data_size_t *used_indices,
                         data_size_t num_used_indices,
                         bool need_meta_data) {
  ASSERT_EQ(num_used_indices, num_samples_);
#pragma omp parallel for schedule(dynamic)
  for (int fid = 0; fid < num_features_; ++fid) {
    bin_data_[fid]->CopySubrow(fullset->bin_data_[fid].get(), used_indices, num_used_indices);
  }

  if (need_meta_data) {
    meta_.Init(fullset->meta_, used_indices, num_used_indices);
  }
}

void Dataset::ConstructHistograms(const std::vector<int8_t> &is_feature_used,
                                  const int *data_indices,
                                  int num_data,
                                  int leaf_idx,
                                  const float *gradients,
                                  const float *hessians,
                                  BinEntry *hist_data) const {

  if (leaf_idx < 0 || num_data < 0 || hist_data == nullptr) { return; }
  bool use_indices = data_indices != nullptr && (num_data < num_samples_);

  std::vector<int> used_features, offsets;
  used_features.reserve(num_features_);
  offsets.reserve(num_features_);
  int num_bin = 0;
  for (int fidx = 0; fidx < num_features_; ++fidx) {
    if (is_feature_used[fidx]) {
      used_features.emplace_back(fidx);
      offsets.emplace_back(num_bin);
    }
    num_bin += mappers_[fidx]->GetNumBin();
  }
  int num_used_feature = static_cast<int>(used_features.size());

#pragma omp parallel for schedule(static)
  for (int fidx = 0; fidx < num_used_feature; ++fidx) {
    int feature = used_features[fidx];
    BinEntry* data_ptr = hist_data + offsets[fidx];
    int num_bin = mappers_[feature]->GetNumBin();
    // std::memset(reinterpret_cast<void*>(data_ptr), 0, num_bin * sizeof(BinEntry));
    for (int i = 0; i < num_bin; ++i) {
      data_ptr[i].Reset();
    }
    if (use_indices) {
      bin_data_[feature]->ConstructHistogram<true>(data_indices, num_data, gradients, hessians,
                                                   meta_.GetLabel(), meta_.GetTreatment(),
                                                   data_ptr);
    } else {
      bin_data_[feature]->ConstructHistogram<false>(data_indices, num_data, gradients, hessians,
                                                    meta_.GetLabel(), meta_.GetTreatment(),
                                                    data_ptr);
    }
  }
}

void Dataset::CreateValid(const Dataset *dataset) {
  num_features_ = dataset->num_features_;
  num_total_bins = dataset->num_total_bins;
  mappers_.clear();
  bin_data_.clear();
  for (int i = 0; i < num_features_; ++i) {
    mappers_.emplace_back(new BinMapper(*(dataset->mappers_[i])));
    bin_data_.emplace_back(
        std::unique_ptr<FeatureBin>(new FeatureBin(num_samples_))
    );
  }
}

void Dataset::DumpBinMapper(const char* text_filename) {
  std::stringstream str_buf;
  C_stringstream(str_buf);
  str_buf << "{\n";
  str_buf << "\"num_features\":" << num_features_ << ',' << '\n';
  str_buf << "\"num_data\":" << num_samples_ << ',' << '\n';
  str_buf << "\"max_bin_by_feature\":[";
  for (int i = 0; i < num_features_ - 1; ++i) {
    str_buf << GetFMapperNum(i) << ',';
  }
  str_buf << GetFMapperNum(num_features_ - 1) << "]," << '\n';

  str_buf << "\"feature_bins\":{" << '\n';
  for (int j = 0; j < num_features_; ++j) {
    const BinMapper* mapper = GetFMapper(j);
    int num_value_bins = mapper->GetNumBin() - mapper->use_missing() - 1;

    str_buf << "  \"f" << j << "\":{";
    str_buf << "\"missing\":" << (mapper->use_missing() ? "true" : "false") << ",";
    if (num_value_bins < 1) {
      if (j == num_features_ - 1)
        str_buf << "\"upper_bounds\":[]}\n }";
      else
        str_buf << "\"upper_bounds\":[]},\n";
      continue;
    }

    str_buf << "\"upper_bounds\":[";
    for (int i = 0; i < num_value_bins - 1; ++i) {
      str_buf << mapper->GetBinUpper(i) << ",";
    }
    if (j == num_features_ - 1)
      str_buf << mapper->GetBinUpper(num_value_bins - 1) << "]}\n }";
    else
      str_buf << mapper->GetBinUpper(num_value_bins - 1) << "]},\n";
  }
  str_buf << "\n}";

  auto writer = VirtualFileWriter::Make(text_filename);
  if (!writer->Init()) {
    Log::Error("Model file %s is not available for writes", text_filename);
  }

  std::string s = str_buf.str();
  writer->Write(s.c_str(), s.size());
}

void Parser::copyTo(double *out_feature, int max_feature_idx, label_t *out_label, treatment_t *out_treat) {
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_samples_; ++i) {
    OMP_LOOP_EX_BEGIN();
      double* f = out_feature + i * max_feature_idx;
      for (std::pair<int, double> p : features_[i]) {
        *(f + p.first) = p.second;
      }
      *(out_label + i) = labels_[i];
      *(out_treat + i) = treatments_[i];
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
}

void LibsvmParser::parseFile(const char *filename, int label_idx, int treatment_idx, bool skip_first_line) {
  TextReader<size_t> model_reader(filename, skip_first_line);
  std::function<void(size_t, const std::vector<std::string>&)>
    process_fun = [&label_idx, &treatment_idx, this](
    data_size_t, const std::vector<std::string>& lines) {

    size_t cur_size = this->features_.size();
    this->features_.insert(this->features_.end(), lines.size(), std::vector<std::pair<int, double>>());
    this->labels_.insert(this->labels_.end(), lines.size(), 0);
    this->treatments_.insert(this->treatments_.end(), lines.size(), 0);

    OMP_INIT_EX();
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < static_cast<data_size_t>(lines.size()); ++i) {
      OMP_LOOP_EX_BEGIN();
        std::vector<std::pair<int, double>> &feature = this->features_[i + cur_size];
        int idx = 0;
        double val = 0.0f;
        const char* str = lines[i].c_str();
        if (label_idx == 0) {
          str = Atof(str, &val);
          this->labels_[i + cur_size] = static_cast<label_t>(val);
          str = SkipSpaceAndTab(str);
        }
        if (treatment_idx == 1) {
          treatment_t t = 0;
          str = Atoi(str, &t);
          this->treatments_[i + cur_size] = t;
          str = SkipSpaceAndTab(str);
        }
        while (*str != '\0') {
          str = Atoi(str, &idx);
          str = SkipSpaceAndTab(str);
          if (*str == ':') {
            ++str;
            str = Atof(str, &val);
            feature.emplace_back(idx - 1, val);
          } else {
            Log::Error("Input format error when parsing as Libsvm at line %d.", i + cur_size + 1);
          }
          str = SkipSpaceAndTab(str);
        }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
  };

  model_reader.ReadAllAndProcessParallel(process_fun);
  num_samples_ = static_cast<int>(features_.size());
}

}  // namespace UTBoost

