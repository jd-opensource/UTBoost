/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#include <iomanip>
#include "UTBoost/tree.h"


namespace UTBoost {

Tree::Tree(int max_leaves, int num_treat)
    :max_leaves_(max_leaves), num_treat_(num_treat) {
  left_child_.resize(max_leaves_ - 1);
  right_child_.resize(max_leaves_ - 1);
  split_feature_.resize(max_leaves_ - 1);
  threshold_in_bin_.resize(max_leaves_ - 1);
  threshold_.resize(max_leaves_ - 1);
  decision_type_.resize(max_leaves_ - 1, 0);
  split_gain_.resize(max_leaves_ - 1);
  leaf_parent_.resize(max_leaves_);
  leaf_value_.resize(max_leaves_ * num_treat_, 0);
  leaf_weight_.resize(max_leaves_);
  leaf_count_.resize(max_leaves_);
  internal_value_.resize((max_leaves_ - 1) * num_treat_, 0);
  internal_weight_.resize(max_leaves_ - 1);
  internal_count_.resize(max_leaves_ - 1);
  leaf_depth_.resize(max_leaves_);
  // root is in the depth 0
  leaf_depth_[0] = 0;
  num_leaves_ = 1;
  // leaf_value_[0] = 0.0f;
  std::fill(leaf_value_.begin(), leaf_value_.begin() + num_treat_, 0.0);
  leaf_weight_[0] = 0.0f;
  leaf_parent_[0] = -1;
  shrinkage_ = 1.0f;
  max_depth_ = -1;
}

int Tree::Split(int leaf, int feature, bin_t threshold_bin,
                double threshold_double, const double* left_value, const double* right_value,
                int left_cnt, int right_cnt, double left_weight, double right_weight,
                float gain, bool missing_as_zero, bool default_left) {
  Split(leaf, feature, left_value, right_value, left_cnt, right_cnt, left_weight, right_weight, gain);
  int new_node_idx = num_leaves_ - 1;
  decision_type_[new_node_idx] = 0;
  // todo:missing?
  SetDecisionType(&decision_type_[new_node_idx], missing_as_zero, kMissingZeroMask);
  SetDecisionType(&decision_type_[new_node_idx], default_left, kDefaultLeftMask);
  threshold_in_bin_[new_node_idx] = threshold_bin;
  threshold_[new_node_idx] = threshold_double;
  ++num_leaves_;
  return num_leaves_ - 1;
}


#define PredictionFun(niter, fidx_in_iter, start_pos, decision_fun, iter_idx,        \
                      data_idx, num_data)                                            \
  std::vector<std::unique_ptr<BinIterator>> iter((niter));                           \
  for (int i = 0; i < (niter); ++i) {                                                \
    iter[i].reset(data->FeatureIterator((fidx_in_iter)));                            \
  }                                                                                  \
  for (data_size_t i = start; i < end; ++i) {                                        \
    int node = 0;                                                                    \
    while (node >= 0) {                                                              \
      node = decision_fun(iter[(iter_idx)]->Get((data_idx)), node,                   \
                          max_bins[node]);                                           \
    }                                                                                \
    for (int j = 0; j < num_treat_; ++j) {                                           \
      score[(data_idx) + (num_data) * j] += leaf_value_[(~node) * (num_treat_) + j]; \
    }                                                                                \
  }                                                                                  \


#define IndexFun(niter, fidx_in_iter, start_pos, decision_fun, iter_idx,             \
                 data_idx)                                                           \
  std::vector<std::unique_ptr<BinIterator>> iter((niter));                           \
  for (int i = 0; i < (niter); ++i) {                                                \
    iter[i].reset(data->FeatureIterator((fidx_in_iter)));                            \
  }                                                                                  \
  for (data_size_t i = start; i < end; ++i) {                                        \
    int node = 0;                                                                    \
    while (node >= 0) {                                                              \
      node = decision_fun(iter[(iter_idx)]->Get((data_idx)), node,                   \
                          max_bins[node]);                                           \
    }                                                                                \
    output[i] = ~node;                                                               \
  }                                                                                  \


void Tree::GetLeafIndex(const Dataset* data, const data_size_t* used_data_indices, data_size_t num_data, int* output) const {
  std::vector<uint32_t> max_bins(num_leaves_ - 1);
  for (int i = 0; i < num_leaves_ - 1; ++i) {
    const int fidx = split_feature_[i];
    max_bins[i] = data->GetFMapperNum(fidx) - 1;
  }
  Threading::For<data_size_t>(0, num_data, 512, [this, &data, output, used_data_indices, &max_bins]
      (int, data_size_t start, data_size_t end) {
    IndexFun(num_leaves_ - 1, split_feature_[i], used_data_indices[start], NumericalDecisionInner, node, used_data_indices[i])
  });
}


void Tree::AddPredictionToScore(const Dataset* data, data_size_t num_data, double* score) const {
  if (num_leaves_ <= 1) {  // only has root node
    for (int i = 0; i < num_treat_; ++i) {
      if (leaf_value_[i] != 0.0f) {
#pragma omp parallel for schedule(static, 512) if (num_data >= 1024)
        for (data_size_t j = 0; j < num_data; ++j) {
          score[i * num_data + j] += leaf_value_[i];
        }
      }
    }
    return;
  }
  std::vector<uint32_t> max_bins(num_leaves_ - 1);
  for (int i = 0; i < num_leaves_ - 1; ++i) {
    const int fidx = split_feature_[i];
    max_bins[i] = data->GetFMapperNum(fidx) - 1;
  }
  Threading::For<data_size_t>(0, num_data, 512, [this, &data, score, &max_bins, &num_data]
      (int, data_size_t start, data_size_t end) {
    PredictionFun(num_leaves_ - 1, split_feature_[i], start, NumericalDecisionInner, node, i, num_data)
  });
}

void Tree::AddPredictionToScore(const Dataset* data,
                                const data_size_t* used_data_indices,
                                data_size_t num_data, double* score) const {
  data_size_t total_num_data = data->GetNumSamples();
  if (num_leaves_ <= 1) {
    for (int i = 0; i < num_treat_; ++i) {
      if (leaf_value_[i] != 0.0f) {
#pragma omp parallel for schedule(static, 512) if (num_data >= 1024)
        for (data_size_t j = 0; j < num_data; ++j) {
          score[i * total_num_data + used_data_indices[j]] += leaf_value_[i];
        }
      }
    }
    return;
  }
  std::vector<uint32_t> max_bins(num_leaves_ - 1);
  for (int i = 0; i < num_leaves_ - 1; ++i) {
    const int fidx = split_feature_[i];
    max_bins[i] = data->GetFMapperNum(fidx) - 1;
  }
  Threading::For<data_size_t>(0, num_data, 512, [this, &data, score, used_data_indices, &max_bins, &total_num_data]
      (int, data_size_t start, data_size_t end) {
    PredictionFun(num_leaves_ - 1, split_feature_[i], used_data_indices[start], NumericalDecisionInner, node, used_data_indices[i], total_num_data)
  });
}

void Tree::RecomputeMaxDepth() {
  if (num_leaves_ == 1) {
    max_depth_ = 0;
  } else {
    if (leaf_depth_.empty()) {
      RecomputeLeafDepths(0, 0);
    }
    max_depth_ = leaf_depth_[0];
    for (int i = 1; i < num_leaves(); ++i) {
      if (max_depth_ < leaf_depth_[i]) max_depth_ = leaf_depth_[i];
    }
  }
}

double Tree::GetUpperBoundValue() const {
  double upper_bound = leaf_value_[0];
  for (int i = 1; i < num_leaves_; ++i) {
    if (leaf_value_[i] > upper_bound) {
      upper_bound = leaf_value_[i];
    }
  }
  return upper_bound;
}

double Tree::GetLowerBoundValue() const {
  double lower_bound = leaf_value_[0];
  for (int i = 1; i < num_leaves_; ++i) {
    if (leaf_value_[i] < lower_bound) {
      lower_bound = leaf_value_[i];
    }
  }
  return lower_bound;
}

std::string Tree::ToString() const {
  std::stringstream str_buf;
  C_stringstream(str_buf);

  str_buf << "num_leaves=" << num_leaves_ << '\n';
  str_buf << "num_treat=" << num_treat_ << '\n';
  str_buf << "split_feature="
          << ArrayToString(split_feature_, num_leaves_ - 1, ' ') << '\n';
  str_buf << "split_gain="
          << ArrayToString(split_gain_, num_leaves_ - 1, ' ') << '\n';
  str_buf << "threshold="
          << ArrayToString(threshold_, num_leaves_ - 1, ' ') << '\n';
  str_buf << "decision_type="
          << ArrayToString(ArrayCast<int8_t, int>(decision_type_), num_leaves_ - 1, ' ') << '\n';
  str_buf << "left_child="
          << ArrayToString(left_child_, num_leaves_ - 1, ' ') << '\n';
  str_buf << "right_child="
          << ArrayToString(right_child_, num_leaves_ - 1, ' ') << '\n';
  str_buf << "leaf_value="
          << ArrayToString(leaf_value_, num_leaves_ * num_treat_, ' ') << '\n';
  str_buf << "leaf_weight="
          << ArrayToString(leaf_weight_, num_leaves_, ' ') << '\n';
  str_buf << "leaf_count="
          << ArrayToString(leaf_count_, num_leaves_, ' ') << '\n';
  str_buf << "internal_value="
          << ArrayToString(internal_value_, (num_leaves_ - 1) * num_treat_, ' ') << '\n';
  str_buf << "internal_weight="
          << ArrayToString(internal_weight_, num_leaves_ - 1, ' ') << '\n';
  str_buf << "internal_count="
          << ArrayToString(internal_count_, num_leaves_ - 1, ' ') << '\n';
  str_buf << "shrinkage=" << shrinkage_ << '\n';
  str_buf << '\n';

  return str_buf.str();
}

std::string Tree::ToJSON() const {
  std::stringstream str_buf;
  C_stringstream(str_buf);
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  str_buf << "\"num_leaves\":" << num_leaves_ << "," << '\n';
  str_buf << "\"shrinkage\":" << shrinkage_ << "," << '\n';
  if (num_leaves_ == 1) {
    str_buf << "\"tree_structure\":{" << "\"leaf_value\":" << leaf_value_[0] << "}" << '\n';
  } else {
    str_buf << "\"tree_structure\":" << NodeToJSON(0) << '\n';
  }
  return str_buf.str();
}


std::string Tree::NodeToJSON(int index) const {
  std::stringstream str_buf;
  C_stringstream(str_buf);
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  if (index >= 0) {
    // non-leaf
    str_buf << "{" << '\n';
    str_buf << "\"split_index\":" << index << "," << '\n';
    str_buf << "\"split_feature\":" << split_feature_[index] << "," << '\n';
    str_buf << "\"split_gain\":" << AvoidInf(split_gain_[index]) << "," << '\n';
    str_buf << "\"threshold\":" << AvoidInf(threshold_[index]) << "," << '\n';
    str_buf << "\"decision_type\":\"<=\"," << '\n';
    if (GetDecisionType(decision_type_[index], kDefaultLeftMask)) {
      str_buf << "\"default_left\":true," << '\n';
    } else {
      str_buf << "\"default_left\":false," << '\n';
    }
    if (GetDecisionType(decision_type_[index], kMissingZeroMask)) {
      str_buf << "\"missing_as_zero\":true," << '\n';
    } else {
      str_buf << "\"missing_as_zero\":false," << '\n';
    }
    str_buf << "\"internal_value\":[" << internal_value_[index * num_treat_];
    for (int i = 1; i < num_treat_; ++i) {
      str_buf << "," << internal_value_[index * num_treat_ + i];
    }
    str_buf << "]," << '\n';
    // str_buf << "\"internal_value\":" << internal_value_[index] << "," << '\n';
    str_buf << "\"internal_weight\":" << internal_weight_[index] << "," << '\n';
    str_buf << "\"internal_count\":" << internal_count_[index] << "," << '\n';
    str_buf << "\"left_child\":" << NodeToJSON(left_child_[index]) << "," << '\n';
    str_buf << "\"right_child\":" << NodeToJSON(right_child_[index]) << '\n';
    str_buf << "}";
  } else {
    // leaf
    index = ~index;
    str_buf << "{" << '\n';
    str_buf << "\"leaf_index\":" << index << "," << '\n';
    str_buf << "\"leaf_value\":[" << leaf_value_[index * num_treat_];
    for (int i = 1; i < num_treat_; ++i) {
      str_buf << "," << leaf_value_[index * num_treat_ + i];
    }
    str_buf << "]," << '\n';
    // str_buf << "\"leaf_value\":" << leaf_value_[index] << "," << '\n';
    str_buf << "\"leaf_weight\":" << leaf_weight_[index] << "," << '\n';
    str_buf << "\"leaf_count\":" << leaf_count_[index] << '\n';
    str_buf << "}";
  }
  return str_buf.str();
}

Tree::Tree(const char* str, size_t* used_len) {
  auto p = str;
  std::unordered_map<std::string, std::string> key_vals;
  const int max_num_line = 22;
  int read_line = 0;
  while (read_line < max_num_line) {
    if (*p == '\r' || *p == '\n') break;
    auto start = p;
    while (*p != '=') ++p;
    std::string key(start, p - start);
    ++p;
    start = p;
    while (*p != '\r' && *p != '\n') ++p;
    key_vals[key] = std::string(start, p - start);
    ++read_line;
    if (*p == '\r') ++p;
    if (*p == '\n') ++p;
  }
  *used_len = p - str;

  if (key_vals.count("num_leaves") <= 0) {
    Log::Error("Tree model should contain num_leaves field");
  }
  Atoi(key_vals["num_leaves"].c_str(), &num_leaves_);

  if (key_vals.count("num_treat") <= 0) {
    Log::Error("Tree model should contain num_treat field");
  }
  Atoi(key_vals["num_treat"].c_str(), &num_treat_);

  if (key_vals.count("leaf_value")) {
    leaf_value_ = StringToArray<double>(key_vals["leaf_value"], ' ', num_leaves_ * num_treat_);
  } else {
    Log::Error("Tree model string format error, should contain leaf_value field");
  }

  if (key_vals.count("shrinkage")) {
    Atof(key_vals["shrinkage"].c_str(), &shrinkage_);
  } else {
    shrinkage_ = 1.0f;
  }

  if ((num_leaves_ <= 1)) {
    return;
  }

  if (key_vals.count("left_child")) {
    left_child_ = StringToArray<int>(key_vals["left_child"], ' ', num_leaves_ - 1);
  } else {
    Log::Error("Tree model string format error, should contain left_child field");
  }

  if (key_vals.count("right_child")) {
    right_child_ = StringToArray<int>(key_vals["right_child"], ' ', num_leaves_ - 1);
  } else {
    Log::Error("Tree model string format error, should contain right_child field");
  }

  if (key_vals.count("split_feature")) {
    split_feature_ = StringToArray<int>(key_vals["split_feature"], ' ', num_leaves_ - 1);
  } else {
    Log::Error("Tree model string format error, should contain split_feature field");
  }

  if (key_vals.count("threshold")) {
    threshold_ = StringToArray<double>(key_vals["threshold"], ' ', num_leaves_ - 1);
  } else {
    Log::Error("Tree model string format error, should contain threshold field");
  }

  if (key_vals.count("split_gain")) {
    split_gain_ = StringToArray<float>(key_vals["split_gain"], ' ', num_leaves_ - 1);
  } else {
    split_gain_.resize(num_leaves_ - 1);
  }

  if (key_vals.count("internal_count")) {
    internal_count_ = StringToArray<int>(key_vals["internal_count"], ' ', num_leaves_ - 1);
  } else {
    internal_count_.resize(num_leaves_ - 1);
  }

  if (key_vals.count("internal_value")) {
    internal_value_ = StringToArray<double>(key_vals["internal_value"], ' ', (num_leaves_ - 1) * num_treat_);
  } else {
    internal_value_.resize(num_leaves_ - 1);
  }

  if (key_vals.count("internal_weight")) {
    internal_weight_ = StringToArray<double>(key_vals["internal_weight"], ' ', num_leaves_ - 1);
  } else {
    internal_weight_.resize(num_leaves_ - 1);
  }

  if (key_vals.count("leaf_weight")) {
    leaf_weight_ = StringToArray<double>(key_vals["leaf_weight"], ' ', num_leaves_);
  } else {
    leaf_weight_.resize(num_leaves_);
  }

  if (key_vals.count("leaf_count")) {
    leaf_count_ = StringToArray<int>(key_vals["leaf_count"], ' ', num_leaves_);
  } else {
    leaf_count_.resize(num_leaves_);
  }

  if (key_vals.count("decision_type")) {
    decision_type_ = StringToArray<int8_t>(key_vals["decision_type"], ' ', num_leaves_ - 1);
  } else {
    decision_type_ = std::vector<int8_t>(num_leaves_ - 1, 0);
  }

  max_depth_ = -1;
}


}  // namespace UTBoost
