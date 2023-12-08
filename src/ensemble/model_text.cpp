/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#include <string>
#include <sstream>
#include <vector>

#include "UTBoost/config.h"
#include "UTBoost/metric.h"
#include "UTBoost/split_criteria.h"
#include "UTBoost/utils/common.h"
#include "UTBoost/utils/file_io.h"

#include "causal_gbm.h"

namespace UTBoost {

const char* kModelVersion = "v0.1";

std::string CausalGBM::DumpModel(int start_iteration, int num_iteration) const {
  std::stringstream str_buf;
  C_stringstream(str_buf);

  str_buf << "{";
  str_buf << "\"name\":\"" << SubModelName() << "\"," << '\n';
  str_buf << "\"version\":\"" << kModelVersion << "\"," << '\n';
  str_buf << "\"num_treat\":" << num_treat_ << "," << '\n';
  str_buf << "\"max_feature_idx\":" << max_feature_idx_ << "," << '\n';
  if (objective_function_ != nullptr) {
    str_buf << "\"objective\":\"" << objective_function_->ToString() << "\",\n";
  }

  str_buf << "\"average_output\":" << (average_output_ ? "true" : "false") << ",\n";

  str_buf << "\"feature_infos\":" << "{";
  bool first_obj = true;
  str_buf << "}," << '\n';

  str_buf << "\"tree_info\":[";
  int num_used_model = static_cast<int>(models_.size());
  int total_iteration = num_used_model;
  start_iteration = std::max(start_iteration, 0);
  start_iteration = std::min(start_iteration, total_iteration);
  if (num_iteration > 0) {
    int end_iteration = start_iteration + num_iteration;
    num_used_model = std::min(end_iteration, num_used_model);
  }
  int start_model = start_iteration;
  for (int i = start_model; i < num_used_model; ++i) {
    if (i > start_model) {
      str_buf << ",";
    }
    str_buf << "{";
    str_buf << "\"tree_index\":" << i << ",";
    str_buf << models_[i]->ToJSON();
    str_buf << "}";
  }
  str_buf << "]" << '\n';

  str_buf << "}" << '\n';

  return str_buf.str();
}

std::string CausalGBM::SaveModelToString(int start_iteration, int num_iteration, int feature_importance_type) const {
  std::stringstream ss;
  C_stringstream(ss);

  // output model type
  ss << SubModelName() << '\n';
  ss << "version=" << kModelVersion << '\n';
  // output number of treatment
  ss << "num_treat=" << num_treat_ << '\n';
  // output max_feature_idx
  ss << "max_feature_idx=" << max_feature_idx_ << '\n';
  // output objective
  if (objective_function_ != nullptr) {
    ss << "objective=" << objective_function_->ToString() << '\n';
  }
  // split criteria
  if (split_criteria_ != nullptr) {
    ss << "split_criteria=" << split_criteria_->ToString() << '\n';
  }

  if (average_output_) {
    ss << "average_output" << '\n';
  }

  int num_used_model = static_cast<int>(models_.size());
  int total_iteration = num_used_model;
  start_iteration = std::max(start_iteration, 0);
  start_iteration = std::min(start_iteration, total_iteration);
  if (num_iteration > 0) {
    int end_iteration = start_iteration + num_iteration;
    num_used_model = std::min(end_iteration, num_used_model);
  }

  int start_model = start_iteration;

  std::vector<std::string> tree_strs(num_used_model - start_model);
  std::vector<size_t> tree_sizes(num_used_model - start_model);
  // output tree models
#pragma omp parallel for schedule(static)
  for (int i = start_model; i < num_used_model; ++i) {
    const int idx = i - start_model;
    tree_strs[idx] = "Tree=" + std::to_string(idx) + '\n';
    tree_strs[idx] += models_[i]->ToString() + '\n';
    tree_sizes[idx] = tree_strs[idx].size();
  }

  ss << "tree_sizes=" << Join(tree_sizes, " ") << '\n';
  ss << '\n';

  for (int i = 0; i < num_used_model - start_model; ++i) {
    ss << tree_strs[i];
    tree_strs[i].clear();
  }
  ss << "end of trees" << "\n";
  std::vector<double> feature_importances = FeatureImportance(
      num_iteration, feature_importance_type);
  // store the importance first
  std::vector<std::pair<size_t, std::string>> pairs;
  for (size_t i = 0; i < feature_importances.size(); ++i) {
    size_t feature_importances_int = static_cast<size_t>(feature_importances[i]);
    if (feature_importances_int > 0) {
      // todo: name
      // pairs.emplace_back(feature_importances_int, feature_names_[i]);
      pairs.emplace_back(feature_importances_int, "f" + std::to_string(i));
    }
  }
  // sort the importance
  std::stable_sort(pairs.begin(), pairs.end(),
                   [](const std::pair<size_t, std::string> &lhs,
                      const std::pair<size_t, std::string> &rhs) {
                     return lhs.first > rhs.first;
                   });
  ss << '\n' << "feature_importances:" << '\n';
  for (size_t i = 0; i < pairs.size(); ++i) {
    ss << pairs[i].second << "=" << std::to_string(pairs[i].first) << '\n';
  }
  if (config_ != nullptr) {
    ss << "\nparameters:" << '\n';
    ss << config_->ToString() << "\n";
    ss << "end of parameters" << '\n';
  }
  return ss.str();
}


bool CausalGBM::SaveModelToFile(int start_iteration, int num_iteration, int feature_importance_type, const char* filename) const {
  /*! \brief File to write models */
  auto writer = VirtualFileWriter::Make(filename);
  if (!writer->Init()) {
    Log::Error("Model file %s is not available for writes", filename);
  }
  std::string str_to_write = SaveModelToString(start_iteration, num_iteration, feature_importance_type);
  auto size = writer->Write(str_to_write.c_str(), str_to_write.size());
  return size > 0;
}

bool CausalGBM::DumpModelToFile(int start_iteration, int num_iteration, const char* filename) const {
  /*! \brief File to write models */
  auto writer = VirtualFileWriter::Make(filename);
  if (!writer->Init()) {
    Log::Error("Model file %s is not available for writes", filename);
  }
  std::string str_to_write = DumpModel(start_iteration, num_iteration);
  auto size = writer->Write(str_to_write.c_str(), str_to_write.size());
  return size > 0;
}


bool CausalGBM::LoadModelFromString(const char* buffer, size_t len) {
  // use serialized string to restore this object
  models_.clear();
  auto c_str = buffer;
  auto p = c_str;
  auto end = p + len;
  std::unordered_map<std::string, std::string> key_vals;
  while (p < end) {
    auto line_len = GetLine(p);
    if (line_len > 0) {
      std::string cur_line(p, line_len);
      if (!StartsWith(cur_line, "Tree=")) {
        auto strs = Split(cur_line, '=');
        if (strs.size() == 1) {
          key_vals[strs[0]] = "";
        } else if (strs.size() == 2) {
          key_vals[strs[0]] = strs[1];
        } else if (strs.size() > 2) {
          Log::Error("Wrong line at model file: %s", cur_line.substr(0, std::min<size_t>(128, cur_line.size())).c_str());
        }
      } else {
        break;
      }
    }
    p += line_len;
    p = SkipNewLine(p);
  }

  // get number of treatment
  if (key_vals.count("num_treat")) {
    Atoi(key_vals["num_treat"].c_str(), &num_treat_);
  } else {
    Log::Error("Model file doesn't specify the number of treatment");
    return false;
  }

  // get max_feature_idx first
  if (key_vals.count("max_feature_idx")) {
    Atoi(key_vals["max_feature_idx"].c_str(), &max_feature_idx_);
  } else {
    Log::Error("Model file doesn't specify max_feature_idx");
    return false;
  }

  // get average_output
  if (key_vals.count("average_output")) {
    average_output_ = true;
  }

  if (key_vals.count("objective")) {
    auto str = key_vals["objective"];
    Config _cfg;
    loaded_objective_.reset(ObjectiveFunction::CreateObjectiveFunction(str, _cfg));
    objective_function_ = loaded_objective_.get();
  }

  if (!key_vals.count("tree_sizes")) {
    while (p < end) {
      auto line_len = GetLine(p);
      if (line_len > 0) {
        std::string cur_line(p, line_len);
        if (StartsWith(cur_line, "Tree=")) {
          p += line_len;
          p = SkipNewLine(p);
          size_t used_len = 0;
          models_.emplace_back(new Tree(p, &used_len));
          p += used_len;
        } else {
          break;
        }
      }
      p = SkipNewLine(p);
    }
  } else {
    std::vector<size_t> tree_sizes = StringToArray<size_t>(key_vals["tree_sizes"].c_str(), ' ');
    std::vector<size_t> tree_boundries(tree_sizes.size() + 1, 0);
    int num_trees = static_cast<int>(tree_sizes.size());
    for (int i = 0; i < num_trees; ++i) {
      tree_boundries[i + 1] = tree_boundries[i] + tree_sizes[i];
      models_.emplace_back(nullptr);
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_trees; ++i) {

      auto cur_p = p + tree_boundries[i];
      auto line_len = GetLine(cur_p);
      std::string cur_line(cur_p, line_len);
      if (StartsWith(cur_line, "Tree=")) {
        cur_p += line_len;
        cur_p = SkipNewLine(cur_p);
        size_t used_len = 0;
        models_[i].reset(new Tree(cur_p, &used_len));
      } else {
        Log::Error("Model format error, expect a tree here. met %s", cur_line.c_str());
      }
    }
  }
  num_iteration_for_pred_ = static_cast<int>(models_.size());
  iter_ = 0;
  return true;
}

std::vector<double> CausalGBM::FeatureImportance(int num_iteration, int importance_type) const {
  int num_used_model = static_cast<int>(models_.size());
  if (num_iteration > 0) {
    num_iteration += 0;
    num_used_model = std::min(num_iteration, num_used_model);
  }

  std::vector<double> feature_importances(max_feature_idx_ + 1, 0.0);
  if (importance_type == 0) {
    for (int iter = 0; iter < num_used_model; ++iter) {
      for (int split_idx = 0; split_idx < models_[iter]->num_leaves() - 1; ++split_idx) {
        if (models_[iter]->split_gain(split_idx) > 0) {
          feature_importances[models_[iter]->split_feature(split_idx)] += 1.0;
        }
      }
    }
  } else if (importance_type == 1) {
    for (int iter = 0; iter < num_used_model; ++iter) {
      for (int split_idx = 0; split_idx < models_[iter]->num_leaves() - 1; ++split_idx) {
        if (models_[iter]->split_gain(split_idx) > 0) {
          feature_importances[models_[iter]->split_feature(split_idx)] += models_[iter]->split_gain(split_idx);
        }
      }
    }
  } else {
    Log::Error("Unknown importance type: only support split=0 and gain=1");
  }
  return feature_importances;
}

}  // namespace UTBoost
