/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#include "UTBoost/config.h"

namespace UTBoost {

void ConfigHelper::Parse(const char *params) {
  std::string str_params(params);
  std::vector<std::string> strs = Split(str_params, '\t', true);
  for (auto &str : strs) {
    std::vector<std::string> kv = Split(str, '=', true);
    if (kv.size() == 2) {
      PushBack(kv[0], kv[1]);
    }
  }
}

void Config::ParseParameters(const char *args) {
    helper_.Clear(); helper_.Parse(args);
    helper_.Init();
    while (helper_.Next()) {
      if (helper_.Name() == std::string("objective"))
        objective = helper_.Val();
      else if (helper_.Name() == std::string("split_criteria"))
        split_criteria = helper_.Val();
      else if (helper_.Name() == std::string("bagging_seed"))
        bagging_seed = atoi(helper_.Val().c_str());
      else if (helper_.Name() == std::string("bagging_fraction"))
        bagging_fraction = atof(helper_.Val().c_str());
      else if (helper_.Name() == std::string("min_data_in_leaf"))
        min_data_in_leaf = atoi(helper_.Val().c_str());
      else if (helper_.Name() == std::string("feature_fraction_seed"))
        feature_fraction_seed = atoi(helper_.Val().c_str());
      else if (helper_.Name() == std::string("feature_fraction"))
        feature_fraction = atof(helper_.Val().c_str());
      else if (helper_.Name() == std::string("feature_fraction_bynode"))
        feature_fraction_bynode = atof(helper_.Val().c_str());
      else if (helper_.Name() == std::string("boost_from_average"))
        boost_from_average = atoi(helper_.Val().c_str()) > 0;
      else if (helper_.Name() == std::string("ensemble"))
        ensemble = helper_.Val();
      else if (helper_.Name() == std::string("num_leaves"))
        num_leaves = atoi(helper_.Val().c_str());
      else if (helper_.Name() == std::string("learning_rate"))
        learning_rate = atof(helper_.Val().c_str());
      else if (helper_.Name() == std::string("num_threads"))
        num_threads = atoi(helper_.Val().c_str());
      else if (helper_.Name() == std::string("scale_treat_weights")) {
        scale_treat_weight.clear();
        std::vector<std::string> weights = Split(helper_.Val(), ',');
        for (auto & weight : weights) {
          scale_treat_weight.emplace_back(atof(weight.c_str()));
        }
      }
      else if (helper_.Name() == std::string("gbm_gain_type"))
        gbm_gain_type_ = atoi(helper_.Val().c_str());
      else if (helper_.Name() == std::string("auto_balance"))
        auto_balance = atoi(helper_.Val().c_str()) > 0;
      else if (helper_.Name() == std::string("seed"))
        seed = atoi(helper_.Val().c_str());
      else if (helper_.Name() == std::string("max_depth"))
        max_depth = atoi(helper_.Val().c_str());
      else if (helper_.Name() == std::string("subsample"))
        subsample = atof(helper_.Val().c_str());
      else if (helper_.Name() == std::string("bagging_freq"))
        bagging_freq = atoi(helper_.Val().c_str());
      else if (helper_.Name() == std::string("colsample"))
        colsample = atof(helper_.Val().c_str());
      else if (helper_.Name() == std::string("min_gain_to_split"))
        min_gain_to_split = atof(helper_.Val().c_str());
      else if (helper_.Name() == std::string("use_honesty"))
        use_honesty = atoi(helper_.Val().c_str()) > 0;
      else if (helper_.Name() == std::string("max_bin"))
        max_bin = atoi(helper_.Val().c_str());
      else if (helper_.Name() == std::string("min_data_in_bin"))
        min_data_in_bin = atoi(helper_.Val().c_str());
      else if (helper_.Name() == std::string("bin_construct_sample_cnt"))
        bin_construct_sample_cnt = atoi(helper_.Val().c_str());
      else if (helper_.Name() == std::string("verbose"))
        verbose = atoi(helper_.Val().c_str()) > 0;
      else if (helper_.Name() == std::string("metric"))
        metric = Split(helper_.Val(), ',');
      else if (helper_.Name() == std::string("effect_constrains")) {
        effect_constrains.clear();
        std::vector<std::string> weights = Split(helper_.Val(), ',');
        for (auto & weight : weights) {
          effect_constrains.emplace_back(atoi(weight.c_str()));
        }
      }
      else
        Log::Debug("Unknown parameter: %s", helper_.Name().c_str());

      if (verbose) {
        Log::ResetLogLevel(LogLevel::Info);
      } else {
        Log::ResetLogLevel(LogLevel::Error);
      }
    }
}

} // namespace UTBoost