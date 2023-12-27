/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 * Created by Junjie Gao on 2023/3/1.
 */

#ifndef UTBOOST_INCLUDE_UTBOOST_CONFIG_H_
#define UTBOOST_INCLUDE_UTBOOST_CONFIG_H_

#include <string>
#include <vector>

#include <cstdlib>

#include "UTBoost/utils/common.h"
#include "UTBoost/definition.h"
#include "UTBoost/utils/log_wrapper.h"


namespace UTBoost{

/*! \brief Helper class to parse c_str type parameter */
class ConfigHelper {
 public:
  /*! \brief Constructor */
  ConfigHelper() { idx_ = 0; }

  /*! \brief clear all parameters */
  inline void Clear() {
    names_.clear(); values_.clear();
    idx_ = 0;
  };

  /*!
   * \brief parse parameters from c string
   * \param params parameters in string
   */
  void Parse(const char* params);

  /*!
   * \brief push back a parameter setting
   * \param name name of parameter
   * \param val value of parameter
   */
  inline void PushBack(const std::string& name, const std::string& val) {
    names_.emplace_back(name); values_.emplace_back(val);
  }

  /*! \brief Set pointer to beginning of the ConfigSaver */
  inline void Init() { idx_ = 0; };

  /*!
   * \brief move iterator to next position
   * \return true if there is value in next position
   */
  inline bool Next() {
    if (idx_ < names_.size()) {
      ++idx_;
      return true;
    } else {
      return false;
    }
  };

  /*!
   * \brief get current name, called after next returns true
   * \return current parameter name
   */
  const std::string& Name() const { return names_[idx_ > 0 ? idx_ - 1 : idx_]; }

  /*!
   * \brief get current value, called after next returns true
   * \return current parameter value
   */
  const std::string& Val() const { return values_[idx_ > 0 ? idx_ - 1 : idx_]; }

 private:
  // parameter names
  std::vector<std::string> names_;
  // parameter values
  std::vector<std::string> values_;
  // used to record current parameter idx
  size_t idx_;
};


struct Config {
 public:
  std::string objective = "logloss";
  std::string split_criteria = "gbm";
  int bagging_seed = 123;
  int min_data_in_leaf = 100;
  double bagging_fraction = 0.5;
  int feature_fraction_seed = 1234;
  double feature_fraction = 0.5;
  double feature_fraction_bynode = 1.0;
  bool boost_from_average = true;
  std::string ensemble = "boost";
  int num_leaves = 7;
  double learning_rate = 0.1;
  int num_threads = 0;
  int gbm_gain_type_ = 0;
  std::vector<double> scale_treat_weight;
  std::vector<int32_t> effect_constrains;
  bool auto_balance = false;
  int seed = 0;
  int max_depth = 5;
  double subsample = 1.0;
  int bagging_freq = 1;
  double colsample = 1.0;
  double min_gain_to_split = 0.0;
  bool use_honesty = true;
  int max_bin = 255;
  int min_data_in_bin = 10;
  int bin_construct_sample_cnt = 200000;
  std::vector<std::string> metric;
  bool verbose = false;

  /*!
   * \brief Parse config from c_str
   * \param args parameter string
   */
  void ParseParameters(const char* args);

  std::string ToString() const { return ""; }

 private:
  // parameter parser
  ConfigHelper helper_;
};

}


#endif //UTBOOST_INCLUDE_UTBOOST_CONFIG_H_
