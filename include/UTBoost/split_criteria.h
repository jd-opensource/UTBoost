/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 * Created by Junjie Gao on 2023/3/1.
 */

#ifndef UTBOOST_INCLUDE_UTBOOST_SPLIT_CRITERIA_H_
#define UTBOOST_INCLUDE_UTBOOST_SPLIT_CRITERIA_H_

#include <vector>

#include <UTBoost/config.h>
#include <UTBoost/dataset.h>
#include <UTBoost/definition.h>
#include "UTBoost/bin.h"

namespace UTBoost {

/*! \brief The interface of split criterion. */
class SplitCriteria {
 public:
  /*!
   * \brief compute split gain given entry
   * \param entry input entry
   * \return split gain
   */
  virtual double GetSplitGains(const BinEntry* entry) const = 0;

  /*!
   * \brief compute leaf output
   * \param entry input entry
   * \return leaf values
   */
  virtual std::vector<double> CalculateLeafOutput(const BinEntry* entry) const = 0;

  /*!
   * \brief compute split score given left, right and root entry
   * \return split score
   */
  virtual double SplitScore(const BinEntry* left, const BinEntry* right, const BinEntry* parent) const = 0;

  virtual std::string ToString() const = 0;

  /*!
   * \brief Create split criteria function
   * \param type Specific type of criteria function
   * \param config Config for criteria function
   */
  static SplitCriteria* Create(const std::string& type, const Config* config);

  SplitCriteria() = default;
  /*! \brief Disable copy */
  SplitCriteria& operator=(const SplitCriteria&) = delete;
  /*! \brief Disable copy */
  SplitCriteria(const SplitCriteria&) = delete;

};

}  // namespace UTBoost

#endif //UTBOOST_INCLUDE_UTBOOST_SPLIT_CRITERIA_H_
