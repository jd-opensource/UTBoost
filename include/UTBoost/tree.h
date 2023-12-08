/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 * Created by Junjie Gao on 2023/3/1.
 */

#ifndef UTBOOST_INCLUDE_UTBOOST_TREE_H_
#define UTBOOST_INCLUDE_UTBOOST_TREE_H_

#include <string>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cmath>

#include "UTBoost/definition.h"
#include "UTBoost/dataset.h"
#include "UTBoost/utils/threading.h"

#define kMissingZeroMask (1)
#define kDefaultLeftMask (2)

namespace UTBoost {

class Tree {
 public:
  /*!
   * \brief Constructor
   * \param max_leaves The number of max leaves
   * \param track_branch_features Whether to keep track of ancestors of leaf nodes
   * \param is_linear Whether the tree has linear models at each leaf
   */
  explicit Tree(int max_leaves, int num_treat);

  /*!
   * \brief Constructor, from a string
   * \param str Model string
   * \param used_len used count of str
   */
  Tree(const char* str, size_t* used_len);

  virtual ~Tree() noexcept = default;

  /*!
   * \brief Performing a split on tree leaves.
   * \param leaf Index of leaf to be split
   * \param feature Index of feature; the converted index after removing useless features
   * \param threshold_bin Threshold(bin) of split
   * \param threshold_double Threshold on feature value
   * \param left_value Model Left child output
   * \param right_value Model Right child output
   * \param left_cnt Count of left child
   * \param right_cnt Count of right child
   * \param left_weight Weight of left child
   * \param right_weight Weight of right child
   * \param gain Split gain
   * \param missing_as_zero whether treated missing as zero
   * \param default_left default direction for missing value
   * \return The index of new leaf.
   */
  int Split(int leaf, int feature, bin_t threshold_bin,
            double threshold_double, const double* left_value, const double* right_value,
            int left_cnt, int right_cnt, double left_weight, double right_weight,
            float gain, bool missing_as_zero, bool default_left);

  /*! \brief Get the output of one leaf */
  inline const double* LeafOutput(int leaf) const {
    return leaf_value_.data() + leaf * num_treat_;
  }

  /*! \brief Set the output of one leaf */
  inline void SetLeafOutput(int leaf, const double* output, int num_output) {
    ASSERT_EQ(num_output, num_treat_);
    for (int i = 0; i < num_treat_; ++i) {
      leaf_value_[leaf * num_treat_ + i] = MaybeRoundToZero(output[i]);
    }
  }

  /*! \brief Set the output of one leaf */
  inline void SetInternalLeafOutput(int leaf, const double* output, int num_output) {
    ASSERT_EQ(num_output, num_treat_);
    for (int i = 0; i < num_treat_; ++i) {
      internal_value_[leaf * num_treat_ + i] = MaybeRoundToZero(output[i]);
    }
  }

  /*!
   * \brief Adding prediction value of this tree model to scores
   * \param data The dataset
   * \param num_data Number of total data
   * \param score Will add prediction to score
   */
  virtual void AddPredictionToScore(const Dataset* data,
                                    data_size_t num_data,
                                    double* score) const;

  /*!
   * \brief Adding prediction value of this tree model to scores
   * \param data The dataset
   * \param used_data_indices Indices of used data
   * \param num_data Number of total data
   * \param score Will add prediction to score
   */
  virtual void AddPredictionToScore(const Dataset* data,
                                    const data_size_t* used_data_indices,
                                    data_size_t num_data, double* score) const;


  virtual void GetLeafIndex(const Dataset* data, const data_size_t* used_data_indices,
                            data_size_t num_data, int* output) const;

  /*!
   * \brief Get upper bound leaf value of this tree model
   */
  double GetUpperBoundValue() const;

  /*!
   * \brief Get lower bound leaf value of this tree model
   */
  double GetLowerBoundValue() const;

  /*!
   * \brief Prediction on one record
   * \param feature_values Feature value of this record
   * \return Prediction result
   */
  inline const double* Predict(const double* feature_values) const;
  inline const double* PredictByMap(const std::unordered_map<int, double>& feature_values) const;

  inline int PredictLeafIndex(const double* feature_values) const;
  inline int PredictLeafIndexByMap(const std::unordered_map<int, double>& feature_values) const;

  /*! \brief Get Number of leaves*/
  inline int num_leaves() const { return num_leaves_; }

  /*! \brief Get depth of specific leaf*/
  inline int leaf_depth(int leaf_idx) const { return leaf_depth_[leaf_idx]; }

  /*! \brief Get parent of specific leaf*/
  inline int leaf_parent(int leaf_idx) const {return leaf_parent_[leaf_idx]; }

  /*! \brief Get feature of specific split (original feature index)*/
  inline int split_feature(int split_idx) const { return split_feature_[split_idx]; }

  /*! \brief Get features on leaf's branch*/
  inline std::vector<int> branch_features(int leaf) const { return branch_features_[leaf]; }

  inline double split_gain(int split_idx) const { return split_gain_[split_idx]; }

  inline double internal_value(int node_idx) const {
    return internal_value_[node_idx];
  }

  inline bool IsNumericalSplit(int node_idx) const {
    return true;
  }

  inline int left_child(int node_idx) const { return left_child_[node_idx]; }

  inline int right_child(int node_idx) const { return right_child_[node_idx]; }

  inline bin_t threshold_in_bin(int node_idx) const {
    return threshold_in_bin_[node_idx];
  }

  /*! \brief Get the number of data points that fall at or below this node*/
  inline int data_count(int node) const { return node >= 0 ? internal_count_[node] : leaf_count_[~node]; }

  /*!
   * \brief Shrinkage for the tree's output
   *        shrinkage rate (a.k.a learning rate) is used to tune the training process
   * \param rate The factor of shrinkage
   */
  virtual inline void Shrinkage(double rate) {
#pragma omp parallel for schedule(static, 1024) if (num_leaves_ >= 2048)
    for (int i = 0; i < num_leaves_ - 1; ++i) {
      for (int j = 0; j < num_treat_; ++j) {
        leaf_value_[i * num_treat_ + j] = MaybeRoundToZero(leaf_value_[i * num_treat_ + j] * rate);
        internal_value_[i * num_treat_ + j] = MaybeRoundToZero(internal_value_[i * num_treat_ + j] * rate);
      }
    }
    for (int i = 0; i < num_treat_; ++i) {
      leaf_value_[(num_leaves_ - 1) * num_treat_ + i] =
          MaybeRoundToZero(leaf_value_[(num_leaves_ - 1) * num_treat_ + i] * rate);
    }

    shrinkage_ *= rate;
  }

  inline double shrinkage() const { return shrinkage_; }

  virtual inline void AddBias(double val, treatment_t treat_id) {
#pragma omp parallel for schedule(static, 1024) if (num_leaves_ >= 2048)
    for (int i = 0; i < num_leaves_ - 1; ++i) {
      leaf_value_[treat_id + i * num_treat_] = MaybeRoundToZero(leaf_value_[treat_id + i * num_treat_] + val);
      internal_value_[treat_id + i * num_treat_] = MaybeRoundToZero(internal_value_[treat_id + i * num_treat_] + val);
    }
    leaf_value_[treat_id + (num_leaves_ - 1) * num_treat_] = MaybeRoundToZero(leaf_value_[treat_id + (num_leaves_ - 1) * num_treat_] + val);
    // force to 1.0
    shrinkage_ = 1.0f;
  }

  virtual inline void AsConstantTree(double val, treatment_t treat_id) {
    num_leaves_ = 1;
    shrinkage_ = 1.0f;
    leaf_value_[treat_id] = val;
  }

  /*! \brief Serialize this object to string*/
  std::string ToString() const;

  /*! \brief Serialize this object to json*/
  std::string ToJSON() const;

  inline static bool IsZero(double fval) {
    return (fval >= -kZeroThreshold && fval <= kZeroThreshold);
  }

  inline static double MaybeRoundToZero(double fval) {
    return IsZero(fval) ? 0 : fval;
  }

  inline static bool GetDecisionType(int8_t decision_type, int8_t mask) {
    return (decision_type & mask) > 0;
  }

  inline static void SetDecisionType(int8_t* decision_type, bool input, int8_t mask) {
    if (input) {
      (*decision_type) |= mask;
    } else {
      (*decision_type) &= (127 - mask);
    }
  }

  inline static int8_t GetMissingType(int8_t decision_type) {
    return (decision_type >> 2) & 3;
  }

  inline static void SetMissingType(int8_t* decision_type, bool input) {
    (*decision_type) &= 1;
    if (input) {
      (*decision_type) |= 3;
    } else {}

  }

  void RecomputeMaxDepth();

  int NextLeafId() const { return num_leaves_; }

  /*! \brief Get the linear model constant term (bias) of one leaf */
  inline double LeafConst(int leaf) const { return leaf_const_[leaf]; }

  /*! \brief Get the linear model coefficients of one leaf */
  inline std::vector<double> LeafCoeffs(int leaf) const { return leaf_coeff_[leaf]; }

  /*! \brief Get the linear model features of one leaf */
  inline std::vector<int> LeafFeaturesInner(int leaf) const {return leaf_features_inner_[leaf]; }

  /*! \brief Get the linear model features of one leaf */
  inline std::vector<int> LeafFeatures(int leaf) const {return leaf_features_[leaf]; }

  /*! \brief Set the linear model features on one leaf */
  inline void SetLeafFeatures(int leaf, const std::vector<int>& features) {
    leaf_features_[leaf] = features;
  }

 protected:
  inline int NumericalDecision(double fval, int node) const {
    bool missing_as_zero = GetDecisionType(decision_type_[node], kMissingZeroMask);
    if (std::isnan(fval)) {
      if (missing_as_zero) {
        fval = 0.0;
      } else if (GetDecisionType(decision_type_[node], kDefaultLeftMask)) {
        return left_child_[node];
      } else {
        return right_child_[node];
      }
    }
    if (fval <= threshold_[node]) {
      return left_child_[node];
    } else {
      return right_child_[node];
    }
  }

  inline int NumericalDecisionInner(bin_t fval, int node, bin_t max_bin) const {
    bool use_missing = !GetDecisionType(decision_type_[node], kMissingZeroMask);
    if (fval == max_bin && use_missing) {
      if (GetDecisionType(decision_type_[node], kDefaultLeftMask)) {
        return left_child_[node];
      } else {
        return right_child_[node];
      }
    }
    if (fval <= threshold_in_bin_[node]) {
      return left_child_[node];
    } else {
      return right_child_[node];
    }
  }

  inline int Decision(double fval, int node) const {
    return NumericalDecision(fval, node);
  }

  inline int DecisionInner(bin_t fval, int node, bin_t default_bin, bin_t max_bin) const {
    return NumericalDecisionInner(fval, node, max_bin);
  }

  inline void Split(int leaf, int feature, const double* left_value, const double* right_value, int left_cnt, int right_cnt,
                    double left_weight, double right_weight, float gain);
  /*!
   * \brief Find leaf index of which record belongs by features
   * \param feature_values Feature value of this record
   * \return Leaf index
   */
  inline int GetLeaf(const double* feature_values) const;
  inline int GetLeafByMap(const std::unordered_map<int, double>& feature_values) const;

  /*! \brief Serialize one node to json*/
  std::string NodeToJSON(int index) const;

  /*! \brief This is used fill in leaf_depth_ after reloading a model*/
  inline void RecomputeLeafDepths(int node = 0, int depth = 0);

  /*! \brief Number of max leaves*/
  int max_leaves_;
  /*! \brief Number of current leaves*/
  int num_leaves_;
  // number of treatment
  int num_treat_;
  // following values used for non-leaf node
  /*! \brief A non-leaf node's left child */
  std::vector<int> left_child_;
  /*! \brief A non-leaf node's right child */
  std::vector<int> right_child_;
  /*! \brief A non-leaf node's split feature, the original index */
  std::vector<int> split_feature_;
  /*! \brief A non-leaf node's split threshold in bin */
  std::vector<bin_t> threshold_in_bin_;
  /*! \brief A non-leaf node's split threshold in feature value */
  std::vector<double> threshold_;
  /*! \brief Store the information for categorical feature handle and missing value handle. */
  std::vector<int8_t> decision_type_;
  /*! \brief A non-leaf node's split gain */
  std::vector<float> split_gain_;
  // used for leaf node
  /*! \brief The parent of leaf */
  std::vector<int> leaf_parent_;
  /*! \brief Output of leaves */
  std::vector<double> leaf_value_;
  /*! \brief weight of leaves */
  std::vector<double> leaf_weight_;
  /*! \brief DataCount of leaves */
  std::vector<int> leaf_count_;
  /*! \brief Output of non-leaf nodes */
  std::vector<double> internal_value_;
  /*! \brief weight of non-leaf nodes */
  std::vector<double> internal_weight_;
  /*! \brief DataCount of non-leaf nodes */
  std::vector<int> internal_count_;
  /*! \brief Depth for leaves */
  std::vector<int> leaf_depth_;
  /*! \brief Features on leaf's branch, original index */
  std::vector<std::vector<int>> branch_features_;
  double shrinkage_;
  int max_depth_;
  /*! \brief coefficients of linear models on leaves */
  std::vector<std::vector<double>> leaf_coeff_;
  /*! \brief constant term (bias) of linear models on leaves */
  std::vector<double> leaf_const_;
  /* \brief features used in leaf linear models; indexing is relative to num_total_features_ */
  std::vector<std::vector<int>> leaf_features_;
  /* \brief features used in leaf linear models; indexing is relative to used_features_ */
  std::vector<std::vector<int>> leaf_features_inner_;

};


inline void Tree::Split(int leaf, int feature, const double* left_value, const double* right_value, int left_cnt, int right_cnt,
                  double left_weight, double right_weight, float gain) {
  int new_node_idx = num_leaves_ - 1;
  // update parent info
  int parent = leaf_parent_[leaf];
  if (parent >= 0) {
    // if cur node is left child
    if (left_child_[parent] == ~leaf) {
      left_child_[parent] = new_node_idx;
    } else {
      right_child_[parent] = new_node_idx;
    }
  }
  // add new node
  split_feature_[new_node_idx] = feature;
  split_gain_[new_node_idx] = gain;
  // add two new leaves
  left_child_[new_node_idx] = ~leaf;
  right_child_[new_node_idx] = ~num_leaves_;
  // update new leaves
  leaf_parent_[leaf] = new_node_idx;
  leaf_parent_[num_leaves_] = new_node_idx;
  // save current leaf value to internal node before change
  internal_weight_[new_node_idx] = leaf_weight_[leaf];
  SetInternalLeafOutput(new_node_idx, LeafOutput(leaf), num_treat_);
  // internal_value_[new_node_idx] = leaf_value_[leaf];
  internal_count_[new_node_idx] = left_cnt + right_cnt;
  SetLeafOutput(leaf, left_value, num_treat_);

  leaf_weight_[leaf] = left_weight;
  leaf_count_[leaf] = left_cnt;
  SetLeafOutput(num_leaves_, right_value, num_treat_);

  leaf_weight_[num_leaves_] = right_weight;
  leaf_count_[num_leaves_] = right_cnt;
  // update leaf depth
  leaf_depth_[num_leaves_] = leaf_depth_[leaf] + 1;
  leaf_depth_[leaf]++;
}

inline const double* Tree::Predict(const double* feature_values) const {
  if (num_leaves_ > 1) {
    int leaf = GetLeaf(feature_values);
    return LeafOutput(leaf);
  } else {
    return leaf_value_.data();
  }
}

inline const double* Tree::PredictByMap(const std::unordered_map<int, double>& feature_values) const {
  if (num_leaves_ > 1) {
    int leaf = GetLeafByMap(feature_values);
    return LeafOutput(leaf);
  } else {
    return leaf_value_.data();
  }
}


inline int Tree::PredictLeafIndex(const double* feature_values) const {
  if (num_leaves_ > 1) {
    int leaf = GetLeaf(feature_values);
    return leaf;
  } else {
    return 0;
  }
}

inline int Tree::PredictLeafIndexByMap(const std::unordered_map<int, double>& feature_values) const {
  if (num_leaves_ > 1) {
    int leaf = GetLeafByMap(feature_values);
    return leaf;
  } else {
    return 0;
  }
}

inline void Tree::RecomputeLeafDepths(int node, int depth) {
  if (node == 0) leaf_depth_.resize(num_leaves());
  if (node < 0) {
    leaf_depth_[~node] = depth;
  } else {
    RecomputeLeafDepths(left_child_[node], depth + 1);
    RecomputeLeafDepths(right_child_[node], depth + 1);
  }
}

inline int Tree::GetLeaf(const double* feature_values) const {
  int node = 0;
  while (node >= 0) {
    node = NumericalDecision(feature_values[split_feature_[node]], node);
  }
  return ~node;
}

inline int Tree::GetLeafByMap(const std::unordered_map<int, double>& feature_values) const {
  int node = 0;
  while (node >= 0) {
    node = NumericalDecision(feature_values.count(split_feature_[node]) > 0 ? feature_values.at(split_feature_[node]) : 0.0f, node);
  }
  return ~node;
}

}  // namespace UTBoost

#endif //UTBOOST_INCLUDE_UTBOOST_TREE_H_
