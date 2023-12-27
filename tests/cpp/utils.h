#ifndef UTBOOST_TESTS_CPP_UTILS_H_
#define UTBOOST_TESTS_CPP_UTILS_H_

#include <vector>
#include <cmath>

#include "UTBoost/utils/random.h"

/*!
 * Creates a batch of Metadata of random values.
 */
void CreateRandomMetadata(int nrows, std::vector<float>* labels, std::vector<int>* treats) {
  UTBoost::Random rand(42);
  labels->reserve(nrows);
  treats->reserve(nrows);
  for (int32_t row = 0; row < nrows; row++) {
    treats->push_back(rand.NextFloat() > 0.5);
    labels->push_back(rand.NextFloat() > 0.7);
  }
}

/*!
 * Creates a dense Dataset of random values.
 */
void CreateRandomDenseData(int nrows, int ncols, std::vector<double>* features, std::vector<float>* labels,
                           std::vector<int>* treats) {
  UTBoost::Random rand(421);
  features->reserve(nrows * ncols);

  for (int32_t row = 0; row < nrows; row++) {
    for (int32_t col = 0; col < ncols; col++) {
      features->push_back(rand.NextFloat());
    }
  }

  CreateRandomMetadata(nrows, labels, treats);
}

/*!
 * Creates a sparse Dataset of random values.
 */
void CreateRandomSparseData(int nrows, int ncols, double sparse_rate, std::vector<double>* features) {
  UTBoost::Random rand(42);
  features->reserve(nrows * ncols);
  for (int32_t row = 0; row < nrows; row++) {
    for (int32_t col = 0; col < ncols; col++) {
      if (rand.NextFloat() < sparse_rate) {
        features->push_back(NAN);
      } else {
        features->push_back(rand.NextFloat());
      }
    }
  }
}

#endif //UTBOOST_TESTS_CPP_UTILS_H_
