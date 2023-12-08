/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 * Created by Junjie Gao on 2023/3/1.
 */

#ifndef UTBOOST_INCLUDE_UTBOOST_DEFINE_H_
#define UTBOOST_INCLUDE_UTBOOST_DEFINE_H_

#include <cstdint>
#include <limits>
#include <functional>

#ifdef __cplusplus
#define UTBOOST_EXTERN_C extern "C"
#else
#define UTBOOST_EXTERN_C
#endif

#ifdef _MSC_VER
#define UTBOOST_EXPORT __declspec(dllexport)
#define UTBOOST_C_EXPORT UTBOOST_EXTERN_C __declspec(dllexport)
#else
#define UTBOOST_EXPORT
#define UTBOOST_C_EXPORT UTBOOST_EXTERN_C
#endif

/*! \brief Type of data size. */
typedef int32_t data_size_t;
/*! \brief Type of label */
typedef float label_t;
/*! \brief Type of weight */
typedef float weight_t;
/*! \brief Type of treatment */
typedef int32_t treatment_t;
/*! \brief Type of score */
typedef float score_t;

typedef uint16_t bin_t;

/*! \brief handle to Dataset */
typedef void* DatasetHandle;
/*! \brief handle to Booster */
typedef void* BoosterHandle;
/*! \brief handle to Parser */
typedef void* ParserHandle;

const double kEpsilon = 1e-10f;

const double kZeroThreshold = 1e-35f;

const score_t kMinScore = -std::numeric_limits<score_t>::infinity();

#define PredictFunction std::function<void(const std::vector<std::pair<int, double>>&, double* output)>


#endif //UTBOOST_INCLUDE_UTBOOST_DEFINE_H_
