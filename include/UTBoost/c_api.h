/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 * Created by Junjie Gao on 2023/3/1.
 */

#ifndef UTBOOST_INCLUDE_UTBOOST_C_API_H_
#define UTBOOST_INCLUDE_UTBOOST_C_API_H_

#include <cstdint>
#include <cstring>
#include "UTBoost/definition.h"

/*!
 * \brief Parse libsvm file
 * \param label_idx label index in file
 * \param treatment_idx treatment indicator index
 * \param num_threads number of threads
 * \param out_num_row pointer to store the number of rows in file
 * \param out pointer to store the handle of the newly created parser
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_ParseLibsvm(const char* filename,
                                     int32_t label_idx,
                                     int32_t treatment_idx,
                                     int32_t num_threads,
                                     int32_t* out_num_row,
                                     ParserHandle* out);

/*!
 * \brief Copy key-value items to array
 * \param max_idx maximum feature index
 * \param features pointer to feature data
 * \param labels pointer to label data
 * \param treatments pointer to treatment indicator
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_MoveLibsvm(ParserHandle handle,
                                    int32_t max_idx,
                                    void* features,
                                    void* labels,
                                    void* treatments);

/*!
 * \brief Free space for Parser.
 * \param handle Handle of parser to be freed
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_FreeParser(ParserHandle handle);

/*!
 * \brief Sets the metadata for a dataset.
 * \param handle The handle of the dataset.
 * \param name The name of the dataset.
 * \param data1d A pointer to the 1-dimensional data.
 * \param num_row The number of rows in the dataset.
 * \param params Additional parameters for the dataset.
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_DatasetSetMeta(DatasetHandle handle,
                                        const char* name,
                                        const void* data1d,
                                        data_size_t num_row,
                                        const char* params);

/*!
 * \brief Creates a dataset using the provided data.
 * \param data2d Pointer to the data
 * \param num_row Number of rows in the dataset
 * \param num_col Number of columns in the dataset
 * \param reference Handle to the reference dataset
 * \param out Pointer to store the handle of the newly created dataset
 * \param params Parameter string
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_CreateDataset(const void* data2d,
                                       data_size_t num_row,
                                       int32_t num_col,
                                       int data_type,
                                       DatasetHandle reference,
                                       DatasetHandle* out,
                                       const char* params);

/*!
 * \brief Create a dataset from CSR matrix.
 * \param indptr Pointer to row headers
 * \param indptr_type Type of indptr
 * \param indices Pointer to column indices
 * \param data Pointer to the data
 * \param data_type Type of data
 * \param nindptr Number of rows in the matrix + 1
 * \param num_col Number of columns
 * \param reference Handle to the reference dataset
 * \param out Pointer to store the handle of the newly created dataset
 * \param params Parameter string
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_CreateDatasetFromCSR(const void* indptr,
                                              int indptr_type,
                                              const int32_t* indices,
                                              const void* data,
                                              int data_type,
                                              int64_t nindptr,
                                              int32_t num_col,
                                              DatasetHandle reference,
                                              DatasetHandle* out,
                                              const char* params);

/*!
 * \brief Load dataset from file with libsvm format.
 * \param filename The name of the file
 * \param params Parameter string
 * \param reference Handle to the reference dataset
 * \param out Pointer to store the handle of the newly created dataset
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_DatasetCreateFromLibsvm(const char* filename,
                                                 int32_t label_idx,
                                                 int32_t treatment_idx,
                                                 const char* params,
                                                 DatasetHandle reference,
                                                 DatasetHandle* out);

/*!
 * \brief Save dataset to text file, intended for debugging use only.
 * \param handle Handle of dataset
 * \param filename The name of the file
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_DatasetDumpMapper(DatasetHandle handle,
                                           const char* filename);

/*!
 * \brief Free space for dataset.
 * \param handle Handle of dataset to be freed
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_DatasetFree(DatasetHandle handle);


/*!
 * \brief Creates a booster for training a model.
 *
 * This function creates a booster for training a model using the given training dataset and parameters.
 * The booster handle is stored in the `out` parameter.
 *
 * \param train_data The handle of the training dataset.
 * \param parameters The parameters for the booster.
 * \param out A pointer to the booster handle.
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_CreateBooster(DatasetHandle train_data,
                                       const char* parameters,
                                       BoosterHandle* out);

/*!
 * \brief Free space for booster.
 * \param handle Handle of booster to be freed
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_BoosterFree(BoosterHandle handle);

/*!
 * \brief Updates the booster with one iteration of training.
 *
 * This function updates the booster identified by the given handle with one iteration of training.
 * The `is_finished` parameter is used to determine if the training process is finished.
 *
 * \param handle The handle of the booster.
 * \param is_finished A pointer to an integer indicating if the training process is finished.
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_BoosterUpdateOneIter(BoosterHandle handle, int* is_finished);

/*!
 * \brief Rollback one iteration.
 * \param handle Handle of booster
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_BoosterRollbackOneIter(BoosterHandle handle);

/*!
 * \brief Get evaluation for training data and validation data.
 * \note
 *   1. You should call ``LGBM_BoosterGetEvalNames`` first to get the names of evaluation metrics.
 *   2. You should pre-allocate memory for ``out_results``, you can get its length by ``LGBM_BoosterGetEvalCounts``.
 * \param handle Handle of booster
 * \param data_idx Index of data, 0: training data, 1: 1st validation data, 2: 2nd validation data and so on
 * \param[out] out_len Length of output result
 * \param[out] out_results Array with evaluation results
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_BoosterGetEval(BoosterHandle handle,
                                        int data_idx,
                                        int* out_len,
                                        double* out_results);

/*!
 * \brief Predicts using the booster for a given matrix of data.
 *
 * This function predicts using the booster identified by the given handle for a given matrix of data.
 * The predictions are stored in the `out_result` array.
 * You should pre-allocate memory for ``out_result``
 *
 * \param handle The handle of the booster.
 * \param data2d A pointer to the 2-dimensional data matrix.
 * \param nrow The number of rows in the data matrix.
 * \param ncol The number of columns in the data matrix.
 * \param data_type Type of data matrix.
 * \param start_iteration The starting iteration for prediction.
 * \param num_iteration The number of iterations for prediction.
 * \param parameter The parameters for prediction.
 * \param out_len[out] A pointer to an integer storing the length of the prediction results.
 * \param out_result[out] A pointer to an array to store the prediction results.
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_BoosterPredictForMat(BoosterHandle handle,
                                              const void* data2d,
                                              int32_t nrow,
                                              int32_t ncol,
                                              int data_type,
                                              int start_iteration,
                                              int num_iteration,
                                              const char* parameter,
                                              int32_t* out_len,
                                              double* out_result);

/*!
 * \brief Save model into file.
 * \param handle Handle of booster
 * \param start_iteration Start index of the iteration that should be saved
 * \param num_iteration Index of the iteration that should be saved, <= 0 means save all
 * \param feature_importance_type Type of feature importance
 * \param filename The name of the file
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_BoosterSaveModel(BoosterHandle handle,
                                          int start_iteration,
                                          int num_iteration,
                                          int feature_importance_type,
                                          const char* filename);

/*!
 * \brief Dump model to JSON.
 * \param handle Handle of booster
 * \param start_iteration Start index of the iteration that should be dumped
 * \param num_iteration Index of the iteration that should be dumped, <= 0 means dump all
 * \param buffer_len String buffer length, if ``buffer_len < out_len``, you should re-allocate buffer
 * \param[out] out_len Actual output length
 * \param[out] out_str JSON format string of model, should pre-allocate memory
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_BoosterDumpModel(BoosterHandle handle,
                                          int start_iteration,
                                          int num_iteration,
                                          int64_t buffer_len,
                                          int64_t* out_len,
                                          char* out_str);

/*!
 * \brief Dump model to JSON.
 * \param handle Handle of booster
 * \param start_iteration Start index of the iteration that should be dumped
 * \param num_iteration Index of the iteration that should be dumped, <= 0 means dump all
 * \param buffer_len String buffer length, if ``buffer_len < out_len``, you should re-allocate buffer
 * \param[out] out_len Actual output length
 * \param[out] out_str JSON format string of model, should pre-allocate memory
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_BoosterDumpModelToFile(BoosterHandle handle,
                                                int start_iteration,
                                                int num_iteration,
                                                const char* filename);

/*!
 * \brief Load an existing booster from model file.
 * \param filename Filename of model
 * \param[out] out_num_iterations Number of iterations of this booster
 * \param[out] out Handle of created booster
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_BoosterCreateFromModelfile(const char* filename,
                                                    int* out_num_iterations,
                                                    int* out_num_treats,
                                                    BoosterHandle* out);

/*!
 * \brief Add new validation data to booster.
 * \param handle Handle of booster
 * \param valid_data Validation dataset
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_BoosterAddValidData(BoosterHandle handle,
                                             DatasetHandle valid_data);

/*!
 * \brief Get model feature importance.
 * \param handle Handle of booster
 * \param num_iteration Number of iterations for which feature importance is calculated, <= 0 means use all
 * \param importance_type Method of importance calculation:
 * \param[out] out_results Result array with feature importance
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_BoosterFeatureImportance(BoosterHandle handle,
                                                  int num_iteration,
                                                  int importance_type,
                                                  double* out_results);

/*!
 * \brief Get number of features.
 * \param handle Handle of booster
 * \param[out] out_len Total number of features
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_BoosterGetNumFeature(BoosterHandle handle, int* out_len);

/*!
 * \brief Get number of treatments.
 * \param handle Handle of booster
 * \param[out] out_len Total number of features
 * \return An integer indicating the success or failure of the operation.
 */
UTBOOST_C_EXPORT int UTB_BoosterGetNumTreatment(BoosterHandle handle, int* out_len);

/*!
 * \brief Get string message of the last error.
 * \return Error information
 */
UTBOOST_C_EXPORT const char* UTB_GetLastError();

#if defined(_MSC_VER)
// exception handle and error msg
static char* LastErrorMsg() { static __declspec(thread) char err_msg[512] = "Everything is fine"; return err_msg; }
#else
static char* LastErrorMsg() { static thread_local char err_msg[512] = "Everything is fine"; return err_msg; }
#endif

/*!
 * \brief Set string message of the last error.
 * \note
 * This will call unsafe ``sprintf`` when compiled using C standards before C99.
 * \param msg Error message
 */
inline void UTB_SetLastError(const char* msg) {
#if !defined(__cplusplus) && (!defined(__STDC__) || (__STDC_VERSION__ < 199901L))
  sprintf(LastErrorMsg(), "%s", msg);  /* NOLINT(runtime/printf) */
#else
  const int err_buf_len = 512;
  snprintf(LastErrorMsg(), err_buf_len, "%s", msg);
#endif
}

#endif //UTBOOST_INCLUDE_UTBOOST_C_API_H_
