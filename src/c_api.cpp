/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 */

#include "UTBoost/c_api.h"
#include "UTBoost/definition.h"
#include "UTBoost/dataset.h"
#include "UTBoost/config.h"
#include "UTBoost/utils/random.h"
#include "UTBoost/utils/common.h"
#include "UTBoost/utils/log_wrapper.h"
#include "UTBoost/utils/omp_wrapper.h"
#include "UTBoost/ensemble_model.h"
#include "UTBoost/objective_function.h"

#include "ensemble/predictor.h"

namespace UTBoost {

inline int APIHandleException(const std::exception& ex) {
  UTB_SetLastError(ex.what());
  return -1;
}

inline int APIHandleException(const std::string& ex) {
  UTB_SetLastError(ex.c_str());
  return -1;
}

#define API_BEGIN() try {
#define API_END() } \
catch(std::exception& ex) { return APIHandleException(ex); } \
catch(std::string& ex) { return APIHandleException(ex); } \
catch(...) { return APIHandleException("unknown exception"); } \
return 0;

class Booster {
 public:
  explicit Booster(const char* filename) {
    boosting_.reset(EnsembleModel::CreateEnsembleModel("boost", filename));
  }

  Booster(const Dataset* train_data, const char* parameters) {
    config_.ParseParameters(parameters);
    OMP_SET_NUM_THREADS(config_.num_threads);
    // create boosting

    boosting_.reset(EnsembleModel::CreateEnsembleModel(config_.ensemble, nullptr));

    train_data_ = train_data;
    CreateObjectiveAndMetrics();
    boosting_->Init(&config_, train_data_, objective_fun_.get(),
                    ConstPtrInVectorWrapper<Metric>(train_metric_));
  }

  void CreateObjectiveAndMetrics() {

    objective_fun_.reset(ObjectiveFunction::CreateObjectiveFunction(config_.objective, config_));
    objective_fun_->Init(train_data_->GetMetaInfo(),train_data_->GetNumSamples());

    train_metric_.clear();
    for (const std::string& metric_type : config_.metric) {
      auto metric = std::unique_ptr<Metric>(
          Metric::CreateMetric(metric_type, config_));
      if (metric == nullptr) { continue; }
      metric->Init(train_data_->GetMetaInfo(), train_data_->GetNumSamples());
      train_metric_.push_back(std::move(metric));
    }
    train_metric_.shrink_to_fit();
  }

  void AddValidData(const Dataset* valid_data) {
    valid_metrics_.emplace_back();
    for (const auto& metric_type : config_.metric) {
      auto metric = std::unique_ptr<Metric>(Metric::CreateMetric(metric_type, config_));
      if (metric == nullptr) { continue; }
      metric->Init(valid_data->GetMetaInfo(), valid_data->GetNumSamples());
      valid_metrics_.back().push_back(std::move(metric));
    }
    valid_metrics_.back().shrink_to_fit();
    boosting_->AddValidDataset(valid_data, ConstPtrInVectorWrapper<Metric>(valid_metrics_.back()));
  }

  bool TrainOneIter() {
    return boosting_->TrainOneIter(nullptr, nullptr);
  }

  void RollbackOneIter() {
    boosting_->RollbackOneIter();
  }

  void Predict(int start_iteration, int num_iteration, int nrow, int ncol,
               std::function<std::vector<std::pair<int, double>>(int row_idx)> get_row_fun,
               const Config& config,
               double* out_result, int32_t* out_len) const {

    Predictor predictor(boosting_.get(), start_iteration, num_iteration);
    int32_t num_pred_in_one_row = boosting_->NumPredictOneRow(start_iteration, num_iteration, false);
    auto pred_fun = predictor.GetPredictFunction();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nrow; ++i) {
      std::vector<std::pair<int, double>> one_row = get_row_fun(i);
      auto pred_wrt_ptr = out_result + static_cast<size_t>(num_pred_in_one_row) * i;
      pred_fun(one_row, pred_wrt_ptr);
    }
    *out_len = num_pred_in_one_row * nrow;
  }

  const EnsembleModel* GetBoosting() const { return boosting_.get(); }

  void SaveModelToFile(int start_iteration, int num_iteration, int feature_importance_type, const char* filename) const {
    boosting_->SaveModelToFile(start_iteration, num_iteration, feature_importance_type, filename);
  }

  void LoadModelFromString(const char* model_str) {
    size_t len = std::strlen(model_str);
    boosting_->LoadModelFromString(model_str, len);
  }

  std::string SaveModelToString(int start_iteration, int num_iteration,
                                int feature_importance_type) const {
    return boosting_->SaveModelToString(start_iteration,
                                        num_iteration, feature_importance_type);
  }

  std::string DumpModel(int start_iteration, int num_iteration) const {
    return boosting_->DumpModel(start_iteration, num_iteration);
  }

 void DumpModelToFile(int start_iteration, int num_iteration, const char* filename) const {
    boosting_->DumpModelToFile(start_iteration, num_iteration, filename);
  }

 private:
  const Dataset* train_data_;
  std::unique_ptr<EnsembleModel> boosting_;
  /*! \brief All configs */
  Config config_;
  /*! \brief Metric for training data */
  std::vector<std::unique_ptr<Metric>> train_metric_;
  /*! \brief Metrics for validation data */
  std::vector<std::vector<std::unique_ptr<Metric>>> valid_metrics_;
  /*! \brief Training objective function */
  std::unique_ptr<ObjectiveFunction> objective_fun_;
};

}  // namespace UTBoost



using namespace UTBoost;

Dataset* BuildDataset(double** sample_values, int num_col, int num_row, int max_bucket, int min_data_in_bucket, data_size_t total_n_row) {
  std::vector<std::unique_ptr<BinMapper>> mappers(num_col);
  TIMER_START(x);
  for (int i = 0; i < num_col; ++i) {
      mappers[i].reset(new BinMapper());
      mappers[i]->FindBoundary(sample_values[i], num_row, max_bucket, min_data_in_bucket);
  }
  TIMER_STOP(x);
  Log::Debug("Construct bin mappers cost %f seconds", TIMER_SEC(x));
  auto dataset = std::unique_ptr<Dataset>(new Dataset());
  dataset->Build(mappers, total_n_row);
  return dataset.release();
}

std::function<std::vector<double>(int row_idx)> DenseRow2Vector(const void* data, int num_col) {
  auto data_ptr = reinterpret_cast<const double*>(data);
  return [=](int row_idx) {

    std::vector<double> ret(num_col);
    auto tmp_ptr = data_ptr + num_col * row_idx;

    for (int i = 0; i < num_col; ++i) {
      ret[i] = (*(tmp_ptr + i));
    }
    return ret;
  };
}


static inline std::vector<int32_t> CreateSampleIndices(data_size_t total_nrow, const Config& config) {
  Random rand(config.seed);
  int sample_cnt = static_cast<int>(total_nrow < config.bin_construct_sample_cnt ? total_nrow : config.bin_construct_sample_cnt);
  return rand.Sample(total_nrow, sample_cnt);
}


int UTB_DatasetFree(DatasetHandle handle) {
  API_BEGIN()
  delete reinterpret_cast<Dataset*>(handle);
  API_END()
}


int UTB_CreateDataset(const void* data2d, data_size_t num_row, int32_t num_col, DatasetHandle reference, DatasetHandle* out, const char* params) {
  API_BEGIN()
  Config config_;
  config_.ParseParameters(params);
  OMP_SET_NUM_THREADS(config_.num_threads);
  auto parse_function = DenseRow2Vector(data2d, num_col);

  std::unique_ptr<Dataset> data_ptr;

  if (reference == nullptr) {
    auto sample_indices = CreateSampleIndices(num_row, config_);
    Log::Debug(
        "Begin construct bin mappers, total %d, using %d" ,
        num_row, static_cast<int>(sample_indices.size())
    );
    std::vector<std::vector<double>> values(num_col);  // values[i] is the i-th feature
    TIMER_START(x);
    for (int idx : sample_indices) {
      auto row_values = parse_function(idx);
      for (size_t j = 0; j < row_values.size(); ++j) {
        values[j].emplace_back(row_values[j]);
      }
    }
    TIMER_STOP(x);
    Log::Debug("Loading samples to construct bin mappers cost %f seconds", TIMER_SEC(x));

    data_ptr.reset(
        BuildDataset(Vector2Ptr(values).data(), num_col,
                     static_cast<int>(sample_indices.size()),
                     config_.max_bin, config_.min_data_in_bin, num_row)
    );
  } else {
    data_ptr.reset(new Dataset(num_row));
    data_ptr->CreateValid(reinterpret_cast<const Dataset*>(reference));
  }

  TIMER_START(y);
  OMP_SET_NUM_THREADS(config_.num_threads);
  #pragma omp parallel for schedule(static)
  for (data_size_t i = 0; i < num_row; ++i) {
    auto row_values = parse_function(i);
    data_ptr->PushRow(i, row_values);
  }
  TIMER_STOP(y);
  Log::Debug("Loading data to Dataset cost %f seconds", TIMER_SEC(y));

  *out = data_ptr.release();

  API_END()
}


int UTB_DatasetSetMeta(DatasetHandle handle, const char* name, const void* data1d, data_size_t num_row, const char* params) {
  API_BEGIN()
  Config config_;
  config_.ParseParameters(params);
  OMP_SET_NUM_THREADS(config_.num_threads);
  auto dataset = reinterpret_cast<Dataset*>(handle);
  if(dataset == nullptr) {
    Log::Error("Dataset is nullptr");
    return -1;
  }
  if (dataset->GetNumSamples() != num_row) {
    Log::Error("Dataset size is not equal to meta size");
    return -1;
  }

  if (std::strcmp(name, "treatment") == 0) {
    auto data_ptr = reinterpret_cast<const treatment_t*>(data1d);
    dataset->SetMetaTreatment(data_ptr, num_row);
  } else {
    auto data_ptr = reinterpret_cast<const float*>(data1d);
    dataset->SetMetaFloat(name, data_ptr, num_row);
  }
  API_END()
}


int UTB_BoosterFree(BoosterHandle handle) {\
  API_BEGIN()
  delete reinterpret_cast<Booster*>(handle);
  API_END()
}

int UTB_CreateBooster(DatasetHandle train_data, const char* parameters, BoosterHandle* out) {
  API_BEGIN()
  auto p_train_data = reinterpret_cast<const Dataset*>(train_data);
  auto ret = std::unique_ptr<Booster>(new Booster(p_train_data, parameters));
  *out = ret.release();
  API_END()
}


int UTB_BoosterUpdateOneIter(BoosterHandle handle, int* is_finished) {
  API_BEGIN()
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  if (ref_booster->TrainOneIter()) {
    *is_finished = 1;
  } else {
    *is_finished = 0;
  }
  API_END()
}


int UTB_BoosterGetEval(BoosterHandle handle, int data_idx, int* out_len, double* out_results) {
  API_BEGIN()
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  auto model = ref_booster->GetBoosting();
  auto result_buf = model->GetEvalAt(data_idx);
  *out_len = static_cast<int>(result_buf.size());
  for (size_t i = 0; i < result_buf.size(); ++i) {
    (out_results)[i] = static_cast<double>(result_buf[i]);
  }
  API_END()
}


std::function<std::vector<double>(int row_idx)> RowFunctionFromDenseMatric(const void* data ,
                                                                           int num_row,
                                                                           int num_col) {
  const double* data_ptr = reinterpret_cast<const double*>(data);
  return [=](int row_idx) {

    std::vector<double> ret(num_col);
    auto tmp_ptr = data_ptr + static_cast<int>(num_col)*row_idx;

    for (int i = 0; i < num_col; ++i) {
      ret[i] = static_cast<double>(*(tmp_ptr + i));
    }
    return ret;
  };
}


std::function<std::vector<std::pair<int, double>>(int row_idx)>
RowPairFunctionFromDenseMatric(const void* data, int num_row, int num_col) {
  auto inner_function = DenseRow2Vector(data, num_col);
  if (inner_function != nullptr) {
    return [inner_function](int row_idx) {
      auto raw_values = inner_function(row_idx);
      std::vector<std::pair<int, double>> ret;
      for (int i = 0; i < static_cast<int>(raw_values.size()); ++i) {
        ret.emplace_back(i, raw_values[i]);
      }
      return ret;
    };
  }
  return nullptr;
}


int UTB_BoosterPredictForMat(BoosterHandle handle,
                             const void* data2d,
                             int32_t nrow,
                             int32_t ncol,
                             int start_iteration,
                             int num_iteration,
                             const char* parameter,
                             int32_t* out_len,
                             double* out_result) {
  API_BEGIN()
  Config config;
  config.ParseParameters(parameter);
  OMP_SET_NUM_THREADS(config.num_threads);
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  auto get_row_fun = RowPairFunctionFromDenseMatric(data2d, nrow, ncol);
  ref_booster->Predict(start_iteration, num_iteration, nrow, ncol, get_row_fun,
                       config, out_result, out_len);
  API_END()
}


int UTB_BoosterSaveModel(BoosterHandle handle,
                         int start_iteration,
                         int num_iteration,
                         int feature_importance_type,
                         const char* filename) {
  API_BEGIN()
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->SaveModelToFile(start_iteration, num_iteration,
                               feature_importance_type, filename);
  API_END()
}

int UTB_BoosterDumpModel(BoosterHandle handle,
                         int start_iteration,
                         int num_iteration,
                         int64_t buffer_len,
                         int64_t* out_len,
                         char* out_str) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  std::string model = ref_booster->DumpModel(start_iteration, num_iteration);
  *out_len = static_cast<int64_t>(model.size()) + 1;
  if (*out_len <= buffer_len) {
    std::memcpy(out_str, model.c_str(), *out_len);
  }
  API_END();
}

int UTB_BoosterDumpModelToFile(BoosterHandle handle,
                               int start_iteration,
                               int num_iteration,
                               const char* filename) {
  API_BEGIN()
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->DumpModelToFile(start_iteration, num_iteration, filename);
  API_END()
}

int UTB_BoosterCreateFromModelfile(const char* filename,
                                   int* out_num_iterations,
                                   int* out_num_treats,
                                   BoosterHandle* out) {
  API_BEGIN()
  auto ret = std::unique_ptr<Booster>(new Booster(filename));
  *out_num_iterations = ret->GetBoosting()->GetCurrentIteration();
  *out_num_treats = ret->GetBoosting()->GetNumTreatment();
  *out = ret.release();
  API_END()
}


int UTB_BoosterAddValidData(BoosterHandle handle,
                            DatasetHandle valid_data) {
  API_BEGIN()
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  Dataset* p_dataset = reinterpret_cast<Dataset*>(valid_data);
  ref_booster->AddValidData(p_dataset);
  API_END()
}

int UTB_BoosterRollbackOneIter(BoosterHandle handle) {
  API_BEGIN()
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->RollbackOneIter();
  API_END()
}

int UTB_BoosterFeatureImportance(BoosterHandle handle,
                                 int num_iteration,
                                 int importance_type,
                                 double* out_results) {

  API_BEGIN()
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  std::vector<double> feature_importances = ref_booster->GetBoosting()->FeatureImportance(num_iteration, importance_type);
  for (size_t i = 0; i < feature_importances.size(); ++i) {
    (out_results)[i] = feature_importances[i];
  }
  API_END()
}

int UTB_BoosterGetNumFeature(BoosterHandle handle, int* out_len) {
  API_BEGIN()
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_len = ref_booster->GetBoosting()->MaxFeatureIdx() + 1;
  API_END()
}

int UTB_DatasetDumpMapper(DatasetHandle handle, const char* filename) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  dataset->DumpBinMapper(filename);
  API_END();
}

const char* UTB_GetLastError() {
  return LastErrorMsg();
}

int UTB_ParseLibsvm(const char* filename,
                    int32_t label_idx,
                    int32_t treatment_idx,
                    int32_t num_threads,
                    int32_t* out_num_row,
                    ParserHandle* out) {
  API_BEGIN();
  OMP_SET_NUM_THREADS(num_threads);
  std::unique_ptr<LibsvmParser> parser;
  parser.reset(new LibsvmParser());
  parser->parseFile(filename, label_idx, treatment_idx, false);
  *out_num_row = parser->num_samples();
  *out = parser.release();
  API_END();
}

int UTB_MoveLibsvm(ParserHandle handle,
                   int32_t max_idx,
                   void* features,
                   void* labels,
                   void* treatments) {
  API_BEGIN();
  auto parser = reinterpret_cast<LibsvmParser*>(handle);
  auto feature_ptr = reinterpret_cast<double*>(features);
  auto label_ptr = reinterpret_cast<label_t*>(labels);
  auto treatment_ptr = reinterpret_cast<treatment_t*>(treatments);
  parser->copyTo(feature_ptr, max_idx, label_ptr, treatment_ptr);
  API_END();
}

int UTB_FreeParser(ParserHandle handle) {
  API_BEGIN();
  auto parser = reinterpret_cast<LibsvmParser*>(handle);
  delete parser;
  API_END();
}
