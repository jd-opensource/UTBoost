#include <gtest/gtest.h>

#include "UTBoost/c_api.h"
#include "UTBoost/dataset.h"

#include "utils.h"

using namespace UTBoost;

class ApiTest : public testing::Test {
 public:
  ApiTest() {
    train = nullptr;
    valid = nullptr;
    nrow = 1000;
    ncol = 10;
    CreateRandomDenseData(nrow, ncol, &feature, &label, &treatment);
    CreateRandomSparseData(nrow, ncol, 0.5, &sparse_feature);
  }

  void SetUp() override {
    const char* params = "max_bin=20\t"
                         "min_data_in_bin=10\t"
                         "num_threads=2\t"
                         "seed=12\t"
                         "bin_construct_sample_cnt=5000\t"
                         "verbose=0";
    // train
    UTB_CreateDataset(feature.data(), nrow, ncol, nullptr, &train, params);
    UTB_DatasetSetMeta(train, "treatment", treatment.data(), nrow, params);
    UTB_DatasetSetMeta(train, "label", label.data(), nrow, params);
    // valid
    UTB_CreateDataset(feature.data(), nrow, ncol, train, &valid, params);
    UTB_DatasetSetMeta(valid, "treatment", treatment.data(), nrow, params);
    UTB_DatasetSetMeta(valid, "label", label.data(), nrow, params);
  }

  void TearDown() override {
    // free dataset
    ASSERT_EQ(UTB_DatasetFree(train), 0);
    ASSERT_EQ(UTB_DatasetFree(valid), 0);
    train = nullptr;
    valid = nullptr;
  }

 protected:
  DatasetHandle train, valid;
  int nrow, ncol;
  std::vector<float> label;
  std::vector<int> treatment;
  std::vector<double> sparse_feature, feature;
};


TEST_F(ApiTest, UTB_CreateDataset) {
  auto* dataset_ptr = reinterpret_cast<Dataset*>(train);
  ASSERT_EQ(dataset_ptr->GetNumSamples(), nrow);
  ASSERT_EQ(dataset_ptr->GetNumFeatures(), ncol);
  for (int i = 0; i < ncol; ++i) {
    ASSERT_GT(dataset_ptr->GetFMapperNum(i), 10);
  }
  ASSERT_EQ(dataset_ptr->GetDistinctTreatNum(), 2);
  ASSERT_TRUE(dataset_ptr->FinshLoad(true));
  auto* valid_ptr = reinterpret_cast<Dataset*>(valid);
  ASSERT_EQ(valid_ptr->GetNumSamples(), nrow);
  ASSERT_EQ(valid_ptr->GetNumFeatures(), ncol);
  ASSERT_TRUE(valid_ptr->FinshLoad(true));
}


TEST_F(ApiTest, UTB_CreateBooster) {
  BoosterHandle booster;
  const char* params = "max_bin=20\t"
                       "min_data_in_bin=10\t"
                       "num_threads=2\t"
                       "seed=12\t"
                       "bin_construct_sample_cnt=5000\t"
                       "verbose=0";
  UTB_CreateBooster(train, params, &booster);
  ASSERT_EQ(UTB_BoosterFree(booster), 0);
}


TEST_F(ApiTest, UTB_BoosterUpdateOneIter) {
  BoosterHandle booster;
  const char* params = "max_bin=20\t"
                       "min_data_in_bin=10\t"
                       "num_threads=2\t"
                       "seed=12\t"
                       "bin_construct_sample_cnt=5000\t"
                       "verbose=0";
  UTB_CreateBooster(train, params, &booster);
  UTB_BoosterAddValidData(booster, valid);
  int is_finished = 0;
  int out_len = 0;
  std::vector<double> ret(2);
  for (int i = 0; i < 10; ++i) {
    UTB_BoosterUpdateOneIter(booster, &is_finished);
    UTB_BoosterGetEval(booster, 0, &out_len, ret.data());
    UTB_BoosterGetEval(booster, 1, &out_len, ret.data());
  }
  ASSERT_EQ(UTB_BoosterFree(booster), 0);
}


TEST_F(ApiTest, UTB_BoosterPredictForMat) {
  BoosterHandle booster;
  const char* params = "max_bin=20\t"
                       "min_data_in_bin=10\t"
                       "num_threads=2\t"
                       "seed=12\t"
                       "bin_construct_sample_cnt=5000\t"
                       "verbose=0";
  UTB_CreateBooster(train, params, &booster);
  int is_finished = 0;
  int max_round = 20;
  int round = 0;
  for (; round < max_round && (!is_finished); ++round) {
    UTB_BoosterUpdateOneIter(booster, &is_finished);
  }
  std::vector<double> results(10 * 2, 0);
  int out_len;
  UTB_BoosterPredictForMat(booster, feature.data(), 10, ncol, 0, round, params, &out_len, results.data());
  UTB_BoosterSaveModel(booster, 0, round, 1, "model.m");
  UTB_BoosterDumpModelToFile(booster, 0, round, "model.json");
  BoosterHandle new_booster;
  int num_iter;
  int num_treat;
  std::vector<double> new_results(10 * 2, 0);
  UTB_BoosterCreateFromModelfile("model.m", &num_iter, &num_treat, &new_booster);
  ASSERT_EQ(max_round, num_iter);
  UTB_BoosterPredictForMat(new_booster, feature.data(), 10, ncol, 0, num_iter, params, &out_len, new_results.data());
  for (int i = 0; i < results.size(); ++i) {
    ASSERT_EQ(results[i], new_results[i]);
  }
  ASSERT_EQ(UTB_BoosterFree(booster), 0);
  ASSERT_EQ(UTB_BoosterFree(new_booster), 0);
}