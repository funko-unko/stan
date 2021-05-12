#include <boost/algorithm/string.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/io/dump.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/sample/standalone_lpg.hpp>
#include <test/test-models/good/services/bernoulli.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/unit/util.hpp>
#include <vector>

class ServicesStandaloneLPG : public ::testing::Test {
 public:
  ServicesStandaloneLPG()
      : logger(logger_ss, logger_ss, logger_ss, logger_ss, logger_ss) {}

  void SetUp() {
    std::fstream data_stream(
        "src/test/test-models/good/services/bernoulli.data.R",
        std::fstream::in);
    stan::io::dump data_var_context(data_stream);
    data_stream.close();
    model = new stan_model(data_var_context);
  }

  void TearDown() { delete model; }

  stan::test::unit::instrumented_interrupt interrupt;
  std::stringstream logger_ss;
  stan::callbacks::stream_logger logger;
  stan_model *model;
};

TEST_F(ServicesStandaloneLPG, lpgDraws_bernoulli) {
  stan::io::stan_csv bern_csv;
  stan::io::stan_csv bern_csv_dia;
  std::stringstream out;
  std::ifstream csv_stream;
  csv_stream.open("src/test/test-models/good/services/bernoulli_fit_lpg.csv");
  bern_csv = stan::io::stan_csv_reader::parse(csv_stream, &out);
  csv_stream.close();
  EXPECT_EQ(12345U, bern_csv.metadata.seed);
  ASSERT_EQ(8, bern_csv.header.size());
  EXPECT_EQ("theta", bern_csv.header[7]);
  ASSERT_EQ(1000, bern_csv.samples.rows());
  ASSERT_EQ(8, bern_csv.samples.cols());

  csv_stream.open("src/test/test-models/good/services/bernoulli_fit_diagnostic.csv");
  bern_csv_dia = stan::io::stan_csv_reader::parse(csv_stream, &out);
  csv_stream.close();
  EXPECT_EQ(12345U, bern_csv_dia.metadata.seed);
  ASSERT_EQ(10, bern_csv_dia.header.size());
  EXPECT_EQ("lp__", bern_csv_dia.header[0]);
  EXPECT_EQ("theta", bern_csv_dia.header[7]);
  EXPECT_EQ("p_theta", bern_csv_dia.header[8]);
  EXPECT_EQ("g_theta", bern_csv_dia.header[9]);
  ASSERT_EQ(1000, bern_csv_dia.samples.rows());
  ASSERT_EQ(10, bern_csv_dia.samples.cols());

  // model bernoulli.stan has 1 param
  std::vector<std::string> param_names;
  std::vector<std::vector<size_t>> param_dimss;
  stan::services::get_model_parameters(*model, param_names, param_dimss);

  EXPECT_EQ(param_names.size(), 1);
  EXPECT_EQ(param_dimss.size(), 1);

  std::stringstream sample_ss;
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  int return_code = stan::services::standalone_lpg(
      *model, bern_csv.samples.middleCols<1>(7), interrupt, logger,
      sample_writer);
  EXPECT_EQ(return_code, stan::services::error_codes::OK);
  EXPECT_EQ(count_matches("lp__", sample_ss.str()), 1);
  EXPECT_EQ(count_matches("theta", sample_ss.str()), 1);
  EXPECT_EQ(count_matches("\n", sample_ss.str()), 1001);
  //Check lp__ from output file
  match_csv_columns(bern_csv.samples, sample_ss.str(), 1000, 1, 0);
  //Check lp__ and gradient from diagnostic file
  Eigen::MatrixXd lpg(1000, 2);
  lpg.col(0) = bern_csv_dia.samples.col(0);
  lpg.col(1) = -bern_csv_dia.samples.col(9);
  match_csv_columns(lpg, sample_ss.str(), 1000, 2, 0);
}

TEST_F(ServicesStandaloneLPG, lpgDraws_empty_draws) {
  const Eigen::MatrixXd draws;
  std::stringstream sample_ss;
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  int return_code = stan::services::standalone_lpg(
      *model, draws, interrupt, logger, sample_writer);
  EXPECT_EQ(return_code, stan::services::error_codes::DATAERR);
  EXPECT_EQ(count_matches("Empty set of draws", logger_ss.str()), 1);
}

TEST_F(ServicesStandaloneLPG, lpgDraws_bad) {
  Eigen::MatrixXd draws(2, 2);
  std::stringstream sample_ss;
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  int return_code = stan::services::standalone_lpg(
      *model, draws, interrupt, logger, sample_writer);
  EXPECT_EQ(return_code, stan::services::error_codes::DATAERR);
  EXPECT_EQ(count_matches("Wrong number of parameter values", logger_ss.str()),
            1);
}
