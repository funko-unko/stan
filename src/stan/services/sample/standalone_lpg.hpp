#ifndef STAN_SERVICES_SAMPLE_STANDALONE_LPG_HPP
#define STAN_SERVICES_SAMPLE_STANDALONE_LPG_HPP

#include <stan/services/sample/standalone_gqs.hpp>
// Piggyback off `standalone_gqs` to reuse
// `get_model_parameters` and the includes
// Choice is between a bit of code duplication,
// this slightly hacky include or a bit of refactoring
#include <stan/model/log_prob_grad.hpp>


namespace stan {
namespace services {

/**
 * Given a set of draws from a fitted model, (re)compute
 * `lp__` and its gradient which are written to callback writer.
 * The gradient is the negative of the one printed into the diagnostic_file.
 * Matrix of draws consists of one row per draw, one column per parameter.
 * Draws are processed one row at a time.
 * Return code indicates success or type of error.
 *
 * @tparam Model model class
 * @param[in] model instantiated model
 * @param[in] draws sequence of draws of constrained parameters
 * @param[in, out] interrupt called every iteration
 * @param[in, out] logger logger to which to write warning and error messages
 * @param[in, out] sample_writer writer to which draws are written
 * @return error code
 */
template <class Model>
int standalone_lpg(const Model &model, const Eigen::MatrixXd &draws,
                        callbacks::interrupt &interrupt,
                        callbacks::logger &logger,
                        callbacks::writer &sample_writer) {
  if (draws.size() == 0) {
    logger.error("Empty set of draws from fitted model.");
    return error_codes::DATAERR;
  }

  std::vector<std::string> p_names;
  model.constrained_param_names(p_names, false, false);

  std::stringstream msg;
  if (p_names.size() != draws.cols()) {
    msg << "Wrong number of parameter values in draws from fitted model.  ";
    msg << "Expecting " << p_names.size() << " columns, ";
    msg << "found " << draws.cols() << " columns.";
    std::string msgstr = msg.str();
    logger.error(msgstr);
    return error_codes::DATAERR;
  }

  std::vector<std::string> col_names;
  model.unconstrained_param_names(col_names, false, false);
  col_names.insert(col_names.begin(), "lp__");
  sample_writer(col_names);

  std::vector<std::string> param_names;
  std::vector<std::vector<size_t>> param_dimss;
  get_model_parameters(model, param_names, param_dimss);

  std::vector<int> dummy_params_i;
  std::vector<double> unconstrained_params_r;
  double log_prob(0);
  std::vector<double> gradient;
  for (size_t i = 0; i < draws.rows(); ++i) {
    dummy_params_i.clear();
    unconstrained_params_r.clear();
    gradient.clear();
    try {
      stan::io::array_var_context context(param_names, draws.row(i),
                                          param_dimss);
      model.transform_inits(context, dummy_params_i, unconstrained_params_r,
                            &msg);
    } catch (const std::exception &e) {
      if (msg.str().length() > 0)
        logger.error(msg);
      logger.error(e.what());
      return error_codes::DATAERR;
    }
    interrupt();  // call out to interrupt and fail
    // Stolen from stan/services/util/initialize.hpp
    std::stringstream log_prob_msg;
    try {
      // we evaluate this with propto=true since we're
      // evaluating with autodiff variables
      log_prob = stan::model::log_prob_grad<true, true>(
          model, unconstrained_params_r, dummy_params_i,
          gradient, &log_prob_msg);
    } catch (const std::exception& e) {
      if (log_prob_msg.str().length() > 0)
        logger.info(log_prob_msg);
      logger.info(e.what());
      // Only change is this to return error code instead of throwing
      return error_codes::DATAERR;
    }
    gradient.insert(gradient.begin(), log_prob);
    sample_writer(gradient);
  }
  return error_codes::OK;
}

}  // namespace services
}  // namespace stan
#endif
