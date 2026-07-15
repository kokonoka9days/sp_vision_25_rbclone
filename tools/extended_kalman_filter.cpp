#include "extended_kalman_filter.hpp"

#include <cmath>
#include <limits>
#include <numeric>

namespace
{
double chi_square_95(Eigen::Index dof)
{
  // Exact 95% upper quantiles for the observation sizes used in this project.
  constexpr double quantiles[] = {
    0.0, 3.841458820694124, 5.991464547107979, 7.814727903251179,
    9.487729036781154, 11.070497693516351, 12.591587243743977};
  if (dof >= 1 && dof <= 6) return quantiles[dof];

  // Wilson-Hilferty approximation keeps the diagnostic meaningful for other dimensions.
  if (dof > 0) {
    constexpr double z95 = 1.6448536269514722;
    const double k = static_cast<double>(dof);
    const double term = 1.0 - 2.0 / (9.0 * k) + z95 * std::sqrt(2.0 / (9.0 * k));
    return k * term * term * term;
  }
  return 0.0;
}
}  // namespace

namespace tools
{
ExtendedKalmanFilter::ExtendedKalmanFilter(
  const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add)
: x(x0), P(P0), I(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())), x_add(x_add)
{
  data["residual_yaw"] = 0.0;
  data["residual_pitch"] = 0.0;
  data["residual_distance"] = 0.0;
  data["residual_angle"] = 0.0;
  data["nis"] = 0.0;
  data["nees"] = 0.0;
  data["nis_fail"] = 0.0;
  data["nees_fail"] = 0.0;
  data["filter_update_rejected"] = 0.0;
  data["recent_nis_failures"] = 0.0;
}

Eigen::VectorXd ExtendedKalmanFilter::predict(const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q)
{
  return predict(F, Q, [&](const Eigen::VectorXd & x) { return F * x; });
}

Eigen::VectorXd ExtendedKalmanFilter::predict(
  const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f)
{
  P = F * P * F.transpose() + Q;
  x = f(x);
  return x;
}

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  return update(z, H, R, [&](const Eigen::VectorXd & x) { return H * x; }, z_subtract);
}

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  const Eigen::VectorXd x_prior = x;
  const Eigen::MatrixXd P_prior = P;
  const Eigen::VectorXd residual = z_subtract(z, h(x_prior));

  const auto record_diagnostics = [&](double nis, double normalized_correction, bool rejected) {
    const double nis_threshold = chi_square_95(residual.size());
    const double correction_threshold = chi_square_95(x_prior.size());
    const bool nis_failed =
      rejected || !std::isfinite(nis) || nis < 0.0 || nis > nis_threshold;
    const bool correction_failed =
      !std::isfinite(normalized_correction) || normalized_correction < 0.0 ||
      normalized_correction > correction_threshold;

    data["nis_fail"] = nis_failed ? 1.0 : 0.0;
    data["nees_fail"] = correction_failed ? 1.0 : 0.0;
    data["filter_update_rejected"] = rejected ? 1.0 : 0.0;
    if (nis_failed) nis_count_++;
    if (correction_failed) nees_count_++;
    total_count_++;
    last_nis = nis;

    recent_nis_failures.push_back(nis_failed ? 1 : 0);
    if (recent_nis_failures.size() > window_size) recent_nis_failures.pop_front();

    const int recent_failures =
      std::accumulate(recent_nis_failures.begin(), recent_nis_failures.end(), 0);
    const double recent_rate =
      static_cast<double>(recent_failures) / recent_nis_failures.size();
    const auto residual_at =
      [&](Eigen::Index i) { return i < residual.size() ? residual[i] : 0.0; };
    data["residual_yaw"] = residual_at(0);
    data["residual_pitch"] = residual_at(1);
    data["residual_distance"] = residual_at(2);
    data["residual_angle"] = residual_at(3);
    data["nis"] = nis;
    // Kept as "nees" for log compatibility; this is a normalized update correction.
    data["nees"] = normalized_correction;
    data["recent_nis_failures"] = recent_rate;
  };

  if (!residual.allFinite()) {
    record_diagnostics(std::numeric_limits<double>::infinity(), 0.0, true);
    return x;
  }

  Eigen::MatrixXd S = H * P_prior * H.transpose() + R;
  S = (0.5 * (S + S.transpose())).eval();
  const Eigen::LDLT<Eigen::MatrixXd> S_ldlt(S);
  if (S_ldlt.info() != Eigen::Success || !S_ldlt.isPositive()) {
    record_diagnostics(std::numeric_limits<double>::infinity(), 0.0, true);
    return x;
  }

  const Eigen::MatrixXd PHt = P_prior * H.transpose();
  const Eigen::MatrixXd solved_PHt = S_ldlt.solve(PHt.transpose());
  const Eigen::VectorXd solved_residual = S_ldlt.solve(residual);
  if (!solved_PHt.allFinite() || !solved_residual.allFinite()) {
    record_diagnostics(std::numeric_limits<double>::infinity(), 0.0, true);
    return x;
  }

  const Eigen::MatrixXd K = solved_PHt.transpose();
  const double nis = residual.dot(solved_residual);
  const Eigen::VectorXd x_candidate = x_add(x_prior, K * residual);

  // Joseph form keeps the posterior covariance symmetric and positive semidefinite.
  // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
  const Eigen::MatrixXd I_KH = I - K * H;
  Eigen::MatrixXd P_candidate =
    I_KH * P_prior * I_KH.transpose() + K * R * K.transpose();
  P_candidate = (0.5 * (P_candidate + P_candidate.transpose())).eval();
  if (!x_candidate.allFinite() || !P_candidate.allFinite()) {
    record_diagnostics(std::numeric_limits<double>::infinity(), 0.0, true);
    return x;
  }

  x = x_candidate;
  P = P_candidate;

  const Eigen::VectorXd correction = x - x_prior;
  const Eigen::LDLT<Eigen::MatrixXd> P_ldlt(P);
  double normalized_correction = std::numeric_limits<double>::infinity();
  if (P_ldlt.info() == Eigen::Success && P_ldlt.isPositive()) {
    const Eigen::VectorXd solved_correction = P_ldlt.solve(correction);
    if (solved_correction.allFinite()) normalized_correction = correction.dot(solved_correction);
  }
  record_diagnostics(nis, normalized_correction, false);

  return x;
}

}  // namespace tools
