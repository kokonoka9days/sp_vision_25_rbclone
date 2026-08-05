#include "prediction_cadence.hpp"

#include <chrono>
#include <cmath>

namespace
{
bool near(double lhs, double rhs) { return std::abs(lhs - rhs) < 1e-9; }
}  // namespace

int main()
{
  const auto t0 = std::chrono::steady_clock::now();
  const auto t20 = t0 + std::chrono::milliseconds(20);

  tools::PredictionCadence no_prediction(0);
  tools::PredictionCadence one_prediction(1);
  tools::PredictionCadence two_predictions(2);
  no_prediction.observe(t0);
  one_prediction.observe(t0);
  two_predictions.observe(t0);
  no_prediction.observe(t20);
  one_prediction.observe(t20);
  two_predictions.observe(t20);

  if (!near(no_prediction.control_period_s(), 0.020)) return 1;
  if (!near(one_prediction.control_period_s(), 0.010)) return 2;
  if (!near(two_predictions.control_period_s(), 0.020 / 3.0)) return 3;

  tools::PredictionCadence adaptive(1);
  adaptive.observe(t0);
  adaptive.observe(t0 + std::chrono::milliseconds(40));
  if (!near(adaptive.control_period_s(), 0.012)) return 4;
  return 0;
}
