#include "buff_target.hpp"

#include <algorithm>
#include <cmath>

namespace auto_buff
{
namespace
{
/** @brief 获取整数符号 @param value 输入值 @return 正数为 1，负数为 -1，零为 0 */
int signum(int value)
{
  if (value > 0) return 1;
  if (value < 0) return -1;
  return 0;
}
}  // namespace

void PhaseDirectionTracker::rebase(double phase, bool preserve_direction)
{
  has_last_phase_ = true;
  last_phase_ = phase;
  reverse_candidate_direction_ = 0;
  reverse_confirm_count_ = 0;
  if (!preserve_direction) {
    deltas_.clear();
    votes_.clear();
    direction_ = 0;
    score_ = 0;
  } else if (direction_ != 0) {
    score_ = direction_ * 6;
  }
}

void PhaseDirectionTracker::reset()
{
  has_last_phase_ = false;
  last_phase_ = 0.0;
  direction_ = 0;
  score_ = 0;
  reverse_candidate_direction_ = 0;
  reverse_confirm_count_ = 0;
  deltas_.clear();
  votes_.clear();
}

void PhaseDirectionTracker::shift_reference(double delta)
{
  if (has_last_phase_) last_phase_ += delta;
}

void PhaseDirectionTracker::update(double phase)
{
  if (!has_last_phase_) {
    rebase(phase, true);
    return;
  }

  const double delta = phase - last_phase_;
  last_phase_ = phase;

  constexpr double min_direction_delta = CV_PI / 900.0;
  constexpr double max_direction_delta = CV_PI / 5.0;
  constexpr double confirm_window_delta = CV_PI / 120.0;
  constexpr int direction_window = 8;
  const int min_window_samples = std::max(2, confirm_intervals_);
  const int confirm_vote_margin = min_window_samples;
  const int confirm_score = min_window_samples;
  constexpr int score_limit = 30;

  if (std::abs(delta) <= min_direction_delta || std::abs(delta) >= max_direction_delta) return;

  const int sample_direction = delta > 0.0 ? 1 : -1;
  deltas_.push_back(delta);
  votes_.push_back(sample_direction);
  while (static_cast<int>(deltas_.size()) > direction_window) {
    deltas_.pop_front();
    votes_.pop_front();
  }
  score_ = std::clamp(score_ + sample_direction, -score_limit, score_limit);

  double window_delta_sum = 0.0;
  int window_vote_sum = 0;
  for (std::size_t i = 0; i < deltas_.size(); ++i) {
    window_delta_sum += deltas_[i];
    window_vote_sum += votes_[i];
  }

  int window_direction = 0;
  if (
    static_cast<int>(deltas_.size()) >= min_window_samples &&
    std::abs(window_delta_sum) >= confirm_window_delta &&
    std::abs(window_vote_sum) >= confirm_vote_margin) {
    const int delta_direction = window_delta_sum > 0.0 ? 1 : -1;
    const int vote_direction = signum(window_vote_sum);
    if (delta_direction == vote_direction) window_direction = delta_direction;
  }

  if (window_direction == 0) {
    reverse_candidate_direction_ = 0;
    reverse_confirm_count_ = 0;
    return;
  }

  if (direction_ == 0 && signum(score_) == window_direction && std::abs(score_) >= confirm_score) {
    direction_ = window_direction;
    reverse_candidate_direction_ = 0;
    reverse_confirm_count_ = 0;
  }
}
}  // namespace auto_buff
