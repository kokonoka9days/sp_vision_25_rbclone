#ifndef AUTO_BUFF__RP_CORE__VISUALIZE_COMPAT_HPP
#define AUTO_BUFF__RP_CORE__VISUALIZE_COMPAT_HPP

#include <optional>
#include <utility>

namespace foxglove::schemas
{
struct Vector3
{
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct Quaternion
{
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double w = 1.0;
};

struct Color
{
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
  float a = 0.0f;
};

struct Pose
{
  Pose() = default;
  explicit Pose(Vector3 value) : position(std::move(value)) {}
  std::optional<Vector3> position;
  std::optional<Quaternion> orientation;
};

struct SpherePrimitive
{
  std::optional<Pose> pose;
  Vector3 size;
  Color color;
};

struct ArrowPrimitive
{
  std::optional<Pose> pose;
  double shaft_length = 0.0;
  double shaft_diameter = 0.0;
  double head_length = 0.0;
  double head_diameter = 0.0;
  Color color;
};
}  // namespace foxglove::schemas

namespace VizTopic
{
#define AUTO_BUFF_NOOP_TOPIC(name) struct name { static bool enabled() { return false; } }
AUTO_BUFF_NOOP_TOPIC(RuneOriPhase);
AUTO_BUFF_NOOP_TOPIC(RuneFilteredPhase);
AUTO_BUFF_NOOP_TOPIC(RuneTrackContinuousPhase);
AUTO_BUFF_NOOP_TOPIC(RuneTrackPhase);
AUTO_BUFF_NOOP_TOPIC(RuneBigFilteredPhase);
AUTO_BUFF_NOOP_TOPIC(RunePlaneVector);
AUTO_BUFF_NOOP_TOPIC(RunePredictError);
AUTO_BUFF_NOOP_TOPIC(RuneExpectYawPitch);
AUTO_BUFF_NOOP_TOPIC(RunePredictPhase);
AUTO_BUFF_NOOP_TOPIC(RunePredictTarget);
AUTO_BUFF_NOOP_TOPIC(PowerRuneCar);
AUTO_BUFF_NOOP_TOPIC(PowerRuneNormalCar);
AUTO_BUFF_NOOP_TOPIC(PowerRuneCamera);
AUTO_BUFF_NOOP_TOPIC(PowerRuneNormalCamera);
#undef AUTO_BUFF_NOOP_TOPIC
}  // namespace VizTopic

class Viz
{
public:
  template<typename Topic, typename... Args>
  static void log(Args &&...) {}

  template<typename Topic, typename... Args>
  static void log_with_time(Args &&...) {}

  template<typename Topic, typename Container>
  static void publish_spheres(const Container &) {}

  template<typename Topic, typename Container>
  static void publish_arrows(const Container &) {}
};

#endif
