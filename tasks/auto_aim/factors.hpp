/// Copyright (c) 2026 Fsk. All Rights Reserved.
#pragma once

#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/base/types.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace auto_aim {

// ==== 补全缺失的类型定义 ====
using ArmorIndex = int;

class TranslationFactor
    : public gtsam::NoiseModelFactorN<gtsam::Point3, gtsam::Vector3,
                                      gtsam::Point3> {
  using Base =
      gtsam::NoiseModelFactorN<gtsam::Point3, gtsam::Vector3, gtsam::Point3>;

public:
  TranslationFactor(const gtsam::SharedNoiseModel &model, gtsam::Key x_pre,
                    gtsam::Key v_pre, gtsam::Key x_cur, double dt);
  gtsam::Vector evaluateError(const gtsam::Point3 &x_pre,
                              const gtsam::Vector3 &v_pre,
                              const gtsam::Point3 &x_cur,
                              gtsam::OptionalMatrixType H1,
                              gtsam::OptionalMatrixType H2,
                              gtsam::OptionalMatrixType H3) const override;

private:
  double dt_;
};

class YawFactor
    : public gtsam::NoiseModelFactorN<gtsam::Rot2, double, gtsam::Rot2> {
  using Base = gtsam::NoiseModelFactorN<gtsam::Rot2, double, gtsam::Rot2>;

public:
  YawFactor(const gtsam::SharedNoiseModel &model, gtsam::Key r_pre,
            gtsam::Key w_pre, gtsam::Key r_cur, double dt);
  gtsam::Vector evaluateError(const gtsam::Rot2 &r_pre, const double &w_pre,
                              const gtsam::Rot2 &r_cur,
                              gtsam::OptionalMatrixType H1,
                              gtsam::OptionalMatrixType H2,
                              gtsam::OptionalMatrixType H3) const override;

private:
  double dt_;
};

class VelocityFactor
    : public gtsam::NoiseModelFactorN<gtsam::Vector3, gtsam::Vector3> {
  using Base = gtsam::NoiseModelFactorN<gtsam::Vector3, gtsam::Vector3>;

public:
  VelocityFactor(const gtsam::SharedNoiseModel &model, gtsam::Key v_pre,
                 gtsam::Key v_cur);

  gtsam::Vector evaluateError(const gtsam::Vector3 &r_pre,
                              const gtsam::Vector3 &r_cur,
                              gtsam::OptionalMatrixType H1,
                              gtsam::OptionalMatrixType H2) const override;
};

class VyawFactor : public gtsam::NoiseModelFactorN<double, double> {
  using Base = gtsam::NoiseModelFactorN<double, double>;

public:
  VyawFactor(const gtsam::SharedNoiseModel &model, gtsam::Key w_pre,
             gtsam::Key w_cur);

  gtsam::Vector evaluateError(const double &w_pre, const double &w_cur,
                              gtsam::OptionalMatrixType H1,
                              gtsam::OptionalMatrixType H2) const override;
};

class ArmorRadiusCenterZFactor
    : public gtsam::NoiseModelFactorN<gtsam::Pose3, double, gtsam::Rot2,
                                      gtsam::Point3> {
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose3, double, gtsam::Rot2,
                                        gtsam::Point3>;

public:
  ArmorRadiusCenterZFactor(const gtsam::SharedNoiseModel &model,
                           gtsam::Key armor_pose_key, gtsam::Key radius_key,
                           gtsam::Key center_yaw_key,
                           gtsam::Key center_point_key,
                           const Eigen::Isometry3d &T_camera_to_odom,
                           ArmorIndex armor_index, double radius_min,
                           double radius_max);

  gtsam::Vector
  evaluateError(const gtsam::Pose3 &armor_pose_camera, const double &radius,
                const gtsam::Rot2 &center_yaw,
                const gtsam::Point3 &center_point, gtsam::OptionalMatrixType H1,
                gtsam::OptionalMatrixType H2, gtsam::OptionalMatrixType H3,
                gtsam::OptionalMatrixType H4) const override;

private:
  Eigen::Isometry3d T_camera_to_odom_;
  ArmorIndex armor_index_;
  double radius_min_, radius_max_;
};

class ArmorRadiusDZFactor
    : public gtsam::NoiseModelFactorN<gtsam::Pose3, double, double, gtsam::Rot2,
                                      gtsam::Point3> {
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose3, double, double,
                                        gtsam::Rot2, gtsam::Point3>;

public:
  ArmorRadiusDZFactor(const gtsam::SharedNoiseModel &model,
                      gtsam::Key armor_pose_key, gtsam::Key radius_key,
                      gtsam::Key dz_key, gtsam::Key center_yaw_key,
                      gtsam::Key center_point_key,
                      const Eigen::Isometry3d &T_camera_to_odom,
                      ArmorIndex armor_index, double radius_min,
                      double radius_max, int armor_numbers = 4);

  gtsam::Vector
  evaluateError(const gtsam::Pose3 &armor_pose_camera, const double &radius,
                const double &dz, const gtsam::Rot2 &center_yaw,
                const gtsam::Point3 &center_point, gtsam::OptionalMatrixType H1,
                gtsam::OptionalMatrixType H2, gtsam::OptionalMatrixType H3,
                gtsam::OptionalMatrixType H4,
                gtsam::OptionalMatrixType H5) const override;

private:
  Eigen::Isometry3d T_camera_to_odom_;
  ArmorIndex armor_index_;
  double radius_min_, radius_max_;
  int armor_numbers_;
};

class ArmorReprojFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3> {
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose3>;

public:
  ArmorReprojFactor(const gtsam::SharedNoiseModel &model,
                    gtsam::Key armor_pose_key, const cv::Mat &camera_matrix,
                    const cv::Mat &distortion_coefficients,
                    const gtsam::Point3 &armor_point, // 修改：直接传入 3D 点
                    Eigen::Vector2d px_point);
  gtsam::Vector evaluateError(const gtsam::Pose3 &armor_pose_camera,
                              gtsam::OptionalMatrixType H) const override;

private:
  gtsam::Point2 px_point_;
  gtsam::Point3 armor_point_;
  gtsam::Cal3DS2 calib_;
};

} // namespace auto_aim