#ifndef TOOLS__YAML_HPP
#define TOOLS__YAML_HPP

#include <yaml-cpp/yaml.h>

#include "tools/logger.hpp"

namespace tools
{
/** @brief 加载 YAML 文件 @param path 文件路径 @return YAML 根节点 @note 文件不存在或格式错误时记录日志并终止进程 */
inline YAML::Node load(const std::string & path)
{
  try {
    return YAML::LoadFile(path);
  } catch (const YAML::BadFile & e) {
    logger()->error("[YAML] Failed to load file: {}", e.what());
    exit(1);
  } catch (const YAML::ParserException & e) {
    logger()->error("[YAML] Parser error: {}", e.what());
    exit(1);
  }
}

/** @brief 从 YAML 节点读取必需字段 @tparam T 目标类型 @param yaml YAML 节点 @param key 字段名 @return 转换后的字段值 @note 字段缺失时记录日志并终止进程 */
template <typename T>
inline T read(const YAML::Node & yaml, const std::string & key)
{
  if (yaml[key]) return yaml[key].as<T>();
  logger()->error("[YAML] {} not found!", key);
  exit(1);
}

}  // namespace tools

#endif  // TOOLS__YAML_HPP
