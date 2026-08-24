#ifndef AUTO_BUFF__RP_CORE__LOG_COMPAT_HPP
#define AUTO_BUFF__RP_CORE__LOG_COMPAT_HPP

#include <sstream>
#include <string>

#include "tools/logger.hpp"

namespace auto_buff::rp
{
enum class LogLevel { INFO, WARNING, ERROR };

class LogLine
{
public:
  explicit LogLine(LogLevel level) : level_(level) {}
  ~LogLine()
  {
    const std::string message = stream_.str();
    switch (level_) {
      case LogLevel::INFO:
        tools::logger()->info("{}", message);
        break;
      case LogLevel::WARNING:
        tools::logger()->warn("{}", message);
        break;
      case LogLevel::ERROR:
        tools::logger()->error("{}", message);
        break;
    }
  }

  template<typename T>
  LogLine & operator<<(const T & value)
  {
    stream_ << value;
    return *this;
  }

private:
  LogLevel level_;
  std::ostringstream stream_;
};
}  // namespace auto_buff::rp

#ifdef LOG
#undef LOG
#endif
#define LOG(level) ::auto_buff::rp::LogLine(::auto_buff::rp::LogLevel::level)

#endif
