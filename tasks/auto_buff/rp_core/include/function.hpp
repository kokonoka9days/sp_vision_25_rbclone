#ifndef AUTO_BUFF__RP_CORE__FUNCTION_HPP
#define AUTO_BUFF__RP_CORE__FUNCTION_HPP

#include <cstdint>
#include <string>

#include "runtime_types.hpp"

namespace function
{
RuneTimestamp getNowTimestamp();
double timestampMinus(const RuneTimestamp & newer, const RuneTimestamp & older);
std::string getLocalTime();
uint64_t to_nanoseconds_since_epoch(const RuneTimestamp & timestamp);
double calculate_delta_phase(const double & new_phase, const double & old_phase);
}  // namespace function

#endif
