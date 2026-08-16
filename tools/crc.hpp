#ifndef TOOLS__CRC_HPP
#define TOOLS__CRC_HPP

#include <cstdint>

namespace tools
{
/**
 * @brief 计算数据的 CRC-8 校验值
 * @param data 待校验数据指针
 * @param len 数据长度，不包含 CRC-8 字节
 * @return CRC-8 校验值
 */
uint8_t get_crc8(const uint8_t * data, uint16_t len);

/**
 * @brief 校验包含 CRC-8 的数据帧
 * @param data 待校验数据指针
 * @param len 数据总长度，包含 CRC-8 字节
 * @return 校验通过返回 true，否则返回 false
 */
bool check_crc8(const uint8_t * data, uint16_t len);

/**
 * @brief 计算数据的 CRC-16 校验值
 * @param data 待校验数据指针
 * @param len 数据长度，不包含 CRC-16 字节
 * @return CRC-16 校验值
 */
uint16_t get_crc16(const uint8_t * data, uint32_t len);

/**
 * @brief 校验包含 CRC-16 的数据帧
 * @param data 待校验数据指针
 * @param len 数据总长度，包含 CRC-16 字节
 * @return 校验通过返回 true，否则返回 false
 */
bool check_crc16(const uint8_t * data, uint32_t len);

}  // namespace tools

#endif  // TOOLS__CRC_HPP
