#include "plotter.hpp"

#include <arpa/inet.h>   // htons, inet_addr
#include <sys/socket.h>  // socket, sendto
#include <unistd.h>      // close

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <thread>

namespace tools
{
namespace
{
std::string default_recording_base_path()
{
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  const auto microseconds =
    std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() % 1000000;
  std::tm local_time{};
  localtime_r(&time, &local_time);

  std::ostringstream path;
  path << "plot_records/" << std::put_time(&local_time, "%Y-%m-%d_%H-%M-%S") << '_'
       << std::setfill('0') << std::setw(6) << microseconds;
  return path.str();
}

void flatten_json(
  const nlohmann::json & value, const std::string & prefix,
  std::map<std::string, std::string> & fields)
{
  if (value.is_object()) {
    for (auto it = value.begin(); it != value.end(); ++it) {
      const auto name = prefix.empty() ? it.key() : prefix + "/" + it.key();
      flatten_json(it.value(), name, fields);
    }
    return;
  }
  if (value.is_array()) {
    for (std::size_t index = 0; index < value.size(); ++index) {
      const auto name =
        prefix.empty() ? std::to_string(index) : prefix + "/" + std::to_string(index);
      flatten_json(value[index], name, fields);
    }
    return;
  }
  if (value.is_null() || value.is_discarded()) return;

  const auto name = prefix.empty() ? std::string("value") : prefix;
  if (value.is_string())
    fields[name] = value.get<std::string>();
  else if (value.is_boolean())
    fields[name] = value.get<bool>() ? "1" : "0";
  else
    fields[name] = value.dump();
}

std::string csv_escape(const std::string & value)
{
  if (value.find_first_of(",\"\r\n") == std::string::npos) return value;

  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('"');
  for (const char c : value) {
    if (c == '"') escaped.push_back('"');
    escaped.push_back(c);
  }
  escaped.push_back('"');
  return escaped;
}

bool read_record(
  const std::string & line, std::size_t line_number, int64_t & time_us, nlohmann::json & data)
{
  try {
    const auto record = nlohmann::json::parse(line);
    if (
      !record.is_object() || !record.contains("time_us") ||
      !record["time_us"].is_number_integer() || !record.contains("data")) {
      std::cerr << "Plotter: invalid record structure at line " << line_number << '\n';
      return false;
    }
    time_us = record["time_us"].get<int64_t>();
    data = record["data"];
    return true;
  } catch (const nlohmann::json::exception & error) {
    std::cerr << "Plotter: invalid record at line " << line_number << ": " << error.what() << '\n';
    return false;
  }
}

}  // namespace

Plotter::Plotter(bool record, std::string host, uint16_t port, std::string record_path)
{
  socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);

  destination_.sin_family = AF_INET;
  destination_.sin_port = ::htons(port);
  destination_.sin_addr.s_addr = ::inet_addr(host.c_str());

  if (record && !start_recording(record_path)) {
    std::cerr << "Plotter: failed to enable recording\n";
  }
}

Plotter::~Plotter()
{
  stop_recording();
  if (socket_ >= 0) ::close(socket_);
}

void Plotter::plot(const nlohmann::json & json)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (record_stream_.is_open()) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - recording_start_);
    record_stream_ << nlohmann::json{{"time_us", elapsed.count()}, {"data", json}}.dump() << '\n';
    // 调试程序可能被直接终止，每个数据点及时落盘可最大限度保留记录。
    record_stream_.flush();
    if (!record_stream_) {
      std::cerr << "Plotter: failed to write recording file " << raw_recording_path_ << '\n';
      record_stream_.close();
    }
  }
  send_unlocked(json);
}

bool Plotter::start_recording(const std::string & path)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (recording_pending_ && !finish_recording_unlocked()) return false;

  std::filesystem::path requested_path = path.empty() ? default_recording_base_path() : path;
  if (requested_path.extension() == ".csv") {
    recording_path_ = requested_path.string();
    requested_path.replace_extension(".jsonl");
    raw_recording_path_ = requested_path.string();
  } else if (requested_path.extension() == ".jsonl") {
    raw_recording_path_ = requested_path.string();
    requested_path.replace_extension(".csv");
    recording_path_ = requested_path.string();
  } else {
    recording_path_ = requested_path.string() + ".csv";
    raw_recording_path_ = requested_path.string() + ".jsonl";
  }

  const std::filesystem::path file_path(recording_path_);
  const auto parent = file_path.parent_path();
  if (!parent.empty()) {
    std::error_code error;
    std::filesystem::create_directories(parent, error);
    if (error) {
      std::cerr << "Plotter: failed to create directory " << parent << ": " << error.message()
                << '\n';
      recording_path_.clear();
      raw_recording_path_.clear();
      recording_pending_ = false;
      return false;
    }
  }

  record_stream_.open(raw_recording_path_, std::ios::out | std::ios::trunc);
  if (!record_stream_.is_open()) {
    std::cerr << "Plotter: failed to open recording file " << raw_recording_path_ << '\n';
    recording_path_.clear();
    raw_recording_path_.clear();
    recording_pending_ = false;
    return false;
  }
  recording_pending_ = true;
  recording_start_ = std::chrono::steady_clock::now();
  return true;
}

bool Plotter::stop_recording()
{
  std::lock_guard<std::mutex> lock(mutex_);
  return finish_recording_unlocked();
}

bool Plotter::recording() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return record_stream_.is_open();
}

std::string Plotter::recording_path() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return recording_path_;
}

bool Plotter::replay(const std::string & path, double playback_speed)
{
  if (!std::isfinite(playback_speed) || playback_speed <= 0.0) {
    std::cerr << "Plotter: playback speed must be greater than zero\n";
    return false;
  }

  std::filesystem::path replay_path(path);
  if (replay_path.extension() == ".csv") replay_path.replace_extension(".jsonl");
  std::ifstream input(replay_path);
  if (!input.is_open()) {
    std::cerr << "Plotter: failed to open replay file " << replay_path << '\n';
    return false;
  }

  using Clock = std::chrono::steady_clock;
  std::string line;
  std::size_t line_number = 0;
  bool has_record = false;
  int64_t first_time_us = 0;
  int64_t previous_time_us = 0;
  auto replay_start = Clock::now();

  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty()) continue;

    nlohmann::json data;
    int64_t time_us = 0;
    if (!read_record(line, line_number, time_us, data)) return false;

    if (time_us < 0) {
      std::cerr << "Plotter: negative timestamp at line " << line_number << '\n';
      return false;
    }
    if (!has_record) {
      has_record = true;
      first_time_us = time_us;
      previous_time_us = time_us;
      replay_start = Clock::now();
    } else if (time_us < previous_time_us) {
      std::cerr << "Plotter: timestamps are not monotonic at line " << line_number << '\n';
      return false;
    }
    previous_time_us = time_us;

    const auto offset = std::chrono::duration<double, std::micro>(
      static_cast<double>(time_us - first_time_us) / playback_speed);
    std::this_thread::sleep_until(
      replay_start + std::chrono::duration_cast<Clock::duration>(offset));

    std::lock_guard<std::mutex> lock(mutex_);
    if (!send_unlocked(data)) {
      std::cerr << "Plotter: UDP send failed at line " << line_number << '\n';
      return false;
    }
  }

  if (!has_record) {
    std::cerr << "Plotter: replay file contains no data\n";
    return false;
  }
  return true;
}

bool Plotter::finish_recording_unlocked()
{
  if (!recording_pending_) return true;
  if (record_stream_.is_open()) record_stream_.close();

  std::ifstream input(raw_recording_path_);
  if (!input.is_open()) {
    std::cerr << "Plotter: failed to reopen recording file " << raw_recording_path_ << '\n';
    return false;
  }

  std::set<std::string> columns;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty()) continue;
    int64_t time_us = 0;
    nlohmann::json data;
    if (!read_record(line, line_number, time_us, data)) return false;
    std::map<std::string, std::string> fields;
    flatten_json(data, "", fields);
    for (const auto & field : fields) columns.insert(field.first);
  }

  const auto temporary_path = recording_path_ + ".tmp";
  std::ofstream csv(temporary_path, std::ios::out | std::ios::trunc);
  if (!csv.is_open()) {
    std::cerr << "Plotter: failed to create CSV file " << temporary_path << '\n';
    return false;
  }

  csv << "time";
  for (const auto & name : columns) csv << ',' << csv_escape(name);
  csv << '\n';

  input.clear();
  input.seekg(0);
  line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty()) continue;
    int64_t time_us = 0;
    nlohmann::json data;
    if (!read_record(line, line_number, time_us, data)) {
      csv.close();
      std::filesystem::remove(temporary_path);
      return false;
    }

    std::map<std::string, std::string> fields;
    flatten_json(data, "", fields);
    csv << std::fixed << std::setprecision(6) << static_cast<double>(time_us) / 1e6;
    for (const auto & name : columns) {
      csv << ',';
      const auto value = fields.find(name);
      if (value != fields.end()) csv << csv_escape(value->second);
    }
    csv << '\n';
  }
  csv.close();
  if (!csv) {
    std::cerr << "Plotter: failed to write CSV file " << temporary_path << '\n';
    std::filesystem::remove(temporary_path);
    return false;
  }

  std::error_code error;
  std::filesystem::rename(temporary_path, recording_path_, error);
  if (error) {
    std::cerr << "Plotter: failed to finalize CSV file " << recording_path_ << ": "
              << error.message() << '\n';
    std::filesystem::remove(temporary_path);
    return false;
  }
  recording_pending_ = false;
  return true;
}

bool Plotter::send_unlocked(const nlohmann::json & json)
{
  auto data = json.dump();
  const auto sent = ::sendto(
    socket_, data.c_str(), data.length(), 0, reinterpret_cast<sockaddr *>(&destination_),
    sizeof(destination_));
  return sent == static_cast<ssize_t>(data.length());
}

}  // namespace tools
