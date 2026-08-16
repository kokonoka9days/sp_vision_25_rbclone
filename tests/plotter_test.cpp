#include "tools/plotter.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace
{
#define CHECK(condition)               \
  do {                                 \
    if (!(condition)) {                \
      std::cerr << #condition << '\n'; \
      return 1;                        \
    }                                  \
  } while (false)

nlohmann::json receive_json(int socket)
{
  char buffer[4096];
  const auto size = ::recv(socket, buffer, sizeof(buffer), 0);
  if (size <= 0) return {};
  return nlohmann::json::parse(buffer, buffer + size);
}

std::vector<std::string> split_csv_line(const std::string & line)
{
  std::vector<std::string> values;
  std::istringstream stream(line);
  std::string value;
  while (std::getline(stream, value, ',')) values.push_back(value);
  if (!line.empty() && line.back() == ',') values.emplace_back();
  return values;
}
}  // namespace

int main()
{
  const int receiver = ::socket(AF_INET, SOCK_DGRAM, 0);
  CHECK(receiver >= 0);

  timeval timeout{};
  timeout.tv_sec = 1;
  CHECK(::setsockopt(receiver, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0);

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  CHECK(::bind(receiver, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0);

  socklen_t address_size = sizeof(address);
  CHECK(::getsockname(receiver, reinterpret_cast<sockaddr *>(&address), &address_size) == 0);
  const uint16_t port = ::ntohs(address.sin_port);

  const auto path = std::filesystem::temp_directory_path() /
                    ("plotter_test_" + std::to_string(::getpid()) + ".csv");
  auto raw_path = path;
  raw_path.replace_extension(".jsonl");
  std::filesystem::remove(path);
  std::filesystem::remove(raw_path);

  tools::Plotter plotter(true, "127.0.0.1", port, path.string());
  CHECK(plotter.recording());
  CHECK(plotter.recording_path() == path.string());

  const nlohmann::json first{{"value", 1}, {"name", "first"}};
  const nlohmann::json second{
    {"value", 2}, {"name", "second"}, {"late", 9}, {"nested", {{"x", 3}}}};
  plotter.plot(first);
  std::this_thread::sleep_for(5ms);
  plotter.plot(second);
  CHECK(plotter.stop_recording());
  CHECK(!plotter.recording());
  CHECK(receive_json(receiver) == first);
  CHECK(receive_json(receiver) == second);

  std::ifstream recording(raw_path);
  std::string line;
  CHECK(std::getline(recording, line));
  const auto first_record = nlohmann::json::parse(line);
  CHECK(first_record["data"] == first);
  CHECK(first_record["time_us"].get<int64_t>() >= 0);
  CHECK(std::getline(recording, line));
  const auto second_record = nlohmann::json::parse(line);
  CHECK(second_record["data"] == second);
  CHECK(second_record["time_us"].get<int64_t>() >= first_record["time_us"].get<int64_t>());
  CHECK(!std::getline(recording, line));

  std::ifstream csv(path);
  CHECK(std::getline(csv, line));
  CHECK(line == "time,late,name,nested/x,value");
  CHECK(std::getline(csv, line));
  auto values = split_csv_line(line);
  CHECK(values.size() == 5);
  CHECK(values[1].empty());
  CHECK(values[2] == "first");
  CHECK(values[3].empty());
  CHECK(values[4] == "1");
  CHECK(std::getline(csv, line));
  values = split_csv_line(line);
  CHECK(values.size() == 5);
  CHECK(values[1] == "9");
  CHECK(values[2] == "second");
  CHECK(values[3] == "3");
  CHECK(values[4] == "2");
  CHECK(!std::getline(csv, line));

  CHECK(plotter.replay(path.string(), 100.0));
  CHECK(receive_json(receiver) == first);
  CHECK(receive_json(receiver) == second);
  CHECK(!plotter.replay(path.string(), 0.0));

  const auto destructor_path = std::filesystem::temp_directory_path() /
                               ("plotter_destructor_test_" + std::to_string(::getpid()) + ".csv");
  auto destructor_raw_path = destructor_path;
  destructor_raw_path.replace_extension(".jsonl");
  std::filesystem::remove(destructor_path);
  std::filesystem::remove(destructor_raw_path);
  {
    tools::Plotter destructor_plotter(true, "127.0.0.1", port, destructor_path.string());
    destructor_plotter.plot(first);
  }
  CHECK(std::filesystem::exists(destructor_path));
  CHECK(receive_json(receiver) == first);

  ::close(receiver);
  std::filesystem::remove(path);
  std::filesystem::remove(raw_path);
  std::filesystem::remove(destructor_path);
  std::filesystem::remove(destructor_raw_path);
  return 0;
}
