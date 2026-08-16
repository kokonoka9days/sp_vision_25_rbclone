#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <string>

#include "tools/plotter.hpp"

namespace
{
void print_usage(const char * program)
{
  std::cerr << "Usage: " << program << " <record.csv|record.jsonl> [speed] [host] [port]\n"
            << "Example: " << program << " plot_records/2026-08-16_19-30-00_123456.jsonl 1.0\n";
}
}  // namespace

int main(int argc, char * argv[])
{
  if (argc < 2 || argc > 5) {
    print_usage(argv[0]);
    return 1;
  }

  try {
    const std::string path = argv[1];
    const double speed = argc >= 3 ? std::stod(argv[2]) : 1.0;
    const std::string host = argc >= 4 ? argv[3] : "127.0.0.1";
    const unsigned long parsed_port = argc >= 5 ? std::stoul(argv[4]) : 9870;
    if (parsed_port > std::numeric_limits<uint16_t>::max()) {
      throw std::out_of_range("port");
    }

    tools::Plotter plotter(host, static_cast<uint16_t>(parsed_port));
    if (!plotter.replay(path, speed)) return 1;
  } catch (const std::exception & error) {
    std::cerr << "plotter_replay: invalid argument: " << error.what() << '\n';
    print_usage(argv[0]);
    return 1;
  }

  return 0;
}
