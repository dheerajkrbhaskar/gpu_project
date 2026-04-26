#pragma once

#include <chrono>
#include <string>

namespace crack {

using Clock = std::chrono::steady_clock;

double elapsedMilliseconds(const Clock::time_point& start, const Clock::time_point& end);

template <typename Func>
inline double measureMilliseconds(Func&& func)
{
    const auto start = Clock::now();
    func();
    return elapsedMilliseconds(start, Clock::now());
}

std::string timestampString();

std::string formatMilliseconds(double value);

} // namespace crack