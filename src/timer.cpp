#include "timer.h"

#include <ctime>
#include <iomanip>
#include <sstream>

using namespace std;
namespace crack {

double elapsedMilliseconds(const Clock::time_point& start, const Clock::time_point& end)
{
    return   chrono::duration<double,   milli>(end - start).count();
}

  string timestampString()
{
    const auto now =   chrono::system_clock::now();
    const   time_t tt =   chrono::system_clock::to_time_t(now);
      tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif

      ostringstream stream;
    stream <<   put_time(&tm, "%Y%m%d_%H%M%S");
    return stream.str();
}

  string formatMilliseconds(double value)
{
      ostringstream stream;
    stream <<   fixed <<   setprecision(2) << value;
    return stream.str();
}

} // namespace crack