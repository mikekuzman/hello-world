#include <chrono>

namespace bec4d {

class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double elapsedSeconds() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - start_).count();
    }

    double elapsedMilliseconds() const {
        return elapsedSeconds() * 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

} // namespace bec4d
