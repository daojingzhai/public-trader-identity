#include <algorithm>
#include <cstdint>
#include <vector>

extern "C" std::uint64_t longest_nondecreasing(
    const std::int64_t* values,
    std::uint64_t count,
    std::uint8_t* keep
) {
    std::vector<std::int64_t> tails;
    std::vector<std::uint32_t> tailIndexes;
    std::vector<std::uint32_t> previous(count, UINT32_MAX);
    tails.reserve(count);
    tailIndexes.reserve(count);

    for (std::uint32_t index = 0; index < count; ++index) {
        const auto position = std::upper_bound(tails.begin(), tails.end(), values[index])
                            - tails.begin();
        if (position > 0) {
            previous[index] = tailIndexes[position - 1];
        }
        if (position == static_cast<std::int64_t>(tails.size())) {
            tails.push_back(values[index]);
            tailIndexes.push_back(index);
        } else {
            tails[position] = values[index];
            tailIndexes[position] = index;
        }
    }

    std::fill(keep, keep + count, 0);
    auto index = tailIndexes.empty() ? UINT32_MAX : tailIndexes.back();
    while (index != UINT32_MAX) {
        keep[index] = 1;
        index = previous[index];
    }
    return tails.size();
}
