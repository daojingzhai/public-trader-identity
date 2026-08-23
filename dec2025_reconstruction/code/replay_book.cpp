#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <unordered_map>
#include <vector>

struct Fenwick {
    std::vector<double> tree;

    explicit Fenwick(std::size_t size) : tree(size + 2, 0.0) {}

    void add(std::int32_t key, double value) {
        for (auto index = static_cast<std::size_t>(key + 1); index < tree.size();
             index += index & -index) {
            tree[index] += value;
        }
    }

    double prefix(std::int32_t key) const {
        if (key < 0) return 0.0;
        auto index = std::min(static_cast<std::size_t>(key + 1), tree.size() - 1);
        double result = 0.0;
        while (index > 0) {
            result += tree[index];
            index -= index & -index;
        }
        return result;
    }

    void clear() {
        std::fill(tree.begin(), tree.end(), 0.0);
    }
};

struct Order {
    std::int32_t key;
    double size;
    bool ask;
};

struct Book {
    std::int32_t scale;
    std::int32_t maxKey;
    std::int64_t initialWarmupNs;
    std::int64_t resetWarmupNs;
    std::unordered_map<std::uint64_t, Order> orders;
    std::vector<double> bidSize;
    std::vector<double> askSize;
    std::set<std::int32_t> bidKeys;
    std::set<std::int32_t> askKeys;
    Fenwick bidNotional;
    Fenwick askNotional;
    std::uint64_t blockNumber = 0;
    std::int64_t lastTime = INT64_MIN;
    std::uint64_t duplicateNew = 0;
    std::uint64_t conflictingNew = 0;
    std::uint64_t missingRemove = 0;
    std::uint64_t missingUpdate = 0;
    std::uint64_t crossedGroups = 0;
    std::uint64_t outOfRange = 0;
    std::uint64_t timeViolations = 0;
    std::uint64_t crossingNewObserved = 0;
    std::uint64_t resetCount = 0;
    std::uint64_t resetInputRows = 0;
    std::uint64_t warmupGroups = 0;
    std::uint64_t warmupInputRows = 0;
    std::uint64_t emptyGroups = 0;
    std::uint64_t discardedLiveOrders = 0;
    std::uint64_t bookGateId = 0;
    std::int64_t lastResetTime = INT64_MIN;

    Book(std::int32_t scaleValue, std::int32_t maximumKey,
         std::int64_t initialWarmupValue, std::int64_t resetWarmupValue)
        : scale(scaleValue), maxKey(maximumKey), initialWarmupNs(initialWarmupValue),
          resetWarmupNs(resetWarmupValue),
          bidSize(maximumKey + 1, 0.0),
          askSize(maximumKey + 1, 0.0), bidNotional(maximumKey + 1),
          askNotional(maximumKey + 1) {
        orders.reserve(4'000'000);
    }

    void clearBook() {
        orders.clear();
        for (const auto key : bidKeys) bidSize[key] = 0.0;
        for (const auto key : askKeys) askSize[key] = 0.0;
        bidKeys.clear();
        askKeys.clear();
        bidNotional.clear();
        askNotional.clear();
    }

    void changeLevel(bool ask, std::int32_t key, double delta) {
        auto& sizes = ask ? askSize : bidSize;
        auto& keys = ask ? askKeys : bidKeys;
        auto& notional = ask ? askNotional : bidNotional;
        const double oldSize = sizes[key];
        const double newSize = oldSize + delta;
        sizes[key] = std::abs(newSize) < 1e-10 ? 0.0 : newSize;
        notional.add(key, delta * static_cast<double>(key) / scale);
        if (oldSize <= 1e-10 && sizes[key] > 1e-10) keys.insert(key);
        if (sizes[key] <= 1e-10) {
            sizes[key] = 0.0;
            keys.erase(key);
        }
    }

    void apply(std::uint64_t oid, bool ask, std::int32_t key, std::uint8_t diff,
               double newSize) {
        if (key < 0 || key > maxKey) {
            ++outOfRange;
            return;
        }
        const auto iterator = orders.find(oid);
        if (diff == 0) {
            if (iterator != orders.end()) {
                const auto& old = iterator->second;
                if (old.key == key && old.ask == ask && std::abs(old.size - newSize) < 1e-9) {
                    ++duplicateNew;
                } else {
                    ++conflictingNew;
                }
                return;
            }
            if ((ask && !bidKeys.empty() && key <= *bidKeys.rbegin()) ||
                (!ask && !askKeys.empty() && key >= *askKeys.begin())) ++crossingNewObserved;
            orders.emplace(oid, Order{key, newSize, ask});
            changeLevel(ask, key, newSize);
            return;
        }
        if (iterator == orders.end()) {
            if (diff == 1) ++missingRemove;
            else ++missingUpdate;
            return;
        }
        const auto old = iterator->second;
        if (diff == 1) {
            changeLevel(old.ask, old.key, -old.size);
            orders.erase(iterator);
            return;
        }
        changeLevel(old.ask, old.key, newSize - old.size);
        iterator->second.size = newSize;
    }
};

extern "C" void* book_create(std::int32_t scale, std::int32_t maxKey,
                             std::int64_t initialWarmupNs, std::int64_t resetWarmupNs) {
    return new Book(scale, maxKey, initialWarmupNs, resetWarmupNs);
}

extern "C" void book_destroy(void* pointer) {
    delete static_cast<Book*>(pointer);
}

extern "C" std::uint64_t book_process(
    void* pointer,
    std::uint64_t count,
    const std::int64_t* times,
    const std::uint64_t* oids,
    const std::uint8_t* asks,
    const std::int32_t* keys,
    const std::uint8_t* diffs,
    const double* sizes,
    const std::int64_t* gateIds,
    std::int64_t* outputTimes,
    std::uint64_t* outputBlocks,
    std::int64_t* outputTimingGates,
    std::uint64_t* outputBookGates,
    std::uint64_t* outputGroupRows,
    double* outputBid,
    double* outputAsk,
    double* outputBidSize,
    double* outputAskSize,
    double* outputBidDepth,
    double* outputAskDepth,
    double* outputBidNear,
    double* outputAskNear,
    double* outputWarmupAgeMs,
    std::uint8_t* outputValid,
    std::uint8_t* outputReset
) {
    auto& book = *static_cast<Book*>(pointer);
    std::uint64_t outputCount = 0;
    std::uint64_t index = 0;
    while (index < count) {
        const auto timestamp = times[index];
        const auto groupStart = index;
        const auto gateId = gateIds[index];
        if (timestamp < book.lastTime) ++book.timeViolations;
        if (book.lastResetTime == INT64_MIN) book.lastResetTime = timestamp;
        while (index < count && times[index] == timestamp && gateIds[index] == gateId) {
            book.apply(oids[index], asks[index], keys[index], diffs[index], sizes[index]);
            ++index;
        }
        const auto groupRows = index - groupStart;
        book.lastTime = std::max(book.lastTime, timestamp);

        bool reset = false;
        if (!book.bidKeys.empty() && !book.askKeys.empty() &&
            *book.bidKeys.rbegin() >= *book.askKeys.begin()) {
            ++book.crossedGroups;
            ++book.resetCount;
            book.resetInputRows += groupRows;
            book.discardedLiveOrders += book.orders.size();
            ++book.bookGateId;
            book.lastResetTime = timestamp;
            book.clearBook();
            reset = true;
        }

        const bool hasBook = !book.bidKeys.empty() && !book.askKeys.empty();
        const auto warmupAgeNs = std::max<std::int64_t>(0, timestamp - book.lastResetTime);
        const auto requiredWarmupNs = book.bookGateId == 0
                                    ? book.initialWarmupNs : book.resetWarmupNs;
        const bool warmed = warmupAgeNs >= requiredWarmupNs;
        if (!reset && !hasBook) ++book.emptyGroups;
        if (!reset && hasBook && !warmed) {
            ++book.warmupGroups;
            book.warmupInputRows += groupRows;
        }

        const double missing = std::numeric_limits<double>::quiet_NaN();
        double bid = missing;
        double ask = missing;
        double bidSize = missing;
        double askSize = missing;
        double bidDepth = missing;
        double askDepth = missing;
        double bidNear = missing;
        double askNear = missing;
        if (hasBook) {
            const auto bidKey = *book.bidKeys.rbegin();
            const auto askKey = *book.askKeys.begin();
            bid = static_cast<double>(bidKey) / book.scale;
            ask = static_cast<double>(askKey) / book.scale;
            bidSize = book.bidSize[bidKey];
            askSize = book.askSize[askKey];
            bidDepth = bid * bidSize;
            askDepth = ask * askSize;
            const double mid = (bid + ask) / 2.0;
            const auto lowerBidKey = static_cast<std::int32_t>(std::ceil(
                mid * (1.0 - 0.0005) * book.scale - 1e-9));
            const auto upperAskKey = static_cast<std::int32_t>(std::floor(
                mid * (1.0 + 0.0005) * book.scale + 1e-9));
            const double totalBid = book.bidNotional.prefix(book.maxKey);
            bidNear = totalBid - book.bidNotional.prefix(lowerBidKey - 1);
            askNear = book.askNotional.prefix(upperAskKey);
        }

        outputTimes[outputCount] = timestamp;
        outputBlocks[outputCount] = book.blockNumber++;
        outputTimingGates[outputCount] = gateId;
        outputBookGates[outputCount] = book.bookGateId;
        outputGroupRows[outputCount] = groupRows;
        outputBid[outputCount] = bid;
        outputAsk[outputCount] = ask;
        outputBidSize[outputCount] = bidSize;
        outputAskSize[outputCount] = askSize;
        outputBidDepth[outputCount] = bidDepth;
        outputAskDepth[outputCount] = askDepth;
        outputBidNear[outputCount] = bidNear;
        outputAskNear[outputCount] = askNear;
        outputWarmupAgeMs[outputCount] = static_cast<double>(warmupAgeNs) / 1e6;
        outputValid[outputCount] = hasBook && warmed;
        outputReset[outputCount] = reset;
        ++outputCount;
    }
    return outputCount;
}

extern "C" void book_metrics(void* pointer, std::uint64_t* output) {
    const auto& book = *static_cast<Book*>(pointer);
    output[0] = book.orders.size();
    output[1] = book.duplicateNew;
    output[2] = book.conflictingNew;
    output[3] = book.missingRemove;
    output[4] = book.missingUpdate;
    output[5] = book.crossedGroups;
    output[6] = book.outOfRange;
    output[7] = book.timeViolations;
    output[8] = book.blockNumber;
    output[9] = book.crossingNewObserved;
    output[10] = book.resetCount;
    output[11] = book.resetInputRows;
    output[12] = book.warmupGroups;
    output[13] = book.warmupInputRows;
    output[14] = book.emptyGroups;
    output[15] = book.discardedLiveOrders;
    output[16] = book.bookGateId;
}
