#if defined(BB_ESMX_CLOCK_ACOUSTIC) && defined(ESP_PLATFORM)
#include "ClockModelVfs.h"

#include "ClockModelAssets.h"
#ifdef BB_ESMX_DICE_ACOUSTIC
#include "DiceAcousticAssets.h"
#endif
#include "esp_vfs.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <span>
#include <string_view>
#include <sys/stat.h>

namespace EsMx::Pilot {
namespace {

constexpr char MountPoint[] = "/bbclock";
constexpr size_t Dimensions = 39;
constexpr size_t PackedHeader = Dimensions * sizeof(float) * 2;
static_assert(sizeof(float) == 4 && std::endian::native == std::endian::little);

struct Asset {
    std::string_view path;
    std::span<const uint8_t> prefix;
    std::span<const uint8_t> packed;

    size_t size() const {
        return prefix.size() + (packed.empty() ? 0 : (packed.size() - PackedHeader) * sizeof(float));
    }

    float reconstruct(size_t index) const {
        const auto feature = (index % Dimensions) * sizeof(float);
        float origin, scale;
        std::memcpy(&origin, packed.data() + feature, sizeof(float));
        std::memcpy(&scale, packed.data() + Dimensions * sizeof(float) + feature, sizeof(float));
        return origin + scale * packed[PackedHeader + index];
    }
};

using namespace ClockModelAssets;
static_assert(sizeof(MeansQ8) == PackedHeader + 90 * Densities * Dimensions);
static_assert(sizeof(VariancesQ8) == sizeof(MeansQ8));

constexpr Asset Files[]{
#ifdef BB_ESMX_DICE_ACOUSTIC
    {"/esmx-dice.dict", DiceAcousticAssets::Dictionary, {}},
    {"/dice.kws", DiceAcousticAssets::Keywords, {}},
#else
    {"/clock.dict", Dictionary, {}},
#endif
    {"/feat.params", Features, {}}, {"/mdef", Definition, {}}, {"/noisedict", Noise, {}},
    {"/mixture_weights", Mixture, {}},
    {"/transition_matrices", Transitions, {}},
    {"/means", MeansPrefix, MeansQ8}, {"/variances", VariancesPrefix, VariancesQ8},
};

// One private directory consumed only by the single recognition worker.
struct Handle {
    const Asset* asset = nullptr;
    size_t offset = 0;
};
std::array<Handle, 6> handles;

const Asset* lookup(std::string_view path) {
    const auto found = std::find_if(std::begin(Files), std::end(Files),
                                    [path](const auto& file) { return path == file.path; });
    return found == std::end(Files) ? nullptr : found;
}

Handle* active(int id) {
    if (id < 0 || static_cast<size_t>(id) >= handles.size() || !handles[id].asset) {
        errno = EBADF;
        return nullptr;
    }
    return &handles[id];
}

int openModel(const char* path, int flags, int) {
    if ((flags & O_ACCMODE) != O_RDONLY || (flags & (O_CREAT | O_TRUNC | O_APPEND))) {
        errno = EROFS;
        return -1;
    }
    const auto* asset = lookup(path);
    if (!asset) {
        errno = ENOENT;
        return -1;
    }
    const auto free = std::find_if(handles.begin(), handles.end(), [](const auto& handle) { return !handle.asset; });
    if (free == handles.end()) {
        errno = EMFILE;
        return -1;
    }
    *free = {asset, 0};
    return static_cast<int>(free - handles.begin());
}

int closeModel(int id) {
    auto* handle = active(id);
    if (!handle) {
        return -1;
    }
    *handle = {};
    return 0;
}

ssize_t readModel(int id, void* destination, size_t bytes) {
    auto* handle = active(id);
    if (!handle) {
        return -1;
    }
    const auto& asset = *handle->asset;
    const auto count = std::min(bytes, asset.size() - handle->offset);
    auto remaining = count;
    auto* dst = static_cast<uint8_t*>(destination);
    auto offset = handle->offset;
    if (remaining && offset < asset.prefix.size()) {
        const auto take = std::min(remaining, asset.prefix.size() - offset);
        std::memcpy(dst, asset.prefix.data() + offset, take);
        dst += take;
        offset += take;
        remaining -= take;
    }
    // fopen/fread may split a float across two callbacks.
    while (remaining) {
        const auto body = offset - asset.prefix.size();
        const auto skip = body % sizeof(float);
        const auto value = asset.reconstruct(body / sizeof(float));
        const auto take = std::min(remaining, sizeof(float) - skip);
        std::memcpy(dst, reinterpret_cast<const uint8_t*>(&value) + skip, take);
        dst += take;
        offset += take;
        remaining -= take;
    }
    handle->offset = offset;
    return static_cast<ssize_t>(count);
}

off_t seekModel(int id, off_t relative, int mode) {
    auto* handle = active(id);
    if (!handle) {
        return -1;
    }
    int64_t origin = 0;
    switch (mode) {
        case SEEK_SET: break;
        case SEEK_CUR: origin = handle->offset; break;
        case SEEK_END: origin = handle->asset->size(); break;
        default: errno = EINVAL; return -1;
    }
    if (relative < -origin || relative > static_cast<int64_t>(handle->asset->size()) - origin) {
        errno = EINVAL;
        return -1;
    }
    handle->offset = static_cast<size_t>(origin + relative);
    return static_cast<off_t>(handle->offset);
}

void fileInfo(const Asset& asset, struct stat* out) {
    *out = {};
    out->st_mode = S_IFREG | 0444;
    out->st_size = static_cast<off_t>(asset.size());
}

int fstatModel(int id, struct stat* out) {
    auto* handle = active(id);
    if (!handle) {
        return -1;
    }
    fileInfo(*handle->asset, out);
    return 0;
}

int statModel(const char* path, struct stat* out) {
    if (path[0] == '\0' || std::string_view(path) == "/") {
        *out = {};
        out->st_mode = S_IFDIR | 0555;
        return 0;
    }
    const auto* asset = lookup(path);
    if (!asset) {
        errno = ENOENT;
        return -1;
    }
    fileInfo(*asset, out);
    return 0;
}

}

const char* mountClockModel() {
    static const bool ready = [] {
        esp_vfs_t fs{};
        fs.flags = ESP_VFS_FLAG_READONLY_FS;
        fs.open = openModel;
        fs.close = closeModel;
        fs.read = readModel;
        fs.lseek = seekModel;
        fs.fstat = fstatModel;
        fs.stat = statModel;
        return esp_vfs_register(MountPoint, &fs, nullptr) == ESP_OK;
    }();
    return ready ? MountPoint : nullptr;
}

}
#endif
