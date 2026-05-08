#include "AclEnvironment.h"

#include <acl/acl.h>

#include <mutex>
#include <unordered_map>

namespace VQCodecRuntime
{
namespace
{

std::mutex gMutex;
size_t gRefCount = 0;
std::unordered_map<int, size_t> gDeviceRefCounts;

} // namespace

bool AclEnvironment::acquire(int deviceId, std::string *error)
{
    std::lock_guard<std::mutex> lock(gMutex);

    if (gRefCount == 0) {
        const aclError initRet = aclInit(nullptr);
        if (initRet != ACL_ERROR_NONE) {
            if (error != nullptr) {
                *error = "aclInit failed, code=" + std::to_string(initRet);
            }
            return false;
        }
    }

    auto deviceIt = gDeviceRefCounts.find(deviceId);
    if (deviceIt == gDeviceRefCounts.end()) {
        const aclError setRet = aclrtSetDevice(deviceId);
        if (setRet != ACL_ERROR_NONE) {
            if (gRefCount == 0) {
                aclFinalize();
            }
            if (error != nullptr) {
                *error = "aclrtSetDevice failed, code=" + std::to_string(setRet);
            }
            return false;
        }
        gDeviceRefCounts.emplace(deviceId, 1);
    } else {
        ++deviceIt->second;
    }

    ++gRefCount;
    return true;
}

void AclEnvironment::release(int deviceId)
{
    std::lock_guard<std::mutex> lock(gMutex);
    if (gRefCount == 0) {
        return;
    }

    auto deviceIt = gDeviceRefCounts.find(deviceId);
    if (deviceIt == gDeviceRefCounts.end()) {
        return;
    }

    if (deviceIt->second > 1) {
        --deviceIt->second;
    } else {
        gDeviceRefCounts.erase(deviceIt);
        aclrtSetDevice(deviceId);
        aclrtResetDevice(deviceId);
    }

    --gRefCount;
    if (gRefCount == 0) {
        aclFinalize();
    }
}

} // namespace VQCodecRuntime
