#pragma once

#include <string>

namespace VQCodecRuntime
{

class AclEnvironment
{
public:
    static bool acquire(int deviceId, std::string *error);
    static void release(int deviceId);
};

} // namespace VQCodecRuntime
