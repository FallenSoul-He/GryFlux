# =============================================================================
# GryFlux Framework - AArch64 Cross-Compilation Toolchain
# =============================================================================
#
# 使用方法：
#   mkdir -p build/aarch64 && cd build/aarch64
#   cmake -DCMAKE_TOOLCHAIN_FILE=../../cmake/aarch64-toolchain.cmake ..
#   cmake --build . -j8
#
# 环境要求：
#   - 安装 aarch64-linux-gnu-gcc 工具链
#   - Ubuntu/Debian: sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
#
# =============================================================================

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# -----------------------------------------------------------------------------
# Toolchain discovery (portable)
#
# 默认使用系统 PATH 中的 aarch64-linux-gnu-* 工具链。
# 如果你用的是自解压的工具链包，把 bin 目录传进来即可：
#   -DTOOLCHAIN_BIN_DIR=/opt/gcc-aarch64/bin
# 或者用不同前缀（少见）：
#   -DTOOLCHAIN_PREFIX=aarch64-none-linux-gnu
# -----------------------------------------------------------------------------
set(TOOLCHAIN_PREFIX "aarch64-linux-gnu" CACHE STRING "Toolchain prefix (without trailing '-')")
set(TOOLCHAIN_BIN_DIR "" CACHE PATH "Optional toolchain bin directory (prepended for discovery)")

if(TOOLCHAIN_BIN_DIR)
    list(PREPEND CMAKE_PROGRAM_PATH "${TOOLCHAIN_BIN_DIR}")
endif()

find_program(CMAKE_C_COMPILER NAMES "${TOOLCHAIN_PREFIX}-gcc" REQUIRED)
find_program(CMAKE_CXX_COMPILER NAMES "${TOOLCHAIN_PREFIX}-g++" REQUIRED)
find_program(CMAKE_AR NAMES "${TOOLCHAIN_PREFIX}-ar" REQUIRED)
find_program(CMAKE_RANLIB NAMES "${TOOLCHAIN_PREFIX}-ranlib" REQUIRED)
find_program(CMAKE_STRIP NAMES "${TOOLCHAIN_PREFIX}-strip" REQUIRED)

# 设置查找路径模式
# NEVER: 不在宿主机系统中查找
# ONLY: 只在目标系统路径中查找
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# 可选：设置目标系统的 sysroot（如果需要）
# 建议在命令行传入（更通用）：
#   -DCMAKE_SYSROOT=/path/to/aarch64-sysroot
if(CMAKE_SYSROOT)
    set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
endif()

# 编译选项
# 注意：AArch64 默认包含 NEON，不需要 -mfpu=neon（这是 ARMv7 32位的选项）
set(CMAKE_C_FLAGS_INIT "-march=armv8-a")
set(CMAKE_CXX_FLAGS_INIT "-march=armv8-a")


# NEON 在 AArch64 中是强制支持的，不需要额外选项

message(STATUS "=================================================")
message(STATUS "Cross-compiling for AArch64")
message(STATUS "=================================================")
message(STATUS "Toolchain prefix: ${TOOLCHAIN_PREFIX}-")
message(STATUS "Toolchain bin dir: ${TOOLCHAIN_BIN_DIR}")
message(STATUS "C Compiler:       ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler:     ${CMAKE_CXX_COMPILER}")
message(STATUS "Sysroot:          ${CMAKE_SYSROOT}")
message(STATUS "Target System:    ${CMAKE_SYSTEM_NAME}")
message(STATUS "Target Processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "=================================================")
