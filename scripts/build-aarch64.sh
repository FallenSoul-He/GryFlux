#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/build-aarch64.sh [options] [-- <extra cmake configure args...>]

Options:
  --enable_profile        -DGRYFLUX_BUILD_PROFILING=1
  --toolchain-bin <path>   Sets -DTOOLCHAIN_BIN_DIR (optional)
  --toolchain-prefix <p>   Default: aarch64-linux-gnu
  -h, --help               Show this help

Examples:
  bash scripts/build-aarch64.sh
  bash scripts/build-aarch64.sh --toolchain-bin /opt/gcc-aarch64/bin
  bash scripts/build-aarch64.sh -- --warn-uninitialized
  bash scripts/build-aarch64.sh -- -DCMAKE_SYSROOT=/opt/aarch64-sysroot
EOF
}

build_dir="${PROJECT_ROOT}/build/aarch64"
do_profile=0
prefix="${PROJECT_ROOT}/install/aarch64"
toolchain_file="${PROJECT_ROOT}/cmake/aarch64-toolchain.cmake"
toolchain_prefix="aarch64-linux-gnu"
toolchain_bin=""
extra_cmake_args=()

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --enable_profile) do_profile=1; shift ;;
    --toolchain-prefix) toolchain_prefix="${2:-}"; shift 2 ;;
    --toolchain-bin) toolchain_bin="${2:-}"; shift 2 ;;
    --) shift; extra_cmake_args+=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

jobs="$(command -v nproc >/dev/null 2>&1 && nproc || echo 8)"

if [[ ! -f "${toolchain_file}" ]]; then
  echo "Toolchain file not found: ${toolchain_file}" >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found in PATH" >&2
  exit 1
fi

echo "=================================================="
echo "GryFlux - AArch64 cross build"
echo "Project:   ${PROJECT_ROOT}"
echo "Build dir: ${build_dir}"
echo "Jobs:      ${jobs}"
echo "Toolchain: ${toolchain_file}"
echo "Prefix:    ${toolchain_prefix}-"
echo "Bin dir:   ${toolchain_bin:-<PATH>}"
echo "Profile:   ${do_profile}"
echo "Dest:      ${prefix}"
echo "=================================================="

mkdir -p "${build_dir}"

CMAKE_CONFIG_ARGS=(
  "-S" "${PROJECT_ROOT}"
  "-B" "${build_dir}"
  "-DCMAKE_TOOLCHAIN_FILE=${toolchain_file}"
  "-DCMAKE_BUILD_TYPE=Release"
  "-DBUILD_TEST=False"
  "-DCMAKE_INSTALL_PREFIX=${prefix}"
  "-DTOOLCHAIN_PREFIX=${toolchain_prefix}"
)

extra_cxxflags=""
if [[ "${do_profile}" -eq 1 ]]; then
  extra_cxxflags="-DGRYFLUX_BUILD_PROFILING=1"
fi

cxx_flags="${CXXFLAGS:-} ${extra_cxxflags}"
CMAKE_CONFIG_ARGS+=("-DCMAKE_CXX_FLAGS:STRING=${cxx_flags}")

if [[ -n "${toolchain_bin}" ]]; then
  CMAKE_CONFIG_ARGS+=("-DTOOLCHAIN_BIN_DIR=${toolchain_bin}")
fi

cmake "${CMAKE_CONFIG_ARGS[@]}" "${extra_cmake_args[@]}"
cmake --build "${build_dir}" --target install --parallel "${jobs}"
