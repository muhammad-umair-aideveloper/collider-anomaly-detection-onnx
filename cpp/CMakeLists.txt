cmake_minimum_required(VERSION 3.15)
project(collider_inference CXX)

# Require C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release build type for single-config generators
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type (Release default)" FORCE)
endif()

# Check and validate ONNX Runtime directory
if(NOT DEFINED ONNXRUNTIME_DIR OR "${ONNXRUNTIME_DIR}" STREQUAL "")
    message(FATAL_ERROR 
        "ONNXRUNTIME_DIR must be specified on CMake configure line.\n"
        "Example: cmake -S cpp -B build -DONNXRUNTIME_DIR=\"C:/onnxruntime\""
    )
endif()

# Normalize path separators
file(TO_CMAKE_PATH "${ONNXRUNTIME_DIR}" ONNXRUNTIME_DIR)

# Validate essential headers and import library
set(ORT_HEADER_PATH "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
set(ORT_LIB_PATH "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
set(ORT_DLL_PATH "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll")

if(NOT EXISTS "${ORT_HEADER_PATH}")
    message(FATAL_ERROR "ONNX Runtime C++ header not found at: ${ORT_HEADER_PATH}")
endif()

if(NOT EXISTS "${ORT_LIB_PATH}")
    message(FATAL_ERROR "ONNX Runtime import library not found at: ${ORT_LIB_PATH}")
endif()

if(NOT EXISTS "${ORT_DLL_PATH}")
    message(FATAL_ERROR "ONNX Runtime runtime DLL not found at: ${ORT_DLL_PATH}")
endif()

message(STATUS "Found ONNX Runtime headers: ${ONNXRUNTIME_DIR}/include")
message(STATUS "Found ONNX Runtime library: ${ORT_LIB_PATH}")
message(STATUS "Found ONNX Runtime DLL:     ${ORT_DLL_PATH}")

# Add inference executable
add_executable(collider_inference inference.cpp)

# Target include directories and linking
target_include_directories(collider_inference PRIVATE "${ONNXRUNTIME_DIR}/include")
target_link_libraries(collider_inference PRIVATE "${ORT_LIB_PATH}")

# Compiler warning flags
if(MSVC)
    target_compile_options(collider_inference PRIVATE /W4 /permissive-)
else()
    target_compile_options(collider_inference PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Define compile-time build configuration string
target_compile_definitions(collider_inference PRIVATE
    BUILD_CONFIG_STRING="$<CONFIG>"
)

# Post-build step: Copy onnxruntime.dll next to executable
add_custom_command(
    TARGET collider_inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_DLL_PATH}"
            $<TARGET_FILE_DIR:collider_inference>
    COMMENT "Copying onnxruntime.dll next to target executable"
)

# Also copy onnxruntime_providers_shared.dll if present
set(ORT_SHARED_PROVIDERS_DLL "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_shared.dll")
if(EXISTS "${ORT_SHARED_PROVIDERS_DLL}")
    add_custom_command(
        TARGET collider_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ORT_SHARED_PROVIDERS_DLL}"
                $<TARGET_FILE_DIR:collider_inference>
        COMMENT "Copying onnxruntime_providers_shared.dll next to target executable"
    )
endif()

