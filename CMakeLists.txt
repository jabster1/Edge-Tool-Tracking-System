cmake_minimum_required(VERSION 3.20)
project(ToolTrackerBenchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Apple Silicon optimisation flags ─────────────────────────────────────────
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    add_compile_options(-O3 -march=armv8-a)
endif()

# ── ONNX Runtime (bundled in third_party/) ────────────────────────────────────
set(ORT_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_DIR}/lib/libonnxruntime.dylib"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/include"
)

# ── OpenCV (Homebrew arm64) ───────────────────────────────────────────────────
# Prefer the Homebrew pkg-config path
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
find_package(OpenCV REQUIRED
    COMPONENTS core imgproc imgcodecs highgui videoio
    HINTS /opt/homebrew/lib/cmake/opencv4
          /opt/homebrew/Cellar/opencv/4.13.0_7/lib/cmake/opencv4
)

message(STATUS "OpenCV version : ${OpenCV_VERSION}")
message(STATUS "ORT include    : ${ORT_DIR}/include")
message(STATUS "ORT lib        : ${ORT_DIR}/lib/libonnxruntime.dylib")

# ── benchmark_cpp ─────────────────────────────────────────────────────────────
add_executable(benchmark_cpp benchmark_cpp.cpp)
target_link_libraries(benchmark_cpp PRIVATE onnxruntime ${OpenCV_LIBS})
target_include_directories(benchmark_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})

# Copy the ONNX Runtime dylib next to the built binary so macOS finds it at
# runtime without needing to set DYLD_LIBRARY_PATH.
add_custom_command(TARGET benchmark_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_DIR}/lib/libonnxruntime.dylib"
        "${ORT_DIR}/lib/libonnxruntime.1.20.1.dylib"
        $<TARGET_FILE_DIR:benchmark_cpp>
    COMMENT "Copying ONNX Runtime dylibs to build directory"
)

# ── tool_tracker_onnx ────────────────────────────────────────────────────────
add_executable(tool_tracker_onnx tool_tracker_onnx.cpp)
target_link_libraries(tool_tracker_onnx PRIVATE onnxruntime ${OpenCV_LIBS})
target_include_directories(tool_tracker_onnx PRIVATE ${OpenCV_INCLUDE_DIRS})

add_custom_command(TARGET tool_tracker_onnx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_DIR}/lib/libonnxruntime.dylib"
        "${ORT_DIR}/lib/libonnxruntime.1.20.1.dylib"
        $<TARGET_FILE_DIR:tool_tracker_onnx>
    COMMENT "Copying ONNX Runtime dylibs to build directory"
)

# ── Install targets ──────────────────────────────────────────────────────────
install(TARGETS benchmark_cpp tool_tracker_onnx
        RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/build/bin)
