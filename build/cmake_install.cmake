# Install script for directory: /Users/jadenbarnwell/Desktop/tool_tracker/Tool-Tracking-System

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/Users/jadenbarnwell/Desktop/tool_tracker/Tool-Tracking-System/build/bin/benchmark_cpp")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/Users/jadenbarnwell/Desktop/tool_tracker/Tool-Tracking-System/build/bin" TYPE EXECUTABLE FILES "/Users/jadenbarnwell/Desktop/tool_tracker/Tool-Tracking-System/build/benchmark_cpp")
  if(EXISTS "$ENV{DESTDIR}/Users/jadenbarnwell/Desktop/tool_tracker/Tool-Tracking-System/build/bin/benchmark_cpp" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/Users/jadenbarnwell/Desktop/tool_tracker/Tool-Tracking-System/build/bin/benchmark_cpp")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/jadenbarnwell/Desktop/tool_tracker/Tool-Tracking-System/third_party/onnxruntime/lib"
      -delete_rpath "/opt/homebrew/lib"
      "$ENV{DESTDIR}/Users/jadenbarnwell/Desktop/tool_tracker/Tool-Tracking-System/build/bin/benchmark_cpp")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}/Users/jadenbarnwell/Desktop/tool_tracker/Tool-Tracking-System/build/bin/benchmark_cpp")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/Users/jadenbarnwell/Desktop/tool_tracker/Tool-Tracking-System/build/bin/tool_tracker_onnx")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/Users/jadenbarnwell/Desktop/tool_tracker/Tool-Tracking-System/build/bin" TYPE EXECUTABLE FILES "/Users/jadenbarnwell/Desktop/tool_tracker/Tool-Tracking-System/build/tool_tracker_onnx")
  if(EXISTS "$ENV{DESTDIR}/Users/jadenbarnwell/Desktop/tool_tracker/Tool-Tracking-System/build/bin/tool_tracker_onnx" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/Users/jadenbarnwell/Desktop/tool_tracker/Tool-Tracking-System/build/bin/tool_tracker_onnx")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/jadenbarnwell/Desktop/tool_tracker/Tool-Tracking-System/third_party/onnxruntime/lib"
      -delete_rpath "/opt/homebrew/lib"
      "$ENV{DESTDIR}/Users/jadenbarnwell/Desktop/tool_tracker/Tool-Tracking-System/build/bin/tool_tracker_onnx")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}/Users/jadenbarnwell/Desktop/tool_tracker/Tool-Tracking-System/build/bin/tool_tracker_onnx")
    endif()
  endif()
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/jadenbarnwell/Desktop/tool_tracker/Tool-Tracking-System/build/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/jadenbarnwell/Desktop/tool_tracker/Tool-Tracking-System/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
