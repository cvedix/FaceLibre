cmake_minimum_required(VERSION 3.14)
project(FaceLibre VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Policies
cmake_policy(SET CMP0079 NEW)

# Find OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn objdetect)

# Find system packages
find_package(ZLIB REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

include(FetchContent)

# Download Drogon with all its dependencies
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_CTL OFF CACHE BOOL "" FORCE)
set(BUILD_ORM OFF CACHE BOOL "" FORCE)
set(BUILD_POSTGRESQL OFF CACHE BOOL "" FORCE)
set(BUILD_MYSQL OFF CACHE BOOL "" FORCE)
set(BUILD_SQLITE OFF CACHE BOOL "" FORCE)
set(BUILD_REDIS OFF CACHE BOOL "" FORCE)
set(BUILD_BROTLI OFF CACHE BOOL "" FORCE)
set(BUILD_YAML_CONFIG OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    drogon
    GIT_REPOSITORY https://github.com/drogonframework/drogon.git
    GIT_TAG v1.9.8
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(drogon)

# Download nlohmann/json (header-only)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/uuid
)

# Source files
set(SOURCES
    src/main.cpp
    src/face_database.cpp
    src/face_service.cpp
)

# Find MySQL
find_package(PkgConfig REQUIRED)
pkg_check_modules(MYSQL REQUIRED mysqlclient)
include_directories(${MYSQL_INCLUDE_DIRS})

# Executable
add_executable(face_api_server ${SOURCES})

target_link_libraries(face_api_server PRIVATE
    ${OpenCV_LIBS}
    drogon
    nlohmann_json::nlohmann_json
    ${MYSQL_LIBRARIES}
)

# Copy models directory to build
add_custom_command(TARGET face_api_server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/models $<TARGET_FILE_DIR:face_api_server>/models
    COMMENT "Copying models to build directory"
)

# Create data directory in build
add_custom_command(TARGET face_api_server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    $<TARGET_FILE_DIR:face_api_server>/data
    COMMENT "Creating data directory"
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
