cmake_minimum_required(VERSION 3.15...3.22)
project(vdb LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Python3 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenVDB REQUIRED)

pybind11_add_module(vdb cpp/vdb.cpp)

target_link_libraries(vdb PRIVATE OpenVDB::openvdb)

set(INSTALL_DIR ${PROJECT_SOURCE_DIR}/src/imvdb)

include(GNUInstallDirs)
install(TARGETS vdb DESTINATION ${INSTALL_DIR})

if (WIN32)
    file(GLOB rts ${CMAKE_BINARY_DIR}/*.dll)
    file(GLOB rt_python ${CMAKE_BINARY_DIR}/python*.dll)
    list(REMOVE_ITEM rts ${rt_python})
    install(FILES ${rts} DESTINATION ${INSTALL_DIR})
else ()
    message(TODO)
endif ()