cmake_minimum_required(VERSION 3.30)

project(
        ${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

find_package(OpenMP REQUIRED)

pybind11_add_module(${SKBUILD_PROJECT_NAME} main.cpp)

if(OPENMP_FOUND)
    target_link_libraries(${SKBUILD_PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

if(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_options(${SKBUILD_PROJECT_NAME} PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-march=native -ffast-math>
        $<$<CXX_COMPILER_ID:Clang>:-march=native -ffast-math>
        $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
    )
endif()

install(TARGETS ${SKBUILD_PROJECT_NAME} DESTINATION .)
install(FILES nndescent_m.pyi DESTINATION .)
