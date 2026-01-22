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

pybind11_add_module(${SKBUILD_PROJECT_NAME} main.cpp)

if(MSVC)
    target_compile_options(nndescent_m PRIVATE /bigobj)
else()
    target_compile_options(nndescent_m PRIVATE -O3 -march=native)
endif()

install(TARGETS ${SKBUILD_PROJECT_NAME} DESTINATION .)
install(FILES nndescent_m.pyi DESTINATION .)
