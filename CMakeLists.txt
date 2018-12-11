cmake_minimum_required(VERSION 3.0.0)

project(NBodyGLCL VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set (CMAKE_CXX_STANDARD 11)

find_package(Qt5 REQUIRED Core Gui)
find_package(OpenCL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Boost REQUIRED)

set(NBodyGLCL_SRC
    main.cpp
)

set(NBodyGLCL_HEADERS
)

include_directories(${Boost_INCLUDE_DIRS})
link_directories (${Boost_LIBRARY_DIRS})

qt5_generate_moc(main.cpp main.moc)

add_executable(NBodyGLCL ${NBodyGLCL_SRC} ${NBodyGLCL_HEADERS} main.moc)

target_link_libraries(NBodyGLCL ${Boost_LIBRARIES})
target_link_libraries(NBodyGLCL Qt5::Core)
target_link_libraries(NBodyGLCL Qt5::Gui)
target_link_libraries(NBodyGLCL OpenCL::OpenCL)
target_link_libraries(NBodyGLCL opengl32)
