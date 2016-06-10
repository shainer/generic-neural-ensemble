project(neural)
cmake_minimum_required(VERSION 2.6)

find_package(Qt4 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${KDE4_ENABLE_EXCEPTIONS}")

# Compile with debug symbols if gcc is employed
if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_C_FLAGS_DEBUG "-ggdb")
endif(CMAKE_COMPILER_IS_GNUCC)

include_directories( ${QT_INCLUDES}
                     ${CMAKE_CURRENT_SOURCE_DIR} )

set(neural_SRCS
    main.cpp
    Neuron.cpp
    Network.cpp
    Link.cpp
    Ensemble.cpp
    ProblemInfo.cpp
    Utils.cpp
    LinkMatrix.cpp
) 

add_executable(neural ${neural_SRCS})
target_link_libraries(neural ${QT_QTCORE_LIBRARY} m)
