cmake_minimum_required(VERSION 3.5)

project(
  NegativeWeightShortestPath
  VERSION 0.1
  LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 20)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	if(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
		add_compile_options(-fsanitize=integer -fsanitize=implicit-integer-truncation)
		add_link_options(-fsanitize=integer -fsanitize=implicit-integer-truncation)
		# add_compile_options(-fsanitize=undefined -fsanitize=address -fno-omit-frame-pointer)
		# add_link_options(-fsanitize=undefined -fsanitize=address -fno-omit-frame-pointer)
	endif()
	if(CMAKE_BUILD_TYPE MATCHES "Debug")
		add_compile_options(-fsanitize=undefined -fsanitize=address -fno-omit-frame-pointer -fsanitize=integer -fsanitize=implicit-integer-truncation)
		add_link_options(-fsanitize=undefined -fsanitize=address -fno-omit-frame-pointer -fsanitize=integer -fsanitize=implicit-integer-truncation)
	endif()
	
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	if(CMAKE_BUILD_TYPE MATCHES "Debug")
		add_compile_options(-fsanitize=undefined -fsanitize=address -fno-omit-frame-pointer)
		add_link_options(-fsanitize=undefined -fsanitize=address -fno-omit-frame-pointer)
	endif()
endif()

add_library(Library
	src/algorithms.h
	src/algorithms.cpp
	src/graph.h
	src/graph.cpp
	src/queries.h
	src/queries.cpp
	src/bcf.h
	src/bcf.cpp
	src/permutations.h
	src/measurement_tool.h
	src/measurements.h
	src/heap.h
	src/create_graph.cpp
	src/gor.h
	src/gor.cpp
	src/config.cpp
	src/config.h
)

add_executable(Main src/main.cpp)
add_executable(UnitTests src/unit_tests.cpp)
add_executable(RuntimeTests src/runtime_tests.cpp)
add_executable(CreateGraph src/create_graph.cpp)

target_link_libraries(Main PRIVATE Library)
target_link_libraries(UnitTests PRIVATE Library)
target_link_libraries(RuntimeTests PRIVATE Library)
target_link_libraries(CreateGraph PRIVATE Library)

enable_testing()
add_test(NAME UnitTests COMMAND UnitTests)
add_test(NAME RuntimeTests COMMAND RuntimeTests)
