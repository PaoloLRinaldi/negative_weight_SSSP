#pragma once

#include <iostream>

// helper macro

#define NOP do { } while (0)

// printing macros
#define NVERBOSE

#if defined(NDEBUG)
#define DEBUG(x) NOP
#else
#define DEBUG(x) do { std::cout << x << std::endl; } while (0)
#endif

#if defined(NVERBOSE)
#define PRINT(x) NOP
#else
#define PRINT(x) do { std::cout << x << std::endl; } while (0)
#endif

#define ERROR(x) do { std::cerr << "Error: " << x << std::endl;\
                      std::exit(EXIT_FAILURE); } while (0)
