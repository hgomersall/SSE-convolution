# Update the module path to include any extra CMAKE modules we might ship.
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(SSE_CONV_TEST)

# Set the minimum required version of cmake
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# Enable warnings and make them errors
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")

# Magic to set GCC-specific compile flags (to turn on optimisation).
set(GCC_FLAGS "-std=c99 -Wall -O3 -msse3 -mtune=core2")
add_definitions( -DSSE3 )

if(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCC_FLAGS}")
endif(CMAKE_COMPILER_IS_GNUCC)
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_FLAGS}")
endif(CMAKE_COMPILER_IS_GNUCXX)

# On x86_64 we need to compile with -fPIC
if(UNIX AND NOT WIN32)
  find_program(CMAKE_UNAME uname /bin /usr/bin /usr/local/bin )
  if(CMAKE_UNAME)
    exec_program(uname ARGS -m OUTPUT_VARIABLE CMAKE_SYSTEM_PROCESSOR)
    set(CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR} 
    	CACHE INTERNAL "processor type (i386 and x86_64)")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
      add_definitions(-fPIC)
    endif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  endif(CMAKE_UNAME)
endif(UNIX AND NOT WIN32)

# Find the pkg-config support macros
find_package(PkgConfig)

# Attempt to locate and use GLib.
pkg_check_modules(GLIB REQUIRED glib-2.0)
include_directories(${GLIB_INCLUDE_DIRS})
link_directories(${GLIB_LIBRARY_DIRS})
set(CMAKE_CFLAGS "${CMAKE_CFLAGS} ${GLIB_CFLAGS}")
set(CMAKE_CXXFLAGS "${CMAKE_CXXFLAGS} ${GLIB_CFLAGS}")

add_library(convolve_funcs SHARED convolve.h convolve.c)

set(_test_convolve_sources
    test_data.h          test_data.c
    test_convolve.c
)

add_executable(test_convolve ${_test_convolve_sources})
target_link_libraries(test_convolve
    convolve_funcs
    ${GLIB_LIBRARIES})

