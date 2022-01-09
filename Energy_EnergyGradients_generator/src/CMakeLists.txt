#
# This script is based on the example used to create static executables
# linking to the static OpenBabel2 library.
#

set(CMAKE_LEGACY_CYGWIN_WIN32 0)  # Hide Cygwin error

# This line is required for cmake backwards compatibility.
cmake_minimum_required(VERSION 2.6)

# Name of your project
project(energyhist)

# Use a loop like openbabel/tools/CMakeLists.txt
set(tools
        energyhist
        hi
   )

################################################################################

set(LOCAL_OB_DATADIR "${CMAKE_SOURCE_DIR}/ob_datadir"
    CACHE PATH "Install dir for OB data customized for MOFs")
# Set up include file for the data directory
configure_file(${CMAKE_SOURCE_DIR}/config_ob_data.h.cmake
  ${CMAKE_BINARY_DIR}/includes/config_ob_data.h)
include_directories(${CMAKE_BINARY_DIR}/includes/)


# Set the path containing OpenBabel2Config.cmake, needed for find_package below.
# There's alternate documentation, too, at http://openbabel.org/dev-api/cmake_project.shtml
# Earlier, this was implemented as a cmake variable, but this path should work more universally
find_path(OpenBabel2_DIR OpenBabel2Config.cmake PATHS
          ${CMAKE_SOURCE_DIR}/../openbabel/installed/lib/cmake/openbabel2/
          )

#
# Find and setup OpenBabel2.
#
find_package(OpenBabel2 REQUIRED)
include_directories(${OpenBabel2_INCLUDE_DIRS})

# Dependencies
#find_package(LibXml2)

# Set compiler flags similar to Open Babel's CMAKE_CXX_FLAGS_RELWITHDEBINFO
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -DNDEBUG")

foreach(tool ${tools})
  # The executable
  add_executable(${tool} ${tool}.cpp supercell.cpp)
  # Link against imported openbabel target
  #target_link_libraries(${executable_target} openbabel ${LIBXML2_LIBRARIES})
  target_link_libraries(${tool} openbabel m)
  # install(TARGETS ${executable_target} DESTINATION bin)
endforeach(tool)
