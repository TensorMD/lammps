
# we use python 3's subprocess module to run the tools and check the output
if(CMAKE_VERSION VERSION_LESS 3.12)
  find_package(PythonInterp 3.5) # Deprecated since version 3.12
  if(PYTHONINTERP_FOUND)
    set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
  endif()
else()
  find_package(Python 3.5 COMPONENTS Interpreter)
endif()

if(Python_EXECUTABLE)
  if(BUILD_LAMMPS_SHELL)
    add_test(NAME LammpsShell
      COMMAND ${Python_EXECUTABLE} -u ${CMAKE_CURRENT_SOURCE_DIR}/test_lammps_shell.py -v
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endif()
else()
  message(STATUS "Skipping Tests for LAMMPS tools: no suitable Python interpreter")
endif()
