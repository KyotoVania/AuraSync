# CMake generated Testfile for 
# Source directory: C:/Users/jeanc/Documents/GitHub/AuraSync/CppSrc
# Build directory: C:/Users/jeanc/Documents/GitHub/AuraSync/CppSrc/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(ave_tests "C:/Users/jeanc/Documents/GitHub/AuraSync/CppSrc/build/bin/Debug/ave_tests.exe")
  set_tests_properties(ave_tests PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/jeanc/Documents/GitHub/AuraSync/CppSrc/CMakeLists.txt;97;add_test;C:/Users/jeanc/Documents/GitHub/AuraSync/CppSrc/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(ave_tests "C:/Users/jeanc/Documents/GitHub/AuraSync/CppSrc/build/bin/Release/ave_tests.exe")
  set_tests_properties(ave_tests PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/jeanc/Documents/GitHub/AuraSync/CppSrc/CMakeLists.txt;97;add_test;C:/Users/jeanc/Documents/GitHub/AuraSync/CppSrc/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(ave_tests "C:/Users/jeanc/Documents/GitHub/AuraSync/CppSrc/build/bin/MinSizeRel/ave_tests.exe")
  set_tests_properties(ave_tests PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/jeanc/Documents/GitHub/AuraSync/CppSrc/CMakeLists.txt;97;add_test;C:/Users/jeanc/Documents/GitHub/AuraSync/CppSrc/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(ave_tests "C:/Users/jeanc/Documents/GitHub/AuraSync/CppSrc/build/bin/RelWithDebInfo/ave_tests.exe")
  set_tests_properties(ave_tests PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/jeanc/Documents/GitHub/AuraSync/CppSrc/CMakeLists.txt;97;add_test;C:/Users/jeanc/Documents/GitHub/AuraSync/CppSrc/CMakeLists.txt;0;")
else()
  add_test(ave_tests NOT_AVAILABLE)
endif()
subdirs("_deps/json-build")
