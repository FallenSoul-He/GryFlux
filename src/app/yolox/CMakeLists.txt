add_library(app_includes INTERFACE)

find_package(OpenCV REQUIRED)

target_include_directories(app_includes INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/source
    ${CMAKE_CURRENT_SOURCE_DIR}/sink
    ${CMAKE_CURRENT_SOURCE_DIR}/tasks
    ${CMAKE_CURRENT_SOURCE_DIR}/package
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
)

# Ascend ACL library configuration
set(ASCEND_TOOLKIT_PATH /usr/local/Ascend/ascend-toolkit/latest)
set(ASCEND_INCLUDE_DIR ${ASCEND_TOOLKIT_PATH}/runtime/include)
set(ASCEND_LIB_DIR ${ASCEND_TOOLKIT_PATH}/runtime/lib64/stub)

# Find ACL libraries
find_library(ACL_LIB ascendcl PATHS ${ASCEND_LIB_DIR} NO_DEFAULT_PATH)

if(ACL_LIB)
    message(STATUS "Found Ascend ACL library: ${ACL_LIB}")
    set(ascend_libs ${ACL_LIB})
else()
    message(WARNING "Ascend ACL library not found at ${ASCEND_LIB_DIR}")
    set(ascend_libs "")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/package ${CMAKE_CURRENT_SOURCE_DIR}/runtime ${OpenCV_INCLUDE_DIRS} ${ASCEND_INCLUDE_DIR})

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/sink/write_consumer APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/source/producer APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/tasks/image_preprocess APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/runtime APP_SRC)

if(ascend_libs)
    aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/tasks/ascend_runner APP_SRC)
    message(STATUS "Including Ascend runner")
endif()

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/tasks/object_detector APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/tasks/res_sender APP_SRC)

add_executable(yolox_stream yolox_stream.cpp ${SRC_DIR} ${APP_SRC})

target_link_libraries(yolox_stream ${app_includes} ${dynamic_libs} ${OpenCV_LIBS} ${rknn_lib} ${ascend_libs})
install(TARGETS yolox_stream RUNTIME DESTINATION ./)
