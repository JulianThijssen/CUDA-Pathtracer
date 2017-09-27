set(DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(BASE
    ${DIR}/Application.cpp
    ${DIR}/Log.cpp
    ${DIR}/Log.h
    ${DIR}/Material.h
    ${DIR}/Mesh.h
    ${DIR}/ModelLoader.h
    ${DIR}/Ray.h
    ${DIR}/Scene.cpp
    ${DIR}/Scene.h
    ${DIR}/Vector3f.h
    ${DIR}/Window.cpp
    ${DIR}/Window.h
    ${DIR}/kernel.cu
    ${DIR}/kernel.cuh
    ${DIR}/glad.c
    PARENT_SCOPE
)

list(APPEND ENGINE_SOURCES
    ${BASE}
)
