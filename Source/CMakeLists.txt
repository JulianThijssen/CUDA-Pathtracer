set(DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(BASE
    ${DIR}/Application.cpp
    ${DIR}/Renderer.h
    ${DIR}/CudaRenderer.h
    ${DIR}/CudaRenderer.cpp
    ${DIR}/Log.cpp
    ${DIR}/Log.h
    ${DIR}/Maths.h
    ${DIR}/Mesh.h
    ${DIR}/Size.h
    ${DIR}/ModelLoader.h
    ${DIR}/Ray.h
    ${DIR}/Window.cpp
    ${DIR}/Window.h
    ${DIR}/glad.c
    ${DIR}/Material.h
    ${DIR}/Vector3f.h
    ${DIR}/BRDF.h
    ${DIR}/Scene.h
    ${DIR}/Scene.cpp
    ${DIR}/Camera.h
    ${DIR}/Camera.cpp
    ${DIR}/kernel.cu
    ${DIR}/kernel.cuh
    PARENT_SCOPE
)
