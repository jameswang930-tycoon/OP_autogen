// Host: ACL runtime 启动 add_custom kernel
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>

extern "C" void add_custom(float* x, float* y, float* z, uint32_t totalLen);

int main()
{
    constexpr uint32_t N = 4096;
    constexpr size_t byteSize = N * sizeof(float);

    aclInit(nullptr);
    aclrtContext context;
    aclrtCreateContext(&context, 0);

    float *xHost, *yHost, *zHost;
    aclrtMallocHost((void**)&xHost, byteSize);
    aclrtMallocHost((void**)&yHost, byteSize);
    aclrtMallocHost((void**)&zHost, byteSize);

    float *xDev, *yDev, *zDev;
    aclrtMalloc((void**)&xDev, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&yDev, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&zDev, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    for (uint32_t i = 0; i < N; i++) {
        xHost[i] = (float)i;
        yHost[i] = (float)(i * 2);
    }
    aclrtMemcpy(xDev, byteSize, xHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDev, byteSize, yHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    add_custom(xDev, yDev, zDev, N);
    aclrtSynchronizeDevice();

    aclrtMemcpy(zHost, byteSize, zDev, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    int errors = 0;
    for (uint32_t i = 0; i < N && errors < 5; i++) {
        if (zHost[i] != xHost[i] + yHost[i]) {
            printf("MISMATCH [%u]\n", i);
            errors++;
        }
    }
    printf("VectorAdd: %s (%u elements, %d errors)\n", errors ? "FAIL" : "PASS", N, errors);

    aclrtFree(xDev); aclrtFree(yDev); aclrtFree(zDev);
    aclrtFreeHost(xHost); aclrtFreeHost(yHost); aclrtFreeHost(zHost);
    aclrtDestroyContext(context);
    aclFinalize();
    return errors;
}
