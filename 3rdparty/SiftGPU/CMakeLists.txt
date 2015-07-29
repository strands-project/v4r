include(ExternalProject)

ExternalProject_Add(SiftGPU
  URL "http://wwwx.cs.unc.edu/~ccwu/cgi-bin/siftgpu.cgi?SiftGPU-V400.zip"
  URL_HASH SHA1=eac08305f14cb35d12a22d493f84f9c21f6b7fdb
  TIMEOUT 30
  CONFIGURE_COMMAND ""
  BUILD_COMMAND make siftgpu_enable_cuda=0
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND ""
)
