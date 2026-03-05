# Rutas base
VCPKG_DIR = C:\vcpkg\packages\libpng_x64-windows
ZLIB_DIR = C:\vcpkg\packages\zlib_x64-windows

CXX = cl
CC = cl
NVCC = nvcc

# Banderas de compilacion (/MD es necesario para enlazar con vcpkg)
CFLAGS = /nologo /O2 /MD /I. /I"$(VCPKG_DIR)\include" /I"$(ZLIB_DIR)\include"
CXXFLAGS = /nologo /O2 /MD /I. /I"$(VCPKG_DIR)\include" /I"$(ZLIB_DIR)\include"
NVCCFLAGS = -O3 -I. -I"$(VCPKG_DIR)\include" -I"$(ZLIB_DIR)\include" -Xcompiler "/MD"

# Rutas a librerias
CUDA_LIB_PATH = /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\lib\x64"
VCPKG_LIB_PATH = /LIBPATH:"$(VCPKG_DIR)\lib" /LIBPATH:"$(ZLIB_DIR)\lib"

# Librerias necesarias
LIBS = cudart.lib cuda.lib libpng16.lib zlib.lib

OBJS = png_io.obj routinesCPU.obj routinesGPU.obj main.obj

%.obj: %.c
	$(CC) $(CFLAGS) /c $< /Fo$@

%.obj: %.cpp
	$(CXX) $(CXXFLAGS) /c $< /Fo$@

%.obj: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

image.exe: $(OBJS)
	$(CXX) /nologo $(OBJS) /Fe:image.exe /link $(CUDA_LIB_PATH) $(VCPKG_LIB_PATH) $(LIBS)

clean:
	del *.obj image.exe *.exp *.lib *.o