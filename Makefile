nproc=$(shell python3 -c 'import multiprocessing; print( max(multiprocessing.cpu_count() - 1, 1))')

CXX := clang++-10
CC := clang-10
TORCH_PATH := /opt/ILLIXR/libtorch

.PHONY: plugin.dbg.so
plugin.dbg.so: build/Debug/Makefile
	make -C build/Debug "-j$(nproc)" && \
	rm -f $@ && \
	ln -s build/Debug/libplugin.so plugin.dbg.so && \
	true

.PHONY: plugin.opt.so
plugin.opt.so: build/RelWithDebInfo/Makefile
	make -C build/RelWithDebInfo "-j$(nproc)" && \
	rm -f $@ && \
	ln -s build/RelWithDebInfo/libplugin.so plugin.opt.so && \
	true

build/Debug/Makefile:
	mkdir -p build/Debug && \
	cd build/Debug && \
	cmake -DCMAKE_BUILD_TYPE=Debug  -DCMAKE_CXX_COMPILER=$(CXX) -DCMAKE_C_COMPILER=$(CC) -DCMAKE_PREFIX_PATH=$(TORCH_PATH) ../.. && \
	true

build/RelWithDebInfo/Makefile:
	mkdir -p build/RelWithDebInfo && \
	cd build/RelWithDebInfo && \
	cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=$(CXX) -DCMAKE_C_COMPILER=$(CC) -DCMAKE_PREFIX_PATH=$(TORCH_PATH) ../.. && \
	true

tests/run:
tests/gdb:

.PHONY: clean
clean:
	touch build && rm -rf build *.so
