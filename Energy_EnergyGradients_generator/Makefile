.PHONY: all test init

all:
	@echo "all target not implemented"

grid: src/energyhist.cpp
	cd bin && make energyhist

test: bin/energyhist
	echo "TODO"

init:
	cd openbabel; \
	mkdir build installed; \
	cd build; \
	cmake -DCMAKE_INSTALL_PREFIX=../installed -DBUILD_GUI=OFF ..; \
	make -j2; \
	make install; \
	cd ../../; \
	mkdir bin; \
	cd bin; \
	cmake ../src/; \
	make
	# Sets up all the cmake details, so that usage is as simple as
	# `bin/executable_name MOF.cif` and re-compilation is as easy as `make bin/executable_name`

eclipse:
	cd bin; \
	cmake -G "Eclipse CDT4 - Unix Makefiles" ../src; \
