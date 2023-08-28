OS = $(shell uname -s)

ifeq (${OS}, Darwin)
  # OS X
	CLANG ?= clang++
	LIB = libflowloss.dylib
	CFLAGS = -O3 -std=c++11 -fPIC
	SHARED_FLAG = -dynamiclib
else ifeq (${OS}, Linux)
  # Linux
 	CLANG ?= clang++
	LIB = libflowloss.so
	CFLAGS = -O3 -std=c++11 -fPIC -fopenmp
	SHARED_FLAG = -shared
else
  $(error Unsupported platform: ${OS})
endif

${LIB}: main.cc
	$(CLANG) $(CFLAGS) -c main.cc -o main.o
	$(CLANG) -fopenmp $(SHARED_FLAG) -o ${LIB} main.o
	#$(CLANG) $(SHARED_FLAG) -o ${LIB} main.o

clean:
	rm -f *.bc *.o *.out ${LIB}
