CXXFLAGS += -g -O2 -std=c++17
CXXFLAGS += -mavx -mavx2 -mavx512f -mavx512pf -mavx512er -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mavx512ifma -mavx512vbmi
CXXFLAGS += -Wall -Wno-reorder -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-label -Werror
CXXFLAGS += -Wno-psabi
LIBS = -libverbs -lrdmacm -lpthread

RLIB_DIRS = $(shell find rlibv2 -maxdepth 3 -type d)
RLIB_FILES = $(foreach dir,$(DIRS),$(wildcard $(dir)/*.hh))

.PHONY: all clean
all: server client local

server: server.cpp $(RLIB_FILES)
	g++ $(CXXFLAGS) -o $@ $< $(LIBS)

client: client.cpp $(RLIB_FILES)
	g++ $(CXXFLAGS) -o $@ $< $(LIBS)

local: local.cpp $(RLIB_FILES)
	g++ $(CXXFLAGS) -o $@ $< -lpthread

clean:
	$(RM) ./server ./client ./local