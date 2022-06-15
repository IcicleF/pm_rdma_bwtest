CXXFLAGS += -g -O2 -std=c++17
CXXFLAGS += -Wall -Wno-reorder -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-label -Werror
LIBS = -libverbs -lrdmacm -lpthread

RLIB_DIRS = $(shell find rlibv2 -maxdepth 3 -type d)
RLIB_FILES = $(foreach dir,$(DIRS),$(wildcard $(dir)/*.hh))

.PHONY: all clean
all: server client

server: server.cpp $(RLIB_FILES)
	g++ $(CXXFLAGS) -o $@ $< $(LIBS)

client: client.cpp $(RLIB_FILES)
	g++ $(CXXFLAGS) -o $@ $< $(LIBS)

clean:
	$(RM) ./server ./client