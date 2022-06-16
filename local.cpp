#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <thread>
#include <atomic>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>

#include "persist.h"

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

static char const *PmDev = "/dev/dax0.0";
static const size_t PageSize = 2ul << 20;
static const u64 MemSize = 128ul << 30;
static const int MaxNThreads = 8;
static int NThreads = 1;
static u32 Granularity = 64;

static const int NTests = 10000000;

u64 thpt[MaxNThreads] = {0};

void parse_inargs(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Test PM I/O bandwidth\n");
        fprintf(stderr, "Usage: %s <NThreads> <Granularity (in bytes)>\n", argv[0]);
        exit(-1);
    }
    NThreads = std::atoi(argv[1]);

    size_t l = strlen(argv[2]);
    Granularity = 1;
    if (argv[2][l - 1] == 'k' || argv[2][l - 1] == 'K') {
        Granularity = 1u << 10;
        argv[2][l - 1] = '\0';
    }
    else if (argv[2][l - 1] == 'm' || argv[2][l - 1] == 'M') {
        Granularity = 1u << 20;
        argv[2][l - 1] = '\0';
    }
    Granularity *= static_cast<u32>(std::atoi(argv[2]));
}

inline void bind_core(uint16_t core)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

std::atomic_int barrier = 0;

void worker(int id, u8 *pm)
{
    bind_core(id);
    u8 *local = new u8[Granularity];

    // Barrier
    barrier.fetch_add(1);
    while (barrier.load() != NThreads + 1);

    const size_t Units = (MemSize / NThreads) / Granularity;
    const size_t Base = Units * Granularity * id;
    for (int i = 0; i < NTests; ++i) {
        memmove_movnt_avx512f_clwb((char *)(pm + Base + (i % Units) * Granularity), (char *)local, Granularity);
        thpt[id]++;
    }

    barrier.fetch_sub(1);
    delete[] local;
}

int main(int argc, char **argv)
{
    parse_inargs(argc, argv);
    bind_core(MaxNThreads);

    int fd = open("/dev/dax0.0", O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "cannot open %s: %s\n", PmDev, strerror(errno));
        exit(-1);
    }
    void *pmbuf = mmap(nullptr, MemSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (pmbuf == (void *)-1) {
        fprintf(stderr, "cannot mmap: %s\n", strerror(errno));
        exit(-1);
    }

    std::thread workers[MaxNThreads];
    for (int i = 0; i < NThreads; ++i)
        workers[i] = std::thread(worker, i, (u8 *)pmbuf);

    barrier.fetch_add(1);
    while (barrier.load() != NThreads + 1);

    auto start_time = std::chrono::steady_clock::now();

    u64 recent = 0;
    for (int i = 0; barrier.load(std::memory_order_relaxed) > 1; ++i) {
        while (true) {
            auto end_time = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count() == 1) {
                start_time = end_time;
                break;
            }
        }
        if (barrier.load(std::memory_order_relaxed) == 1)
            break;

        u64 tot = 0;
        for (int j = 0; j < NThreads; ++j)
            tot += thpt[j];
        u64 delta = tot - recent;
        recent = tot;

        double thpt_in_gb = (delta * Granularity) / 1e9;
        printf("%.3lf GB/s\n", thpt_in_gb);
    }

    for (int i = 0; i < NThreads; ++i)
        workers[i].join();

    return 0;
}
