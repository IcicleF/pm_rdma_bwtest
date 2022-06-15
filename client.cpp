#include <cstdlib>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <thread>
#include <chrono>

#include "rlibv2/lib.hh"
#include "common.h"

using namespace rdmaio;
using namespace rdmaio::rmem;
using namespace rdmaio::qp;

static const int QpTimeout = 2;
static const size_t LocalMemSize = 1ul << 30;

static const int MaxNThreads = 8;
static int NThreads = 1;
static u32 Granularity = 64;

static const int NTests = 10000000;
static const int Batch = 8;
static const auto IOMode = IBV_WR_RDMA_READ;

Arc<RC> qps[MaxNThreads];
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

void worker(int id, u8 *buf)
{
    bind_core(id);

    // Barrier
    barrier.fetch_add(1);
    while (barrier.load() != NThreads + 1);

    const size_t Units = (ServerMemSize / NThreads) / Granularity;
    const size_t Base = Units * Granularity * id;
    for (int i = 0; i < NTests + Batch; ++i) {
        if (i < NTests) {
            qps[id]->send_normal(
                {
                    .op = IOMode,
                    .flags = (i + 1) % Batch == 0 ? IBV_SEND_SIGNALED : 0,
                    .len = Granularity,
                    .wr_id = 0
                },
                {
                    .local_addr = reinterpret_cast<RMem::raw_ptr_t>(buf + (i % (Batch * 2)) * Granularity),
                    .remote_addr = Base + (i % Units) * Granularity,
                    .imm_data = 0
                }
            );
        }
        if (i >= Batch && (i + 1) % Batch == 0) {
            qps[id]->wait_one_comp();
            thpt[id] += Batch;
        }
    }

    barrier.fetch_sub(1);
}

int main(int argc, char **argv)
{
    parse_inargs(argc, argv);
    bind_core(MaxNThreads);

    auto nic = RNic::create(RNicInfo::query_dev_names().at(UseNixIdx)).value();
    for (int i = 0; i < NThreads; ++i)
        qps[i] = RC::create(nic, QPConfig().set_timeout(QpTimeout)).value();

    ConnectManager cm(ServerAddr);
    if (cm.wait_ready(1000000, 2) == IOCode::Timeout) {
        fprintf(stderr, "connection timed out.\n");
        exit(-1);
    }

    auto aligned_alloc_fn = [](u64 size) -> RMem::raw_ptr_t {
        void *buf = nullptr;
        int rc = posix_memalign(&buf, 4096, size);
        if (rc)
            return 0;
        return reinterpret_cast<RMem::raw_ptr_t>(buf);
    };
    auto local_mem = Arc<RMem>(new RMem(LocalMemSize * NThreads, aligned_alloc_fn));
    auto local_mr = RegHandler::create(local_mem, nic).value();
    u8 *local_buf = (u8 *)(local_mem->raw_ptr);
    
    auto fetch_res = cm.fetch_remote_mr(RegMemName);
    rmem::RegAttr remote_attr = std::get<1>(fetch_res.desc);

    for (int i = 0; i < NThreads; ++i) {
        cm.cc_rc("client-qp" + std::to_string(i), qps[i], RegNicName, QPConfig().set_timeout(QpTimeout));

        qps[i]->bind_remote_mr(remote_attr);
        qps[i]->bind_local_mr(local_mr->get_reg_attr().value());
    }

    std::thread workers[MaxNThreads];
    for (int i = 0; i < NThreads; ++i)
        workers[i] = std::thread(worker, i, local_buf + i * LocalMemSize);

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
