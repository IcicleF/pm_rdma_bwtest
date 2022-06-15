#include <cstdlib>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>

#include "rlibv2/lib.hh"
#include "common.h"

using namespace rdmaio;
using namespace rdmaio::rmem;

static const size_t PageSize = 2ul << 20;
static char const *PmDev = "/dev/dax0.0";

int main(int argc, char **argv)
{
    RCtrl ctrl(PORT);

    auto nic = RNic::create(RNicInfo::query_dev_names().at(UseNixIdx)).value();
    ctrl.opened_nics.reg(RegNicName, nic);

    int fd = -1;
    auto pm_alloc_fn = [&fd](u64 size) -> RMem::raw_ptr_t { 
        int fd = open(PmDev, O_RDWR);
        if (fd < 0) {
            fprintf(stderr, "cannot open %s: %s\n", PmDev, strerror(errno));
            exit(-1);
        }
        void *buf = nullptr;
        u64 sz = ((size - 1) / PageSize + 1) * PageSize;
        
        buf = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (buf == (void *)-1) {
            fprintf(stderr, "cannot mmap: %s\n", strerror(errno));
            exit(-1);
        }
        return reinterpret_cast<RMem::raw_ptr_t>(buf);
    };
    auto pm_dealloc_fn = [&fd](RMem::raw_ptr_t ptr, u64 size) {
        if (ptr != 0)
            munmap((void *)ptr, size);
        close(fd);
    };
    ctrl.registered_mrs.create_then_reg(
        RegMemName, Arc<RMem>(new RMem(ServerMemSize, pm_alloc_fn, pm_dealloc_fn)),
        ctrl.opened_nics.query(RegNicName).value()
    );

    u64 *reg_mem = (u64 *)(ctrl.registered_mrs.query(RegMemName).value()->get_reg_attr().value().buf);
    // memset(reg_mem, 0, MemSize);
    
    ctrl.start_daemon();
    printf("server started.\n");

    while (true) {
        printf("press any key to terminate ...\n");
        getchar();

        printf("press Enter to confirm ->");
        fflush(stdout);
        char c = getchar();
        if (c == '\n') {
            printf("terminate.\n");
            break;
        }
        printf("\n");
    } 

    return 0;
}
