#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api.h"
#include "sfpi.h"
#include "debug/dprint.h"

namespace sfpi {
template< int ITERATIONS = 8 >
sfpi_inline void compute() {
    for (int i = 0; i < ITERATIONS; i++) {
        vFloat in = dst_reg[i];
        vFloat a = in + 1.0f;
        vFloat out = a;
        dst_reg[i] = out;
    }
}
}

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    DPRINT_MATH(DPRINT << "n_tiles: " << n_tiles << ENDL());

    // We are going to read from these two circular buffers
    constexpr auto cb_in0 = tt::CB::c_in0;
    // and write to the output circular buffer
    constexpr auto cb_out0 =  tt::CB::c_out0;

    init_sfpu(cb_in0);

    // Loop over all the tiles and perform the computation
    for(uint32_t i = 0; i < n_tiles; i++) {
        // Make sure there is a valid register we can use.
        // acquire_dst();
        // Wait until there is a tile in both input circular buffers
        cb_wait_front(cb_in0, 1);
        // Add the tiles from the input circular buffers and write the result to the destination register
        // add_tiles(cb_in0, cb_in1, 0, 0, dst_reg);
        copy_tile(cb_in0, 0, 0);
        tile_regs_acquire();

        MATH((sfpi::compute<16>()));

        tile_regs_commit();

        cb_pop_front(cb_in0, 1);

        tile_regs_wait();

        // Make sure there is space in the output circular buffer
        cb_reserve_back(cb_out0, 1);
        pack_tile(0, cb_out0);
        // Mark the output tile as ready and pop the input tiles
        cb_push_back(cb_out0, 1);
        // Release the held register
        // release_dst();
        tile_regs_release();
    }
}
}
