
#include <cstdint>

void kernel_main()
{
    // Read parameters from the kernel arguments
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_tile = get_arg_val<uint32_t>(2);

    // The circular buffers to read the tiles into
    constexpr uint32_t cb_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_in1 = tt::CB::c_in1;

    // Get the tile size used in the circular buffers. We assume the
    // circular buffers are created with the same tile size as the DRAM
    // buffers. (Whis is most of the cases)
    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    // Create address generators for the input buffers. This is much faster
    // then doing plain DRAM reads.
    // Setting the page size to be tile_size_bytes works because we set it up
    // explicitly in host code. This is usually a good idea as it makes coding
    // easy. But may not be the most efficient way to do it in all cases.
    const InterleavedAddrGenFast<true> a = {
        .bank_base_address = a_addr,          // The base address of the buffer
        .page_size = tile_size_bytes,         // The size of a buffer page
        .data_format = DataFormat::Float16_b, // The data format of the buffer
    };

    // Now we loop over all the tiles and read them into the circular buffers
    for(uint32_t i = 0; i < n_tiles; i++) {
        // First we make sure there is space in the circular buffers
        cb_reserve_back(cb_in0, 1);
        uint32_t cb_in0_addr = get_write_ptr(cb_in0);
        noc_async_read_tile(start_tile + i, a, cb_in0_addr); // read the tile into the circular buffer

        // NOTE: Since circular buffers are backed by SRAM, we can actually access them by
        // casting the address to a pointer. This is not helpful in most cases as the CPU
        // is quite slow compared to the tensor/simd engines. But useful for debugging.
        // uint16_t* ptr = (uint16_t*)cb_in0_addr;
        // DPRINT << "cb_in0_addr: " << ptr << " " << *ptr;

        noc_async_read_barrier(); // Wait until tile reads are done
        cb_push_back(cb_in0, 1);
    }
}
