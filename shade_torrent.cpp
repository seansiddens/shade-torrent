#include <iostream>
#include "common/core_coord.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "common/bfloat16.hpp"
#include "common/work_split.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>
#include <chrono>

using namespace tt;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t TILE_HEIGHT = 32;

std::shared_ptr<Buffer> MakeBuffer(Device *device, uint32_t size, uint32_t page_size, bool sram)
{
    InterleavedBufferConfig config{
        .device= device,
        .size = size,
        .page_size = page_size,
        .buffer_type = (sram ? BufferType::L1 : BufferType::DRAM)
    };
    return CreateBuffer(config);
}

// Allocate a buffer on DRAM or SRAM. Assuming the buffer holds BFP16 data.
// A tile on Tenstorrent is 32x32 elements, given us using BFP16, we need 2 bytes per element.
// Making the tile size 32x32x2 = 2048 bytes.
// @param device: The device to allocate the buffer on.
// @param n_tiles: The number of tiles to allocate.
// @param sram: If true, allocate the buffer on SRAM, otherwise allocate it on DRAM.
std::shared_ptr<Buffer> MakeBufferBFP16(Device *device, uint32_t n_tiles, bool sram)
{
    constexpr uint32_t tile_size = sizeof(bfloat16) * TILE_WIDTH * TILE_HEIGHT;
    // For simplicity, all DRAM buffers have page size = tile size.
    const uint32_t page_tiles = sram ? n_tiles : 1;
    return MakeBuffer(device, tile_size * n_tiles, page_tiles * tile_size, sram);
}

CBHandle MakeCircularBuffer(Program& program, const CoreSpec& core, tt::CB cb, uint32_t size, uint32_t page_size, tt::DataFormat format)
{
    CircularBufferConfig cb_src0_config = CircularBufferConfig(
        size,
        {{
            cb,
            format
    }})
    .set_page_size(cb, page_size);
    return CreateCircularBuffer(program, core, cb_src0_config);
}

// Circular buffers are Tenstorrent's way of communicating between the data movement and the compute kernels.
// kernels queue tiles into the circular buffer and takes them when they are ready. The circular buffer is
// backed by SRAM. There can be multiple circular buffers on a single Tensix core.
// @param program: The program to create the circular buffer on.
// @param core: The core to create the circular buffer on.
// @param cb: Which circular buffer to create (c_in0, c_in1, c_out0, c_out1, etc..). This is just an ID
// @param n_tiles: The number of tiles the circular buffer can hold.
CBHandle MakeCircularBufferBFP16(Program& program, const CoreSpec& core, tt::CB cb, uint32_t n_tiles)
{
    constexpr uint32_t tile_size = sizeof(bfloat16) * TILE_WIDTH * TILE_HEIGHT;
    return MakeCircularBuffer(program, core, cb, n_tiles * tile_size, tile_size, tt::DataFormat::Float16_b);
}

std::string next_arg(int& i, int argc, char **argv)
{
    if(i + 1 >= argc) {
        std::cerr << "Expected argument after " << argv[i] << std::endl;
        exit(1);
    }
    return argv[++i];
}

void help(std::string_view program_name)
{
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "This program demonstrates how to add two vectors using tt-Metalium.\n";
    std::cout << "\n";
    std::cout << "Options:\n";
    std::cout << "  --device, -d <device_id>  Specify the device to run the program on. Default is 0.\n";
    std::cout << "  --seed, -s <seed>         Specify the seed for the random number generator. Default is random.\n";
    exit(0);
}

int main(int argc, char **argv)
{
    int seed = std::random_device{}();
    int device_id = 0;

    // Quick and dirty argument parsing.
    for(int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];
        if(arg == "--device" || arg == "-d") {
            device_id = std::stoi(next_arg(i, argc, argv));
        }
        else if(arg == "--seed" || arg == "-s") {
            seed = std::stoi(next_arg(i, argc, argv));
        }
        else if(arg == "--help" || arg == "-h") {
            help(argv[0]);
            return 0;
        }
        else {
            std::cout << "Unknown argument: " << arg << std::endl;
            help(argv[0]);
        }
    }

    Device *device = CreateDevice(device_id);

    Program program = CreateProgram();
    // constexpr CoreCoord core = {0, 0};
    constexpr uint32_t num_cores = 32;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    tt::log_info("num_cores_x: {}, num_cores_y: {}", num_cores_x, num_cores_y);
    auto core_set = num_cores_to_corerange_set({0, 0}, num_cores, {num_cores_x, num_cores_y});
    tt::log_info("core_set: {}", core_set);
    std::cout << "Total cores: " << (*core_set.begin()).size() << std::endl;

    CommandQueue& cq = device->command_queue();
    const uint32_t n_tiles = 1024 * 8;
    auto tiles_per_core = n_tiles / num_cores;
    tt::log_info("n_tiles: {}", n_tiles);
    tt::log_info("tiles per core: {}", tiles_per_core);
    const uint32_t tile_size = TILE_WIDTH * TILE_HEIGHT;

    // Create 3 buffers on DRAM. These will hold the input and output data. A and B are the input buffers, C is the output buffer.
    auto a = MakeBufferBFP16(device, n_tiles, false);
    auto c = MakeBufferBFP16(device, n_tiles, false);

    std::mt19937 rng(seed);
    // std::vector<uint32_t> a_data = create_random_vector_of_bfloat16(tile_size * n_tiles * 2, 10, rng());
    std::vector<uint32_t> a_data = create_constant_vector_of_bfloat16(tile_size * n_tiles * 2, 1.0f);

    const uint32_t tiles_per_cb = 4;
    tt::log_info("tiles_per_cb: {}", tiles_per_cb);
    // Create 3 circular buffers. These will be used by the data movement kernels to stream data into the compute cores and for the compute cores to stream data out.
    CBHandle cb_a = MakeCircularBufferBFP16(program, core_set, tt::CB::c_in0, tiles_per_cb);
    CBHandle cb_c = MakeCircularBufferBFP16(program, core_set, tt::CB::c_out0, tiles_per_cb);

    EnqueueWriteBuffer(cq, a, a_data, false);
    tt::log_info("Wrote input buffer to DRAM");

    // A Tensix core is made up with 5 processors. 2 data movement processors, and 3 compute processors. The 2 data movement
    // processors act independent to other cores. And the 3 compute processors act together (hence 1 kerenl for compute).
    // There is no need to explicitly parallelize the compute kernels. Unlike traditional CPU/GPU style SPMD programming,
    // the 3 compute processors moves data from SRAM into the FPU(tensor engine)/SFPU(SIMD engine), operates on the data, and
    // move it back to SRAM. The data movement processors moves data from the NoC, or in our case, the DRAM, into the SRAM.
    //
    // The vector add example consists of 3 kernels. `interleaved_tile_read` reads tiles from the input buffers A and B
    // into 2 circular buffers. `add` reads tiles from the circular buffers, adds them together, and dumps the result into
    // a third circular buffer. `tile_write` reads tiles from the third circular buffer and writes them to the output buffer C.
    auto reader = CreateKernel(
        program,
        "tt_metal/programming_examples/personal/shade-torrent/kernels/reader.cpp",
        core_set,
        DataMovementConfig {.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    auto writer = CreateKernel(
        program,
        "tt_metal/programming_examples/personal/shade-torrent/kernels/writer.cpp",
        core_set,
        DataMovementConfig {.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );
    auto compute = CreateKernel(
        program,
        "tt_metal/programming_examples/personal/shade-torrent/kernels/compute.cpp",
        core_set,
        ComputeConfig{
            .dst_full_sync_en = true,
            .math_approx_mode = false,
            .compile_args = {},
            .defines = {}
        }
    );

    // Set runtime args for each core.
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Set the runtime arguments for the kernels. This also registers
        // the kernels with the program.
        SetRuntimeArgs(program, reader, core, {
            a->address(),
            tiles_per_core,
            num_tiles_written
        });
        SetRuntimeArgs(program, writer, core, {
            c->address(),
            tiles_per_core,
            num_tiles_written
        });
        SetRuntimeArgs(program, compute, core, {
            tiles_per_core,
        });

        num_tiles_written += tiles_per_core;
    }


    // Add timing variables
    auto start_time = std::chrono::high_resolution_clock::now();
    
    EnqueueProgram(cq, program, true);
    Finish(cq);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Calculate metrics
    double latency_ms = duration.count() / 1000.0;  // Convert to milliseconds
    
    // Calculate data size and bandwidth
    size_t total_bytes = n_tiles * TILE_WIDTH * TILE_HEIGHT * sizeof(bfloat16);
    double bandwidth_mbps = (total_bytes) / (duration.count() * 1e-6) / 1e6;
    

    // Read the output buffer.
    std::vector<uint32_t> c_data;
    EnqueueReadBuffer(cq, c, c_data, true);

    // Print partial results so we can see the output is correct (plus or minus some error due to BFP16 precision)
    std::cout << "Partial results: (note we are running under BFP16. It's going to be less accurate)\n";
    // size_t n = std::min((size_t)tile_size, (size_t)tile_size * n_tiles);
    bfloat16* a_bf16 = reinterpret_cast<bfloat16*>(a_data.data());
    bfloat16* c_bf16 = reinterpret_cast<bfloat16*>(c_data.data());
    for(int i = 0; i < tile_size * n_tiles; i++) {
        // auto c_val = c_bf16[i].to_float();
        auto tile_idx = i % tile_size;
        // std::cout << i << " - tile: " << tile_idx << " in: " << a_bf16[i].to_float() << " out: " << c_bf16[i].to_float() << "\n";
        if (c_bf16[i].to_float() != 6.0f) {
            std::cout << "Element " << i << " is incorrect: " << c_bf16[i].to_float() << " instead of 2.0!\n";
            break;
        }
    }
    std::cout << std::flush;

    // Print performance metrics
    std::cout << "\nPerformance Metrics:" << std::endl;
    std::cout << "Number of tiles: " << n_tiles << std::endl;
    std::cout << "Total # of cores: " << num_cores << std::endl;
    std::cout << "Total Execution Time: " << latency_ms << " ms" << std::endl;
    std::cout << "Latency per tile: " << latency_ms / n_tiles << " ms/tile" << std::endl;
    std::cout << "Total Bandwidth: " << bandwidth_mbps << " MB/s" << std::endl;
    std::cout << "Bandwidth per tile: " << bandwidth_mbps / n_tiles << " MB/s/tile" << std::endl;
    std::cout << "Data processed: " << total_bytes / 1024.0 / 1024.0 << " MB" << std::endl;

    // Finally, we close the device.
    CloseDevice(device);
    return 0;
}
