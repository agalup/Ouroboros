#include <iostream>

#include "device/Ouroboros_impl.cuh"
#include "device/MemoryInitialization.cuh"
#include "InstanceDefinitions.cuh"
#include "Utility.h"

#define TEST_MULTI

template <typename MemoryManagerType>
__global__ void d_testAllocation(MemoryManagerType* mm, int** verification_ptr, int num_allocations, int allocation_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;

	verification_ptr[tid] = reinterpret_cast<int*>(mm->malloc(allocation_size));
}


// run 1 thread per warp: group allocation for entire warp
template <typename MemoryManagerType>
__global__ void d_test_warp_Allocation(MemoryManagerType* mm, int** verification_ptr, int num_allocations, int allocation_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;

    if ((threadIdx.x % 32) == 0)
	    verification_ptr[tid] = reinterpret_cast<int*>(mm->malloc(32 * allocation_size));

    __syncthreads();
    verification_ptr[tid] = reinterpret_cast<int*>(reinterpret_cast<char*>(verification_ptr[(threadIdx.x/32)*32]) +
    ((threadIdx.x%32) * allocation_size));
}

__global__ void d_testWriteToMemory(int** verification_ptr, int num_allocations, int allocation_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	
	auto ptr = verification_ptr[tid];

	for(auto i = 0; i < (allocation_size / sizeof(int)); ++i)
	{
		ptr[i] = tid;
	}
}

__global__ void d_testReadFromMemory(int** verification_ptr, int num_allocations, int allocation_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	
	if(threadIdx.x == 0 && blockIdx.x == 0)
		printf("Test Read!\n");
	
	auto ptr = verification_ptr[tid];

	for(auto i = 0; i < (allocation_size / sizeof(int)); ++i)
	{
		if(ptr[i] != tid)
		{
			printf("%d - %d | We got a wrong value here! %d vs %d\n", threadIdx.x, blockIdx.x, ptr[i], tid);
			return;
		}
	}
}

template <typename MemoryManagerType>
__global__ void d_testFree(MemoryManagerType* mm, int** verification_ptr, int num_allocations)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;

	mm->free(verification_ptr[tid]);
}


// run 1 thread per warp: group allocation for entire warp
template <typename MemoryManagerType>
__global__ void d_test_warp_Free(MemoryManagerType* mm, int** verification_ptr, int num_allocations)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;

    if (threadIdx.x % 32 == 0)
	    mm->free(verification_ptr[tid]);
}

int main(int argc, char* argv[])
{
	std::cout << "Usage: num_allocations allocation_size_in_bytes blockSize\n";
	int num_allocations {10000};
	int allocation_size_byte {16};
	int num_iterations {10};
	int blockSize {256};
    int allthreads {1};
	if(argc >= 2){
		num_allocations = atoi(argv[1]);
		if(argc >= 3){
			allocation_size_byte = atoi(argv[2]);
            if (argc >= 4){
                blockSize = atoi(argv[3]);
                if (argc >= 5){
                    allthreads = atoi(argv[4]);
                }
            }
		}
	}
	allocation_size_byte = Ouro::alignment(allocation_size_byte, sizeof(int));
	std::cout << "Number of Allocations: " << num_allocations << " | Allocation Size: " << allocation_size_byte << " | Iterations: " << num_iterations << std::endl;
    if (allthreads){
        printf("All threads per warp\n");
    }else{
        printf("One thread per warp\n");
    }

	#ifdef TEST_PAGES

	#ifdef TEST_VIRTUALARRAY
	std::cout << "Testing page-based memory manager - Virtualized Array!\n";
	#ifndef TEST_MULTI
	using MemoryManagerType = OuroVAPQ;
	#else
	using MemoryManagerType = MultiOuroVAPQ;
	#endif
	#elif TEST_VIRTUALLIST
	std::cout << "Testing page-based memory manager - Virtualized List!\n";
	#ifndef TEST_MULTI
	using MemoryManagerType = OuroVLPQ;
	#else
	using MemoryManagerType = MultiOuroVLPQ;
	#endif
	#else
	std::cout << "Testing page-based memory manager - Standard!\n";
	#ifndef TEST_MULTI
	using MemoryManagerType = OuroPQ;
	#else
	using MemoryManagerType = MultiOuroPQ;
	#endif
	#endif

	#elif TEST_CHUNKS

	#ifdef TEST_VIRTUALARRAY
	std::cout << "Testing chunk-based memory manager - Virtualized Array!\n";
	#ifndef TEST_MULTI
	using MemoryManagerType = OuroVACQ;
	#else
	using MemoryManagerType = MultiOuroVACQ;
	#endif
	#elif TEST_VIRTUALLIST
	std::cout << "Testing chunk-based memory manager - Virtualized List!\n";
	#ifndef TEST_MULTI
	using MemoryManagerType = OuroVLCQ;
	#else
	using MemoryManagerType = MultiOuroVLCQ;
	#endif
	#else
	std::cout << "Testing chunk-based memory manager - Standard!\n";
	#ifndef TEST_MULTI
	using MemoryManagerType = OuroCQ;
	#else
	using MemoryManagerType = MultiOuroCQ;
	#endif
	#endif

	#endif

	size_t instantitation_size = 4 * 1024ULL * 1024ULL * 1024ULL;
	MemoryManagerType memory_manager;
	memory_manager.initialize(instantitation_size);

	int** d_memory{nullptr};
	HANDLE_ERROR(cudaMalloc(&d_memory, sizeof(int*) * num_allocations));
    printf("num_allocations %d\n", num_allocations * sizeof(int*));

	int gridSize {Ouro::divup(num_allocations, blockSize)};
	float timing_allocation{0.0f};
	float timing_free{0.0f};
	cudaEvent_t start, end;
	for(auto i = 0; i < num_iterations; ++i)
	{
		start_clock(start, end);

        if (allthreads){
		    d_testAllocation <MemoryManagerType> <<<gridSize, blockSize>>>(memory_manager.getDeviceMemoryManager(), d_memory, num_allocations, allocation_size_byte);
        }else{
		    d_test_warp_Allocation <MemoryManagerType> <<<gridSize, blockSize>>>(memory_manager.getDeviceMemoryManager(), d_memory, num_allocations, allocation_size_byte);
        }

		timing_allocation += end_clock(start, end);

		HANDLE_ERROR(cudaDeviceSynchronize());

		d_testWriteToMemory<<<gridSize, blockSize>>>(d_memory, num_allocations, allocation_size_byte);

		HANDLE_ERROR(cudaDeviceSynchronize());

		d_testReadFromMemory<<<gridSize, blockSize>>>(d_memory, num_allocations, allocation_size_byte);

		HANDLE_ERROR(cudaDeviceSynchronize());

		start_clock(start, end);

        if (allthreads){
		    d_testFree <MemoryManagerType> <<<gridSize, blockSize>>>(memory_manager.getDeviceMemoryManager(), d_memory,
            num_allocations);
        }else{
		    d_test_warp_Free <MemoryManagerType> <<<gridSize, blockSize>>>(memory_manager.getDeviceMemoryManager(), d_memory, num_allocations);
        }

		timing_free += end_clock(start, end);

		HANDLE_ERROR(cudaDeviceSynchronize());
	}
	timing_allocation /= num_iterations;
	timing_free /= num_iterations;

	std::cout << "Timing Allocation: " << timing_allocation << "ms" << std::endl;
    int num_alloc_per_sec = (1000.0 * num_allocations)/timing_allocation;
    if (num_alloc_per_sec/(1000*1000*1000) > 0){
        std::cout << "# allocations per sec: " << num_alloc_per_sec/1000000000 << "G" << std::endl;
    }else if (num_alloc_per_sec/(1000*1000) > 0){
        std::cout << "# allocations per sec: " << num_alloc_per_sec/1000000 << "M" << std::endl;
    }else if (num_alloc_per_sec/1000 > 0){
        std::cout << "# allocations per sec: " << num_alloc_per_sec/1000 << std::endl;
    }else{
        std::cout << "# allocations per sec: " << num_alloc_per_sec << std::endl;
    }
	std::cout << "Timing       Free: " << timing_free << "ms" << std::endl;
	std::cout << "Testcase DONE!\n";

    return 0;
}
