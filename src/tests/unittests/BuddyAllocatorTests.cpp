// Copyright 2019 The Dawn Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include "dawn_native/BuddyAllocator.h"

using namespace dawn_native;

// Verify the buddy allocator with a basic test.
TEST(BuddyAllocatorTests, SingleBlock) {
    // After one 32 byte allocation:
    //
    //  Level          --------------------------------
    //      0       32 |               A              |
    //                 --------------------------------
    //
    constexpr uint64_t maxBlockSize = 32;
    BuddyAllocator allocator(maxBlockSize);

    // Check that we cannot allocate a oversized block.
    ASSERT_EQ(allocator.Allocate(maxBlockSize * 2), INVALID_OFFSET);

    // Check that we cannot allocate a zero sized block.
    ASSERT_EQ(allocator.Allocate(0u), INVALID_OFFSET);

    // Allocate the block.
    uint64_t blockOffset = allocator.Allocate(maxBlockSize);
    ASSERT_EQ(blockOffset, 0u);

    // Check that we are full.
    ASSERT_EQ(allocator.Allocate(maxBlockSize), INVALID_OFFSET);
    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 0u);

    // Deallocate the block.
    allocator.Deallocate(blockOffset);
    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 1u);
}

// Verify multiple allocations succeeds using a buddy allocator.
TEST(BuddyAllocatorTests, MultipleBlocks) {
    // Fill every level in the allocator (order-n = 2^n)
    const uint64_t maxBlockSize = (1ull << 16);
    for (uint64_t order = 1; (1ull << order) <= maxBlockSize; order++) {
        BuddyAllocator allocator(maxBlockSize);

        uint64_t blockSize = (1ull << order);
        for (uint32_t blocki = 0; blocki < (maxBlockSize / blockSize); blocki++) {
            ASSERT_EQ(allocator.Allocate(blockSize), blockSize * blocki);
        }
    }
}

// Verify that a single allocation succeeds using a buddy allocator.
TEST(BuddyAllocatorTests, SingleSplitBlock) {
    //  After one 8 byte allocation:
    //
    //  Level          --------------------------------
    //      0       32 |               S              |
    //                 --------------------------------
    //      1       16 |       S       |       F      |        S - split
    //                 --------------------------------        F - free
    //      2       8  |   A   |   F   |       |      |        A - allocated
    //                 --------------------------------
    //
    constexpr uint64_t maxBlockSize = 32;
    BuddyAllocator allocator(maxBlockSize);

    // Allocate block (splits two blocks).
    uint64_t blockOffset = allocator.Allocate(8);
    ASSERT_EQ(blockOffset, 0u);
    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 2u);

    // Deallocate block (merges two blocks).
    allocator.Deallocate(blockOffset);
    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 1u);

    // Check that we cannot allocate a block that is oversized.
    ASSERT_EQ(allocator.Allocate(maxBlockSize * 2), INVALID_OFFSET);

    // Re-allocate the largest block allowed after merging.
    blockOffset = allocator.Allocate(maxBlockSize);
    ASSERT_EQ(blockOffset, 0u);

    allocator.Deallocate(blockOffset);
    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 1u);
}

// Verify that a multiple allocated blocks can be removed in the free-list.
TEST(BuddyAllocatorTests, MultipleSplitBlocks) {
    //  After four 16 byte allocations:
    //
    //  Level          --------------------------------
    //      0       32 |               S              |
    //                 --------------------------------
    //      1       16 |       S       |       S      |        S - split
    //                 --------------------------------        F - free
    //      2       8  |   Aa  |   Ab  |  Ac  |   Ad  |        A - allocated
    //                 --------------------------------
    //
    constexpr uint64_t maxBlockSize = 32;
    BuddyAllocator allocator(maxBlockSize);

    // Populates the free-list with four blocks at Level2.

    // Allocate "a" block (two splits).
    constexpr uint64_t blockSizeInBytes = 8;
    uint64_t blockOffsetA = allocator.Allocate(blockSizeInBytes);
    ASSERT_EQ(blockOffsetA, 0u);
    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 2u);

    // Allocate "b" block.
    uint64_t blockOffsetB = allocator.Allocate(blockSizeInBytes);
    ASSERT_EQ(blockOffsetB, blockSizeInBytes);
    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 1u);

    // Allocate "c" block (three splits).
    uint64_t blockOffsetC = allocator.Allocate(blockSizeInBytes);
    ASSERT_EQ(blockOffsetC, blockOffsetB + blockSizeInBytes);
    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 1u);

    // Allocate "d" block.
    uint64_t blockOffsetD = allocator.Allocate(blockSizeInBytes);
    ASSERT_EQ(blockOffsetD, blockOffsetC + blockSizeInBytes);
    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 0u);

    // Deallocate "d" block.
    // FreeList[Level2] = [BlockD] -> x
    allocator.Deallocate(blockOffsetD);
    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 1u);

    // Deallocate "b" block.
    // FreeList[Level2] = [BlockB] -> [BlockD] -> x
    allocator.Deallocate(blockOffsetB);
    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 2u);

    // Deallocate "c" block (one merges).
    // FreeList[Level1] = [BlockCD] -> x
    // FreeList[Level2] = [BlockB] -> x
    allocator.Deallocate(blockOffsetC);
    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 2u);

    // Deallocate "a" block (two merges).
    // FreeList[Level0] = [BlockABCD] -> x
    allocator.Deallocate(blockOffsetA);
    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 1u);
}

// Verify the buddy allocator can handle allocations of various sizes.
TEST(BuddyAllocatorTests, MultipleSplitBlockIncreasingSize) {
    //  After four Level4-to-Level1 byte then one L4 block allocations:
    //
    //  Level          -----------------------------------------------------------------
    //      0      512 |                               S                               |
    //                 -----------------------------------------------------------------
    //      1      256 |               S               |               A               |
    //                 -----------------------------------------------------------------
    //      2      128 |       S       |       A       |               |               |
    //                 -----------------------------------------------------------------
    //      3       64 |   S   |   A   |       |       |       |       |       |       |
    //                 -----------------------------------------------------------------
    //      4       32 | A | F |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
    //                 -----------------------------------------------------------------
    //
    constexpr uint64_t maxBlockSize = 512;
    BuddyAllocator allocator(maxBlockSize);

    ASSERT_EQ(allocator.Allocate(32), 0ull);
    ASSERT_EQ(allocator.Allocate(64), 64ull);
    ASSERT_EQ(allocator.Allocate(128), 128ull);
    ASSERT_EQ(allocator.Allocate(256), 256ull);

    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 1u);

    // Fill in the last free block.
    ASSERT_EQ(allocator.Allocate(32), 32ull);

    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 0u);

    // Check if we're full.
    ASSERT_EQ(allocator.Allocate(32), INVALID_OFFSET);
}

// Verify very small allocations using a larger allocator works correctly.
TEST(BuddyAllocatorTests, MultipleSplitBlocksVariableSizes) {
    //  After allocating four pairs of one 64 byte block and one 32 byte block.
    //
    //  Level          -----------------------------------------------------------------
    //      0      512 |                               S                               |
    //                 -----------------------------------------------------------------
    //      1      256 |               S               |               S               |
    //                 -----------------------------------------------------------------
    //      2      128 |       S       |       S       |       S       |       F       |
    //                 -----------------------------------------------------------------
    //      3       64 |   A   |   S   |   A   |   A   |   S   |   A   |       |       |
    //                 -----------------------------------------------------------------
    //      4       32 |   |   | A | A |   |   |   |   | A | A |   |   |   |   |   |   |
    //                 -----------------------------------------------------------------
    //
    constexpr uint64_t maxBlockSize = 512;
    BuddyAllocator allocator(maxBlockSize);

    ASSERT_EQ(allocator.Allocate(64), 0ull);
    ASSERT_EQ(allocator.Allocate(32), 64ull);

    ASSERT_EQ(allocator.Allocate(64), 128ull);
    ASSERT_EQ(allocator.Allocate(32), 96ull);

    ASSERT_EQ(allocator.Allocate(64), 192ull);
    ASSERT_EQ(allocator.Allocate(32), 256ull);

    ASSERT_EQ(allocator.Allocate(64), 320ull);
    ASSERT_EQ(allocator.Allocate(32), 288ull);

    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 1u);
}

// Verify the buddy allocator can deal with bad fragmentation.
TEST(BuddyAllocatorTests, MultipleSplitBlocksInterleaved) {
    //  Allocate every leaf then de-allocate every other of those allocations.
    //
    //  Level          -----------------------------------------------------------------
    //      0      512 |                               S                               |
    //                 -----------------------------------------------------------------
    //      1      256 |               S               |               S               |
    //                 -----------------------------------------------------------------
    //      2      128 |       S       |       S       |        S       |        S     |
    //                 -----------------------------------------------------------------
    //      3       64 |   S   |   S   |   S   |   S   |   S   |   S   |   S   |   S   |
    //                 -----------------------------------------------------------------
    //      4       32 | A | F | A | F | A | F | A | F | A | F | A | F | A | F | A | F |
    //                 -----------------------------------------------------------------
    //
    constexpr uint64_t maxBlockSize = 512;
    BuddyAllocator allocator(maxBlockSize);

    // Allocate leaf blocks
    constexpr uint64_t minBlockSizeInBytes = 32;
    std::vector<uint64_t> blockOffsets;
    for (uint64_t i = 0; i < maxBlockSize / minBlockSizeInBytes; i++) {
        blockOffsets.push_back(allocator.Allocate(minBlockSizeInBytes));
    }

    // Free every other leaf block.
    for (size_t count = 1; count < blockOffsets.size(); count += 2) {
        allocator.Deallocate(blockOffsets[count]);
    }

    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 8u);
}

// Verify the buddy allocator can deal with multiple allocations with mixed alignments.
TEST(BuddyAllocatorTests, SameSizeVariousAlignment) {
    //  After two 8 byte allocations with 16 byte alignment then one 8 byte allocation with 8 byte
    //  alignment.
    //
    //  Level          --------------------------------
    //      0       32 |               S              |
    //                 --------------------------------
    //      1       16 |       S       |       S      |       S - split
    //                 --------------------------------       F - free
    //      2       8  |   Aa  |   F   |  Ab   |  Ac  |       A - allocated
    //                 --------------------------------
    //
    BuddyAllocator allocator(32);

    // Allocate Aa (two splits).
    ASSERT_EQ(allocator.Allocate(8, 16), 0u);
    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 2u);

    // Allocate Ab (skip Aa buddy due to alignment and perform another split).
    ASSERT_EQ(allocator.Allocate(8, 16), 16u);

    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 2u);

    // Check that we cannot fit another.
    ASSERT_EQ(allocator.Allocate(8, 16), INVALID_OFFSET);

    // Allocate Ac (zero splits and Ab's buddy is now the first free block).
    ASSERT_EQ(allocator.Allocate(8, 8), 24u);

    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 1u);
}

// Verify the buddy allocator can deal with multiple allocations with equal alignments.
TEST(BuddyAllocatorTests, VariousSizeSameAlignment) {
    //  After two 8 byte allocations with 4 byte alignment then one 16 byte allocation with 4 byte
    //  alignment.
    //
    //  Level          --------------------------------
    //      0       32 |               S              |
    //                 --------------------------------
    //      1       16 |       S       |       Ac     |       S - split
    //                 --------------------------------       F - free
    //      2       8  |   Aa  |   Ab  |              |       A - allocated
    //                 --------------------------------
    //
    constexpr uint64_t maxBlockSize = 32;
    constexpr uint64_t alignment = 4;
    BuddyAllocator allocator(maxBlockSize);

    // Allocate block Aa (two splits)
    ASSERT_EQ(allocator.Allocate(8, alignment), 0u);
    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 2u);

    // Allocate block Ab (Aa's buddy)
    ASSERT_EQ(allocator.Allocate(8, alignment), 8u);

    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 1u);

    // Check that we can still allocate Ac.
    ASSERT_EQ(allocator.Allocate(16, alignment), 16ull);

    ASSERT_EQ(allocator.ComputeTotalNumOfFreeBlocksForTesting(), 0u);
}