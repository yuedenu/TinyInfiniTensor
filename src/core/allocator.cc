#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量1
        // =================================== 作业 ===================================
        auto target{recycle_map_.end()};
        for (auto it = recycle_map_.begin(); it != recycle_map_.end(); ++it) {
            if (it->second >= size) {
                // valid block
                if (target == recycle_map_.end()) {
                    // the first available block
                    target = it;
                } else if (target->second > it->second) {
                    // try to find the smallest block
                    target = it;
                }
            }
        }

        size_t addr{0};
        if (target != recycle_map_.end()) {
            // alloc from recycled blocks
            addr = target->first + target->second - size;
            target->second -= size;
            if (target->second == 0) {
                recycle_map_.erase(target);
            }
        } else {
            // alloc from raw memory bank
            addr = used;
            used += size;
        }

        peak = used;
        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
                recycle_map_.insert(std::make_pair(addr, size));
        // try to merge recycled blocks
        // time complexity O(n)
        for (auto it = recycle_map_.begin(); it != recycle_map_.end();) {
            auto found{recycle_map_.find(it->first + it->second)};
            if(found != recycle_map_.end()) {
                it->second += found->second;
                recycle_map_.erase(found);
            } else if (it->first + it->second == used) {
                used -= it->second;
                it = recycle_map_.erase(it);
            } else {
                ++it;
            }
        }
        peak = used;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
