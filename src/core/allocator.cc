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
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        auto f = 0;
        auto min = fb_map.begin();
        for(auto i = fb_map.begin(); i != fb_map.end(); i++){
            if(i->second >= size){
                if(f == 0){
                    f = 1;
                    min = i;
                }else if(min->second > i->second) min = i;
            }
        }
        auto addr = 0;
        if(f == 0){
            addr = used;
            used += size;
        }else {
            addr = min->first + min->second - size;
            min->second -= size;
            if(min->second == 0) fb_map.erase(min);
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
        for(auto i = fb_map.begin(); i != fb_map.end(); i++){
            if(i->first + i->second == addr){
                i->second += size;
                auto j = i+1;
                if(j != fb_map.end() && i->first + i->second == j->first){
                    i->second += j->second;
                    fb_map.erase(j);
                }
                if(i->first + i->second == used){
                    used -= i->second;
                    fb_map.erase(i);
                }
                break;
            }
        }
        if(i == fb_map.end()){
            if(addr + size == used) used -= size;
            else fb_map.insert(std::make_pair(addr,size));
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
