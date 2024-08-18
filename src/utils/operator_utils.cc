#include "utils/operator_utils.h"
#include "core/common.h"
#include "core/runtime.h"
#include <algorithm>
#include <cstddef>

namespace infini {

/**
 * @brief 双向广播是处理不同形状张量进行算术操作的常用技术。允许形状不同的张量通过扩展维度（在某些情况下重复元素）来让他们
 * 有相同的形状，从而进行逐元素操作。
 * 对齐形状：从后往前对齐两个张量的形状，如果一个张量的维度数少于另一个，则在前面用1填充较小维度的张量
 * 检查维度兼容性：如果两个张量在某个维度上大小相等，或其中一个维度的大小为1，则在该维度上可以进行广播
 *          如果在某个维度上大小不相等且都不为1，则广播失败
 * 计算广播后的形状：对齐后的维度中，每个位置上的维度取两者中的较大值
 * @param A 
 * @param B 
 * @return Shape 
 */
Shape infer_broadcast(const Shape &A, const Shape &B) {

    // =================================== 作业 ===================================
    // TODO：对 A 和 B 进行双向广播，返回广播后的形状。
    // REF: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    // =================================== 作业 ===================================
    
    Shape result(std::max(A.size(), B.size()), 1);
    for (int i = 0; i < static_cast<int>(result.size()); ++i) {
        int A_idx{static_cast<int>(A.size()) - i - 1};
        int B_idx{static_cast<int>(B.size()) - i - 1};
        if (A_idx >= 0 && B_idx >= 0) {
            result.at(result.size() - i - 1) = std::max(A.at(A_idx), B.at(B_idx));
        } else if (A_idx >= 0) {
            result.at(result.size() - i - 1) = A.at(A_idx);
        } else {
            result.at(result.size() - i - 1) = B.at(B_idx);
        }
    }
    return result;
}

int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini
