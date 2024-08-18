#include "operators/transpose.h"
#include <cstddef>

namespace infini
{
    TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                               vector<int> permute)
        : OperatorObj(OpType::Transpose, {input}, {output})
    {
        auto rank = input->getRank();
        if (permute.empty())
        {
            for (size_t i = 0; i < rank; ++i)
            {
                transposePermute[i] = i;
            }
        }
        else
        {
            IT_ASSERT(rank == permute.size());
            transposePermute = std::move(permute);
        }
        IT_ASSERT(checkValid(graph));
    }

/**
 * @brief inferShape是推断张量经过转置操作后的形状，根据输入张量的维度和给定的维度排列（transposePermute)来计算输出张量的维度。
 * 
 * @param inputs 
 * @return optional<vector<Shape>> 
 */
    optional<vector<Shape>> TransposeObj::inferShape(const TensorVec &inputs)
    {
        const auto A = inputs[0];
        auto input_dim = A->getDims(); // input_dim获取输入张量A的维度列表，是一个包含张量各个维度大小的向量
        auto output_dim = input_dim; // 初始化输出张量的维度，初始与输入张量的维度相同。
        int rank = A->getRank(); // 获取输入张量的秩，表示维度的数量

        // =================================== 作业 ===================================
        // TODO：修改 output_dim，返回正确的 transpose 后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Transpose.html#transpose-21
        // =================================== 作业 ===================================
        // 计算转置后的维度
        for (int i = 0; i < rank; ++i) {
            output_dim[i] = input_dim[transposePermute[i]];
        }
        return vector<Shape>{output_dim};
    }

    std::string TransposeObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }
}; // namespace infini
