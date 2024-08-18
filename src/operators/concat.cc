#include "operators/concat.h"
#include "utils/operator_utils.h"
#include <cstddef>

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

/**
 * @brief concat属于张量运算的算子，是常见的张量操作。
 * 功能：将两个或多个张量tensor在指定维度上拼接在一起。
 * 例如一个张量(2, 3)，也就是有两维，是一个矩阵，第0维2行，第1维3列，这是一个2行3列的矩阵。
 * 如果有两个（2，3）的张量，如果在第0维拼接，concat(tensor1, tensor2, axis=0)，结果将是（4，3）的tensor
 * 如果在第1维拼接，将是一个(2,6)的张量。
 * @param inputs 
 * @return optional<vector<Shape>> 
 */
optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    Shape dims = inputs[0]->getDims(); 
    auto rank = inputs[0]->getRank(); 

    // =================================== 作业 ===================================
    // TODO：修改 dims，返回正确的 concat 后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
    // =================================== 作业 ===================================
    for (size_t i = 1; i < inputs.size(); ++i) {
        dims.at(dim) += inputs[i]->getDims().at(dim);
    }
    return {{dims}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini
