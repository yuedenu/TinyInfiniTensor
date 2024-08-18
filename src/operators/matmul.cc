#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        
        const auto& A = inputs[0];
        const auto& B = inputs[1];
        Shape result{infer_broadcast(A->getDims(), B->getDims())};

        if (transA) {
            result.at(result.size() - 2) = A->getDims().at(A->getDims().size() - 1);
        } else {
            result.at(result.size() - 2) = A->getDims().at(A->getDims().size() - 2);
        }

        if (transB) {
            result.at(result.size() - 1) = B->getDims().at(B->getDims().size() - 2);
        } else {
            result.at(result.size() - 1) = B->getDims().at(B->getDims().size() - 1);
        }

        return {{result}};
    }

} // namespace infini