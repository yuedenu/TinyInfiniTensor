#include "core/graph.h"
#include "core/common.h"
#include "core/op_type.h"
#include "core/runtime.h"
#include "operators/transpose.h"
#include "operators/matmul.h"
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        if (ops.empty()) {
            return;
        }

        IT_ASSERT(topo_sort() == true);
        bool optimized_this_pass{false};
        for (size_t i = 0; i < ops.size();) {
            optimized_this_pass = false;
            auto& op{ops.at(i)};
            // rule 1, 2
            if (op->type == OpType::Transpose) {
                const auto& perm{as<TransposeObj>(op)->getPermute()};
                for (auto& succ: op->successors) {
                    auto succ_ptr{succ.lock()};
                    if (!succ_ptr) {
                        continue;
                    }

                    // rule 1
                    if (succ_ptr->type == OpType::Transpose) {
                        const auto& succ_perm{as<TransposeObj>(succ_ptr)->getPermute()};
                        if (perm == succ_perm) {
                            for (auto& succ2: succ_ptr->successors) {
                                auto succ2_ptr{succ2.lock()};
                                if (!succ2_ptr) {
                                    continue;
                                }

                                // release tensor
                                for (auto t: op->outputs) {
                                    removeTensor(t);
                                }
                                for (auto t: succ_ptr->outputs) {
                                    removeTensor(t);
                                }

                                // reconnect
                                for (size_t i = 0; i < succ2_ptr->inputs.size(); ++i) {
                                    if (succ2_ptr->inputs[i] == succ_ptr->outputs[0]) {
                                        succ2_ptr->inputs[i] = op->inputs[0];
                                    }
                                }
                                for (auto& input: succ2_ptr->inputs) {
                                    input->removeTarget(op);
                                    input->addTarget(succ2_ptr);
                                }
                                succ2_ptr->removePredecessors(succ_ptr);
                            }

                            // remove from ops
                            for (auto it = ops.begin(); it != ops.end();) {
                                if (*it == op) {
                                    ops.erase(it);
                                    break;
                                } else {
                                    ++it;
                                }
                            }

                            for (auto it = ops.begin(); it != ops.end();) {
                                if (*it == succ_ptr) {
                                    ops.erase(it);
                                    break;
                                } else {
                                    ++it;
                                }
                            }
                            // The graph structure has been modified, we optimize from the begining
                            i = 0;
                            optimized_this_pass = true;
                        }
                    } else if (succ_ptr->type == OpType::MatMul) {
                        // rule 2
                        if ((perm.at(perm.size() - 1) == static_cast<int>(perm.size()) - 2) &&
                            (perm.at(perm.size() - 2) == static_cast<int>(perm.size()) - 1)) {
                            // can been trans
                            if (op->outputs[0] == succ_ptr->inputs[0]) {
                                // trans A
                                as<MatmulObj>(succ_ptr)->setTransA(true);
                                succ_ptr->inputs[0] = op->inputs[0];
                                succ_ptr->inputs[0]->removeTarget(op);
                                succ_ptr->inputs[0]->addTarget(succ_ptr);
                            } else {
                                // trans B
                                as<MatmulObj>(succ_ptr)->setTransB(true);
                                succ_ptr->inputs[1] = op->inputs[0];
                                succ_ptr->inputs[1]->removeTarget(op);
                                succ_ptr->inputs[1]->addTarget(succ_ptr);
                            }
                            // release tensor
                            removeTensor(op->outputs[0]);
                            // reconnect
                            succ_ptr->removePredecessors(op);

                            // remove from ops
                            for (auto it = ops.begin(); it != ops.end();) {
                                if (*it == op) {
                                    ops.erase(it);
                                    break;
                                } else {
                                    ++it;
                                }
                            }
                            // The graph structure has been modified, we optimize from the begining
                            i = 0;
                            optimized_this_pass = true;
                        }
                    }
                }
            }

            if(!optimized_this_pass) {
                ++i;
            }
        }

        IT_ASSERT(topo_sort() == true);
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        size_t size{0};
        for (auto tensor: tensors) {
            size += tensor->getBytes();
        }

        allocator.alloc(size);
        size_t offset{0};
        for (auto tensor: tensors) {
            tensor->setDataBlob(make_ref<BlobObj>(runtime,
                reinterpret_cast<void*>(reinterpret_cast<char*>(allocator.getPtr()) + offset)));
            offset += tensor->getBytes();
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini