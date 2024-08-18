#include "operators/unary.h"

namespace infini
{
    UnaryObj::UnaryObj(OpType type, GraphObj *graph, Tensor input, Tensor output)
        : OperatorObj(type, {input}, {output})
    {
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> UnaryObj::inferShape(const TensorVec &inputs)
    {
        const auto A = inputs[0];
        return {{A->getDims()}};
    }

    std::string UnaryObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }
/**
 * c++类的构造函数，定义了一个名为ClipObj的类对象。用于初始化ClipObj类的对象。
 * ClipObj::表明ClipObj这个构造函数是这个类的成员
 */
    ClipObj::ClipObj(GraphObj *graph, Tensor input, Tensor output,
                     std::optional<float> min, std::optional<float> max)
        : OperatorObj(OpType::Clip, {input}, {output}), minValue(min),
          maxValue(max)
    {
        IT_ASSERT(checkValid(graph));
    }
    
    optional<vector<Shape>> ClipObj::inferShape(const TensorVec &inputs) // TensorVec == vector<Tensor>
    {
        /**
         *  inputs是一个存放张量的向量。&表示是引用传递，表示传递的是参数本身，函数内部对参数的修改将直接影响原始变量。 
         *  optional<vector<Shape>> 是c++17引入的std::optional用法，是一个模板类，表示一个变量可能包含值，也可能不包含值，可以用来避免使用指针或特殊值（如nullptr或-1）来表示无值的情况
         *  vector<Shape>是一个包含Shape类型对象的数组
         *  optional<vector<Shape>>组合起来使用表明返回对象可能是一个Shape类型的数组或者是一个无值（nullptr或者-1）
         *  ClipObj::inferShape是ClipObj的成员函数，::是作用于解析操作符，用于告诉编译器inferShape函数属于ClipObj类。
         *  在类内部声明，在类外部实现。
         */
        /**
         * 张量的形状通常描述张量在每个维度上的大小
         */
        // =================================== 作业 ===================================
        // TODO：返回经过 clip 操作后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Clip.html#clip-13
        // =================================== 作业 ===================================
        // clip时一种用于限制张量中元素数值范围的操作。具体说，clip会将张量中的每个元素限制在一个指定的最小值和最大值之间，如果元素的值小于最小值，就将其设为最小值，如果大于最大值
        // 设为最大值。如果在这个范围内，就保持不变
        if (inputs.empty()) {
            return std::nullopt;
        }
        vector<Shape> shapes;
        for (const auto &tensor : inputs) {
            Shape shape = tensor->getDims();
            shapes.push_back(shape);
        }
        return shapes;
    }

    std::string ClipObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

    CastObj::CastObj(GraphObj *graph, Tensor input, Tensor output, CastType type)
        : OperatorObj(OpType::Cast, {input}, {output}), castType(type)
    {
        IT_ASSERT(checkValid(graph));
    }

/**
 * @brief 将一个tensor从一个数据类型转换为另一种数据类型，通常是为了匹配模型的输入要求或进行特定的数值运算
 * cast操作通常是无损的，但是从浮点类型转换为整数类型时，可能会丢失精度，因为浮点数的小数部分会被舍弃
 * @param inputs 
 * @return vector<DataType> 
 */
    vector<DataType> CastObj::inferDataType(const TensorVec &inputs) const
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 cast 操作后, 输出 tensor 的数目和数据类型
        // REF_FILE: src/core/operator.cc
        // REF: https://onnx.ai/onnx/operators/onnx__Cast.html#cast-21
        // =================================== 作业 ===================================
        // return vector(numOutputs(), getOutputDataType());
        return vector(numOutputs(), getOutputDataType());
    }

    optional<vector<Shape>> CastObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 cast 操作后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Cast.html#cast-21
        // =================================== 作业 ===================================
        // if (inputs.empty()) {
        //     return std::nullopt;
        // }

        vector<Shape> outputShape;

        for (auto &tensor : inputs) {
            Shape shapes = tensor->getDims();
            outputShape.push_back(shapes);
        }

        return outputShape;
    }

    std::string CastObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

    DataType CastObj::getOutputDataType() const
    {
        switch (castType)
        {
        case CastType::Float2Float16:
            return DataType::Float16;
        case CastType::Float2Int64:
            return DataType::Int64;
        case CastType::Float2Int32:
            return DataType::Int32;
        case CastType::Float2Int16:
            return DataType::Int16;
        case CastType::Float2Int8:
            return DataType::Int8;
        case CastType::Int322Float:
            return DataType::Float32;
        case CastType::Int322Int8:
            return DataType::Int8;
        case CastType::Int322Int16:
            return DataType::Int16;
        case CastType::Int162Float:
            return DataType::Float32;
        case CastType::Int162Int32:
            return DataType::Int32;
        case CastType::Int82Float:
            return DataType::Float32;
        case CastType::Int82Int16:
            return DataType::Int16;
        case CastType::Int82Int32:
            return DataType::Int32;
        case CastType::Uint82Float:
            return DataType::Float32;
        case CastType::Uint82Int32:
            return DataType::Int32;
        case CastType::Uint82Int64:
            return DataType::Int64;
        case CastType::Int322Int64:
            return DataType::Int64;
        case CastType::Int642Int32:
            return DataType::Int32;
        case CastType::Int642Uint32:
            return DataType::UInt32;
        case CastType::Int642Float:
            return DataType::Float32;
        case CastType::Uint322Int64:
            return DataType::Int64;
        case CastType::Float162Float:
            return DataType::Float32;
        case CastType::BFloat162Float:
            return DataType::Float32;
        case CastType::Float2BFloat16:
            return DataType::BFloat16;
        case CastType::Float2Float:
            return DataType::Float32;
        default:
            IT_TODO_HALT();
        }
    }
}; // namespace infini
