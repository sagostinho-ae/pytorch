#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API bool EliminateCommonSubexpression(
    const std::shared_ptr<Graph>& graph,
    const std::vector<std::string>& modules_as_onnx_functions =
        std::vector<std::string>());
}
} // namespace torch
