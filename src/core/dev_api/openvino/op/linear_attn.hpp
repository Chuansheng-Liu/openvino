// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/variable.hpp"

namespace ov {
namespace op {

// This is an experimental operation that is implemented in the plugins.
// Do not use in user applications, backward compatibility is not guaranteed in future releases.
class OPENVINO_API LinearAttention : public ov::op::Op {
public:
    OPENVINO_OP("LinearAttention");

    LinearAttention() = default;

    LinearAttention(const ov::OutputVector& args);
    LinearAttention(const ov::OutputVector& args, const std::shared_ptr<ov::op::util::Variable>& variable);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    void set_out_type(int index, const ov::element::Type& output_type);

    std::shared_ptr<ov::op::util::Variable> get_variable() const { return m_variable; }

protected:
    std::vector<ov::element::Type> m_output_type = {ov::element::dynamic, ov::element::dynamic, ov::element::dynamic};
    std::shared_ptr<ov::op::util::Variable> m_variable;
};

}  // namespace op
}  // namespace ov
