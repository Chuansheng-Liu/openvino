// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/util/variable_extension.hpp"

namespace ov {
namespace op {

// This is an experimental operation that is implemented in the plugins.
// Do not use in user applications, backward compatibility is not guaranteed in future releases.
class OPENVINO_API LinearAttention : public ov::op::Op, public ov::op::util::VariableExtension {
public:
    OPENVINO_OP("LinearAttention");

    LinearAttention() = default;

    LinearAttention(const ov::OutputVector& args);
    LinearAttention(const ov::OutputVector& args, const std::shared_ptr<ov::op::util::Variable>& variable);
    LinearAttention(const ov::OutputVector& args, const std::shared_ptr<ov::op::util::Variable>& variable, bool output_snapshots);
    LinearAttention(const ov::OutputVector& args, const std::shared_ptr<ov::op::util::Variable>& variable, bool output_snapshots, int64_t snapshot_max_seq);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    std::string get_variable_id() const override {
        OPENVINO_ASSERT(m_variable, "Variable is not initialized. Variable_id is unavailable");
        return m_variable->get_info().variable_id;
    }

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    void set_out_type(int index, const ov::element::Type& output_type);

    /// Enable per-step state snapshot output (output[2]).
    /// When enabled, the op produces 3 outputs:
    ///   output[0]: attention output  [B, S, num_v_heads, head_v_dim]
    ///   output[1]: final state       [B, num_v_heads, head_k_dim, head_v_dim]
    ///   output[2]: state snapshots   [B, S, num_v_heads, head_k_dim, head_v_dim]
    void set_output_snapshots(bool enable) { m_output_snapshots = enable; }
    bool get_output_snapshots() const { return m_output_snapshots; }
    void set_snapshot_max_seq(int64_t v) { m_snapshot_max_seq = v; }
    int64_t get_snapshot_max_seq() const { return m_snapshot_max_seq; }

protected:
    bool m_output_snapshots = false;
    int64_t m_snapshot_max_seq = 0;  // 0 = use input seq len; >0 = cap snapshot S dim
    std::vector<ov::element::Type> m_output_type = {ov::element::dynamic, ov::element::dynamic, ov::element::dynamic};
};

}  // namespace op
}  // namespace ov
