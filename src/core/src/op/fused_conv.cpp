// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fused_conv.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {

FusedConv::FusedConv(const ov::OutputVector& args, const std::shared_ptr<ov::op::util::Variable>& variable)
    : ov::op::Op(args) {
    m_variable = variable;
    constructor_validate_and_infer_types();
}

FusedConv::FusedConv(const ov::OutputVector& args, const std::shared_ptr<ov::op::util::Variable>& variable, bool output_snapshots)
    : ov::op::Op(args) {
    m_variable = variable;
    m_output_snapshots = output_snapshots;
    constructor_validate_and_infer_types();
}

FusedConv::FusedConv(const ov::OutputVector& args, const std::shared_ptr<ov::op::util::Variable>& variable, bool output_snapshots, int64_t snapshot_max_seq)
    : ov::op::Op(args) {
    m_variable = variable;
    m_output_snapshots = output_snapshots;
    m_snapshot_max_seq = snapshot_max_seq;
    constructor_validate_and_infer_types();
}

bool FusedConv::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(FusedConv_visit_attributes);

    visitor.on_attribute("output_snapshots", m_output_snapshots);
    visitor.on_attribute("snapshot_max_seq", m_snapshot_max_seq);
    visitor.on_attribute("variable_id", m_variable);
    OPENVINO_ASSERT(m_variable, "Variable is not initialized.");

    auto variable_info = m_variable->get_info();
    visitor.on_attribute("variable_type", variable_info.data_type);
    visitor.on_attribute("variable_shape", variable_info.data_shape);
    m_variable->update(variable_info);

    return true;
}

void FusedConv::validate_and_infer_types() {
    OV_OP_SCOPE(FusedConv_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 5,
                          "FusedConv expects 5 inputs, but it has ",
                          get_input_size());
    OPENVINO_ASSERT(m_variable, "Variable is not initialized.");

    // input[0]: [B, conv_dim, S]
    const auto& input_rank = get_input_partial_shape(0).rank();
    NODE_VALIDATION_CHECK(this,
                          input_rank.is_dynamic() || input_rank.get_length() == 3,
                          "Rank of `input` should be 3, but it is ",
                          input_rank);

    // input[1]: [conv_dim, kernel_size]
    const auto& weight_rank = get_input_partial_shape(1).rank();
    NODE_VALIDATION_CHECK(this,
                          weight_rank.is_dynamic() || weight_rank.get_length() == 2,
                          "Rank of `conv_weight` should be 2, but it is ",
                          weight_rank);

    // input[2]: [B] (i32/i64)
    const auto& beam_rank = get_input_partial_shape(2).rank();
    NODE_VALIDATION_CHECK(this,
                          beam_rank.is_dynamic() || beam_rank.get_length() == 1,
                          "Rank of `beam_idx` should be 1, but it is ",
                          beam_rank);

    // input[3]: [B, conv_dim, kernel_size]
    const auto& state_rank = get_input_partial_shape(3).rank();
    NODE_VALIDATION_CHECK(this,
                          state_rank.is_dynamic() || state_rank.get_length() == 3,
                          "Rank of `initial_state` should be 3, but it is ",
                          state_rank);

    const auto& mode_rank = get_input_partial_shape(4).rank();
    NODE_VALIDATION_CHECK(this,
                          mode_rank.is_dynamic() || mode_rank.get_length() == 0 || mode_rank.get_length() == 1,
                          "Rank of `state_update_mode` should be 0 or 1, but it is ",
                          mode_rank);
    const auto& mode_type = get_input_element_type(4);
    NODE_VALIDATION_CHECK(this,
                          mode_type.is_dynamic() || mode_type == ov::element::i32 || mode_type == ov::element::i64 || mode_type == ov::element::boolean,
                          "Element type of `state_update_mode` should be i32, i64, or boolean, but it is ",
                          mode_type);

    const auto& variable_info = m_variable->get_info();
    const auto& variable_shape = variable_info.data_shape;
    const auto& variable_type = variable_info.data_type;
    const auto& initial_shape = get_input_partial_shape(3);
    const auto& initial_type = get_input_element_type(3);

    OPENVINO_ASSERT(variable_shape.relaxes(initial_shape),
                    "The shape specified in the Variable has to extend (relax) the shape inferred from `initial_state`.",
                    " Variable shape: ",
                    variable_shape,
                    " initial_state shape: ",
                    initial_shape);
    OPENVINO_ASSERT(variable_type.is_dynamic() || initial_type == variable_type,
                    "The type specified in the Variable is not compatible with `initial_state`.",
                    " Variable type: ",
                    variable_type,
                    " initial_state type: ",
                    initial_type);

    // output[0]: same shape as input[0] = [B, conv_dim, S]
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    // output[1]: same shape as input[3] = [B, conv_dim, kernel_size]
    set_output_type(1, variable_type, variable_shape);

    if (m_output_snapshots) {
        // output[2]: per-step state snapshots [B, S, conv_dim, kernel_size]
        // Cap S at snapshot_max_seq to prevent huge allocation during prefill.
        const auto& in_ps = get_input_partial_shape(0);  // [B, conv_dim, S]
        ov::PartialShape snap_ps;
        if (in_ps.rank().is_static() && initial_shape.rank().is_static() &&
            in_ps.rank().get_length() == 3 && initial_shape.rank().get_length() == 3) {
            auto snap_s = in_ps[2];
            if (m_snapshot_max_seq > 0) {
                snap_s = m_snapshot_max_seq;
            }
            // [batch, seq_len, conv_dim, kernel_size]
            snap_ps = {in_ps[0], snap_s, initial_shape[1], initial_shape[2]};
        } else {
            snap_ps = ov::PartialShape::dynamic(4);
        }
        set_output_type(2, variable_type, snap_ps);
    }
}

std::shared_ptr<ov::Node> FusedConv::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<FusedConv>(new_args, m_variable, m_output_snapshots, m_snapshot_max_seq);
}

}  // namespace op
}  // namespace ov
