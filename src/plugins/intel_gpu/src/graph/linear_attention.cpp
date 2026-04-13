// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "linear_attention_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include "to_string_utils.h"
#include <string>
#include <vector>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(linear_attention)

layout linear_attention_inst::calc_output_layout(linear_attention_node const& node, kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template<typename ShapeType>
std::vector<layout> linear_attention_inst::calc_output_layouts(linear_attention_node const& node, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<linear_attention>();
    const auto& all_inputs = node.get_input_layouts();
    const auto num_outputs = desc->output_size();
    if (all_inputs.size() != 7)
        OPENVINO_THROW("linear_attention must have 7 inputs");
    // query, key, value, g, beta, initial_states, state_update_mode
    auto query_layout = impl_param.get_input_layout(0);
    auto value_layout = impl_param.get_input_layout(2);
    auto out_ps = value_layout.get_partial_shape();
    const auto& q_ps = query_layout.get_partial_shape();
    if (out_ps.rank().is_static() && q_ps.rank().is_static() && out_ps.rank().get_length() == 4 && q_ps.rank().get_length() == 4) {
        out_ps[0] = q_ps[0];
        out_ps[1] = q_ps[1];
    }
    std::vector<layout> output_layouts;
    output_layouts.emplace_back(out_ps, value_layout.data_type, value_layout.format);
    if (num_outputs >= 2) {
        output_layouts.push_back(impl_param.get_input_layout(5));
    }
    if (num_outputs >= 3) {
        // output[2]: per-step state snapshots [B, S, num_v_heads, head_k_dim, head_v_dim]
        auto initial_states_layout = impl_param.get_input_layout(5);
        const auto& h_ps = initial_states_layout.get_partial_shape();
        ov::PartialShape snap_ps;
        if (q_ps.rank().is_static() && h_ps.rank().is_static() &&
            q_ps.rank().get_length() == 4 && h_ps.rank().get_length() == 4) {
            auto snap_s = q_ps[1];
            if (desc->snapshot_max_seq > 0) {
                snap_s = desc->snapshot_max_seq;
            }
            snap_ps = {q_ps[0], snap_s, h_ps[1], h_ps[2], h_ps[3]};
        } else {
            snap_ps = ov::PartialShape::dynamic(5);
        }
        output_layouts.emplace_back(snap_ps, initial_states_layout.data_type, format::bfzyx);
    }
    return output_layouts;
}

template std::vector<layout> linear_attention_inst::calc_output_layouts<ov::PartialShape>(linear_attention_node const& node, const kernel_impl_params& impl_param);

std::string linear_attention_inst::to_string(linear_attention_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite linear_attention_info;
    linear_attention_info.add("query", node.input(0).id());
    linear_attention_info.add("key", node.input(1).id());
    linear_attention_info.add("value", node.input(2).id());
    linear_attention_info.add("g", node.input(3).id());
    linear_attention_info.add("beta", node.input(3).id());

    node_info->add("linear_attention_info", linear_attention_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

linear_attention_inst::typed_primitive_inst(network& network, linear_attention_node const& node) : parent(network, node) { }
}  // namespace cldnn
