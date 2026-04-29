// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "linear_attention_inst.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct LinearAttentionOpt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::linear_attention::opt")
    explicit LinearAttentionOpt(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<linear_attention>());

        // Only activate when env var is set
        static const bool opt_enabled = [] {
            const char* env = std::getenv("OV_GENAI_USE_LA_OPT");
            return env && std::string(env) == "1";
        }();
        if (!opt_enabled) {
            return false;
        }

        // Only supports K_HEAD_DIMS=128
        const auto& q_layout = node.get_input_layout(0);
        const auto& q_shape = q_layout.get_partial_shape();
        if (q_shape.rank().get_length() < 4 || q_shape[3].is_dynamic() || q_shape[3].get_length() != 128) {
            return false;
        }

        static constexpr std::array supported_fmts = {
            format::bfyx,
        };

        static constexpr std::array supported_types = {
            ov::element::f16,
            ov::element::f32,
        };

        for (size_t i = 0; i < node.get_dependencies().size(); i++) {
            const auto& in_layout = node.get_input_layout(i);
            if (!one_of(in_layout.format, supported_fmts) || !one_of(in_layout.data_type, supported_types)) {
                return false;
            }
        }

        const auto& out_layout = node.get_output_layout(0);
        if (!one_of(out_layout.format, supported_fmts) || !one_of(out_layout.data_type, supported_types)) {
            return false;
        }
        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
