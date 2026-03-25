// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/ivariable_state.hpp"

#include "openvino/core/except.hpp"
#include "openvino/runtime/itensor.hpp"

ov::IVariableState::IVariableState(const std::string& name) : m_name(name) {}

ov::IVariableState::~IVariableState() = default;

const std::string& ov::IVariableState::get_name() const {
    return m_name;
}

void ov::IVariableState::reset() {
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::IVariableState::set_state(const ov::SoPtr<ov::ITensor>& state) {
    m_state = state;
}

ov::SoPtr<ov::ITensor> ov::IVariableState::get_state() const {
    return m_state;
}

ov::Shape ov::IVariableState::get_shape() const {
    auto state = get_state();
    if (state) {
        return state->get_shape();
    }
    return {};
}

void ov::IVariableState::set_shape(const ov::Shape& /*shape*/) {
    OPENVINO_NOT_IMPLEMENTED;
}
