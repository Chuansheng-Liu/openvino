// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides ov::VariableState.
 * @file openvino/runtime/variable_state.hpp
 */

#pragma once

#include <memory>
#include <string>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {

class InferRequest;
class IVariableState;

/**
 * @brief VariableState class
 * @ingroup ov_runtime_cpp_api
 */
class OPENVINO_RUNTIME_API VariableState {
    std::shared_ptr<ov::IVariableState> _impl;
    std::shared_ptr<void> _so;

    /**
     * @brief Constructs VariableState from the initialized std::shared_ptr.
     * @param impl Initialized shared pointer.
     * @param so Optional: plugin to use. This is required to ensure that VariableState can work properly even if a
     * plugin object is destroyed.
     */
    VariableState(const std::shared_ptr<ov::IVariableState>& impl, const std::shared_ptr<void>& so);

    friend class ov::InferRequest;

public:
    /**
     * @brief Default constructor.
     */
    VariableState() = default;

    /**
     * @brief Destructor that preserves unloading order of implementation object and reference to the library.
     */
    ~VariableState();

    /**
     * @brief Resets internal variable state for relevant infer request
     * to a value specified as default for the corresponding ReadValue node.
     */
    void reset();

    /**
     * @brief Gets the name of the current variable state. If length of an array is not enough, the name is truncated by
     * len, null terminator is inserted as well. `variable_id` from the corresponding `ReadValue` is used as variable
     * state name.
     * @return A string representing state name.
     */
    std::string get_name() const;

    /**
     * @brief Returns the value of the variable state.
     * @return A tensor representing a state.
     */
    Tensor get_state() const;

    /**
     * @brief Sets the new state for the next inference.
     * @param state The current state to set.
     */
    void set_state(const Tensor& state);

    /**
     * @brief Returns the shape of the variable state.
     * On GPU devices, avoids the expensive device-to-host data copy
     * that get_state() would trigger.
     * @return The shape of the state tensor.
     */
    Shape get_shape() const;

    /**
     * @brief Sets the shape of the variable state in-place.
     * On GPU devices with pre-allocated buffers, this is a zero-copy operation
     * that reinterprets the existing device buffer with the new shape.
     * Useful for efficiently trimming KV cache in speculative decoding.
     * @param shape The new shape. Must not exceed the current buffer allocation.
     */
    void set_shape(const Shape& shape);
};

}  // namespace ov
