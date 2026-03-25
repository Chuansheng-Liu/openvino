// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime IVariableState interface
 * @file openvino/runtime/ivariable_state.hpp
 */

#pragma once

#include <memory>
#include <string>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {

/**
 * @interface IVariableState
 * @brief Minimal interface for variable state implementation
 * @ingroup ov_dev_api_variable_state_api
 */
class OPENVINO_RUNTIME_API IVariableState : public std::enable_shared_from_this<IVariableState> {
public:
    explicit IVariableState(const std::string& name);

    /**
     * @brief Gets a variable state name
     * @return A string representing variable state name
     */
    virtual const std::string& get_name() const;

    /**
     * @brief Reset internal variable state for relevant infer request, to a value specified as
     * default for according `ReadValue` node
     */
    virtual void reset();

    /**
     * @brief Sets the new state for the next inference
     * @param newState A new state
     */
    virtual void set_state(const ov::SoPtr<ov::ITensor>& state);

    /**
     * @brief Returns the value of the variable state.
     * @return The value of the variable state
     */
    virtual ov::SoPtr<ov::ITensor> get_state() const;

    /**
     * @brief Returns the shape of the variable state without necessarily copying state data.
     * GPU-accelerated implementations return the shape from device-side metadata,
     * avoiding an expensive device-to-host data transfer.
     * @return The shape of the variable state tensor
     */
    virtual ov::Shape get_shape() const;

    /**
     * @brief Sets the shape of the variable state in-place.
     * For GPU devices with pre-allocated buffers, this is a zero-copy operation
     * that reinterprets the existing device buffer with the new shape.
     * Useful for efficiently trimming KV cache tails in speculative decoding.
     * @param shape The new shape. The total byte size must not exceed the current allocation.
     * @note Default implementation throws ov::NotImplemented.
     *       GPU plugin overrides this for zero-copy reshape.
     */
    virtual void set_shape(const ov::Shape& shape);

protected:
    /**
     * @brief A default dtor
     */
    virtual ~IVariableState();

    std::string m_name;
    ov::SoPtr<ov::ITensor> m_state;
};

}  // namespace ov
