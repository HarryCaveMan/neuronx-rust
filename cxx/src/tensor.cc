#include "tensor.h"

using std::unique_ptr;
using std::move;
using std::string;
using std::unordered_map;
using std::static_cast;

namespace neuronx_rs::data {

    TensorResult Tensor::empty(nrt_tensor_info_t *info) {
        nrt_tensor_t *handle{nullptr};
        NRT_STATUS status{nrt_tensor_allocate_empty(info->name,&handle)};
        if (status != NRT_SUCCESS) {
            return TensorResult{nullptr,static_cast<uint32_t>(status)};
        }
        unique_ptr<nrt_tensor_t, TensorHandleDestructor> handle_ptr{handle};
        string name{info->name};
        unique_ptr<uint32_t[]> shape{unique_ptr<uint32_t[]>(new uint32_t[info->ndim])};
        for (uint32_t i{0}; i < info->ndim; i++) {
            shape[i] = info->shape[i];
        }
        return TensorResult {
            unique_ptr<Tensor>(new Tensor(
                info->name,
                move(handle_ptr),
                info->size,
                move(shape),
                info->ndim,
                info->dtype
            )),
            static_cast<uint32_t>(status)
        };
    }

    uint32_t Tensor::attach_buffer(void *buffer) {
        if (!_handle) {
            return static_cast<uint32_t>(NRT_INVALID_HANDLE);
        }
        return static_cast<uint32_t>(nrt_tensor_attach_buffer(_handle.get(), buffer, _size));
    }

    TensorSetResult TensorSet::empty() {
        nrt_tensor_set_t *handle{nullptr};
        NRT_STATUS status{nrt_allocate_tensor_set(&handle)};
        if (status != NRT_SUCCESS) {
            return TensorSetResult{nullptr, static_cast<uint32_t>(status)};
        } else {
            unique_ptr<nrt_tensor_set_t, TensorSetHandleDestructor> handle_ptr{handle};
            return TensorSetResult {
                unique_ptr<TensorSet>(new TensorSet(move(handle_ptr))),
                static_cast<uint32_t>(status)
            };
        }
    }

    TensorSetResult TensorSet::empty_from_info_array(nrt_tensor_info_array_t *info_array) {
        nrt_tensor_set_t *handle{nullptr};
        NRT_STATUS status{nrt_allocate_tensor_set(&handle)};
        if (status != NRT_SUCCESS) {
            return TensorSetResult{nullptr, static_cast<uint32_t>(status)};
        }
        unique_ptr<nrt_tensor_set_t, TensorSetHandleDestructor> handle_ptr{handle};
        unordered_map<string, unique_ptr<Tensor>> tensors;
        for (uint64_t i{0}; i < info_array->tensor_count; i++) {
            nrt_tensor_info_t *info{info_array->tensor_array[i]};
            TensorResult tensor_result{Tensor::empty(info)};
            if (!tensor_result.success()) {
                return TensorSetResult{nullptr, tensor_result.status};
            }
            tensors[info->name] = tensor_result.value;
            status = nrt_add_tensor_to_tensor_set(handle_ptr.get(), info->name, tensors[info->name]->handle());
            if (status != NRT_SUCCESS) {
                return TensorSetResult{nullptr,static_cast<uint32_t>(status)};
            }
        }
        return TensorSetResult {
            unique_ptr<TensorSet>(new TensorSet(
                move(handle_ptr),
                info_array->tensor_count,
                move(tensors)
            )),
            static_cast<uint32_t>(status)
        };
    }

    uint32_t TensorSet::add(unique_ptr<Tensor> tensor) {
        NRT_STATUS status{NRT_FAILURE};
        if (!_handle || !tensor || !tensor->handle()) {
            return static_cast<uint32_t>(NRT_INVALID_HANDLE);
        }
        status = nrt_add_tensor_to_tensor_set(_handle.get(), tensor->name().c_str(), tensor->handle());
        if (status == NRT_SUCCESS) {
            _tensors[tensor->name()] = move(tensor);
            _size++;
        }
        return static_cast<uint32_t>(status);
    }

    IoTensorsResult IoTensors::empty_from_info_array(nrt_tensor_info_array_t *info_array) {
        TensorSetResult inputs_result{TensorSet::empty()};
        if(!inputs_result.success()) {
            return IoTensorsResult{nullptr, inputs_result.status};
        }
        TensorSetResult outputs_result{TensorSet::empty()};
        if(!outputs_result.success()) {
            return IoTensorsResult{nullptr, outputs_result.status};
        }
        for (uint64_t i{0}; i < info_array->tensor_count; i++) {
            nrt_tensor_info_t *info{info_array->tensor_array[i]};
            unique_ptr<Tensor> tensor{Tensor::empty(info)};
            if (info->usage == NRT_TENSOR_USAGE_INPUT) {
                inputs_result.value->add(move(tensor));
            } else {
                outputs_result.value->add(move(tensor));
            }
        }
        return IoTensorsResult {
            unique_ptr<IoTensors>(new IoTensors(move(inputs_result.value), move(outputs_result.value))),
            static_cast<uint32_t>(NRT_SUCCESS)
        };
    }

    uint32_t IoTensors::bind(const string &name, uint32_t usage, void *buffer) {
        nrt_tensor_usage_t usage_selector{static_cast<nrt_tensor_usage_t>(usage)};
        NRT_STATUS status{NRT_FAILURE};
        if (usage_selector == NRT_TENSOR_USAGE_INPUT) {
            Tensor * tensor{_inputs->get_tensor(name)};
            return tensor ? tensor->attach_buffer(buffer) : static_cast<uint32_t>(NRT_INVALID_HANDLE);
        } else if (usage_selector == NRT_TENSOR_USAGE_OUTPUT) {
            Tensor * tensor{_outputs->get_tensor(name)};
            return tensor ? tensor->attach_buffer(buffer) : static_cast<uint32_t>(NRT_INVALID_HANDLE);
        }
        else {
            return static_cast<uint32_t>(NRT_INVALID);
        }
    }
} // namespace neuronx_rs::data