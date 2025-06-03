#include "tensor.h"
#include <stdexcept>       // For runtime_error

using std::unique_ptr;
using std::move;
using std::runtime_error;
using std::static_cast;
using std::string;
using std::unordered_map;

namespace neuronx_rs::data {

    unique_ptr<Tensor> Tensor::empty(nrt_tensor_info_t *info) {
        nrt_tensor_t *handle = nullptr;
        NRT_STATUS status = nrt_tensor_allocate_empty(info->name,&handle);
        if (status != NRT_SUCCESS) {
            throw runtime_error("Failed to create empty tensor");
        }
        unique_ptr<nrt_tensor_t, TensorHandleDestructor> handle_ptr{handle};
        string name{info->name};
        unique_ptr<uint32_t[]> shape = unique_ptr<uint32_t[]>(new uint32_t[info->ndim]);
        for (uint32_t i = 0; i < info->ndim; ++i) {
            shape[i] = info->shape[i];
        }
        return unique_ptr<Tensor>(new Tensor(
            info->name,
            move(handle_ptr),
            info->size,
            move(shape),
            info->ndim,
            info->dtype
        ));
    }

    uint32_t Tensor::attach_buffer(void *buffer) {
        if (!_handle) {
            throw runtime_error("Tensor handle is not initialized");
        }
        return static_cast<uint32_t>(nrt_tensor_attach_buffer(_handle.get(), buffer, _size));
    }

    unique_ptr<TensorSet> TensorSet::empty() {
        nrt_tensor_set_t *handle = nullptr;
        NRT_STATUS status = nrt_allocate_tensor_set(&handle);
        if (status != NRT_SUCCESS) {
            throw runtime_error("Failed to create empty tensor set");
        }
        unique_ptr<nrt_tensor_set_t, TensorSetHandleDestructor> handle_ptr{handle};
        return unique_ptr<TensorSet>(new TensorSet(move(handle_ptr)));
    }

    unique_ptr<TensorSet> TensorSet::empty_from_info_array(nrt_tensor_info_array_t *info_array) {
        nrt_tensor_set_t *handle = nullptr;
        NRT_STATUS status = nrt_allocate_tensor_set(&handle);
        if (status != NRT_SUCCESS) {
            throw runtime_error("Failed to create empty tensor set");
        }
        unique_ptr<nrt_tensor_set_t, TensorSetHandleDestructor> handle_ptr{handle};
        unordered_map<string, unique_ptr<Tensor>> tensors;
        for (uint64_t i = 0; i < info_array->tensor_count; i++) {
            nrt_tensor_info_t *info = info_array->tensor_array[i];
            tensors[info->name] = Tensor::empty(info);
            status = nrt_add_tensor_to_tensor_set(handle_ptr.get(), info->name, tensors[info->name]->handle());
            if (status != NRT_SUCCESS) {
                throw runtime_error("Failed to add tensor to tensor set");
            }
        }
        return unique_ptr<TensorSet>(new TensorSet(
            move(handle_ptr),
            info_array->tensor_count,
            move(tensors)
        ));
    }

    NRT_STATUS TensorSet::add(unique_ptr<Tensor> tensor) {
        NRT_STATUS status;
        if (!_handle || !tensor || !tensor->handle()) {
            return NRT_INVALID_HANDLE;
        }
        status = nrt_add_tensor_to_tensor_set(_handle.get(), tensor->name().c_str(), tensor->handle());
        if (status == NRT_SUCCESS) {
            _tensors[tensor->name()] = move(tensor);
            _size++;
        }
        return status;
    }

    unique_ptr<IoTensors> IoTensors::empty_from_info_array(nrt_tensor_info_array_t *info_array) {
        unique_ptr<TensorSet> inputs = TensorSet::empty();
        unique_ptr<TensorSet> outputs = TensorSet::empty();
        for (uint64_t i = 0; i < info_array->tensor_count; ++i) {
            nrt_tensor_info_t *info = info_array->tensor_array[i];
            unique_ptr<Tensor> tensor = Tensor::empty(info);
            if (info->usage == NRT_TENSOR_USAGE_INPUT) {
                inputs->add(move(tensor));
            } else {
                outputs->add(move(tensor));
            }
        }
        return unique_ptr<IoTensors>(new IoTensors(move(inputs), move(outputs)));
    }
} // namespace neuronx_rs::data