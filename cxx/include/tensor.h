#pragma once

#include <nrt/nrt.h>              // nrt_tensor_allocate_empty,nrt_tensor_free,nrt_tensor_attach_buffer
                                  // nrt_allocate_tensor_set,nrt_destroy_tensor_set
#include <nrt/nrt_experimental.h> // nrt_tensor_info_t,nrt_tensor_info_array_t,nrt_tensor_t,nrt_tensor_set_t
#include <cstdint>                // uint32_t
#include <memory>                 // unique_ptr
#include <string>                 // string
#include <unordered_map>          // unordered_map
#include <vector>                // vector
#include "rust/cxx.h"             // rust::Slice

using std::string;
using std::unique_ptr;
using std::unordered_map;
using std::vector;
using std::move;

namespace neuronx_rs::data {

    struct TensorInfoResult {
        unique_ptr<TensorInfo> value;
        uint32_t status;
        bool success() const { return status == static_cast<uint32_t>(NRT_SUCCESS); }
    };
    struct TensorResult {
        unique_ptr<Tensor> value;
        uint32_t status;
        bool success() const { return status == static_cast<uint32_t>(NRT_SUCCESS); }
    };
    struct TensorSetResult {
        unique_ptr<TensorSet> value;
        uint32_t status;
        bool success() const { return status == static_cast<uint32_t>(NRT_SUCCESS); }
    };
    struct IoTensorsResult {
        unique_ptr<IoTensors> value;
        uint32_t status;
        bool success() const { return status == static_cast<uint32_t>(NRT_SUCCESS); }
    };

    class TensorInfo {
        public:
            const string& name() const { return _name; }
            size_t size() const { return _size; }
            uint32_t usage() const { return static_cast<uint32_t>(_usage); }
            uint32_t dtype() const { return static_cast<uint32_t>(_dtype); }
            uint32_t ndim() const { return _ndim; }
            const uint32_t* shape() const { return _shape.get(); }
            const rust::Slice<uint32_t> shape_slice() const {return rust::Slice<uint32_t>(_shape.get(), _ndim);}
            // Factory method to create from nrt_tensor_info_t
            static TensorInfoResult from_nrt_tensor_info_array(const nrt_tensor_info_t *info) {
                if(!info) return TensorInfoResult(nullptr, static_cast<uint32_t>(NRT_INVALID_HANDLE));
                unique_ptr<uint32_t[]> shape{unique_ptr<uint32_t[]>(new uint32_t[info->ndim])};
                for (uint32_t i{0}; i < info->ndim; i++) {
                    shape[i] = info->shape[i];
                }
                return TensorInfoResult{
                    unique_ptr<TensorInfo>(
                        new TensorInfo(
                            info->name,
                            info->size,
                            info->usage,
                            info->dtype,
                            info->ndim,
                            move(shape)
                        )
                    ),
                    static_cast<uint32_t>(NRT_SUCCESS)
                };
            }
            static TensorInfoResult from_tensor(Tensor *tensor) {
                if (!tensor) return TensorInfoResult(nullptr, static_cast<uint32_t>(NRT_INVALID_HANDLE));
                unique_ptr<uint32_t[]> shape{unique_ptr<uint32_t[]>(new uint32_t[tensor->ndim()])};
                for (uint32_t i{0}; i < tensor->ndim(); i++) {
                    shape[i] = tensor->shape()[i];
                }
                return TensorInfoResult{
                    unique_ptr<TensorInfo>(
                        new TensorInfo(
                            tensor->name(),
                            tensor->size(),
                            NRT_TENSOR_USAGE_INPUT, // Default usage, can be changed later
                            tensor->dtype(),
                            tensor->ndim(),
                            move(shape)
                        )
                    ),
                    static_cast<uint32_t>(NRT_SUCCESS)
                };
            }
        private:
            TensorInfo(
                const string &name,
                size_t size,
                nrt_tensor_usage_t usage,
                nrt_dtype_t dtype,
                uint32_t ndim,
                unique_ptr<uint32_t[]> shape
            ) : _name(name), _size(size), _usage(usage), _dtype(dtype), _ndim(ndim), _shape(move(shape)) {}

            string _name;
            size_t _size;
            nrt_tensor_usage_t _usage;
            nrt_dtype_t _dtype;
            uint32_t _ndim;
            unique_ptr<uint32_t[]> _shape;
            
    };

    class Tensor {
        public:
            static TensorResult empty(nrt_tensor_info_t *info);
            const string& name() const { return _name; }
            nrt_tensor_t* handle() const {return _handle.get();}
            size_t size() const { return _size; }
            const uint32_t* shape() const {return _shape.get();}
            uint32_t ndim() const { return _ndim; }
            nrt_dtype_t dtype() const { return _dtype; }
            uint32_t attach_buffer(void *buffer);            
            
        private:

            struct TensorHandleDestructor {
                void operator()(nrt_tensor_t *handle) const {
                    if (handle) nrt_tensor_free(&handle);
                }
            };

            Tensor(
                const string name,
                unique_ptr<nrt_tensor_t, TensorHandleDestructor> handle,
                size_t size, 
                unique_ptr<uint32_t[]> shape,
                uint32_t ndim,
                nrt_dtype_t dtype
            ) : _name(name), _handle(move(handle)), _size(size), 
                _shape(move(shape)), _ndim(ndim), _dtype(dtype) {}
            
            string _name;
            unique_ptr<nrt_tensor_t, TensorHandleDestructor> _handle;
            size_t _size{0};
            unique_ptr<uint32_t[]> _shape{nullptr};
            uint32_t _ndim{0};
            nrt_dtype_t _dtype{NRT_DTYPE_UNKNOWN};
    };

    class TensorSet {
        public:
            TensorSetResult empty();
            TensorSetResult empty_from_info_array(nrt_tensor_info_array_t *info_array);
            nrt_tensor_set_t* handle() const { return _handle.get(); }
            size_t size() const { return _size; }
            Tensor* get_tensor(const string &name) const {
                auto it = _tensors.find(name);
                return (it != _tensors.end()) ? it->second.get() : nullptr;
            }
            uint32_t add(unique_ptr<Tensor> tensor);
        private:
            struct TensorSetHandleDestructor {
                void operator()(nrt_tensor_set_t *handle) const {
                    if (handle) nrt_destroy_tensor_set(&handle);
                }
            };
            TensorSet(
                unique_ptr<nrt_tensor_set_t, TensorSetHandleDestructor> handle,
                size_t size=0,
                unordered_map<string,unique_ptr<Tensor>> tensors= unordered_map<string,unique_ptr<Tensor>>{}
                
            ) : _handle{move(handle)},_size{size},_tensors{tensors} {}

            size_t _size;
            unordered_map<string,unique_ptr<Tensor>> _tensors;
            unique_ptr<nrt_tensor_set_t, TensorSetHandleDestructor> _handle;
    };

    class IoTensors {
        public:
            static IoTensorsResult empty_from_info_array(nrt_tensor_info_array_t *info_array);
            TensorSet* inputs() const { return _inputs.get(); }
            TensorSet* outputs() const { return _outputs.get(); }

            // Public buffer mutation methods only accepts sized types
            // Interface currently supports (via overloading):
            // - std::vector<uint8_t>
            // - rust::Slice<uint8_t>

            unique_ptr<TensorInfo[]> get_input_tensor_info();
            unique_ptr<TensorInfo[]> get_output_tensor_info();
            rust::Slice<TensorInfo> get_input_tensor_info_slice() {
                unique_ptr<TensorInfo[]> info{get_input_tensor_info()};
                return rust::Slice<TensorInfo>(info.get(), _inputs->size());
            }
            rust::Slice<TensorInfo> get_output_tensor_info_slice() {
                unique_ptr<TensorInfo[]> info{get_output_tensor_info()};
                return rust::Slice<TensorInfo>(info.get(), _outputs->size());
            }
            uint32_t bind(const string &name, uint32_t usage, vector<uint8_t> &buffer) {
                if (buffer.empty()) {
                    return static_cast<uint32_t>(NRT_INVALID_HANDLE);
                }
                return bind(name, usage, static_cast<void*>(buffer.data()), buffer.size());
            }
            
            // Rust FFI exposed methods
            uint32_t bind(const string &name, uint32_t usage, rust::Slice<uint8_t> buffer) {
                return bind(name, usage, static_cast<void*>(buffer.data()), buffer.size());
            }

        private:
            uint32_t bind(const string &name, uint32_t usage, void *buffer, size_t size);
            IoTensors(unique_ptr<TensorSet> inputs, unique_ptr<TensorSet> outputs)
                : _inputs(move(inputs)), _outputs(move(outputs)) {}
            unique_ptr<TensorSet> _inputs;
            unique_ptr<TensorSet> _outputs;
    };
}