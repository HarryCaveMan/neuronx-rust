#pragma once

#include <nrt/nrt.h>              // nrt_tensor_allocate_empty,nrt_tensor_free,nrt_tensor_attach_buffer
                                  // nrt_allocate_tensor_set,nrt_destroy_tensor_set
#include <nrt/nrt_experimental.h> // nrt_tensor_info_t,nrt_tensor_info_array_t,nrt_tensor_t,nrt_tensor_set_t
#include <cstdint>                // uint32_t
#include <memory>                 // unique_ptr
#include <string>                 // string
#include <unordered_map>          // unordered_map
#include "rust/cxx.h"             // rust::Slice

using std::string;
using std::unique_ptr;
using std::unordered_map;
using std::move;

namespace neuronx_rs::data {

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
            uint32_t bind(const string &name, uint32_t usage, void *buffer);
            // Rust FFI exposed methods
            uint32_t bind_slice(const string &name, uint32_t usage, rust::Slice<uint8_t> slice) {
                return bind(name, usage, static_cast<void*>(slice.data()));
            }

        private:
            IoTensors(unique_ptr<TensorSet> inputs, unique_ptr<TensorSet> outputs)
                : _inputs(move(inputs)), _outputs(move(outputs)) {}
            unique_ptr<TensorSet> _inputs;
            unique_ptr<TensorSet> _outputs;
    };
}