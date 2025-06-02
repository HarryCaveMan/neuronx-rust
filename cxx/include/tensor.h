#pragma once

#include <nrt/nrt.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <string>
#include <stdexcept>

using std::unique_ptr;
using std::move;
using std::string;
using std::unordered_map;


namespace neuron_rs::data {

    class Tensor {
        public:
            static unique_ptr<Tensor> empty(nrt_tensor_info_t *info);
            const string& name() const { return _name; }
            nrt_tensor_t* handle() const { return _handle.get(); }
            size_t size() const { return _size; }
            const uint32_t* shape() const {
                if (!_shape) {
                    throw std::runtime_error("Tensor shape is not initialized or has been invalidated");
                }
                return _shape.get();
            }
            uint32_t ndim() const { return _ndim; }
            nrt_dtype_t dtype() const { return _dtype; }
            uint32_t attach_buffer(void *buffer);
            
            
        private:
            struct TensorHandleDestructor {
                void operator()(nrt_tensor_t *handle) const {
                    if (handle) nrt_free_tensor(&handle);
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
            static unique_ptr<TensorSet> empty();
            static unique_ptr<TensorSet> empty_from_info_array(nrt_tensor_info_array_t *info_array);
            nrt_tensor_set_t* handle() const { return _handle.get(); }
            size_t size() const { return _size; }
            Tensor* get_tensor(const string &name) const {
                auto it = _tensors.find(name);
                return (it != _tensors.end()) ? it->second.get() : nullptr;
            }
            NRT_STATUS add(unique_ptr<Tensor> tensor);
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
            static unique_ptr<IoTensors> empty_from_info_array(nrt_tensor_info_array_t *info_array);
            TensorSet* inputs() const { return _inputs.get(); }
            TensorSet* outputs() const { return _outputs.get(); }

        private:
            IoTensors(unique_ptr<TensorSet> inputs, unique_ptr<TensorSet> outputs)
                : _inputs(move(inputs)), _outputs(move(outputs)) {}
            unique_ptr<TensorSet> _inputs;
            unique_ptr<TensorSet> _outputs;
    };
}