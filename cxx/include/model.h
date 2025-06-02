#pragma once

#include <nrt/nrt.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <string>
#include "tensor.h"
#include "rust/cxx.h"

using std::unique_ptr;
using std::move;
using std::unordered_map;
using std::string;
using neuron_rs::data::IoTensors;

namespace neuron_rs::model {

    class Model {
        public:
            static unique_ptr<Model> from_neff_file(
                const std::string &path,
                int32_t start_nc = -1,
                int32_t nc_count = -1
            );            
            static unique_ptr<Model> from_neff_buffer(
                const void *data,
                size_t size,
                int32_t start_nc = -1,
                int32_t nc_count = -1
            );
            nrt_model_t* handle() const { return _handle.get(); }
            uint32_t bind(const string &name,nrt_tensor_usage_t usage, void *buffer);
            uint32_t execute();

        private:
            struct ModelHandleDestructor {
                void operator()(nrt_model_t *handle) const {
                    if (handle) nrt_unload(handle);
                }
            };

            Model(
                unique_ptr<nrt_model_t,ModelHandleDestructor> handle,
                unique_ptr<IoTensors> io_tensors
            )
            : _handle{move(handle)}, _io_tensors{move(io_tensors)} {}

            unique_ptr<nrt_model_t,ModelHandleDestructor> _handle;
            unique_ptr<IoTensors> _io_tensors;
    };
}