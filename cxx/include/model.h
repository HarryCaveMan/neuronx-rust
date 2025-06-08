#pragma once

#include <nrt/nrt.h>       // For NRT types and functions
#include <cstdint>         // For int32_t, uint32_t
#include <memory>          // For unique_ptr
#include <string>          // For string
#include "rust/cxx.h"      // For rust::Str, rust::Slice
#include "tensor.h"      // For IoTensors, IoTensorsResult

using std::unique_ptr;
using std::move;
using std::string;
using std::static_cast;
using neuronx_rs::data::IoTensors;
using neuronx_rs::data::IoTensorsResult;

namespace neuronx_rs::model {

    struct ModelResult {
        unique_ptr<Model> value;
        uint32_t status;
        bool success() const { return status == static_cast<uint32_t>(NRT_SUCCESS); }
    };

    class Model {
        public:
            static ModelResult from_neff_file(
                const std::string &path,
                int32_t start_nc = -1,
                int32_t nc_count = -1
            );            
            static ModelResult from_neff_buffer(
                const void *data,
                size_t size,
                int32_t start_nc = -1,
                int32_t nc_count = -1
            );
            nrt_model_t* handle() const { return _handle.get(); }

            //rust ffi exposed methods
            static ModelResult load(
                const rust::Str path,
                int32_t start_nc = -1,
                int32_t nc_count = -1
            ) {
                return from_neff_file(path, start_nc, nc_count);
            }
            IoTensorsResult get_new_io_tensors();
            uint32_t execute(IoTensors *io_tensors);

        private:
            struct ModelHandleDestructor {
                void operator()(nrt_model_t *handle) const {
                    if (handle) nrt_unload(handle);
                }
            };

            Model(
                unique_ptr<nrt_model_t,ModelHandleDestructor> handle
            )
            : _handle{move(handle)} {}

            unique_ptr<nrt_model_t,ModelHandleDestructor> _handle;
    };
}