#pragma once

#include <nrt/nrt.h>       // For NRT types and functions
#include <cstdint>         // For int32_t, uint32_t
#include <memory>          // For unique_ptr
#include <string>          // For string
#include "tensor.h"        // For IoTensors
#include "rust/cxx.h"      // For rust::Str, rust::Slice

namespace neuronx_rs::model {

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
            uint32_t bind(const string &name, uint32_t usage, void *buffer);

            //rust ffi exposed methods
            static unique_ptr<Model> load(
                const rust::Str path,
                int32_t start_nc = -1,
                int32_t nc_count = -1
            ) {
                return from_neff_file(path, start_nc, nc_count);
            }            
            uint32_t bind_slice(const string &name, uint32_t usage,rust::Slice<uint32_t> slice) {
                return bind(name, usage, static_cast<void*>(slice.data()));
            }
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