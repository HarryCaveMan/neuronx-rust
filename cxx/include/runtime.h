#pragma once

#include <nrt/nrt.h>
#include <cstdint>  // For uint32_t
#include <stdexcept>  // For std::runtime_error
#include <string>  // For std::string
#include <memory>  // For std::unique_ptr
#include "rust/cxx.h"  // For rust::Str, rust::Slice

using std::static_cast;
using std::string;
using std::runtime_error;
using std::unique_ptr;

namespace neuronx_rs::runtime {

    struct NrtVersionResult {
        NrtVersion value;
        uint32_t status;
        bool success() const {return status == static_cast<uint32_t>(NRT_SUCCESS);}
    };

    struct Uint32Result {
        uint32_t value;
        uint32_t status;
        bool success() const {return status == static_cast<uint32_t>(NRT_SUCCESS);}
    };

    class NrtVersion {
        public:
            static NrtVersionResult get() {
                nrt_version_t version;
                NRT_STATUS status = nrt_get_version(&version);                
                return NrtVersionResult{
                    NrtVersion(version),
                    static_cast<uint32_t>(status)
                };
            }           
            uint64_t major() const { return _version.rt_major; }
            uint64_t minor() const { return _version.rt_minor; }
            uint64_t patch() const { return _version.rt_patch; }
            uint64_t maintenance() const { return _version.rt_maintenance; }
            string detail() const { return string(_version.rt_detail); }
            string git_hash() const { return string(_version.git_hash); }
            
        private:
            NrtVersion(nrt_version_t version) : _version(version) {}
            nrt_version_t _version;
    };
    // Initialize neuron runtime
    // Returns NRT status code (0 for success)
    inline uint32_t neuronx_init() {return static_cast<uint32_t>(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW,"",""));}
    // Close neuron runtime
    // Returns NRT status code (0 for success)
    inline uint32_t neuronx_close() {return static_cast<uint32_t>(nrt_close());}
    // Get the version of the neuron runtime
    inline NrtVersionResult neuronx_version() {return NrtVersion::get();}
    // Get the number of neuron cores on this hardware
    inline Uint32Result neuronx_get_nc_count() {
        uint32_t count{0};
        NRT_STATUS status = nrt_get_nc_count(&count);
        return Uint32Result{count, static_cast<uint32_t>(status)};
    }
    // Get the number of visible neuron cores
    inline Uint32Result neuronx_get_visible_nc_count() {
        uint32_t count{0};
        NRT_STATUS status = nrt_get_visible_nc_count(&count);
        return Uint32Result{count, static_cast<uint32_t>(status)};
    }
}