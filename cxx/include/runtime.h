#pragma once

#include <nrt/nrt.h>
#include <cstdint>  // For uint32_t

using std::static_cast;

namespace neuronx_rs::runtime {
    // Initialize neuron runtime
    // Returns NRT status code (0 for success)
    uint32_t neuronx_init() {return static_cast<uint32_t>(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW,"",""));}
    // Close neuron runtime
    // Returns NRT status code (0 for success)
    uint32_t neuronx_close() {return static_cast<uint32_t>(nrt_close());}
}