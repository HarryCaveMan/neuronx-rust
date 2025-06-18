/*
 * Copyright 2021, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */

#pragma once

#include "nrt/nrt.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Enable profiling for a model
 *
 * @param model[in]     - model to profile
 * @param filename[in]  - file to save profile information to
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_profile_start(nrt_model_t *model, const char *filename);

/** Collect results and disable profiling for a model
 *
 * @param filename[in] - file that contains profile information from nrt_profile_start
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_profile_stop(const char *filename);

/* Begin tracing/profiling
 *
 * Users of this API must set options through environment variables:
 * 
 * - NEURON_RT_INSPECT_ENABLE: Set to 1 to enable system and device profiles.
 *   For control over which profile types are captured, use NEURON_RT_INSPECT_SYSTEM_PROFILE 
 *   and NEURON_RT_INSPECT_DEVICE_PROFILE.
 * - NEURON_RT_INSPECT_OUTPUT_DIR: The directory where captured profile data will be saved to.
 *   Defaults to ./output.
 * - NEURON_RT_INSPECT_SYSTEM_PROFILE: Set to 0 to disable the capture of system profiles. 
 *   Defaults to 1 when NEURON_RT_INSPECT_ENABLE is set to 1.
 * - NEURON_RT_INSPECT_DEVICE_PROFILE: Set to 0 to disable the capture of device profiles.
 *   Defaults to 1 when NEURON_RT_INSPECT_ENABLE is set to 1.
 * - NEURON_RT_INSPECT_ON_FAIL: Set to 1 to enable dumping of device profiles in case of an error 
 *   during graph execution. Defaults to 0.
 * 
 * @return NRT_SUCCESS on success
 */
NRT_STATUS nrt_inspect_begin();


/* Stop tracing/profiling and dump profile data.
 * Does nothing if `duration` is given to nrt_inspect_begin() and already elapsed
 *
 * @return NRT_SUCCESS on success
 */
NRT_STATUS nrt_inspect_stop();

#ifdef __cplusplus
}
#endif
