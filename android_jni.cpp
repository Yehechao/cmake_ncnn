#include <jni.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "android_api.h"

#ifndef NCNN_API_JNI_CLASS_NAME
#define NCNN_API_JNI_CLASS_NAME "matrix_ncnn/app/NcnnApi"
#endif

namespace {

thread_local std::vector<float> g_input_buffer;
thread_local std::vector<float> g_output_buffer;
thread_local std::vector<jfloat> g_packed_buffer;

// 结构化 FloatArray 接口：
// 返回 [success, count, det0(7), det1(7), ...]
jfloatArray nativeRunObb(JNIEnv* env, jobject /* thiz */, jfloatArray flat_data, jint rows, jint cols) {
    if (!flat_data || rows <= 0 || cols <= 0) {
        jfloatArray result = env->NewFloatArray(NCNNAPI_OBB_HEADER_FIELDS);
        if (result) {
            const jfloat header[NCNNAPI_OBB_HEADER_FIELDS] = {0.0f, 0.0f};
            env->SetFloatArrayRegion(result, 0, NCNNAPI_OBB_HEADER_FIELDS, header);
        }
        return result;
    }

    const jsize required = rows * cols;
    if (env->GetArrayLength(flat_data) < required) {
        jfloatArray result = env->NewFloatArray(NCNNAPI_OBB_HEADER_FIELDS);
        if (result) {
            const jfloat header[NCNNAPI_OBB_HEADER_FIELDS] = {0.0f, 0.0f};
            env->SetFloatArrayRegion(result, 0, NCNNAPI_OBB_HEADER_FIELDS, header);
        }
        return result;
    }

    if (g_input_buffer.size() < static_cast<size_t>(required)) {
        g_input_buffer.resize(static_cast<size_t>(required));
    }

    jfloat* critical = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(flat_data, nullptr));
    if (critical) {
        std::memcpy(g_input_buffer.data(), critical, static_cast<size_t>(required) * sizeof(float));
        env->ReleasePrimitiveArrayCritical(flat_data, critical, JNI_ABORT);
    } else {
        env->GetFloatArrayRegion(flat_data, 0, required, g_input_buffer.data());
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            jfloatArray result = env->NewFloatArray(NCNNAPI_OBB_HEADER_FIELDS);
            if (result) {
                const jfloat header[NCNNAPI_OBB_HEADER_FIELDS] = {0.0f, 0.0f};
                env->SetFloatArrayRegion(result, 0, NCNNAPI_OBB_HEADER_FIELDS, header);
            }
            return result;
        }
    }

    // 这里是输出上限，不是类别上限；仅用于保护 JNI 缓冲写入边界
    const int maxDetections = 64;
    const int payloadSize = maxDetections * NCNNAPI_OBB_FIELDS_PER_DET;
    if (g_output_buffer.size() < static_cast<size_t>(payloadSize)) {
        g_output_buffer.resize(static_cast<size_t>(payloadSize));
    }

    int outCount = 0;
    const bool success = ncnnapi_run_obb_struct(
        g_input_buffer.data(),
        rows,
        cols,
        g_output_buffer.data(),
        maxDetections,
        &outCount
    );

    const int safeCount = std::max(0, std::min(maxDetections, outCount));
    const int outLen = NCNNAPI_OBB_HEADER_FIELDS + safeCount * NCNNAPI_OBB_FIELDS_PER_DET;
    jfloatArray result = env->NewFloatArray(outLen);
    if (!result) {
        return nullptr;
    }

    if (g_packed_buffer.size() < static_cast<size_t>(outLen)) {
        g_packed_buffer.resize(static_cast<size_t>(outLen));
    }

    g_packed_buffer[0] = success ? 1.0f : 0.0f;
    g_packed_buffer[1] = static_cast<jfloat>(safeCount);
    if (safeCount > 0) {
        std::memcpy(
            g_packed_buffer.data() + NCNNAPI_OBB_HEADER_FIELDS,
            g_output_buffer.data(),
            static_cast<size_t>(safeCount * NCNNAPI_OBB_FIELDS_PER_DET) * sizeof(jfloat)
        );
    }

    env->SetFloatArrayRegion(result, 0, outLen, g_packed_buffer.data());
    return result;
}

jboolean nativeLoadObbModel(JNIEnv* env,
                            jobject /* thiz */,
                            jstring param_path,
                            jstring bin_path,
                            jint size,
                            jfloat conf,
                            jfloat iou,
                            jboolean use_gpu) {
    if (!param_path || !bin_path) {
        return JNI_FALSE;
    }

    const char* param_utf = env->GetStringUTFChars(param_path, nullptr);
    const char* bin_utf = env->GetStringUTFChars(bin_path, nullptr);
    const bool ok = ncnnapi_load_obb_model(param_utf, bin_utf, size, conf, iou, use_gpu == JNI_TRUE, -1);
    env->ReleaseStringUTFChars(param_path, param_utf);
    env->ReleaseStringUTFChars(bin_path, bin_utf);
    return ok ? JNI_TRUE : JNI_FALSE;
}

jboolean nativeLoadObbModelWithThreads(JNIEnv* env,
                                       jobject /* thiz */,
                                       jstring param_path,
                                       jstring bin_path,
                                       jint size,
                                       jfloat conf,
                                       jfloat iou,
                                       jboolean use_gpu,
                                       jint num_threads) {
    if (!param_path || !bin_path) {
        return JNI_FALSE;
    }

    const char* param_utf = env->GetStringUTFChars(param_path, nullptr);
    const char* bin_utf = env->GetStringUTFChars(bin_path, nullptr);
    const bool ok = ncnnapi_load_obb_model(param_utf, bin_utf, size, conf, iou, use_gpu == JNI_TRUE, num_threads);
    env->ReleaseStringUTFChars(param_path, param_utf);
    env->ReleaseStringUTFChars(bin_path, bin_utf);
    return ok ? JNI_TRUE : JNI_FALSE;
}

void nativeRelease(JNIEnv* /* env */, jobject /* thiz */) {
    ncnnapi_release();
}

jboolean nativeIsGpuActive(JNIEnv* /* env */, jobject /* thiz */) {
    return isGpuActive() ? JNI_TRUE : JNI_FALSE;
}

static const JNINativeMethod kMethods[] = {
    {"nativeLoadObbModel", "(Ljava/lang/String;Ljava/lang/String;IFFZ)Z", reinterpret_cast<void*>(nativeLoadObbModel)},
    {"nativeLoadObbModel", "(Ljava/lang/String;Ljava/lang/String;IFFZI)Z", reinterpret_cast<void*>(nativeLoadObbModelWithThreads)},
    {"nativeRunObb", "([FII)[F", reinterpret_cast<void*>(nativeRunObb)},
    {"isGpuActive", "()Z", reinterpret_cast<void*>(nativeIsGpuActive)},
    {"nativeRelease", "()V", reinterpret_cast<void*>(nativeRelease)},
};

} // namespace

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* /* reserved */) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }

    jclass clazz = env->FindClass(NCNN_API_JNI_CLASS_NAME);
    if (!clazz) {
        return JNI_ERR;
    }

    if (env->RegisterNatives(clazz, kMethods, sizeof(kMethods) / sizeof(kMethods[0])) != JNI_OK) {
        env->DeleteLocalRef(clazz);
        return JNI_ERR;
    }

    env->DeleteLocalRef(clazz);
    return JNI_VERSION_1_6;
}
