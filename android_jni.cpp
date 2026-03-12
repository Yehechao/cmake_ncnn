#include <jni.h>

#include <vector>

#include "android_api.h"

#ifndef NCNN_API_JNI_CLASS_NAME
#define NCNN_API_JNI_CLASS_NAME "matrix_ncnn/app/NcnnApi"
#endif

namespace {

jstring nativeRunObb(JNIEnv* env, jobject /* thiz */, jfloatArray flat_data, jint rows, jint cols) {
    if (!flat_data) {
        return env->NewStringUTF("{\"success\":false,\"error\":\"flat_data is null\"}");
    }

    const jsize len = env->GetArrayLength(flat_data);
    const jsize required = rows * cols;
    if (len < required) {
        return env->NewStringUTF("{\"success\":false,\"error\":\"input data length is smaller than rows*cols\"}");
    }

    std::vector<float> buffer(static_cast<size_t>(required));
    env->GetFloatArrayRegion(flat_data, 0, required, buffer.data());
    const char* json = ncnnapi_run_obb(buffer.data(), rows, cols);
    return env->NewStringUTF(json ? json : "{\"success\":false,\"error\":\"native returned null\"}");
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
    {"nativeRunObb", "([FII)Ljava/lang/String;", reinterpret_cast<void*>(nativeRunObb)},
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
