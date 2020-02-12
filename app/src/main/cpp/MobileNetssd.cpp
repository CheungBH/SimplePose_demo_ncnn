#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>

// ncnn
#include "include/opencv.h"
#include "MobileNetSSD_deploy.id.h"   //这里看成自己的id.h
#include <sys/time.h>
#include <unistd.h>
#include "include/net.h"
#include <android/log.h>
#define TAG "learn JNI"
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,TAG ,__VA_ARGS__) // 定义LOGW类型

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static ncnn::Mat ncnn_param;
static ncnn::Mat ncnn_bin;
static ncnn::Net ncnn_net;

extern "C" {


// public native boolean Init(byte[] words,byte[] param, byte[] bin);　　原函数形式（c++） 以下形式为ndk的c++形式
JNIEXPORT jboolean JNICALL
Java_com_example_cheungbh_SimplePose_SimplePose_Init(JNIEnv *env, jobject obj, jbyteArray param, jbyteArray bin) {
    __android_log_print(ANDROID_LOG_DEBUG, "MobileNetssd", "enter the jni func");
    // init param
    {
        int len = env->GetArrayLength(param);
        ncnn_param.create(len, (size_t) 1u);
        env->GetByteArrayRegion(param, 0, len, (jbyte *) ncnn_param);
        int ret = ncnn_net.load_param((const unsigned char *) ncnn_param);
        __android_log_print(ANDROID_LOG_DEBUG, "MobileNetssd", "load_param %d %d", ret, len);
    }

    // init bin
    {
        int len = env->GetArrayLength(bin);
        ncnn_bin.create(len, (size_t) 1u);
        env->GetByteArrayRegion(bin, 0, len, (jbyte *) ncnn_bin);
        int ret = ncnn_net.load_model((const unsigned char *) ncnn_bin);
        __android_log_print(ANDROID_LOG_DEBUG, "MobileNetssd", "load_model %d %d", ret, len);
    }

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;   //线程 这里可以修改
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;

    ncnn::set_default_option(opt);

    return JNI_TRUE;
}

// public native String Detect(Bitmap bitmap);
JNIEXPORT jfloatArray JNICALL Java_com_example_cheungbh_SimplePose_SimplePose_Detect(JNIEnv* env, jobject thiz, jobject bitmap)
{
    // ncnn from bitmap
    ncnn::Mat in;
    {
        AndroidBitmapInfo info;
        AndroidBitmap_getInfo(env, bitmap, &info);
//        int origin_w = info.width;
//        int origin_h = info.height;
//        int width = 300;
//        int height = 300;
        int width = info.width;
        int height = info.height;
        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
            return NULL;

        void* indata;
        AndroidBitmap_lockPixels(env, bitmap, &indata);
        // 把像素转换成data，并指定通道顺序
        // 因为图像预处理每个网络层输入的数据格式不一样一般为300*300 128*128等等所以这类需要一个resize的操作可以在cpp中写，也可以是java读入图片时有个resize操作
//      in = ncnn::Mat::from_pixels_resize((const unsigned char*)indata, ncnn::Mat::PIXEL_RGBA2RGB, origin_w, origin_h, width, height);

        in = ncnn::Mat::from_pixels((const unsigned char*)indata, ncnn::Mat::PIXEL_RGBA2RGB, width, height);

        // 下面一行为debug代码
        //__android_log_print(ANDROID_LOG_DEBUG, "MobilenetssdJniIn", "Mobilenetssd_predict_has_input1, in.w: %d; in.h: %d", in.w, in.h);
        AndroidBitmap_unlockPixels(env, bitmap);
    }

    // ncnn_net
    std::vector<float> cls_scores;
    {
        // 减去均值和乘上比例（这个数据和前面的归一化图片预处理形式一一对应）
        const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
        const float scale[3] = {0.007843f, 0.007843f, 0.007843f};

        in.substract_mean_normalize(mean_vals, scale);// 归一化
        LOGW("input shape: %d dimensions: %d %d %d\n", in.dims, in.h, in.w, in.c);

        ncnn::Extractor ex = ncnn_net.create_extractor();

        ex.input("data", in);

        ncnn::Mat resnetv1b_conv0_fwd_out;
        int lay1 = ex.extract("resnetv1b_conv0_fwd_out", resnetv1b_conv0_fwd_out);
        LOGW("The return value is: %d", lay1);
        LOGW("output shape: %d dimensions: %d %d %d\n", resnetv1b_conv0_fwd_out.dims, resnetv1b_conv0_fwd_out.h, resnetv1b_conv0_fwd_out.w, resnetv1b_conv0_fwd_out.c);

        ncnn::Mat resnetv1b_pool0_fwd;
        int lay2 = ex.extract("resnetv1b_pool0_fwd", resnetv1b_pool0_fwd);
        LOGW("The return value is: %d", lay2);
        LOGW("output shape: %d dimensions: %d %d %d\n", resnetv1b_pool0_fwd.dims, resnetv1b_pool0_fwd.h, resnetv1b_pool0_fwd.w, resnetv1b_pool0_fwd.c);

        ncnn::Mat splitncnn_0;
        int lay3 = ex.extract("splitncnn_0", splitncnn_0);
        LOGW("Current layer is splitncnn_0");
        LOGW("The return value is: %d", lay3);
        LOGW("output shape: %d dimensions: %d %d %d\n", splitncnn_0.dims, splitncnn_0.h, splitncnn_0.w, splitncnn_0.c);



        ncnn::Mat out;
        int a = ex.extract("conv3_fwd", out);
        LOGW("The return value is: %d", a);
        LOGW("output shape: %d dimensions: %d %d %d\n", out.dims, out.h, out.w, out.c);


        int output_wsize = out.w;
        int output_hsize = out.h;
        LOGW("The width is %d\n", output_wsize);
        LOGW("The height is %d\n", output_hsize);
//        printf(output_hsize)

        //输出整理
        jfloat *output[output_wsize * output_hsize];   // float类型
        for(int i = 0; i< out.h; i++) {
            for (int j = 0; j < out.w; j++) {
                output[i*output_wsize + j] = &out.row(i)[j];
            }
        }
        //建立float数组 长度为 output_wsize * output_hsize,如果只是ouput_size相当于只有一行的out的数据那就是一个object检测数据
        jfloatArray jOutputData = env->NewFloatArray(output_wsize * output_hsize);
        if (jOutputData == nullptr) return nullptr;
        env->SetFloatArrayRegion(jOutputData, 0,  output_wsize * output_hsize,
                                 reinterpret_cast<const jfloat *>(*output));
        return jOutputData;
    }
}
}
