#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

#define NUM_CATEGORIES 3
#define NUM_OUTPUT 4
#define BOTTOM_NUM 6

template <typename TypeParam>
class SoftmaxProductLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SoftmaxProductLayerTest()
      : blob_bottom_(new Blob<Dtype>(BOTTOM_NUM, 3, 5, 2)),
        blob_bottom2_(new Blob<Dtype>(BOTTOM_NUM/2, 2, 1, 1)),
        blob_top_(new Blob<Dtype>()),
        blob_top2_(new Blob<Dtype>()),
        blob_top3_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    for (int n = 0; n < BOTTOM_NUM; ++n) {
      this->blob_bottom2_->mutable_cpu_data()[n] = caffe_rng_rand() % (NUM_CATEGORIES * NUM_OUTPUT);
    }
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom2_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_vec_.push_back(blob_top2_);
    blob_top_vec_.push_back(blob_top3_);
  }
  virtual ~SoftmaxProductLayerTest() {
    delete blob_bottom_; delete blob_bottom2_;
    delete blob_top_; delete blob_top2_; delete blob_top3_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top2_;
  Blob<Dtype>* const blob_top3_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxProductLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxProductLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SoftmaxProductParameter* softmax_product_param =
      layer_param.mutable_softmax_product_param();
  softmax_product_param->set_num_output(NUM_OUTPUT);
  softmax_product_param->set_num_categories(NUM_CATEGORIES);
  shared_ptr<SoftmaxProductLayer<Dtype> > layer(
      new SoftmaxProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), BOTTOM_NUM);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), NUM_OUTPUT);

  EXPECT_EQ(this->blob_top2_->num(), BOTTOM_NUM);
  EXPECT_EQ(this->blob_top2_->height(), 1);
  EXPECT_EQ(this->blob_top2_->width(), 1);
  EXPECT_EQ(this->blob_top2_->channels(), 1);

  EXPECT_EQ(this->blob_top3_->num(), BOTTOM_NUM);
  EXPECT_EQ(this->blob_top3_->height(), 1);
  EXPECT_EQ(this->blob_top3_->width(), 1);
  EXPECT_EQ(this->blob_top3_->channels(), 1);
}

TYPED_TEST(SoftmaxProductLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    SoftmaxProductParameter* softmax_product_param =
        layer_param.mutable_softmax_product_param();
    softmax_product_param->set_num_output(NUM_OUTPUT);
    softmax_product_param->set_num_categories(NUM_CATEGORIES);
    softmax_product_param->mutable_weight_filler()->set_type("uniform");
    softmax_product_param->mutable_weight_filler()->set_min(-1);
    softmax_product_param->mutable_weight_filler()->set_max(1);
    SoftmaxProductLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
        &(this->blob_top_vec_), 0);

  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
