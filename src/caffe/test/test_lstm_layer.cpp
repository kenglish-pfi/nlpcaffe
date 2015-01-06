#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define NUM_CELLS 5

using std::min;
using std::max;

namespace caffe {

template <typename TypeParam>
class LstmLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  LstmLayerTest()
      : epsilon_(Dtype(1e-5)),
        blob_bottom_(new Blob<Dtype>()),
        blob_bottom2_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1601);
    blob_bottom_->Reshape(NUM_CELLS, 1, 1, 1);
    blob_bottom2_->Reshape(NUM_CELLS, 1, 1, 1);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    GaussianFiller<Dtype> filler2(filler_param);
    filler.Fill(this->blob_bottom_);
    filler2.Fill(this->blob_bottom2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom2_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_vec_.push_back(blob_top2_);
  }
  virtual ~LstmLayerTest() { delete blob_bottom_; delete blob_bottom2_; delete blob_top_; delete blob_top2_; }
  void ReferenceLstmForward(const Blob<Dtype>& blob_bottom,
      const LayerParameter& layer_param, Blob<Dtype>* blob_top);

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top2_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

template <typename TypeParam>
void LstmLayerTest<TypeParam>::ReferenceLstmForward(
    const Blob<Dtype>& blob_bottom, const LayerParameter& layer_param,
    Blob<Dtype>* blob_top) {
  typedef typename TypeParam::Dtype Dtype;
  blob_top->Reshape(blob_bottom.num(), blob_bottom.channels(),
      blob_bottom.height(), blob_bottom.width());
  Dtype* top_data = blob_top->mutable_cpu_data();
  LstmParameter lstm_param = layer_param.lstm_param();
  //Dtype alpha = lstm_param.alpha();
  //Dtype beta = lstm_param.beta();
  //int size = lstm_param.local_size();
  for (int n = 0; n < blob_bottom.num(); ++n) {
    for (int c = 0; c < blob_bottom.channels(); ++c) {
      for (int h = 0; h < blob_bottom.height(); ++h) {
        for (int w = 0; w < blob_bottom.width(); ++w) {
          //int c_start = c - (size - 1) / 2;
          //int c_end = min(c_start + size, blob_bottom.channels());
          //c_start = max(c_start, 0);
          //Dtype scale = 1.;
          //for (int i = c_start; i < c_end; ++i) {
            //Dtype value = blob_bottom.data_at(n, i, h, w);
            //scale += value * value * alpha / size;
          //}
          //*(top_data + blob_top->offset(n, c, h, w)) =
            //blob_bottom.data_at(n, c, h, w) / pow(scale, beta);
        }
      }
    }
  }
}

TYPED_TEST_CASE(LstmLayerTest, TestDtypesAndDevices);

TYPED_TEST(LstmLayerTest, TestSetupAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LstmParameter* lstm_param = layer_param.mutable_lstm_param();
  lstm_param->set_num_cells(NUM_CELLS);
  lstm_param->mutable_input_bias_filler()->set_type("constant");
  lstm_param->mutable_input_bias_filler()->set_value(0.1);
  lstm_param->mutable_input_gate_bias_filler()->set_type("constant");
  lstm_param->mutable_input_gate_bias_filler()->set_value(0.1);
  lstm_param->mutable_forget_gate_bias_filler()->set_type("constant");
  lstm_param->mutable_forget_gate_bias_filler()->set_value(0.1);
  lstm_param->mutable_output_gate_bias_filler()->set_type("constant");
  lstm_param->mutable_output_gate_bias_filler()->set_value(0.1);
  lstm_param->mutable_input_weight_filler()->set_type("constant");
  lstm_param->mutable_input_weight_filler()->set_value(0.1);
  lstm_param->mutable_input_gate_weight_filler()->set_type("constant");
  lstm_param->mutable_input_gate_weight_filler()->set_value(0.1);
  lstm_param->mutable_forget_gate_weight_filler()->set_type("constant");
  lstm_param->mutable_forget_gate_weight_filler()->set_value(0.1);
  lstm_param->mutable_output_gate_weight_filler()->set_type("constant");
  lstm_param->mutable_output_gate_weight_filler()->set_value(0.1);
  LstmLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), NUM_CELLS);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

//TYPED_TEST(LRNLayerTest, TestForwardAcrossChannels) {
  //typedef typename TypeParam::Dtype Dtype;
  //LayerParameter layer_param;
  //LRNLayer<Dtype> layer(layer_param);
  //layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  //layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  //Blob<Dtype> top_reference;
  //this->ReferenceLRNForward(*(this->blob_bottom_), layer_param,
      //&top_reference);
  //for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    //EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                //this->epsilon_);
  //}
//}

TYPED_TEST(LstmLayerTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LstmParameter* lstm_param = layer_param.mutable_lstm_param();
  lstm_param->set_num_cells(NUM_CELLS);
  lstm_param->mutable_input_bias_filler()->set_type("constant");
  lstm_param->mutable_input_bias_filler()->set_value(0.1);
  lstm_param->mutable_input_gate_bias_filler()->set_type("constant");
  lstm_param->mutable_input_gate_bias_filler()->set_value(0.1);
  lstm_param->mutable_forget_gate_bias_filler()->set_type("constant");
  lstm_param->mutable_forget_gate_bias_filler()->set_value(0.1);
  lstm_param->mutable_output_gate_bias_filler()->set_type("constant");
  lstm_param->mutable_output_gate_bias_filler()->set_value(0.1);
  lstm_param->mutable_input_weight_filler()->set_type("constant");
  lstm_param->mutable_input_weight_filler()->set_value(0.1);
  lstm_param->mutable_input_gate_weight_filler()->set_type("constant");
  lstm_param->mutable_input_gate_weight_filler()->set_value(0.1);
  lstm_param->mutable_forget_gate_weight_filler()->set_type("constant");
  lstm_param->mutable_forget_gate_weight_filler()->set_value(0.1);
  lstm_param->mutable_output_gate_weight_filler()->set_type("constant");
  lstm_param->mutable_output_gate_weight_filler()->set_value(0.1);

  LstmLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  for (int i = 0; i < this->blob_top2_->count(); ++i) {
    this->blob_top2_->mutable_cpu_diff()[i] = 1.;
  }
  //std::cout << this->blob_top2_->count() << " size\n";
  //std::cout << this->blob_top2_->mutable_cpu_diff()[0] << " diff\n";
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 &(this->blob_bottom_vec_));
  // for (int i = 0; i < this->blob_bottom_->count(); ++i) {
  //   std::cout << "CPU diff " << this->blob_bottom_->cpu_diff()[i]
  //       << std::endl;
  // }
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

//TYPED_TEST(LRNLayerTest, TestSetupWithinChannel) {
  //typedef typename TypeParam::Dtype Dtype;
  //LayerParameter layer_param;
  //layer_param.mutable_lrn_param()->set_norm_region(
      //LRNParameter_NormRegion_WITHIN_CHANNEL);
  //layer_param.mutable_lrn_param()->set_local_size(3);
  //LRNLayer<Dtype> layer(layer_param);
  //layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  //EXPECT_EQ(this->blob_top_->num(), 2);
  //EXPECT_EQ(this->blob_top_->channels(), 7);
  //EXPECT_EQ(this->blob_top_->height(), 3);
  //EXPECT_EQ(this->blob_top_->width(), 3);
//}

//TYPED_TEST(LRNLayerTest, TestForwardWithinChannel) {
  //typedef typename TypeParam::Dtype Dtype;
  //LayerParameter layer_param;
  //layer_param.mutable_lrn_param()->set_norm_region(
      //LRNParameter_NormRegion_WITHIN_CHANNEL);
  //layer_param.mutable_lrn_param()->set_local_size(3);
  //LRNLayer<Dtype> layer(layer_param);
  //layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  //layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  //Blob<Dtype> top_reference;
  //this->ReferenceLRNForward(*(this->blob_bottom_), layer_param,
      //&top_reference);
  //for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    //EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                //this->epsilon_);
  //}
//}

//TYPED_TEST(LRNLayerTest, TestGradientWithinChannel) {
  //typedef typename TypeParam::Dtype Dtype;
  //LayerParameter layer_param;
  //layer_param.mutable_lrn_param()->set_norm_region(
      //LRNParameter_NormRegion_WITHIN_CHANNEL);
  //layer_param.mutable_lrn_param()->set_local_size(3);
  //LRNLayer<Dtype> layer(layer_param);
  //GradientChecker<Dtype> checker(1e-2, 1e-2);
  //layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  //layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  //for (int i = 0; i < this->blob_top_->count(); ++i) {
    //this->blob_top_->mutable_cpu_diff()[i] = 1.;
  //}
  //checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      //&(this->blob_top_vec_));
//}


}  // namespace caffe
