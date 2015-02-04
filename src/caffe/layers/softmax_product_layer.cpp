#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  num_output_ = this->layer_param_.softmax_product_param().num_output();
  num_categories_ = this->layer_param_.softmax_product_param().num_categories();
  input_dim_ = bottom[0]->count() / bottom[0]->num();
  K_ = input_dim_;

  N_ = num_output_;
  this->blobs_.resize(1);
  this->blobs_[0].reset(new Blob<Dtype>(num_categories_, 1, N_, K_));

  // fill the weights
  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
      this->layer_param_.softmax_product_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());

  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void SoftmaxProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // // Figure out the dimensions
  CHECK_EQ(bottom[0]->num(), bottom[1]->count()) << "There must be one label per input vector";
  (*top)[0]->Reshape(bottom[0]->num(), N_, 1, 1);
  (*top)[1]->Reshape(bottom[0]->num(), 1, 1, 1);
  (*top)[2]->Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void SoftmaxProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* labels = bottom[1]->cpu_data();

  for (int n = 0; n < bottom[0]->num(); ++n) {
    const int label_batch_size = bottom[1]->num();
    const int label_batch = n % label_batch_size;
    const int label_sentence_pos = n / label_batch_size;
    const int label_idx = bottom[1]->offset(label_batch, label_sentence_pos);
    const int label = static_cast<int>(labels[label_idx] + Dtype(0.5));

    const int category = label % num_categories_;
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1, N_, K_, (Dtype)1.,
        bottom_data + bottom[0]->offset(n),
        weight + this->blobs_[0]->offset(category), (Dtype)0.,
        top_data + (*top)[0]->offset(n));
    (*top)[1]->mutable_cpu_data()[n] = label / num_categories_;
    (*top)[2]->mutable_cpu_data()[n] = category;
  }
}

template <typename Dtype>
void SoftmaxProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  const Dtype* labels = (*bottom)[1]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype * weight_diff = this->blobs_[0]->mutable_cpu_diff();

  caffe_set((*bottom)[0]->count(), Dtype(0), bottom_diff);
  caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);

  for (int n = 0; n < (*bottom)[0]->num(); ++n) {
    const int label_batch_size = (*bottom)[1]->num();
    const int label_batch = n % label_batch_size;
    const int label_sentence_pos = n / label_batch_size;
    const int label_idx = (*bottom)[1]->offset(label_batch, label_sentence_pos);
    const int category = static_cast<int>(labels[label_idx] + Dtype(0.5)) % num_categories_;
    // Gradient with respect to weight
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, 1, (Dtype)1.,
        top_diff + top[0]->offset(n),
        bottom_data + (*bottom)[0]->offset(n), (Dtype)1.,
        weight_diff + this->blobs_[0]->offset(category));
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, K_, N_, (Dtype)1.,
          top_diff + top[0]->offset(n),
          weight + this->blobs_[0]->offset(category), (Dtype)0.,
          bottom_diff + (*bottom)[0]->offset(n));
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxProductLayer);
#endif

INSTANTIATE_CLASS(SoftmaxProductLayer);

}  // namespace caffe
