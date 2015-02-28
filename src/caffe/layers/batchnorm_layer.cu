#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

inline int offset(int n, int bottom_size) {
  return bottom_size * n;
}
template <typename Dtype>
void BatchnormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
    /*Forward_cpu(bottom, top);*/
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* gamma_data = this->blobs_[0]->gpu_data();
  const Dtype* beta_data = this->blobs_[1]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();

  Dtype* mean_data = batch_mean_.mutable_gpu_data();
  Dtype* variance_data = batch_variance_.mutable_gpu_data();
  Dtype* buffer = buffer_blob_.mutable_gpu_data();

  caffe_gpu_set(bottom_size_, Dtype(0), mean_data);
  caffe_gpu_set(bottom_size_, Dtype(0), variance_data);

  for (int n = 0; n < num_; ++n) {
    caffe_gpu_add(bottom_size_, bottom_data + offset(n, bottom_size_), mean_data,
        mean_data);
    caffe_gpu_powx(bottom_size_, bottom_data + offset(n, bottom_size_), Dtype(2.0), buffer);
    caffe_gpu_add(bottom_size_, buffer, variance_data, variance_data);
  }
  caffe_gpu_scale(bottom_size_, Dtype(1) / Dtype(num_), mean_data, mean_data);
  caffe_gpu_scale(bottom_size_, Dtype(1) / Dtype(num_), variance_data,
      variance_data);

  caffe_gpu_powx(bottom_size_, mean_data, Dtype(2.0), buffer);
  caffe_gpu_sub(bottom_size_, variance_data, buffer, variance_data);
  caffe_gpu_add_scalar(bottom_size_, var_epsilon_, variance_data);
  caffe_gpu_powx(bottom_size_, variance_data, Dtype(0.5), variance_data);

  for (int n = 0; n < num_; ++n) {
    caffe_gpu_sub(bottom_size_, bottom_data + offset(n, bottom_size_), mean_data, buffer);
    caffe_gpu_div(bottom_size_, buffer, variance_data, buffer);
    caffe_gpu_mul(bottom_size_, buffer, gamma_data, buffer);
    caffe_gpu_add(bottom_size_, buffer, beta_data,
        top_data + offset(n, bottom_size_));
  }
}

template <typename Dtype>
void BatchnormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
    /*Backward_cpu(top, propagate_down, bottom);*/
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* variance_data = batch_variance_.gpu_data();
  const Dtype* gamma_data = this->blobs_[0]->gpu_data();
  const Dtype* beta_data = this->blobs_[1]->gpu_data();

  Dtype* dl_dvar = batch_variance_.mutable_gpu_diff();
  Dtype* dl_dmean = batch_mean_.mutable_gpu_diff();
  Dtype* buffer = buffer_blob_.mutable_gpu_data();

  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  Dtype* gamma_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* beta_diff = this->blobs_[1]->mutable_gpu_diff();

  caffe_gpu_set((*bottom)[0]->count(), Dtype(0), bottom_diff);
  caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), gamma_diff);
  caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), beta_diff);
  caffe_gpu_set(bottom_size_, Dtype(0), dl_dvar);
  caffe_gpu_set(bottom_size_, Dtype(0), dl_dmean);

  for (int n = 0; n < num_; ++n) {
    // fill gamma_diff
    caffe_gpu_sub(bottom_size_, top_data + offset(n, bottom_size_), beta_data,
        buffer);
    caffe_gpu_div(bottom_size_, buffer, gamma_data,
        buffer);
    caffe_gpu_mul(bottom_size_, buffer, top_diff + offset(n, bottom_size_),
        buffer);
    caffe_gpu_add(bottom_size_, buffer, gamma_diff, gamma_diff);

    // fill beta_diff
    caffe_gpu_add(bottom_size_, top_diff + offset(n, bottom_size_), beta_diff, beta_diff);
  }

  // fill bottom_diff direct term
  for (int n = 0; n < num_; ++n) {
    caffe_gpu_mul(bottom_size_, top_diff + offset(n, bottom_size_), gamma_data, buffer);
    caffe_gpu_div(bottom_size_, buffer, variance_data, buffer);
    caffe_gpu_add(bottom_size_, buffer, bottom_diff + offset(n, bottom_size_),
        bottom_diff + offset(n, bottom_size_));
  }

  // fill bottom_diff variance contribution term
  for (int n = 0; n < num_; ++n) {
    caffe_gpu_sub(bottom_size_, top_data + offset(n, bottom_size_), beta_data, buffer);
    caffe_gpu_mul(bottom_size_, buffer, variance_data, buffer);
    caffe_gpu_mul(bottom_size_, buffer, top_diff + offset(n, bottom_size_), buffer);
    caffe_gpu_add(bottom_size_, buffer, dl_dvar, dl_dvar);
  }
  caffe_gpu_powx(bottom_size_, variance_data, Dtype(-3.0), buffer);
  caffe_gpu_mul(bottom_size_, dl_dvar, buffer, dl_dvar);
  caffe_gpu_scale(bottom_size_, Dtype(-0.5), dl_dvar, dl_dvar);
  for (int n = 0; n < num_; ++n) {
    caffe_gpu_sub(bottom_size_, top_data + offset(n, bottom_size_), beta_data, buffer);
    caffe_gpu_div(bottom_size_, buffer, gamma_data, buffer);
    caffe_gpu_mul(bottom_size_, buffer, variance_data, buffer);
    caffe_gpu_scale(bottom_size_, Dtype(2) / Dtype(num_), buffer, buffer);
    caffe_gpu_mul(bottom_size_, buffer, dl_dvar, buffer);
    caffe_gpu_add(bottom_size_, buffer, bottom_diff + offset(n, bottom_size_),
        bottom_diff + offset(n, bottom_size_));
  }

  // fill bottom_diff mean contribution term
  for (int n = 0; n < num_; ++n) {
    caffe_gpu_mul(bottom_size_, top_diff + offset(n, bottom_size_), gamma_data, buffer);
    caffe_gpu_div(bottom_size_, buffer, variance_data, buffer);
    caffe_gpu_sub(bottom_size_, dl_dmean, buffer, dl_dmean);
  }
  caffe_gpu_scale(bottom_size_, Dtype(1) / Dtype(num_), dl_dmean, dl_dmean);
  for (int n = 0; n < num_; ++n) {
    caffe_gpu_add(bottom_size_, dl_dmean, bottom_diff + offset(n, bottom_size_),
        bottom_diff + offset(n, bottom_size_));
  }
}

INSTANTIATE_CLASS(BatchnormLayer);

}  // namespace caffe
