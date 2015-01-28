#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void WordvecLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  WordvecParameter wordvec_param = this->layer_param_.wordvec_param();
  CHECK((wordvec_param.has_dimension()))
      << "wordvec_param.has_dimension()";
  CHECK((wordvec_param.has_vocab_size()))
      << "wordvec_param.has_vocab_size()";

  num_ = bottom[0]->num();
  dimension_ = wordvec_param.dimension();
  vocab_size_ = wordvec_param.vocab_size();
  sentence_length_ = bottom[0]->channels();

  this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(
        1, vocab_size_, 1, dimension_));

  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
      wordvec_param.weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), false);
}

template <typename Dtype>
void WordvecLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK((this->layer_param_.bottom_size() == 1 || this->layer_param_.bottom_size() == 0))
      << "Wordvec must have no more than one bottom";
  CHECK((this->layer_param_.top_size() == 1 || this->layer_param_.top_size() == 0))
      << "Wordvec must have no more than one top";
  (*top)[0]->Reshape(sentence_length_ * num_, dimension_, 1, 1);
}

template <typename Dtype>
void WordvecLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* weights = this->blobs_[0]->cpu_data();
  Dtype* bottom_data = bottom[0]->mutable_cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();

  for (int i = 0; i < sentence_length_; ++i) {
    for (int n = 0; n < num_; ++n) {
      const int idx = n + i * num_;
      const int word = static_cast<int>(bottom_data[idx] + Dtype(0.5));
      caffe_copy(dimension_, weights + word * dimension_, top_data + (*top)[0]->offset(idx));
    }
  }
}

template <typename Dtype>
void WordvecLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  Dtype* weights_diff = this->blobs_[0]->mutable_cpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();

  caffe_set(this->blobs_[0]->count(), Dtype(0), weights_diff);

  for (int i = 0; i < sentence_length_; ++i) {
    for (int n = 0; n < num_; ++n) {
      const int idx = n + i * num_;
      const int word = static_cast<int>(bottom_data[idx] + Dtype(0.5));
      caffe_add(dimension_, top_diff + top[0]->offset(idx),
        weights_diff + word * dimension_,
        weights_diff + word * dimension_);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(WordvecLayer);
#endif

INSTANTIATE_CLASS(WordvecLayer);

}  // namespace caffe
