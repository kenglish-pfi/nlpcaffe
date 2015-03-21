#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    for (int n = 0; n < num_concats_; ++n) {
      caffe_copy(bottom_concat_axis * concat_input_size_,
          bottom_data + n * bottom_concat_axis * concat_input_size_,
          top_data + (n * top_concat_axis + offset_concat_axis)
              * concat_input_size_);
    }
    offset_concat_axis += bottom_concat_axis;
  }
  //<<<<<<<<<<<<<<<<<<
  /*} else if (concat_dim_ == 1) {*/
    /*int offset_channel = 0;*/
    /*Dtype* top_buffer_data = top_buffer_.mutable_gpu_data();*/
    /*for (int i = 0; i < bottom.size(); ++i) {*/
      /*const Dtype* bottom_data = bottom[i]->gpu_data();*/
      /*int num_elem =*/
        /*bottom[i]->channels() * height_ * width_;*/
      /*int top_offset = num_ * height_ * width_ * offset_channel;*/
      /*caffe_gpu_transpose(num_, num_elem, bottom_data, top_buffer_data + top_offset);*/
      /*offset_channel += bottom[i]->channels();*/
    /*}*/
    /*int total_num_elem = height_ * width_ * offset_channel;*/
    /*caffe_gpu_transpose(total_num_elem, num_, top_buffer_data, top_data);*/
  /*} else if (concat_dim_ == 2) {*/
    /*int offset_height = 0;*/
    /*for (int i = 0; i < bottom.size(); ++i) {*/
      /*const Dtype* bottom_data = bottom[i]->gpu_data();*/
      /*int num_elem =*/
        /*bottom[i]->height()*bottom[i]->width();*/
      /*for (int n = 0; n < num_; ++n) {*/
        /*for (int c = 0; c < channels_; ++c) {*/
          /*caffe_copy(num_elem, bottom_data+bottom[i]->offset(n, c),*/
            /*top_data + top[0]->offset(n, c, offset_height));*/
        /*}*/
      /*}*/
      /*offset_height += bottom[i]->height();*/
    /*}*/
  /*} else {*/
    /*LOG(FATAL) << "concat_dim along dim" << concat_dim_ <<*/
      /*" not implemented yet";*/
  //>>>>>>>>>>>>>>>>>
}

template <typename Dtype>
void ConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  //<<<<<<<<<<<<<<<<<<
  /*if (concat_dim_ == 0) {*/
    /*int offset_num = 0;*/
    /*for (int i = 0; i < bottom->size(); ++i) {*/
      /*Blob<Dtype>* blob = (*bottom)[i];*/
      /*if (propagate_down[i]) {*/
        /*Dtype* bottom_diff = blob->mutable_gpu_diff();*/
        /*caffe_copy(blob->count(), top_diff + top[0]->offset(offset_num),*/
                       /*bottom_diff);*/
      /*}*/
      /*offset_num += blob->num();*/
    /*}*/
  /*} else if (concat_dim_ == 1) {*/
    /*int offset_channel = 0;*/
    /*Dtype* top_buffer_diff = top_buffer_.mutable_gpu_diff();*/
    /*int total_num_elem = height_ * width_ * channels_;*/
    /*caffe_gpu_transpose(num_, total_num_elem, top_diff, top_buffer_diff);*/
    /*for (int i = 0; i < bottom->size(); ++i) {*/
      /*if (propagate_down[i]) {*/
        /*Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();*/
        /*int top_offset = num_ * height_ * width_ * offset_channel;*/
        /*int num_elem = (*bottom)[i]->channels() * height_ * width_;*/
        /*caffe_gpu_transpose(num_elem, num_, top_buffer_diff + top_offset, bottom_diff);*/
      /*}*/
      /*offset_channel += (*bottom)[i]->channels();*/
    /*}*/
  /*} else if (concat_dim_ == 2) {*/
    /*int offset_width = 0;*/
    /*for (int i = 0; i < bottom->size(); ++i) {*/
      /*Blob<Dtype>* blob = (*bottom)[i];*/
      /*if (propagate_down[i]) {*/
        /*Dtype* bottom_diff = blob->mutable_gpu_diff();*/
        /*int num_elem = blob->height()*blob->width();*/
        /*for (int n = 0; n < num_; ++n) {*/
          /*for (int c = 0; c < channels_; ++c) {*/
            /*caffe_copy(num_elem, top_diff + top[0]->offset(n, c, offset_width),*/
                       /*bottom_diff + blob->offset(n, c));*/
          /*}*/
        /*}*/
      /*}*/
      /*offset_width += blob->width();*/
  //<<<<<<<<<<<<<<<<<<
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  for (int i = 0; i < bottom.size(); ++i) {
    if (!propagate_down[i]) { continue; }
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    for (int n = 0; n < num_concats_; ++n) {
      caffe_copy(bottom_concat_axis * concat_input_size_, top_diff +
          (n * top_concat_axis + offset_concat_axis) * concat_input_size_,
          bottom_diff + n * bottom_concat_axis * concat_input_size_);
    }
    offset_concat_axis += bottom_concat_axis;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConcatLayer);

}  // namespace caffe
