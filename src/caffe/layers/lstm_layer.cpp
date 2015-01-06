#include <vector>

//#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
//#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
inline Dtype sigmoid_diff(Dtype x) {
  return x * (1. - x);
}

template <typename Dtype>
void LstmLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  LstmParameter lstm_param = this->layer_param_.lstm_param();
  CHECK((lstm_param.has_num_blocks()))
      << "lstm_param.has_num_blocks()";
  CHECK((lstm_param.has_cells_per_block()))
      << "lstm_param.has_cells_per_block()";
  CHECK((lstm_param.has_input_weight_filler()))
      << "lstm_param.has_input_weight_filler()";
  CHECK((lstm_param.has_input_bias_filler()))
      << "lstm_param.has_input_bias_filler()";
  CHECK((lstm_param.has_input_gate_weight_filler()))
      << "lstm_param.has_input_gate_weight_filler()";
  CHECK((lstm_param.has_input_gate_bias_filler()))
      << "lstm_param.has_input_gate_bias_filler()";
  CHECK((lstm_param.has_forget_gate_weight_filler()))
      << "lstm_param.has_forget_gate_weight_filler()";
  CHECK((lstm_param.has_forget_gate_bias_filler()))
      << "lstm_param.has_forget_gate_bias_filler()";
  CHECK((lstm_param.has_output_gate_weight_filler()))
      << "lstm_param.has_output_gate_weight_filler()";
  CHECK((lstm_param.has_output_gate_bias_filler()))
      << "lstm_param.has_output_gate_bias_filler()";
  channels_ = lstm_param.num_blocks();
  height_ = lstm_param.cells_per_block();
  const int input_data_size = bottom[0]->count();
  CHECK((channels_ * height_ == input_data_size))
      << "input dimension must match total number of cells " << input_data_size << ", " << channels_ * height_;

  this->blobs_.resize(8);
  this->blobs_[0].reset(new Blob<Dtype>(
      1, channels_, height_, input_data_size));
  shared_ptr<Filler<Dtype> > input_weight_filler(GetFiller<Dtype>(
      this->layer_param_.lstm_param().input_weight_filler()));
  input_weight_filler->Fill(this->blobs_[0].get());

  this->blobs_[2].reset(new Blob<Dtype>(
      1, channels_, 1, input_data_size));
  shared_ptr<Filler<Dtype> > input_gate_weight_filler(GetFiller<Dtype>(
      this->layer_param_.lstm_param().input_gate_weight_filler()));
  input_gate_weight_filler->Fill(this->blobs_[2].get());

  this->blobs_[4].reset(new Blob<Dtype>(
      1, channels_, 1, input_data_size));
  shared_ptr<Filler<Dtype> > forget_gate_weight_filler(GetFiller<Dtype>(
      this->layer_param_.lstm_param().forget_gate_weight_filler()));
  forget_gate_weight_filler->Fill(this->blobs_[4].get());

  this->blobs_[6].reset(new Blob<Dtype>(
      1, channels_, 1, input_data_size));
  shared_ptr<Filler<Dtype> > output_gate_weight_filler(GetFiller<Dtype>(
      this->layer_param_.lstm_param().output_gate_weight_filler()));
  output_gate_weight_filler->Fill(this->blobs_[6].get());


  this->blobs_[1].reset(new Blob<Dtype>(1, channels_, height_, 1));
  shared_ptr<Filler<Dtype> > input_bias_filler(GetFiller<Dtype>(
      this->layer_param_.lstm_param().input_bias_filler()));
  input_bias_filler->Fill(this->blobs_[1].get());

  this->blobs_[3].reset(new Blob<Dtype>(1, channels_, 1, 1));
  shared_ptr<Filler<Dtype> > input_gate_bias_filler(GetFiller<Dtype>(
      this->layer_param_.lstm_param().input_gate_bias_filler()));
  input_gate_bias_filler->Fill(this->blobs_[3].get());

  this->blobs_[5].reset(new Blob<Dtype>(1, channels_, 1, 1));
  shared_ptr<Filler<Dtype> > forget_gate_bias_filler(GetFiller<Dtype>(
      this->layer_param_.lstm_param().forget_gate_bias_filler()));
  forget_gate_bias_filler->Fill(this->blobs_[5].get());

  this->blobs_[7].reset(new Blob<Dtype>(1, channels_, 1, 1));
  shared_ptr<Filler<Dtype> > output_gate_bias_filler(GetFiller<Dtype>(
      this->layer_param_.lstm_param().output_gate_bias_filler()));
  output_gate_bias_filler->Fill(this->blobs_[7].get());


  //Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void LstmLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  (*top)[0]->Reshape(1,channels_, height_, 1);
  (*top)[1]->Reshape(1,channels_, height_, 1);
  //num_ = bottom[0]->num();
  //height_ = bottom[0]->height();
  //width_ = bottom[0]->width();
  //CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    //" convolution kernel.";
  //// TODO: generalize to handle inputs of different shapes.
  //for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    //CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    //CHECK_EQ(channels_, bottom[bottom_id]->channels())
        //<< "Inputs must have same channels.";
    //CHECK_EQ(height_, bottom[bottom_id]->height())
        //<< "Inputs must have same height.";
    //CHECK_EQ(width_, bottom[bottom_id]->width())
        //<< "Inputs must have same width.";
  //}
  //// Shape the tops.
  //height_out_ =
      //(height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  //width_out_ = (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
  //for (int top_id = 0; top_id < top->size(); ++top_id) {
    //(*top)[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  //}
  //// Prepare the matrix multiplication computation.
  //// Each input will be convolved as a single GEMM.
  //M_ = num_output_ / group_;
  //K_ = channels_ * kernel_h_ * kernel_w_ / group_;
  //N_ = height_out_ * width_out_;
  //// The im2col result buffer will only hold one image at a time to avoid
  //// overly large memory usage.
  //col_buffer_.Reshape(
      //1, channels_ * kernel_h_ * kernel_w_, height_out_, width_out_);
  //for (int top_id = 0; top_id < top->size(); ++top_id) {
    //(*top)[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  //}
  //// Set up the all ones "bias multiplier" for adding biases by BLAS
  //if (bias_term_) {
    //bias_multiplier_.Reshape(1, 1, 1, N_);
    //caffe_set(N_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  //}
}

template <typename Dtype>
void LstmLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* prev_state_data = bottom[1]->cpu_data();
  const int input_data_size = bottom[0]->count();

  const Dtype* input_weight = this->blobs_[0]->cpu_data();
  const Dtype* input_bias = this->blobs_[1]->cpu_data();
  const Dtype* input_gate_weight = this->blobs_[2]->cpu_data();
  const Dtype* input_gate_bias = this->blobs_[3]->cpu_data();
  const Dtype* forget_gate_weight = this->blobs_[4]->cpu_data();
  const Dtype* forget_gate_bias = this->blobs_[5]->cpu_data();
  const Dtype* output_gate_weight = this->blobs_[6]->cpu_data();
  const Dtype* output_gate_bias = this->blobs_[7]->cpu_data();

  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* next_state_data = (*top)[1]->mutable_cpu_data();
  
  for (int i = 0; i < channels_; i++) {
    for (int j = 0; j < height_; j++) {
      const int idx = i * height_ + j;

      Dtype input = input_bias[idx];
      Dtype input_gate_mult = input_gate_bias[i];
      Dtype forget_gate_mult = forget_gate_bias[i];
      Dtype output_gate_mult = output_gate_bias[i];
      for (int k = 0; k < input_data_size; k++) {
        input += input_weight[k + idx*input_data_size] * input_data[k];
        input_gate_mult += input_gate_weight[k + i * input_data_size] * input_data[k];
        forget_gate_mult += forget_gate_weight[k + i * input_data_size] * input_data[k];
        output_gate_mult += output_gate_weight[k + i * input_data_size] * input_data[k];
      }
      input = sigmoid(input);
      input_gate_mult = sigmoid(input_gate_mult);
      forget_gate_mult = sigmoid(forget_gate_mult);
      output_gate_mult = sigmoid(output_gate_mult);

      next_state_data[idx] = prev_state_data[idx] * forget_gate_mult + input * input_gate_mult;
      top_data[idx] = next_state_data[idx] * output_gate_mult;
    }
  }
}

template <typename Dtype>
void LstmLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < 2; ++i) {
    caffe_set((*bottom)[i]->count(), Dtype(0), (*bottom)[i]->mutable_cpu_diff());
  }
  const Dtype* input_data = (*bottom)[0]->cpu_data();
  Dtype* input_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* prev_state_data = (*bottom)[1]->cpu_data();
  Dtype* prev_state_diff = (*bottom)[1]->mutable_cpu_diff();
  const int input_data_size = (*bottom)[0]->count();

  const Dtype* input_weight = this->blobs_[0]->cpu_data();
  const Dtype* input_bias = this->blobs_[1]->cpu_data();
  const Dtype* input_gate_weight = this->blobs_[2]->cpu_data();
  const Dtype* input_gate_bias = this->blobs_[3]->cpu_data();
  const Dtype* forget_gate_weight = this->blobs_[4]->cpu_data();
  const Dtype* forget_gate_bias = this->blobs_[5]->cpu_data();
  const Dtype* output_gate_weight = this->blobs_[6]->cpu_data();
  const Dtype* output_gate_bias = this->blobs_[7]->cpu_data();


  for (int i = 0; i < 8; ++i) {
    caffe_set(this->blobs_[i]->count(), Dtype(0), this->blobs_[i]->mutable_cpu_diff());
  }


  Dtype* input_weight_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* input_bias_diff = this->blobs_[1]->mutable_cpu_diff();
  Dtype* input_gate_weight_diff = this->blobs_[2]->mutable_cpu_diff();
  Dtype* input_gate_bias_diff = this->blobs_[3]->mutable_cpu_diff();
  Dtype* forget_gate_weight_diff = this->blobs_[4]->mutable_cpu_diff();
  Dtype* forget_gate_bias_diff = this->blobs_[5]->mutable_cpu_diff();
  Dtype* output_gate_weight_diff = this->blobs_[6]->mutable_cpu_diff();
  Dtype* output_gate_bias_diff = this->blobs_[7]->mutable_cpu_diff();

  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* next_state_data = top[1]->cpu_data();
  const Dtype* next_state_diff = top[1]->cpu_diff();

  for (int i = 0; i < channels_; i++) {
    for (int j = 0; j < height_; j++) {
      const int idx = i * height_ + j;

      Dtype input = input_bias[idx];
      Dtype input_gate_mult = input_gate_bias[i];
      Dtype forget_gate_mult = forget_gate_bias[i];
      Dtype output_gate_mult = output_gate_bias[i];
      for (int k = 0; k < input_data_size; k++) {
        input += input_weight[k + idx*input_data_size] * input_data[k];
        input_gate_mult += input_gate_weight[k + i * input_data_size] * input_data[k];
        forget_gate_mult += forget_gate_weight[k + i * input_data_size] * input_data[k];
        output_gate_mult += output_gate_weight[k + i * input_data_size] * input_data[k];
      }
      input = sigmoid(input);
      input_gate_mult = sigmoid(input_gate_mult);
      forget_gate_mult = sigmoid(forget_gate_mult);
      output_gate_mult = sigmoid(output_gate_mult);

      const Dtype next_state_tot_diff = next_state_diff[idx] + output_gate_mult * top_diff[idx];

      prev_state_diff[idx] += next_state_tot_diff * forget_gate_mult;

      forget_gate_bias_diff[i] += next_state_tot_diff * prev_state_data[idx] * sigmoid_diff(forget_gate_mult);
      output_gate_bias_diff[i] += top_diff[idx] * next_state_data[idx] * sigmoid_diff(output_gate_mult);
      input_gate_bias_diff[i] += next_state_tot_diff * input * sigmoid_diff(input_gate_mult);
      input_bias_diff[idx] += next_state_tot_diff * input_gate_mult * sigmoid_diff(input);

      for (int k = 0; k < input_data_size; k++) {
        input_weight_diff[k + idx * input_data_size] += sigmoid_diff(input) * next_state_tot_diff * input_gate_mult * input_data[k];
        input_gate_weight_diff[k + i * input_data_size] += sigmoid_diff(input_gate_mult) * next_state_tot_diff * input * input_data[k];
        forget_gate_weight_diff[k + i * input_data_size] += sigmoid_diff(forget_gate_mult) * prev_state_data[idx] * next_state_tot_diff * input_data[k];
        output_gate_weight_diff[k + i * input_data_size] += sigmoid_diff(output_gate_mult) * top_diff[idx] * next_state_data[idx] * input_data[k];
        input_diff[k] += sigmoid_diff(input) * next_state_tot_diff * input_gate_mult * input_weight[k + idx * input_data_size];
        input_diff[k] += sigmoid_diff(input_gate_mult) * next_state_tot_diff * input * input_gate_weight[k + i * input_data_size];
        input_diff[k] += sigmoid_diff(forget_gate_mult) * next_state_tot_diff * prev_state_data[idx] * forget_gate_weight[k + i * input_data_size];
        input_diff[k] += sigmoid_diff(output_gate_mult) * top_diff[idx] * next_state_data[idx] * output_gate_weight[k + i * input_data_size];
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(LstmLayer);
#endif

INSTANTIATE_CLASS(LstmLayer);

}  // namespace caffe
