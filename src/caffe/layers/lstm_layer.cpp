#include <vector>
#include "caffe/layer.hpp"
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
inline Dtype tanh(Dtype x) {
  Dtype exp2x = exp(2 * x);
  return abs(x) < Dtype(5) ? ((exp2x - Dtype(1)) / (exp2x + Dtype(1))) : (x > 0 ? Dtype(1) : Dtype(-1));
}

template <typename Dtype>
inline Dtype tanh_diff(Dtype x) {
  return (1. - x * x);
}

template <typename Dtype>
void LstmLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  LstmParameter lstm_param = this->layer_param_.lstm_param();
  CHECK((lstm_param.has_num_cells()))
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
  channels_ = lstm_param.num_cells();
  const int input_data_size = (bottom[0]->channels() *
                               bottom[0]->width() *
                               bottom[0]->height());

  this->blobs_.resize(8);
  this->blobs_[0].reset(new Blob<Dtype>(
      1, channels_, 1, input_data_size + 1));
  shared_ptr<Filler<Dtype> > input_weight_filler(GetFiller<Dtype>(
      this->layer_param_.lstm_param().input_weight_filler()));
  input_weight_filler->Fill(this->blobs_[0].get());

  this->blobs_[2].reset(new Blob<Dtype>(
      1, channels_, 1, input_data_size + 1));
  shared_ptr<Filler<Dtype> > input_gate_weight_filler(GetFiller<Dtype>(
      this->layer_param_.lstm_param().input_gate_weight_filler()));
  input_gate_weight_filler->Fill(this->blobs_[2].get());

  this->blobs_[4].reset(new Blob<Dtype>(
      1, channels_, 1, input_data_size + 1));
  shared_ptr<Filler<Dtype> > forget_gate_weight_filler(GetFiller<Dtype>(
      this->layer_param_.lstm_param().forget_gate_weight_filler()));
  forget_gate_weight_filler->Fill(this->blobs_[4].get());

  this->blobs_[6].reset(new Blob<Dtype>(
      1, channels_, 1, input_data_size + 1));
  shared_ptr<Filler<Dtype> > output_gate_weight_filler(GetFiller<Dtype>(
      this->layer_param_.lstm_param().output_gate_weight_filler()));
  output_gate_weight_filler->Fill(this->blobs_[6].get());


  this->blobs_[1].reset(new Blob<Dtype>(1, channels_, 1, 1));
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

  for (int i = 0; i < channels_; i++) {
    this->blobs_[0]->mutable_cpu_data()[(i + 1) * (input_data_size + 1) - 1] = 0;
    this->blobs_[2]->mutable_cpu_data()[(i + 1) * (input_data_size + 1) - 1] = Dtype(this->layer_param_.lstm_param().input_gate_cell_weight_filler());
    this->blobs_[4]->mutable_cpu_data()[(i + 1) * (input_data_size + 1) - 1] = Dtype(this->layer_param_.lstm_param().forget_gate_cell_weight_filler());
    this->blobs_[6]->mutable_cpu_data()[(i + 1) * (input_data_size + 1) - 1] = Dtype(this->layer_param_.lstm_param().output_gate_cell_weight_filler());
  }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void LstmLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK((this->layer_param_.bottom_size() == 2 || this->layer_param_.bottom_size() == 0))
      << "LSTM must have a data and cell bottom";
  CHECK((this->layer_param_.top_size() == 2 || this->layer_param_.top_size() == 0))
      << "LSTM must have a data and cell top";
  num_ = bottom[0]->num();
  (*top)[0]->Reshape(num_, channels_, 1, 1);
  (*top)[1]->Reshape(num_, channels_, 1, 1);
}

template <typename Dtype>
void LstmLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* prev_state_data = bottom[1]->cpu_data();
  const int input_data_size = (bottom[0]->channels() *
                               bottom[0]->width() *
                               bottom[0]->height());

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

  for (int n = 0; n < num_; n++) {
    const int state_offset = n * channels_;
    const int input_offset = n * input_data_size;
    for (int i = 0; i < channels_; i++) {
      const int cid = (i + 1) * (input_data_size + 1) - 1;
      Dtype input = input_bias[i];
      Dtype input_gate_mult = input_gate_bias[i] + input_gate_weight[cid] * prev_state_data[state_offset + i];
      Dtype forget_gate_mult = forget_gate_bias[i] + forget_gate_weight[cid] * prev_state_data[state_offset + i];
      Dtype output_gate_mult = output_gate_bias[i] + output_gate_weight[cid] * prev_state_data[state_offset + i];
      for (int k = 0; k < input_data_size; k++) {
        const int wid = k + i * (input_data_size + 1);
        input += input_weight[wid] * input_data[input_offset + k];
        input_gate_mult += input_gate_weight[wid] * input_data[input_offset + k];
        forget_gate_mult += forget_gate_weight[wid] * input_data[input_offset + k];
        output_gate_mult += output_gate_weight[wid] * input_data[input_offset + k];
      }
      input = tanh(input);
      input_gate_mult = sigmoid(input_gate_mult);
      forget_gate_mult = sigmoid(forget_gate_mult);
      output_gate_mult = sigmoid(output_gate_mult);

      next_state_data[state_offset + i] = prev_state_data[state_offset + i] * forget_gate_mult + input * input_gate_mult;
      top_data[state_offset + i] = next_state_data[state_offset + i] * output_gate_mult;
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
  const int input_data_size = (*bottom)[0]->channels() * (*bottom)[0]->width() * (*bottom)[0]->height();

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

  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* next_state_data = top[1]->cpu_data();
  const Dtype* next_state_diff = top[1]->cpu_diff();

  for (int n = 0; n < num_; n++) {
    const int state_offset = n * channels_;
    const int input_offset = n * input_data_size;
    for (int i = 0; i < channels_; i++) {
      const int cid = (i + 1) * (input_data_size + 1) - 1;
      Dtype input = input_bias[i];
      Dtype input_gate_mult = input_gate_bias[i] + input_gate_weight[cid] * prev_state_data[state_offset + i];
      Dtype forget_gate_mult = forget_gate_bias[i] + forget_gate_weight[cid] * prev_state_data[state_offset + i];
      Dtype output_gate_mult = output_gate_bias[i] + output_gate_weight[cid] * prev_state_data[state_offset + i];
      for (int k = 0; k < input_data_size; k++) {
        const int wid = k + i * (input_data_size + 1);
        input += input_weight[wid] * input_data[input_offset + k];
        input_gate_mult += input_gate_weight[wid] * input_data[input_offset + k];
        forget_gate_mult += forget_gate_weight[wid] * input_data[input_offset + k];
        output_gate_mult += output_gate_weight[wid] * input_data[input_offset + k];
      }
      input = tanh(input);
      input_gate_mult = sigmoid(input_gate_mult);
      forget_gate_mult = sigmoid(forget_gate_mult);
      output_gate_mult = sigmoid(output_gate_mult);

      const Dtype next_state = next_state_data[state_offset + i];
      const Dtype next_state_tot_diff = next_state_diff[state_offset + i] + output_gate_mult * top_diff[state_offset + i];

      prev_state_diff[state_offset + i] += next_state_tot_diff * forget_gate_mult;

      forget_gate_bias_diff[i] += next_state_tot_diff * prev_state_data[state_offset + i] * sigmoid_diff(forget_gate_mult);
      output_gate_bias_diff[i] += top_diff[state_offset + i] * next_state * sigmoid_diff(output_gate_mult);
      input_gate_bias_diff[i] += next_state_tot_diff * input * sigmoid_diff(input_gate_mult);
      input_bias_diff[i] += next_state_tot_diff * input_gate_mult * tanh_diff(input);

      input_gate_weight_diff[cid] += sigmoid_diff(input_gate_mult) * next_state_tot_diff * input * prev_state_data[state_offset + i];
      forget_gate_weight_diff[cid] += sigmoid_diff(forget_gate_mult) * prev_state_data[state_offset + i] * next_state_tot_diff * prev_state_data[state_offset + i];
      output_gate_weight_diff[cid] += sigmoid_diff(output_gate_mult) * top_diff[state_offset + i] * next_state * prev_state_data[state_offset + i];
      prev_state_diff[state_offset + i] += sigmoid_diff(input_gate_mult) * next_state_tot_diff * input * input_gate_weight[cid];
      prev_state_diff[state_offset + i] += sigmoid_diff(forget_gate_mult) * next_state_tot_diff * prev_state_data[state_offset + i] * forget_gate_weight[cid];
      prev_state_diff[state_offset + i] += sigmoid_diff(output_gate_mult) * top_diff[state_offset + i] * next_state * output_gate_weight[cid];

      for (int k = 0; k < input_data_size; k++) {
        const int wid = k + i * (input_data_size + 1);
        input_weight_diff[wid] += tanh_diff(input) * next_state_tot_diff * input_gate_mult * input_data[input_offset + k];
        input_gate_weight_diff[wid] += sigmoid_diff(input_gate_mult) * next_state_tot_diff * input * input_data[input_offset + k];
        forget_gate_weight_diff[wid] += sigmoid_diff(forget_gate_mult) * prev_state_data[state_offset + i] * next_state_tot_diff * input_data[input_offset + k];
        output_gate_weight_diff[wid] += sigmoid_diff(output_gate_mult) * top_diff[state_offset + i] * next_state * input_data[input_offset + k];
        input_diff[input_offset + k] += tanh_diff(input) * next_state_tot_diff * input_gate_mult * input_weight[wid];
        input_diff[input_offset + k] += sigmoid_diff(input_gate_mult) * next_state_tot_diff * input * input_gate_weight[wid];
        input_diff[input_offset + k] += sigmoid_diff(forget_gate_mult) * next_state_tot_diff * prev_state_data[state_offset + i] * forget_gate_weight[wid];
        input_diff[input_offset + k] += sigmoid_diff(output_gate_mult) * top_diff[state_offset + i] * next_state * output_gate_weight[wid];
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(LstmLayer);
#endif

INSTANTIATE_CLASS(LstmLayer);

}  // namespace caffe
