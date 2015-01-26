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
  CHECK((lstm_param.has_input_gate_weight_filler()))
      << "lstm_param.has_input_gate_weight_filler()";
  CHECK((lstm_param.has_forget_gate_weight_filler()))
      << "lstm_param.has_forget_gate_weight_filler()";
  CHECK((lstm_param.has_output_gate_weight_filler()))
      << "lstm_param.has_output_gate_weight_filler()";
  M_ = lstm_param.num_cells();
  K_ = (bottom[0]->channels() *
        bottom[0]->width() *
        bottom[0]->height());
  N_ = bottom[0]->num();
  num_ = K_;
  channels_ = M_;

  this->blobs_.resize(4);
  for (int i=0; i<4; i++) {
      this->blobs_[i].reset(new Blob<Dtype>(
          1, M_, 1, K_));
  }

  shared_ptr<Filler<Dtype> > input_weight_filler(GetFiller<Dtype>(
      this->layer_param_.lstm_param().input_weight_filler()));
  input_weight_filler->Fill(this->blobs_[0].get());

  shared_ptr<Filler<Dtype> > input_gate_weight_filler(GetFiller<Dtype>(
      this->layer_param_.lstm_param().input_gate_weight_filler()));
  input_gate_weight_filler->Fill(this->blobs_[1].get());

  shared_ptr<Filler<Dtype> > forget_gate_weight_filler(GetFiller<Dtype>(
      this->layer_param_.lstm_param().forget_gate_weight_filler()));
  forget_gate_weight_filler->Fill(this->blobs_[2].get());

  shared_ptr<Filler<Dtype> > output_gate_weight_filler(GetFiller<Dtype>(
      this->layer_param_.lstm_param().output_gate_weight_filler()));
  output_gate_weight_filler->Fill(this->blobs_[3].get());

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
  gates_buffer_.Reshape(N_, 4 * M_, 1, 1);
  (*top)[0]->Reshape(N_, M_, 1, 1);
  (*top)[1]->Reshape(N_, M_, 1, 1);
}

template <typename Dtype>
void LstmLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* prev_state_data = bottom[1]->cpu_data();

  const Dtype* input_weight = this->blobs_[0]->cpu_data();
  const Dtype* input_gate_weight = this->blobs_[1]->cpu_data();
  const Dtype* forget_gate_weight = this->blobs_[2]->cpu_data();
  const Dtype* output_gate_weight = this->blobs_[3]->cpu_data();

  Dtype* next_hidden_state = (*top)[0]->mutable_cpu_data();
  Dtype* next_memory_state = (*top)[1]->mutable_cpu_data();

  Dtype* gates_data = gates_buffer_.mutable_cpu_data();

  Dtype* input_gates = gates_data + M_ * N_ * 0;
  Dtype* forget_gates = gates_data + M_ * N_ * 1;
  Dtype* output_gates = gates_data + M_ * N_ * 2;
  Dtype* input_values = gates_data + M_ * N_ * 3;

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
    (Dtype)1., input_weight, input_data,
    (Dtype)0., input_values);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
    (Dtype)1., input_gate_weight, input_data,
    (Dtype)0., input_gates);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
    (Dtype)1., forget_gate_weight, input_data,
    (Dtype)0., forget_gates);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
    (Dtype)1., output_gate_weight, input_data,
    (Dtype)0., output_gates);

  for (int i = 0; i < N_; ++i) {
    for (int j = 0; j < M_; ++j) {
      const int idx = j + i * M_;
      const int idy = i + j * N_;
      next_memory_state[idy] = prev_state_data[idy] * sigmoid(forget_gates[idx]) +
          sigmoid(input_gates[idx]) * tanh(input_values[idx]);
      next_hidden_state[idy] = next_memory_state[idy] *
          sigmoid(output_gates[idx]);
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
  const Dtype* input_gate_weight = this->blobs_[1]->cpu_data();
  const Dtype* forget_gate_weight = this->blobs_[2]->cpu_data();
  const Dtype* output_gate_weight = this->blobs_[3]->cpu_data();

  for (int i = 0; i < 4; ++i) {
    caffe_set(this->blobs_[i]->count(), Dtype(0), this->blobs_[i]->mutable_cpu_diff());
  }

  Dtype* input_weight_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* input_gate_weight_diff = this->blobs_[1]->mutable_cpu_diff();
  Dtype* forget_gate_weight_diff = this->blobs_[2]->mutable_cpu_diff();
  Dtype* output_gate_weight_diff = this->blobs_[3]->mutable_cpu_diff();

  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* next_memory_state = top[1]->cpu_data();
  const Dtype* next_state_diff = top[1]->cpu_diff();

  for (int n = 0; n < num_; n++) {
    const int state_offset = n * channels_;
    const int input_offset = n * input_data_size;
    for (int i = 0; i < channels_; i++) {
      Dtype input = Dtype(0);
      Dtype input_gate_mult = Dtype(0);
      Dtype forget_gate_mult = Dtype(0);
      Dtype output_gate_mult = Dtype(0);
      for (int k = 0; k < input_data_size; k++) {
        const int wid = k + i * (input_data_size + 0);
        input += input_weight[wid] * input_data[input_offset + k];
        input_gate_mult += input_gate_weight[wid] * input_data[input_offset + k];
        forget_gate_mult += forget_gate_weight[wid] * input_data[input_offset + k];
        output_gate_mult += output_gate_weight[wid] * input_data[input_offset + k];
      }
      input = tanh(input);
      input_gate_mult = sigmoid(input_gate_mult);
      forget_gate_mult = sigmoid(forget_gate_mult);
      output_gate_mult = sigmoid(output_gate_mult);

      const Dtype next_state = next_memory_state[state_offset + i];
      const Dtype next_state_tot_diff = next_state_diff[state_offset + i] + output_gate_mult * top_diff[state_offset + i];

      prev_state_diff[state_offset + i] += next_state_tot_diff * forget_gate_mult;

      for (int k = 0; k < input_data_size; k++) {
        const int wid = k + i * (input_data_size + 0);
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
