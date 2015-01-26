#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// atomicAdd does not support doubles
template<typename T>
__device__ T myAtomicAdd(T *address, T val){
    return;
};

template<>
__device__ float myAtomicAdd<float>(float *address, float val){
    return atomicAdd(address, val);
};

template<>
__device__ double myAtomicAdd<double>(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

template <typename Dtype>
__device__ Dtype cuda_sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
__device__ Dtype cuda_sigmoid_diff(Dtype x) {
  return x * (1. - x);
}

template <typename Dtype>
__device__ Dtype cuda_tanh(Dtype x) {
  Dtype exp2x = exp(2 * x);
  return abs(x) < Dtype(5) ? ((exp2x - Dtype(1)) / (exp2x + Dtype(1))) : (x > 0 ? Dtype(1) : Dtype(-1));
}

template <typename Dtype>
__device__ Dtype cuda_tanh_diff(Dtype x) {
  return (1. - x * x);
}

template <typename Dtype>
__global__ void LstmForward(
      const Dtype* input_weight,
      const Dtype* input_bias,
      const Dtype* input_gate_weight,
      const Dtype* input_gate_bias,
      const Dtype* forget_gate_weight,
      const Dtype* forget_gate_bias,
      const Dtype* output_gate_weight,
      const Dtype* output_gate_bias,
      const Dtype* input_data,
      Dtype* top_data,
      const Dtype* prev_state_data,
      Dtype* next_state_data,
      const int input_data_size,
      const int channels) {    
  const int n = blockIdx.x;
  const int i = threadIdx.x;

  const int state_offset = n * channels;
  const int input_offset = n * input_data_size;
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
  input = cuda_tanh(input);
  input_gate_mult = cuda_sigmoid(input_gate_mult);
  forget_gate_mult = cuda_sigmoid(forget_gate_mult);
  output_gate_mult = cuda_sigmoid(output_gate_mult);

  next_state_data[state_offset + i] = prev_state_data[state_offset + i] * forget_gate_mult + input * input_gate_mult;
  top_data[state_offset + i] = next_state_data[state_offset + i] * output_gate_mult;
}

template <typename Dtype>
__global__ void LstmBackward(
      const Dtype* input_weight,
      const Dtype* input_bias,
      const Dtype* input_gate_weight,
      const Dtype* input_gate_bias,
      const Dtype* forget_gate_weight,
      const Dtype* forget_gate_bias,
      const Dtype* output_gate_weight,
      const Dtype* output_gate_bias,
      Dtype* input_weight_diff,
      Dtype* input_bias_diff,
      Dtype* input_gate_weight_diff,
      Dtype* input_gate_bias_diff,
      Dtype* forget_gate_weight_diff,
      Dtype* forget_gate_bias_diff,
      Dtype* output_gate_weight_diff,
      Dtype* output_gate_bias_diff,
      const Dtype* top_diff,
      const Dtype* next_state_diff,
      Dtype* input_diff,
      Dtype* prev_state_diff,
      const Dtype* input_data,
      const Dtype* prev_state_data,
      const Dtype* next_state_data,
      const int input_data_size,
      const int channels) {    
  const int n = blockIdx.x;
  const int i = threadIdx.x;

  const int state_offset = n * channels;
  const int input_offset = n * input_data_size;
  
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
  input = cuda_tanh(input);
  input_gate_mult = cuda_sigmoid(input_gate_mult);
  forget_gate_mult = cuda_sigmoid(forget_gate_mult);
  output_gate_mult = cuda_sigmoid(output_gate_mult);

  const Dtype next_state = next_state_data[state_offset + i];
  const Dtype next_state_tot_diff = next_state_diff[state_offset + i] + output_gate_mult * top_diff[state_offset + i];

  myAtomicAdd(&prev_state_diff[state_offset + i], next_state_tot_diff * forget_gate_mult);

  myAtomicAdd(&forget_gate_bias_diff[i], next_state_tot_diff * prev_state_data[state_offset + i] * cuda_sigmoid_diff(forget_gate_mult));
  myAtomicAdd(&output_gate_bias_diff[i], top_diff[state_offset + i] * next_state * cuda_sigmoid_diff(output_gate_mult));
  myAtomicAdd(&input_gate_bias_diff[i], next_state_tot_diff * input * cuda_sigmoid_diff(input_gate_mult));
  myAtomicAdd(&input_bias_diff[i], next_state_tot_diff * input_gate_mult * cuda_tanh_diff(input));

  myAtomicAdd(&input_gate_weight_diff[cid], cuda_sigmoid_diff(input_gate_mult) * next_state_tot_diff * input * prev_state_data[state_offset + i]);
  myAtomicAdd(&forget_gate_weight_diff[cid], cuda_sigmoid_diff(forget_gate_mult) * prev_state_data[state_offset + i] * next_state_tot_diff * prev_state_data[state_offset + i]);
  myAtomicAdd(&output_gate_weight_diff[cid], cuda_sigmoid_diff(output_gate_mult) * top_diff[state_offset + i] * next_state * prev_state_data[state_offset + i]);
  myAtomicAdd(&prev_state_diff[state_offset + i], cuda_sigmoid_diff(input_gate_mult) * next_state_tot_diff * input * input_gate_weight[cid]);
  myAtomicAdd(&prev_state_diff[state_offset + i], cuda_sigmoid_diff(forget_gate_mult) * next_state_tot_diff * prev_state_data[state_offset + i] * forget_gate_weight[cid]);
  myAtomicAdd(&prev_state_diff[state_offset + i], cuda_sigmoid_diff(output_gate_mult) * top_diff[state_offset + i] * next_state * output_gate_weight[cid]);

  for (int k = 0; k < input_data_size; k++) {
    const int wid = k + i * (input_data_size + 1);
    myAtomicAdd(&input_weight_diff[wid], cuda_tanh_diff(input) * next_state_tot_diff * input_gate_mult * input_data[input_offset + k]);
    myAtomicAdd(&input_gate_weight_diff[wid], cuda_sigmoid_diff(input_gate_mult) * next_state_tot_diff * input * input_data[input_offset + k]);
    myAtomicAdd(&forget_gate_weight_diff[wid], cuda_sigmoid_diff(forget_gate_mult) * prev_state_data[state_offset + i] * next_state_tot_diff * input_data[input_offset + k]);
    myAtomicAdd(&output_gate_weight_diff[wid], cuda_sigmoid_diff(output_gate_mult) * top_diff[state_offset + i] * next_state * input_data[input_offset + k]);
    myAtomicAdd(&input_diff[input_offset + k], cuda_tanh_diff(input) * next_state_tot_diff * input_gate_mult * input_weight[wid]);
    myAtomicAdd(&input_diff[input_offset + k], cuda_sigmoid_diff(input_gate_mult) * next_state_tot_diff * input * input_gate_weight[wid]);
    myAtomicAdd(&input_diff[input_offset + k], cuda_sigmoid_diff(forget_gate_mult) * next_state_tot_diff * prev_state_data[state_offset + i] * forget_gate_weight[wid]);
    myAtomicAdd(&input_diff[input_offset + k], cuda_sigmoid_diff(output_gate_mult) * top_diff[state_offset + i] * next_state * output_gate_weight[wid]);
  }
}

template <typename Dtype>
void LstmLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // const Dtype* input_data = bottom[0]->gpu_data();
  // const Dtype* prev_state_data = bottom[1]->gpu_data();
  // const int input_data_size = (bottom[0]->channels() *
  //                              bottom[0]->width() *
  //                              bottom[0]->height());

  // const Dtype* input_weight = this->blobs_[0]->gpu_data();
  // const Dtype* input_bias = this->blobs_[1]->gpu_data();
  // const Dtype* input_gate_weight = this->blobs_[2]->gpu_data();
  // const Dtype* input_gate_bias = this->blobs_[3]->gpu_data();
  // const Dtype* forget_gate_weight = this->blobs_[4]->gpu_data();
  // const Dtype* forget_gate_bias = this->blobs_[5]->gpu_data();
  // const Dtype* output_gate_weight = this->blobs_[6]->gpu_data();
  // const Dtype* output_gate_bias = this->blobs_[7]->gpu_data();

  // Dtype* top_data = (*top)[0]->mutable_gpu_data();
  // Dtype* next_state_data = (*top)[1]->mutable_gpu_data();

  // LstmForward<Dtype><<<num_, channels_>>>(
  //     input_weight,
  //     input_bias,
  //     input_gate_weight,
  //     input_gate_bias,
  //     forget_gate_weight,
  //     forget_gate_bias,
  //     output_gate_weight,
  //     output_gate_bias,
  //     input_data,
  //     top_data,
  //     prev_state_data,
  //     next_state_data,
  //     input_data_size,
  //     channels_);

  // CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void LstmLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  // for (int i = 0; i < 2; ++i) {
  //   caffe_gpu_set((*bottom)[i]->count(), Dtype(0), (*bottom)[i]->mutable_gpu_diff());
  // }
  // const Dtype* input_data = (*bottom)[0]->gpu_data();
  // Dtype* input_diff = (*bottom)[0]->mutable_gpu_diff();
  // const Dtype* prev_state_data = (*bottom)[1]->gpu_data();
  // Dtype* prev_state_diff = (*bottom)[1]->mutable_gpu_diff();
  // const int input_data_size = (*bottom)[0]->channels() * (*bottom)[0]->width() * (*bottom)[0]->height();

  // const Dtype* input_weight = this->blobs_[0]->gpu_data();
  // const Dtype* input_bias = this->blobs_[1]->gpu_data();
  // const Dtype* input_gate_weight = this->blobs_[2]->gpu_data();
  // const Dtype* input_gate_bias = this->blobs_[3]->gpu_data();
  // const Dtype* forget_gate_weight = this->blobs_[4]->gpu_data();
  // const Dtype* forget_gate_bias = this->blobs_[5]->gpu_data();
  // const Dtype* output_gate_weight = this->blobs_[6]->gpu_data();
  // const Dtype* output_gate_bias = this->blobs_[7]->gpu_data();

  // for (int i = 0; i < 8; ++i) {
  //   caffe_gpu_set(this->blobs_[i]->count(), Dtype(0), this->blobs_[i]->mutable_gpu_diff());
  // }

  // Dtype* input_weight_diff = this->blobs_[0]->mutable_gpu_diff();
  // Dtype* input_bias_diff = this->blobs_[1]->mutable_gpu_diff();
  // Dtype* input_gate_weight_diff = this->blobs_[2]->mutable_gpu_diff();
  // Dtype* input_gate_bias_diff = this->blobs_[3]->mutable_gpu_diff();
  // Dtype* forget_gate_weight_diff = this->blobs_[4]->mutable_gpu_diff();
  // Dtype* forget_gate_bias_diff = this->blobs_[5]->mutable_gpu_diff();
  // Dtype* output_gate_weight_diff = this->blobs_[6]->mutable_gpu_diff();
  // Dtype* output_gate_bias_diff = this->blobs_[7]->mutable_gpu_diff();

  // const Dtype* top_diff = top[0]->gpu_diff();
  // const Dtype* next_state_data = top[1]->gpu_data();
  // const Dtype* next_state_diff = top[1]->gpu_diff();

  // LstmBackward<Dtype><<<num_, channels_>>>(
  //     input_weight,
  //     input_bias,
  //     input_gate_weight,
  //     input_gate_bias,
  //     forget_gate_weight,
  //     forget_gate_bias,
  //     output_gate_weight,
  //     output_gate_bias,
  //     input_weight_diff,
  //     input_bias_diff,
  //     input_gate_weight_diff,
  //     input_gate_bias_diff,
  //     forget_gate_weight_diff,
  //     forget_gate_bias_diff,
  //     output_gate_weight_diff,
  //     output_gate_bias_diff,
  //     top_diff,
  //     next_state_diff,
  //     input_diff,
  //     prev_state_diff,
  //     input_data,
  //     prev_state_data,
  //     next_state_data,
  //     input_data_size,
  //     channels_);

  // CUDA_POST_KERNEL_CHECK;


//   Dtype max_input_diff = Dtype(0.0);
//   Dtype max_prev_diff = Dtype(0.0);
//   for (int n = 0; n < num_; ++n) {
//     for (int k = 0; k < input_data_size; ++k) {
//       Dtype* input_diff_cpu = (*bottom)[0]->mutable_cpu_diff();
//       input_diff_cpu[k + n * input_data_size] = std::max(Dtype(-0.001), std::min(Dtype(0.001), input_diff_cpu[k + n * input_data_size]));
//       /*max_input_diff = std::max(max_input_diff, input_diff_cpu[k]);*/
//     }

//     for (int k = 0; k < channels_; ++k) {
//       Dtype* prev_state_diff_cpu = (*bottom)[1]->mutable_cpu_diff();
//       prev_state_diff_cpu[k + n * channels_] = std::max(Dtype(-0.001), std::min(Dtype(0.001), prev_state_diff_cpu[k + n * channels_]));
//       /*max_prev_diff = std::max(max_prev_diff, prev_state_diff_cpu[k]);*/
//     }
//   }
//   // std::cout << "input_diff: " << max_input_diff << "\n";
//   /*std::cout << "prev_state_diff: " << max_prev_diff << "\n";*/
}

INSTANTIATE_CLASS(LstmLayer);

}  // namespace caffe
