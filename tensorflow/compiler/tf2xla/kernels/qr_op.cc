/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/qr.h"
#include "xla/hlo/builder/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {

class QROp : public XlaOpKernel {
 public:
  explicit QROp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("full_matrices", &full_matrices_));
  }
  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);

    if (input_shape.dims() < 2) {
      ctx->ReportError(errors::InvalidArgument(
          "QR decomposition requires an input with rank at least 2, but got rank ",
          input_shape.dims()));
      return;
    }

    xla::XlaOp q, r;
    xla::QrExplicit(ctx->Input(0), full_matrices_, q, r);
    ctx->SetOutput(0, q);
    ctx->SetOutput(1, r);
  }

 private:
  // If true, compute full-sized q and r. If false, compute only the leading P
  // columns of q.
  bool full_matrices_;
};

REGISTER_XLA_OP(Name("Qr"), QROp);

}  // namespace
}  // namespace tensorflow
