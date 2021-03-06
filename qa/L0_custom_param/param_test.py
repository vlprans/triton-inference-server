#!/usr/bin/python

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import os
import sys
from builtins import range
import tritongrpcclient as grpcclient
import tritonhttpclient as httpclient
from tritonclientutils.utils import np_to_triton_dtype

FLAGS = None

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                       help='Enable verbose output')
   parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8000',
                       help='Inference server URL. Default is localhost:8000.')
   parser.add_argument('-i', '--protocol', type=str, required=False, default='http',
                       help='Protocol ("http"/"grpc") used to ' +
                       'communicate with inference service. Default is "http".')

   FLAGS = parser.parse_args()
   if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
      print("unexpected protocol \"{}\", expects \"http\" or \"grpc\"".format(FLAGS.protocol))
      exit(1)

   client_util = httpclient if FLAGS.protocol == "http" else grpcclient

   model_name = "param"

   # Create the inference context for the model.
   client = client_util.InferenceServerClient(FLAGS.url, FLAGS.verbose)

   # Input tensor can be any size int32 vector...
   input_data = np.zeros(shape=1, dtype=np.int32)

   inputs = [client_util.InferInput(
                  "INPUT", input_data.shape, np_to_triton_dtype(input_data.dtype))]
   inputs[0].set_data_from_numpy(input_data)

   results = client.infer(model_name, inputs)

   print(results)

   params = results.as_numpy("OUTPUT")
   if params is None:
      print("error: expected 'OUTPUT'")
      sys.exit(1)

   if params.size != 5:
      print("error: expected 5 output strings, got {}".format(params.size))
      sys.exit(1)

   # Element type returned is different between HTTP and GRPC client.
   # The former is str and the latter is bytes
   params = [p if type(p) == str else p.decode('utf8') for p in params]
   p0 = params[0]
   if not p0.startswith("INPUT=0"):
      print("error: expected INPUT=0 string, got {}".format(p0))
      sys.exit(1)

   p1 = params[1]
   if not p1.startswith("server_0="):
      print("error: expected server_0 parameter, got {}".format(p1))
      sys.exit(1)

   p2 = params[2]
   if not p2.startswith("server_1="):
      print("error: expected server_1 parameter, got {}".format(p2))
      sys.exit(1)
   if not p2.endswith("L0_custom_param/models"):
      print("error: expected model-repository to end with L0_custom_backend/models, got {}".format(p2));
      sys.exit(1)

   # configuration param values can be returned in any order.
   p3 = params[3]
   p4 = params[4]
   if p3.startswith("param1"):
      p3, p4 = p4, p3

   if p3 != "param0=value0":
      print("error: expected param0=value0, got {}".format(p3));
      sys.exit(1)

   if p4 != "param1=value1":
      print("error: expected param1=value1, got {}".format(p4));
      sys.exit(1)
