intrinsics/gen/textureLoad/fe222a.wgsl:29:24 warning: use of deprecated intrinsic
  var res: vec4<f32> = textureLoad(arg_0, 1);
                       ^^^^^^^^^^^

Texture1D<float4> arg_0 : register(t0, space1);

void textureLoad_fe222a() {
  float4 res = arg_0.Load(int2(1, 0));
}

struct tint_symbol {
  float4 value : SV_Position;
};

tint_symbol vertex_main() {
  textureLoad_fe222a();
  const tint_symbol tint_symbol_1 = {float4(0.0f, 0.0f, 0.0f, 0.0f)};
  return tint_symbol_1;
}

void fragment_main() {
  textureLoad_fe222a();
  return;
}

[numthreads(1, 1, 1)]
void compute_main() {
  textureLoad_fe222a();
  return;
}
