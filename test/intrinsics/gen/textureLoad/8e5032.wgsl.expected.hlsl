intrinsics/gen/textureLoad/8e5032.wgsl:29:24 warning: use of deprecated intrinsic
  var res: vec4<u32> = textureLoad(arg_0, vec2<i32>(), 1);
                       ^^^^^^^^^^^

Texture2DArray<uint4> arg_0 : register(t0, space1);

void textureLoad_8e5032() {
  uint4 res = arg_0.Load(int4(0, 0, 1, 0));
}

struct tint_symbol {
  float4 value : SV_Position;
};

tint_symbol vertex_main() {
  textureLoad_8e5032();
  const tint_symbol tint_symbol_1 = {float4(0.0f, 0.0f, 0.0f, 0.0f)};
  return tint_symbol_1;
}

void fragment_main() {
  textureLoad_8e5032();
  return;
}

[numthreads(1, 1, 1)]
void compute_main() {
  textureLoad_8e5032();
  return;
}
