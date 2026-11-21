#!/usr/bin/env python3
"""Generate dispatch_pc.c (1036-slot GLI dispatch table) from ios-dispatch.txt."""
import sys

MAP = {
    'glEnable': 'rpc_Enable', 'glDisable': 'rpc_Disable',
    'glViewport': 'rpc_Viewport', 'glScissor': 'rpc_Scissor',
    'glClearColor': 'rpc_ClearColor', 'glClear': 'rpc_Clear',
    'glColorMask': 'rpc_ColorMask',
    'glBlendFunc': 'rpc_BlendFunc', 'glBlendFuncSeparate': 'rpc_BlendFuncSeparate',
    'glBlendEquation': 'rpc_BlendEquation', 'glBlendEquationSeparateEXT': 'rpc_BlendEquationSeparate',
    'glBlendColor': 'rpc_BlendColor',
    'glDepthMask': 'rpc_DepthMask', 'glDepthFunc': 'rpc_DepthFunc',
    'glStencilFunc': 'rpc_StencilFunc', 'glStencilFuncSeparate': 'rpc_StencilFuncSeparate',
    'glStencilOp': 'rpc_StencilOp', 'glStencilOpSeparate': 'rpc_StencilOpSeparate',
    'glStencilMask': 'rpc_StencilMask', 'glStencilMaskSeparate': 'rpc_StencilMaskSeparate',
    'glCullFace': 'rpc_CullFace', 'glFrontFace': 'rpc_FrontFace',
    'glActiveTexture': 'rpc_ActiveTexture', 'glPixelStorei': 'rpc_PixelStorei',
    'glFlush': 'rpc_Flush', 'glFinish': 'rpc_Finish',
    'glGetError': 'rpc_GetError', 'glGetString': 'rpc_GetString',
    'glGetIntegerv': 'rpc_GetIntegerv', 'glGetFloatv': 'rpc_GetFloatv', 'glGetBooleanv': 'rpc_GetBooleanv',
    'glCreateShaderObjectARB': 'rpc_CreateShader', 'glShaderSourceARB': 'rpc_ShaderSource',
    'glCompileShaderARB': 'rpc_CompileShader', 'glCreateProgramObjectARB': 'rpc_CreateProgram',
    'glAttachObjectARB': 'rpc_AttachShader', 'glBindAttribLocationARB': 'rpc_BindAttribLocation',
    'glLinkProgramARB': 'rpc_LinkProgram', 'glUseProgramObjectARB': 'rpc_UseProgram',
    'glDeleteObjectARB': 'rpc_DeleteObjectARB', 'glGetObjectParameterivARB': 'rpc_GetObjectParameterivARB',
    'glGetInfoLogARB': 'rpc_GetShaderInfoLog',
    'glGetShaderiv': 'rpc_GetShaderiv', 'glGetProgramiv': 'rpc_GetProgramiv',
    'glGetShaderInfoLog': 'rpc_GetShaderInfoLog', 'glGetProgramInfoLog': 'rpc_GetProgramInfoLog',
    'glGetUniformLocationARB': 'rpc_GetUniformLocation', 'glGetAttribLocationARB': 'rpc_GetAttribLocation',
    'glUniform1i': 'rpc_Uniform1i', 'glUniform2i': 'rpc_Uniform2i', 'glUniform3i': 'rpc_Uniform3i', 'glUniform4i': 'rpc_Uniform4i',
    'glUniform1iv': 'rpc_Uniform1iv', 'glUniform2iv': 'rpc_Uniform2iv', 'glUniform3iv': 'rpc_Uniform3iv', 'glUniform4iv': 'rpc_Uniform4iv',
    'glUniform1f': 'rpc_Uniform1f', 'glUniform2f': 'rpc_Uniform2f', 'glUniform3f': 'rpc_Uniform3f', 'glUniform4f': 'rpc_Uniform4f',
    'glUniform1fv': 'rpc_Uniform1fv', 'glUniform2fv': 'rpc_Uniform2fv', 'glUniform3fv': 'rpc_Uniform3fv', 'glUniform4fv': 'rpc_Uniform4fv',
    'glUniformMatrix2fv': 'rpc_UniformMatrix2fv', 'glUniformMatrix3fv': 'rpc_UniformMatrix3fv', 'glUniformMatrix4fv': 'rpc_UniformMatrix4fv',
    'glGenBuffers': 'rpc_GenBuffers', 'glDeleteBuffers': 'rpc_DeleteBuffers',
    'glBindBuffer': 'rpc_BindBuffer', 'glBufferData': 'rpc_BufferData', 'glBufferSubData': 'rpc_BufferSubData',
    'glGenVertexArraysAPPLE': 'rpc_GenVertexArrays', 'glDeleteVertexArraysAPPLE': 'rpc_DeleteVertexArrays',
    'glBindVertexArrayAPPLE': 'rpc_BindVertexArray',
    'glEnableVertexAttribArrayARB': 'rpc_EnableVertexAttribArray', 'glDisableVertexAttribArrayARB': 'rpc_DisableVertexAttribArray',
    'glVertexAttribPointerARB': 'rpc_VertexAttribPointer', 'glVertexAttrib4fARB': 'rpc_VertexAttrib4f',
    'glDrawArrays': 'rpc_DrawArrays', 'glDrawElements': 'rpc_DrawElements',
    'glGenTextures': 'rpc_GenTextures', 'glDeleteTextures': 'rpc_DeleteTextures',
    'glBindTexture': 'rpc_BindTexture', 'glTexParameteri': 'rpc_TexParameteri', 'glTexParameterf': 'rpc_TexParameterf',
    'glTexImage2D': 'rpc_TexImage2D', 'glTexSubImage2D': 'rpc_TexSubImage2D',
    'glGenerateMipmapEXT': 'rpc_GenerateMipmap',
    'glGenFramebuffersEXT': 'rpc_GenFramebuffers', 'glDeleteFramebuffersEXT': 'rpc_DeleteFramebuffers',
    'glBindFramebufferEXT': 'rpc_BindFramebuffer', 'glFramebufferTexture2DEXT': 'rpc_FramebufferTexture2D',
    'glCheckFramebufferStatusEXT': 'rpc_CheckFramebufferStatus',
    'glGenRenderbuffersEXT': 'rpc_GenRenderbuffers', 'glDeleteRenderbuffersEXT': 'rpc_DeleteRenderbuffers',
    'glBindRenderbufferEXT': 'rpc_BindRenderbuffer', 'glRenderbufferStorageEXT': 'rpc_RenderbufferStorage',
    'glFramebufferRenderbufferEXT': 'rpc_FramebufferRenderbuffer',
    'glBlitFramebufferEXT': 'rpc_BlitFramebuffer', 'glDiscardFramebufferEXT': 'rpc_DiscardFramebuffer',
    'glFinishObjectAPPLE': 'rpc_FinishObjectAPPLE', 'glTestObjectAPPLE': 'rpc_TestObjectAPPLE',
}

slots = {}
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        slots[int(parts[0])] = parts[1]

out = ['#include <stdint.h>\n', 'extern void sgl_unhandled(int slot, const char *name);\n']
for fn in sorted(set(MAP.values())):
    out.append(f'extern void {fn}(void);\n')

for i in range(1036):
    name = slots.get(i, f'slot_{i}')
    if name not in MAP:
        out.append(f'static uintptr_t stub_{i}(void *r) {{ (void)r; sgl_unhandled({i}, "{name}"); return 0; }}\n')

out.append('\nvoid *const sgl_dispatch[1036] = {\n')
for i in range(1036):
    name = slots.get(i, f'slot_{i}')
    target = MAP.get(name, f'stub_{i}')
    out.append(f'\t(void *){target},\n')
out.append('};\n')

with open(sys.argv[2], 'w', newline='\n') as f:
    f.writelines(out)
print(f'Generated {sys.argv[2]}: {sum(1 for n in slots.values() if n in MAP)} mapped RPC slots')
