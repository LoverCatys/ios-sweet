/*
 * sweetgl-pc: Host-GPU OpenGL ES 3.0 RPC shim for iOS 16 CoreAnimation on QEMU.
 *
 * Intercepts EAGLContext + CoreAnimation's GLI dispatch table (1036 slots) and
 * streams batched GLES 3.0 commands + IOSurface textures over /dev/sweetgpu
 * (virtio-console) directly to the Windows host GPU (AMD Radeon RX 6600 XT).
 */
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <objc/NSObjCRuntime.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <IOSurface/IOSurfaceRef.h>
#include <OpenGLES/ES3/gl.h>
#include <OpenGLES/ES3/glext.h>

#define SGL_SLOTS 1036
#define TX_BUF_CAP (512 * 1024)

/* RPC opcodes (must match tools/gpu-host.py) */
enum {
	OP_PING                 = 1,
	OP_ENABLE               = 2,
	OP_DISABLE              = 3,
	OP_VIEWPORT             = 4,
	OP_SCISSOR              = 5,
	OP_CLEAR_COLOR          = 6,
	OP_CLEAR                = 7,
	OP_COLOR_MASK           = 8,
	OP_BLEND_FUNC_SEP       = 9,
	OP_BLEND_EQ_SEP         = 10,
	OP_BLEND_COLOR          = 11,
	OP_DEPTH_MASK           = 12,
	OP_DEPTH_FUNC           = 13,
	OP_STENCIL_FUNC_SEP     = 14,
	OP_STENCIL_OP_SEP       = 15,
	OP_STENCIL_MASK_SEP     = 16,
	OP_CULL_FACE            = 17,
	OP_FRONT_FACE           = 18,
	OP_ACTIVE_TEXTURE       = 19,
	OP_PIXEL_STOREI         = 20,
	OP_FLUSH                = 21,
	OP_FINISH               = 22,
	OP_GET_ERROR            = 23,
	OP_GET_STRING           = 24,
	OP_GET_INTEGERV         = 25,
	OP_GET_FLOATV           = 26,
	OP_GET_BOOLEANV         = 27,

	OP_CREATE_SHADER        = 30,
	OP_SHADER_SOURCE        = 31,
	OP_COMPILE_SHADER       = 32,
	OP_DELETE_SHADER        = 33,
	OP_GET_SHADERIV         = 34,
	OP_GET_SHADER_INFO_LOG  = 35,
	OP_CREATE_PROGRAM       = 36,
	OP_ATTACH_SHADER        = 37,
	OP_BIND_ATTRIB_LOC      = 38,
	OP_LINK_PROGRAM         = 39,
	OP_USE_PROGRAM          = 40,
	OP_DELETE_PROGRAM       = 41,
	OP_GET_PROGRAMIV        = 42,
	OP_GET_PROGRAM_INFO_LOG = 43,
	OP_GET_UNIFORM_LOC      = 44,
	OP_GET_ATTRIB_LOC       = 45,
	OP_UNIFORM_IV           = 46,
	OP_UNIFORM_FV           = 47,
	OP_UNIFORM_MAT          = 48,

	OP_GEN_BUFFERS          = 50,
	OP_DELETE_BUFFERS       = 51,
	OP_BIND_BUFFER          = 52,
	OP_BUFFER_DATA          = 53,
	OP_BUFFER_SUB_DATA      = 54,
	OP_GEN_VERTEX_ARRAYS    = 55,
	OP_DELETE_VERTEX_ARRAYS = 56,
	OP_BIND_VERTEX_ARRAY    = 57,
	OP_ENABLE_VATTRIB       = 58,
	OP_DISABLE_VATTRIB      = 59,
	OP_VATTRIB_PTR          = 60,
	OP_DRAW_ARRAYS          = 61,
	OP_DRAW_ELEMENTS        = 62,
	OP_VATTRIB_4F           = 63,

	OP_GEN_TEXTURES         = 70,
	OP_DELETE_TEXTURES      = 71,
	OP_BIND_TEXTURE         = 72,
	OP_TEX_PARAM_I          = 73,
	OP_TEX_PARAM_F          = 74,
	OP_TEX_IMAGE_2D         = 75,
	OP_TEX_SUB_IMAGE_2D     = 76,
	OP_GENERATE_MIPMAP      = 77,

	OP_GEN_FRAMEBUFFERS     = 80,
	OP_DELETE_FRAMEBUFFERS  = 81,
	OP_BIND_FRAMEBUFFER     = 82,
	OP_FRAMEBUFFER_TEX2D    = 83,
	OP_CHECK_FB_STATUS      = 84,
	OP_GEN_RENDERBUFFERS    = 85,
	OP_DELETE_RENDERBUFFERS = 86,
	OP_BIND_RENDERBUFFER    = 87,
	OP_RENDERBUFFER_STORAGE = 88,
	OP_FRAMEBUFFER_RBO      = 89,
	OP_BLIT_FRAMEBUFFER     = 90,
	OP_INVALIDATE_FB        = 91,

	OP_PRESENT_TEX          = 95,
};

extern void *const sgl_dispatch[SGL_SLOTS];

static int gpu_fd = -1;
static pthread_mutex_t rpc_lock = PTHREAD_MUTEX_INITIALIZER;
static uint8_t tx_buf[TX_BUF_CAP];
static size_t tx_len = 0;

/* Track bound element array buffer and active texture target for GL_TEXTURE_RECTANGLE */
static GLuint bound_ebo = 0;
static GLuint bound_tex[32];
static int active_unit = 0;
static GLuint display_tex_id = 0;
static uint32_t display_w = 0, display_h = 0;

static void
sgl_log(const char *fmt, ...)
{
	static int lfd = -1;
	if (lfd < 0) {
		lfd = open("/dev/null", O_WRONLY | O_CLOEXEC);
	}
	char line[512];
	va_list ap;
	va_start(ap, fmt);
	int n = vsnprintf(line, sizeof(line) - 2, fmt, ap);
	va_end(ap);
	if (n > 0) {
		if (n > (int)sizeof(line) - 2) n = (int)sizeof(line) - 2;
		line[n++] = '\n';
		(void)write(lfd, line, (size_t)n);
	}
}

static void
write_all(int fd, const void *buf, size_t n)
{
	const uint8_t *p = (const uint8_t *)buf;
	while (n > 0) {
		ssize_t w = write(fd, p, n);
		if (w <= 0) break;
		p += w;
		n -= (size_t)w;
	}
}

static bool
read_all(int fd, void *buf, size_t n)
{
	uint8_t *p = (uint8_t *)buf;
	while (n > 0) {
		ssize_t r = read(fd, p, n);
		if (r <= 0) return false;
		p += r;
		n -= (size_t)r;
	}
	return true;
}

static void
rpc_flush_locked(void)
{
	if (tx_len > 0 && gpu_fd >= 0) {
		write_all(gpu_fd, tx_buf, tx_len);
		tx_len = 0;
	}
}

static void
rpc_emit_locked(uint16_t op, const void *payload, uint32_t len)
{
	if (gpu_fd < 0) return;
	uint8_t hdr[6];
	memcpy(hdr, &op, 2);
	memcpy(hdr + 2, &len, 4);
	if (tx_len + 6 + len > TX_BUF_CAP) {
		rpc_flush_locked();
	}
	if (6 + len <= TX_BUF_CAP) {
		memcpy(tx_buf + tx_len, hdr, 6);
		tx_len += 6;
		if (len > 0) {
			memcpy(tx_buf + tx_len, payload, len);
			tx_len += len;
		}
	} else {
		write_all(gpu_fd, hdr, 6);
		if (len > 0) write_all(gpu_fd, payload, len);
	}
}

static void
rpc_send(uint16_t op, const void *payload, uint32_t len)
{
	pthread_mutex_lock(&rpc_lock);
	rpc_emit_locked(op, payload, len);
	pthread_mutex_unlock(&rpc_lock);
}

static bool
rpc_call(uint16_t op, const void *payload, uint32_t len, void *reply, size_t reply_len)
{
	pthread_mutex_lock(&rpc_lock);
	rpc_emit_locked(op, payload, len);
	rpc_flush_locked();
	bool ok = (gpu_fd >= 0) && read_all(gpu_fd, reply, reply_len);
	pthread_mutex_unlock(&rpc_lock);
	return ok;
}

static GLenum
norm_tgt(GLenum t)
{
	return (t == 0x84F5 /* GL_TEXTURE_RECTANGLE */) ? GL_TEXTURE_2D : t;
}

void
sgl_unhandled(int slot, const char *name)
{
	static uint8_t seen[SGL_SLOTS];
	if (slot >= 0 && slot < SGL_SLOTS && !seen[slot]) {
		seen[slot] = 1;
		sgl_log("sweetgl-pc: unhandled slot %d (%s)", slot, name);
	}
}

/* ---- GLES 3.0 RPC handlers ---- */

void rpc_Enable(void *r, GLenum cap) { (void)r; rpc_send(OP_ENABLE, &cap, 4); }
void rpc_Disable(void *r, GLenum cap) { (void)r; rpc_send(OP_DISABLE, &cap, 4); }
void rpc_Viewport(void *r, GLint x, GLint y, GLsizei w, GLsizei h) {
	(void)r; int32_t a[4] = {x, y, w, h}; rpc_send(OP_VIEWPORT, a, sizeof(a));
}
void rpc_Scissor(void *r, GLint x, GLint y, GLsizei w, GLsizei h) {
	(void)r; int32_t a[4] = {x, y, w, h}; rpc_send(OP_SCISSOR, a, sizeof(a));
}
void rpc_ClearColor(void *r, GLfloat cr, GLfloat cg, GLfloat cb, GLfloat ca) {
	(void)r; float a[4] = {cr, cg, cb, ca}; rpc_send(OP_CLEAR_COLOR, a, sizeof(a));
}
void rpc_Clear(void *r, GLbitfield mask) { (void)r; rpc_send(OP_CLEAR, &mask, 4); }
void rpc_ColorMask(void *r, GLboolean cr, GLboolean cg, GLboolean cb, GLboolean ca) {
	(void)r; uint8_t a[4] = {cr, cg, cb, ca}; rpc_send(OP_COLOR_MASK, a, 4);
}
void rpc_BlendFunc(void *r, GLenum s, GLenum d) {
	(void)r; uint32_t a[4] = {s, d, s, d}; rpc_send(OP_BLEND_FUNC_SEP, a, sizeof(a));
}
void rpc_BlendFuncSeparate(void *r, GLenum sr, GLenum dr, GLenum sa, GLenum da) {
	(void)r; uint32_t a[4] = {sr, dr, sa, da}; rpc_send(OP_BLEND_FUNC_SEP, a, sizeof(a));
}
void rpc_BlendEquation(void *r, GLenum m) {
	(void)r; uint32_t a[2] = {m, m}; rpc_send(OP_BLEND_EQ_SEP, a, sizeof(a));
}
void rpc_BlendEquationSeparate(void *r, GLenum mr, GLenum ma) {
	(void)r; uint32_t a[2] = {mr, ma}; rpc_send(OP_BLEND_EQ_SEP, a, sizeof(a));
}
void rpc_BlendColor(void *r, GLfloat cr, GLfloat cg, GLfloat cb, GLfloat ca) {
	(void)r; float a[4] = {cr, cg, cb, ca}; rpc_send(OP_BLEND_COLOR, a, sizeof(a));
}
void rpc_DepthMask(void *r, GLboolean f) { (void)r; uint8_t v = f; rpc_send(OP_DEPTH_MASK, &v, 1); }
void rpc_DepthFunc(void *r, GLenum f) { (void)r; rpc_send(OP_DEPTH_FUNC, &f, 4); }
void rpc_StencilFunc(void *r, GLenum f, GLint ref, GLuint m) {
	(void)r; uint32_t a[4] = {GL_FRONT_AND_BACK, f, (uint32_t)ref, m}; rpc_send(OP_STENCIL_FUNC_SEP, a, sizeof(a));
}
void rpc_StencilFuncSeparate(void *r, GLenum face, GLenum f, GLint ref, GLuint m) {
	(void)r; uint32_t a[4] = {face, f, (uint32_t)ref, m}; rpc_send(OP_STENCIL_FUNC_SEP, a, sizeof(a));
}
void rpc_StencilOp(void *r, GLenum sf, GLenum df, GLenum dp) {
	(void)r; uint32_t a[4] = {GL_FRONT_AND_BACK, sf, df, dp}; rpc_send(OP_STENCIL_OP_SEP, a, sizeof(a));
}
void rpc_StencilOpSeparate(void *r, GLenum face, GLenum sf, GLenum df, GLenum dp) {
	(void)r; uint32_t a[4] = {face, sf, df, dp}; rpc_send(OP_STENCIL_OP_SEP, a, sizeof(a));
}
void rpc_StencilMask(void *r, GLuint m) {
	(void)r; uint32_t a[2] = {GL_FRONT_AND_BACK, m}; rpc_send(OP_STENCIL_MASK_SEP, a, sizeof(a));
}
void rpc_StencilMaskSeparate(void *r, GLenum face, GLuint m) {
	(void)r; uint32_t a[2] = {face, m}; rpc_send(OP_STENCIL_MASK_SEP, a, sizeof(a));
}
void rpc_CullFace(void *r, GLenum m) { (void)r; rpc_send(OP_CULL_FACE, &m, 4); }
void rpc_FrontFace(void *r, GLenum m) { (void)r; rpc_send(OP_FRONT_FACE, &m, 4); }
void rpc_ActiveTexture(void *r, GLenum t) {
	(void)r;
	if (t >= GL_TEXTURE0 && t < GL_TEXTURE0 + 32) active_unit = (int)(t - GL_TEXTURE0);
	rpc_send(OP_ACTIVE_TEXTURE, &t, 4);
}
void rpc_PixelStorei(void *r, GLenum pn, GLint p) {
	(void)r; int32_t a[2] = {(int32_t)pn, p}; rpc_send(OP_PIXEL_STOREI, a, sizeof(a));
}
void rpc_Flush(void *r) {
	(void)r;
	pthread_mutex_lock(&rpc_lock);
	if (display_tex_id) {
		uint32_t p[3] = {display_tex_id, display_w, display_h};
		rpc_emit_locked(OP_PRESENT_TEX, p, sizeof(p));
	}
	rpc_emit_locked(OP_FLUSH, NULL, 0);
	rpc_flush_locked();
	pthread_mutex_unlock(&rpc_lock);
}
void rpc_Finish(void *r) {
	(void)r;
	uint32_t ok = 0;
	pthread_mutex_lock(&rpc_lock);
	if (display_tex_id) {
		uint32_t p[3] = {display_tex_id, display_w, display_h};
		rpc_emit_locked(OP_PRESENT_TEX, p, sizeof(p));
	}
	rpc_emit_locked(OP_FINISH, NULL, 0);
	rpc_flush_locked();
	if (gpu_fd >= 0) (void)read_all(gpu_fd, &ok, 4);
	pthread_mutex_unlock(&rpc_lock);
}
GLenum rpc_GetError(void *r) {
	(void)r;
	uint32_t err = 0;
	rpc_call(OP_GET_ERROR, NULL, 0, &err, 4);
	return err;
}
const GLubyte *rpc_GetString(void *r, GLenum name) {
	(void)r;
	static char cache[8][512];
	static int idx = 0;
	uint32_t slen = 0;
	pthread_mutex_lock(&rpc_lock);
	rpc_emit_locked(OP_GET_STRING, &name, 4);
	rpc_flush_locked();
	char *out = cache[(idx++) & 7];
	out[0] = 0;
	if (gpu_fd >= 0 && read_all(gpu_fd, &slen, 4)) {
		uint32_t keep = slen < 511 ? slen : 511;
		if (keep > 0) (void)read_all(gpu_fd, out, keep);
		out[keep] = 0;
		while (slen > keep) {
			uint8_t sink[64];
			uint32_t drop = (slen - keep) > 64 ? 64 : (slen - keep);
			(void)read_all(gpu_fd, sink, drop);
			keep += drop;
		}
	}
	pthread_mutex_unlock(&rpc_lock);
	return (const GLubyte *)out;
}
void rpc_GetIntegerv(void *r, GLenum pn, GLint *data) {
	(void)r;
	if (!data) return;
	uint32_t cnt = 0;
	pthread_mutex_lock(&rpc_lock);
	rpc_emit_locked(OP_GET_INTEGERV, &pn, 4);
	rpc_flush_locked();
	if (gpu_fd >= 0 && read_all(gpu_fd, &cnt, 4) && cnt > 0 && cnt <= 16) {
		(void)read_all(gpu_fd, data, cnt * 4);
	}
	pthread_mutex_unlock(&rpc_lock);
}
void rpc_GetFloatv(void *r, GLenum pn, GLfloat *data) {
	(void)r;
	if (!data) return;
	uint32_t cnt = 0;
	pthread_mutex_lock(&rpc_lock);
	rpc_emit_locked(OP_GET_FLOATV, &pn, 4);
	rpc_flush_locked();
	if (gpu_fd >= 0 && read_all(gpu_fd, &cnt, 4) && cnt > 0 && cnt <= 16) {
		(void)read_all(gpu_fd, data, cnt * 4);
	}
	pthread_mutex_unlock(&rpc_lock);
}
void rpc_GetBooleanv(void *r, GLenum pn, GLboolean *data) {
	(void)r;
	if (!data) return;
	uint32_t cnt = 0;
	pthread_mutex_lock(&rpc_lock);
	rpc_emit_locked(OP_GET_BOOLEANV, &pn, 4);
	rpc_flush_locked();
	if (gpu_fd >= 0 && read_all(gpu_fd, &cnt, 4) && cnt > 0 && cnt <= 16) {
		(void)read_all(gpu_fd, data, cnt);
	}
	pthread_mutex_unlock(&rpc_lock);
}

/* Shaders & Programs */
GLuint rpc_CreateShader(void *r, GLenum type) {
	(void)r; uint32_t id = 0; rpc_call(OP_CREATE_SHADER, &type, 4, &id, 4); return id;
}
void rpc_ShaderSource(void *r, GLuint sh, GLsizei count, const GLchar *const *str, const GLint *len) {
	(void)r;
	size_t total = 0;
	for (GLsizei i = 0; i < count; i++) {
		if (!str[i]) continue;
		total += (len && len[i] >= 0) ? (size_t)len[i] : strlen(str[i]);
	}
	uint8_t *buf = malloc(8 + total);
	if (!buf) return;
	uint32_t u_sh = sh, u_tot = (uint32_t)total;
	memcpy(buf, &u_sh, 4);
	memcpy(buf + 4, &u_tot, 4);
	size_t off = 8;
	for (GLsizei i = 0; i < count; i++) {
		if (!str[i]) continue;
		size_t l = (len && len[i] >= 0) ? (size_t)len[i] : strlen(str[i]);
		memcpy(buf + off, str[i], l);
		off += l;
	}
	rpc_send(OP_SHADER_SOURCE, buf, (uint32_t)(8 + total));
	free(buf);
}
void rpc_CompileShader(void *r, GLuint sh) { (void)r; rpc_send(OP_COMPILE_SHADER, &sh, 4); }
void rpc_DeleteShader(void *r, GLuint sh) { (void)r; rpc_send(OP_DELETE_SHADER, &sh, 4); }
void rpc_GetShaderiv(void *r, GLuint sh, GLenum pn, GLint *p) {
	(void)r; uint32_t a[2] = {sh, pn}; int32_t v = 0;
	rpc_call(OP_GET_SHADERIV, a, 8, &v, 4); if (p) *p = v;
}
void rpc_GetShaderInfoLog(void *r, GLuint sh, GLsizei mx, GLsizei *len, GLchar *log) {
	(void)r; uint32_t slen = 0;
	pthread_mutex_lock(&rpc_lock);
	rpc_emit_locked(OP_GET_SHADER_INFO_LOG, &sh, 4);
	rpc_flush_locked();
	if (gpu_fd >= 0 && read_all(gpu_fd, &slen, 4)) {
		uint32_t keep = (mx > 0 && slen < (uint32_t)mx) ? slen : (mx > 0 ? (uint32_t)(mx - 1) : 0);
		if (keep > 0 && log) (void)read_all(gpu_fd, log, keep);
		if (log && mx > 0) log[keep] = 0;
		if (len) *len = (GLsizei)keep;
		uint32_t rem = slen - keep;
		while (rem > 0) { uint8_t sink[64]; uint32_t d = rem > 64 ? 64 : rem; (void)read_all(gpu_fd, sink, d); rem -= d; }
	}
	pthread_mutex_unlock(&rpc_lock);
}
GLuint rpc_CreateProgram(void *r) {
	(void)r; uint32_t id = 0; rpc_call(OP_CREATE_PROGRAM, NULL, 0, &id, 4); return id;
}
void rpc_AttachShader(void *r, GLuint pr, GLuint sh) {
	(void)r; uint32_t a[2] = {pr, sh}; rpc_send(OP_ATTACH_SHADER, a, 8);
}
void rpc_BindAttribLocation(void *r, GLuint pr, GLuint idx, const GLchar *name) {
	(void)r; uint32_t nl = name ? (uint32_t)strlen(name) : 0;
	uint8_t *b = malloc(12 + nl); if (!b) return;
	memcpy(b, &pr, 4); memcpy(b + 4, &idx, 4); memcpy(b + 8, &nl, 4);
	if (nl) memcpy(b + 12, name, nl);
	rpc_send(OP_BIND_ATTRIB_LOC, b, 12 + nl); free(b);
}
void rpc_LinkProgram(void *r, GLuint pr) { (void)r; rpc_send(OP_LINK_PROGRAM, &pr, 4); }
void rpc_UseProgram(void *r, GLuint pr) { (void)r; rpc_send(OP_USE_PROGRAM, &pr, 4); }
void rpc_DeleteProgram(void *r, GLuint pr) { (void)r; rpc_send(OP_DELETE_PROGRAM, &pr, 4); }
void rpc_DeleteObjectARB(void *r, GLuint obj) { rpc_DeleteProgram(r, obj); rpc_DeleteShader(r, obj); }
void rpc_GetProgramiv(void *r, GLuint pr, GLenum pn, GLint *p) {
	(void)r; uint32_t a[2] = {pr, pn}; int32_t v = 0;
	rpc_call(OP_GET_PROGRAMIV, a, 8, &v, 4); if (p) *p = v;
}
void rpc_GetProgramInfoLog(void *r, GLuint pr, GLsizei mx, GLsizei *len, GLchar *log) {
	(void)r; uint32_t slen = 0;
	pthread_mutex_lock(&rpc_lock);
	rpc_emit_locked(OP_GET_PROGRAM_INFO_LOG, &pr, 4);
	rpc_flush_locked();
	if (gpu_fd >= 0 && read_all(gpu_fd, &slen, 4)) {
		uint32_t keep = (mx > 0 && slen < (uint32_t)mx) ? slen : (mx > 0 ? (uint32_t)(mx - 1) : 0);
		if (keep > 0 && log) (void)read_all(gpu_fd, log, keep);
		if (log && mx > 0) log[keep] = 0;
		if (len) *len = (GLsizei)keep;
		uint32_t rem = slen - keep;
		while (rem > 0) { uint8_t sink[64]; uint32_t d = rem > 64 ? 64 : rem; (void)read_all(gpu_fd, sink, d); rem -= d; }
	}
	pthread_mutex_unlock(&rpc_lock);
}
void rpc_GetObjectParameterivARB(void *r, GLuint obj, GLenum pn, GLint *p) {
	if (pn == GL_COMPILE_STATUS || pn == GL_SHADER_TYPE || pn == GL_SHADER_SOURCE_LENGTH)
		rpc_GetShaderiv(r, obj, pn, p);
	else
		rpc_GetProgramiv(r, obj, pn, p);
}
GLint rpc_GetUniformLocation(void *r, GLuint pr, const GLchar *name) {
	(void)r; uint32_t nl = name ? (uint32_t)strlen(name) : 0;
	uint8_t *b = malloc(8 + nl); if (!b) return -1;
	memcpy(b, &pr, 4); memcpy(b + 4, &nl, 4); if (nl) memcpy(b + 8, name, nl);
	int32_t loc = -1; rpc_call(OP_GET_UNIFORM_LOC, b, 8 + nl, &loc, 4); free(b); return loc;
}
GLint rpc_GetAttribLocation(void *r, GLuint pr, const GLchar *name) {
	(void)r; uint32_t nl = name ? (uint32_t)strlen(name) : 0;
	uint8_t *b = malloc(8 + nl); if (!b) return -1;
	memcpy(b, &pr, 4); memcpy(b + 4, &nl, 4); if (nl) memcpy(b + 8, name, nl);
	int32_t loc = -1; rpc_call(OP_GET_ATTRIB_LOC, b, 8 + nl, &loc, 4); free(b); return loc;
}

static void send_uiv(GLint loc, uint32_t comp, GLsizei cnt, const GLint *v) {
	if (loc < 0 || cnt <= 0) return;
	uint32_t bytes = (uint32_t)(comp * (uint32_t)cnt * 4);
	uint8_t *b = malloc(12 + bytes); if (!b) return;
	uint32_t c32 = (uint32_t)cnt;
	memcpy(b, &loc, 4); memcpy(b + 4, &comp, 4); memcpy(b + 8, &c32, 4);
	memcpy(b + 12, v, bytes);
	rpc_send(OP_UNIFORM_IV, b, 12 + bytes); free(b);
}
static void send_ufv(GLint loc, uint32_t comp, GLsizei cnt, const GLfloat *v) {
	if (loc < 0 || cnt <= 0) return;
	uint32_t bytes = (uint32_t)(comp * (uint32_t)cnt * 4);
	uint8_t *b = malloc(12 + bytes); if (!b) return;
	uint32_t c32 = (uint32_t)cnt;
	memcpy(b, &loc, 4); memcpy(b + 4, &comp, 4); memcpy(b + 8, &c32, 4);
	memcpy(b + 12, v, bytes);
	rpc_send(OP_UNIFORM_FV, b, 12 + bytes); free(b);
}
static void send_umat(GLint loc, uint32_t dim, GLsizei cnt, GLboolean tr, const GLfloat *v) {
	if (loc < 0 || cnt <= 0) return;
	uint32_t bytes = (uint32_t)(dim * dim * (uint32_t)cnt * 4);
	uint8_t *b = malloc(16 + bytes); if (!b) return;
	uint32_t c32 = (uint32_t)cnt, tr32 = tr ? 1 : 0;
	memcpy(b, &loc, 4); memcpy(b + 4, &dim, 4); memcpy(b + 8, &c32, 4); memcpy(b + 12, &tr32, 4);
	memcpy(b + 16, v, bytes);
	rpc_send(OP_UNIFORM_MAT, b, 16 + bytes); free(b);
}
void rpc_Uniform1i(void *r, GLint l, GLint v0) { (void)r; send_uiv(l, 1, 1, &v0); }
void rpc_Uniform2i(void *r, GLint l, GLint v0, GLint v1) { (void)r; GLint v[2]={v0,v1}; send_uiv(l, 2, 1, v); }
void rpc_Uniform3i(void *r, GLint l, GLint v0, GLint v1, GLint v2) { (void)r; GLint v[3]={v0,v1,v2}; send_uiv(l, 3, 1, v); }
void rpc_Uniform4i(void *r, GLint l, GLint v0, GLint v1, GLint v2, GLint v3) { (void)r; GLint v[4]={v0,v1,v2,v3}; send_uiv(l, 4, 1, v); }
void rpc_Uniform1iv(void *r, GLint l, GLsizei c, const GLint *v) { (void)r; send_uiv(l, 1, c, v); }
void rpc_Uniform2iv(void *r, GLint l, GLsizei c, const GLint *v) { (void)r; send_uiv(l, 2, c, v); }
void rpc_Uniform3iv(void *r, GLint l, GLsizei c, const GLint *v) { (void)r; send_uiv(l, 3, c, v); }
void rpc_Uniform4iv(void *r, GLint l, GLsizei c, const GLint *v) { (void)r; send_uiv(l, 4, c, v); }
void rpc_Uniform1f(void *r, GLint l, GLfloat v0) { (void)r; send_ufv(l, 1, 1, &v0); }
void rpc_Uniform2f(void *r, GLint l, GLfloat v0, GLfloat v1) { (void)r; GLfloat v[2]={v0,v1}; send_ufv(l, 2, 1, v); }
void rpc_Uniform3f(void *r, GLint l, GLfloat v0, GLfloat v1, GLfloat v2) { (void)r; GLfloat v[3]={v0,v1,v2}; send_ufv(l, 3, 1, v); }
void rpc_Uniform4f(void *r, GLint l, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) { (void)r; GLfloat v[4]={v0,v1,v2,v3}; send_ufv(l, 4, 1, v); }
void rpc_Uniform1fv(void *r, GLint l, GLsizei c, const GLfloat *v) { (void)r; send_ufv(l, 1, c, v); }
void rpc_Uniform2fv(void *r, GLint l, GLsizei c, const GLfloat *v) { (void)r; send_ufv(l, 2, c, v); }
void rpc_Uniform3fv(void *r, GLint l, GLsizei c, const GLfloat *v) { (void)r; send_ufv(l, 3, c, v); }
void rpc_Uniform4fv(void *r, GLint l, GLsizei c, const GLfloat *v) { (void)r; send_ufv(l, 4, c, v); }
void rpc_UniformMatrix2fv(void *r, GLint l, GLsizei c, GLboolean t, const GLfloat *v) { (void)r; send_umat(l, 2, c, t, v); }
void rpc_UniformMatrix3fv(void *r, GLint l, GLsizei c, GLboolean t, const GLfloat *v) { (void)r; send_umat(l, 3, c, t, v); }
void rpc_UniformMatrix4fv(void *r, GLint l, GLsizei c, GLboolean t, const GLfloat *v) { (void)r; send_umat(l, 4, c, t, v); }

/* Buffers, VAOs, Draw */
void rpc_GenBuffers(void *r, GLsizei n, GLuint *ids) {
	(void)r; if (n <= 0 || !ids) return; uint32_t u = (uint32_t)n;
	rpc_call(OP_GEN_BUFFERS, &u, 4, ids, (size_t)n * 4);
}
void rpc_DeleteBuffers(void *r, GLsizei n, const GLuint *ids) {
	(void)r; if (n <= 0 || !ids) return;
	uint8_t *b = malloc(4 + (size_t)n * 4); if (!b) return;
	uint32_t u = (uint32_t)n; memcpy(b, &u, 4); memcpy(b + 4, ids, (size_t)n * 4);
	rpc_send(OP_DELETE_BUFFERS, b, 4 + u * 4); free(b);
}
void rpc_BindBuffer(void *r, GLenum tgt, GLuint buf) {
	(void)r;
	if (tgt == GL_ELEMENT_ARRAY_BUFFER) bound_ebo = buf;
	uint32_t a[2] = {tgt, buf}; rpc_send(OP_BIND_BUFFER, a, 8);
}
void rpc_BufferData(void *r, GLenum tgt, GLsizeiptr sz, const void *data, GLenum usage) {
	(void)r;
	uint32_t s32 = (uint32_t)sz, has = data ? 1 : 0;
	uint32_t hdr[4] = {tgt, s32, usage, has};
	uint32_t tot = 16 + (has ? s32 : 0);
	uint8_t *b = malloc(tot); if (!b) return;
	memcpy(b, hdr, 16); if (has && s32) memcpy(b + 16, data, s32);
	rpc_send(OP_BUFFER_DATA, b, tot); free(b);
}
void rpc_BufferSubData(void *r, GLenum tgt, GLintptr off, GLsizeiptr sz, const void *data) {
	(void)r; if (!data || sz <= 0) return;
	uint32_t o32 = (uint32_t)off, s32 = (uint32_t)sz;
	uint32_t hdr[3] = {tgt, o32, s32};
	uint8_t *b = malloc(12 + s32); if (!b) return;
	memcpy(b, hdr, 12); memcpy(b + 12, data, s32);
	rpc_send(OP_BUFFER_SUB_DATA, b, 12 + s32); free(b);
}
void rpc_GenVertexArrays(void *r, GLsizei n, GLuint *ids) {
	(void)r; if (n <= 0 || !ids) return; uint32_t u = (uint32_t)n;
	rpc_call(OP_GEN_VERTEX_ARRAYS, &u, 4, ids, (size_t)n * 4);
}
void rpc_DeleteVertexArrays(void *r, GLsizei n, const GLuint *ids) {
	(void)r; if (n <= 0 || !ids) return;
	uint8_t *b = malloc(4 + (size_t)n * 4); if (!b) return;
	uint32_t u = (uint32_t)n; memcpy(b, &u, 4); memcpy(b + 4, ids, (size_t)n * 4);
	rpc_send(OP_DELETE_VERTEX_ARRAYS, b, 4 + u * 4); free(b);
}
void rpc_BindVertexArray(void *r, GLuint vao) { (void)r; rpc_send(OP_BIND_VERTEX_ARRAY, &vao, 4); }
void rpc_EnableVertexAttribArray(void *r, GLuint idx) { (void)r; rpc_send(OP_ENABLE_VATTRIB, &idx, 4); }
void rpc_DisableVertexAttribArray(void *r, GLuint idx) { (void)r; rpc_send(OP_DISABLE_VATTRIB, &idx, 4); }
void rpc_VertexAttribPointer(void *r, GLuint idx, GLint sz, GLenum tp, GLboolean norm, GLsizei stride, const void *ptr) {
	(void)r;
	uint8_t b[28];
	uint32_t a[5] = {idx, (uint32_t)sz, tp, norm ? 1u : 0u, (uint32_t)stride};
	uint64_t p64 = (uint64_t)(uintptr_t)ptr;
	memcpy(b, a, 20); memcpy(b + 20, &p64, 8);
	rpc_send(OP_VATTRIB_PTR, b, 28);
}
void rpc_VertexAttrib4f(void *r, GLuint idx, GLfloat x, GLfloat y, GLfloat z, GLfloat w) {
	(void)r; uint8_t b[20]; memcpy(b, &idx, 4); float f[4]={x,y,z,w}; memcpy(b+4, f, 16);
	rpc_send(OP_VATTRIB_4F, b, 20);
}
void rpc_DrawArrays(void *r, GLenum mode, GLint first, GLsizei count) {
	(void)r; int32_t a[3] = {(int32_t)mode, first, count}; rpc_send(OP_DRAW_ARRAYS, a, 12);
}
void rpc_DrawElements(void *r, GLenum mode, GLsizei count, GLenum type, const void *indices) {
	(void)r;
	uint32_t inline_len = 0;
	if (bound_ebo == 0 && indices != NULL && count > 0) {
		uint32_t esz = (type == GL_UNSIGNED_BYTE) ? 1 : (type == GL_UNSIGNED_SHORT ? 2 : 4);
		inline_len = (uint32_t)count * esz;
	}
	uint8_t *b = malloc(24 + inline_len); if (!b) return;
	uint32_t hdr[3] = {mode, (uint32_t)count, type};
	uint64_t off64 = (uint64_t)(uintptr_t)indices;
	memcpy(b, hdr, 12); memcpy(b + 12, &off64, 8); memcpy(b + 20, &inline_len, 4);
	if (inline_len) memcpy(b + 24, indices, inline_len);
	rpc_send(OP_DRAW_ELEMENTS, b, 24 + inline_len); free(b);
}

/* Textures & Framebuffers */
static uint32_t bpp_for(GLenum fmt, GLenum type) {
	if (type == GL_UNSIGNED_SHORT_5_6_5 || type == GL_UNSIGNED_SHORT_4_4_4_4 || type == GL_UNSIGNED_SHORT_5_5_5_1) return 2;
	uint32_t comps = 4;
	if (fmt == GL_RED || fmt == GL_ALPHA || fmt == GL_LUMINANCE) comps = 1;
	else if (fmt == GL_RG || fmt == GL_LUMINANCE_ALPHA) comps = 2;
	else if (fmt == GL_RGB) comps = 3;
	uint32_t tb = (type == GL_FLOAT || type == GL_UNSIGNED_INT) ? 4 : (type == GL_HALF_FLOAT || type == GL_UNSIGNED_SHORT ? 2 : 1);
	return comps * tb;
}

void rpc_GenTextures(void *r, GLsizei n, GLuint *ids) {
	(void)r; if (n <= 0 || !ids) return; uint32_t u = (uint32_t)n;
	rpc_call(OP_GEN_TEXTURES, &u, 4, ids, (size_t)n * 4);
}
void rpc_DeleteTextures(void *r, GLsizei n, const GLuint *ids) {
	(void)r; if (n <= 0 || !ids) return;
	uint8_t *b = malloc(4 + (size_t)n * 4); if (!b) return;
	uint32_t u = (uint32_t)n; memcpy(b, &u, 4); memcpy(b + 4, ids, (size_t)n * 4);
	rpc_send(OP_DELETE_TEXTURES, b, 4 + u * 4); free(b);
}
void rpc_BindTexture(void *r, GLenum tgt, GLuint tex) {
	(void)r;
	GLenum nt = norm_tgt(tgt);
	if (nt == GL_TEXTURE_2D) bound_tex[active_unit] = tex;
	uint32_t a[2] = {nt, tex}; rpc_send(OP_BIND_TEXTURE, a, 8);
}
void rpc_TexParameteri(void *r, GLenum tgt, GLenum pn, GLint p) {
	(void)r; int32_t a[3] = {(int32_t)norm_tgt(tgt), (int32_t)pn, p}; rpc_send(OP_TEX_PARAM_I, a, 12);
}
void rpc_TexParameterf(void *r, GLenum tgt, GLenum pn, GLfloat p) {
	(void)r; uint8_t b[12]; uint32_t t = norm_tgt(tgt); memcpy(b, &t, 4); memcpy(b+4, &pn, 4); memcpy(b+8, &p, 4);
	rpc_send(OP_TEX_PARAM_F, b, 12);
}
void rpc_TexImage2D(void *r, GLenum tgt, GLint lvl, GLint ifmt, GLsizei w, GLsizei h, GLint border, GLenum fmt, GLenum type, const void *px) {
	(void)r;
	uint32_t has = px ? 1 : 0;
	uint32_t bytes = (has && w > 0 && h > 0) ? (uint32_t)w * (uint32_t)h * bpp_for(fmt, type) : 0;
	uint32_t hdr[9] = {norm_tgt(tgt), (uint32_t)lvl, (uint32_t)ifmt, (uint32_t)w, (uint32_t)h, (uint32_t)border, fmt, type, bytes};
	uint8_t *b = malloc(36 + bytes); if (!b) return;
	memcpy(b, hdr, 36); if (bytes) memcpy(b + 36, px, bytes);
	rpc_send(OP_TEX_IMAGE_2D, b, 36 + bytes); free(b);
}
void rpc_TexSubImage2D(void *r, GLenum tgt, GLint lvl, GLint x, GLint y, GLsizei w, GLsizei h, GLenum fmt, GLenum type, const void *px) {
	(void)r; if (!px || w <= 0 || h <= 0) return;
	uint32_t bytes = (uint32_t)w * (uint32_t)h * bpp_for(fmt, type);
	uint32_t hdr[9] = {norm_tgt(tgt), (uint32_t)lvl, (uint32_t)x, (uint32_t)y, (uint32_t)w, (uint32_t)h, fmt, type, bytes};
	uint8_t *b = malloc(36 + bytes); if (!b) return;
	memcpy(b, hdr, 36); memcpy(b + 36, px, bytes);
	rpc_send(OP_TEX_SUB_IMAGE_2D, b, 36 + bytes); free(b);
}
void rpc_GenerateMipmap(void *r, GLenum tgt) { (void)r; GLenum nt = norm_tgt(tgt); rpc_send(OP_GENERATE_MIPMAP, &nt, 4); }
void rpc_GenFramebuffers(void *r, GLsizei n, GLuint *ids) {
	(void)r; if (n <= 0 || !ids) return; uint32_t u = (uint32_t)n;
	rpc_call(OP_GEN_FRAMEBUFFERS, &u, 4, ids, (size_t)n * 4);
}
void rpc_DeleteFramebuffers(void *r, GLsizei n, const GLuint *ids) {
	(void)r; if (n <= 0 || !ids) return;
	uint8_t *b = malloc(4 + (size_t)n * 4); if (!b) return;
	uint32_t u = (uint32_t)n; memcpy(b, &u, 4); memcpy(b + 4, ids, (size_t)n * 4);
	rpc_send(OP_DELETE_FRAMEBUFFERS, b, 4 + u * 4); free(b);
}
void rpc_BindFramebuffer(void *r, GLenum tgt, GLuint fb) {
	(void)r; uint32_t a[2] = {tgt, fb}; rpc_send(OP_BIND_FRAMEBUFFER, a, 8);
}
void rpc_FramebufferTexture2D(void *r, GLenum tgt, GLenum att, GLenum tt, GLuint tex, GLint lvl) {
	(void)r; uint32_t a[5] = {tgt, att, norm_tgt(tt), tex, (uint32_t)lvl};
	rpc_send(OP_FRAMEBUFFER_TEX2D, a, 20);
}
GLenum rpc_CheckFramebufferStatus(void *r, GLenum tgt) {
	(void)r; uint32_t st = GL_FRAMEBUFFER_COMPLETE;
	rpc_call(OP_CHECK_FB_STATUS, &tgt, 4, &st, 4); return st;
}
void rpc_GenRenderbuffers(void *r, GLsizei n, GLuint *ids) {
	(void)r; if (n <= 0 || !ids) return; uint32_t u = (uint32_t)n;
	rpc_call(OP_GEN_RENDERBUFFERS, &u, 4, ids, (size_t)n * 4);
}
void rpc_DeleteRenderbuffers(void *r, GLsizei n, const GLuint *ids) {
	(void)r; if (n <= 0 || !ids) return;
	uint8_t *b = malloc(4 + (size_t)n * 4); if (!b) return;
	uint32_t u = (uint32_t)n; memcpy(b, &u, 4); memcpy(b + 4, ids, (size_t)n * 4);
	rpc_send(OP_DELETE_RENDERBUFFERS, b, 4 + u * 4); free(b);
}
void rpc_BindRenderbuffer(void *r, GLenum tgt, GLuint rb) {
	(void)r; uint32_t a[2] = {tgt, rb}; rpc_send(OP_BIND_RENDERBUFFER, a, 8);
}
void rpc_RenderbufferStorage(void *r, GLenum tgt, GLenum ifmt, GLsizei w, GLsizei h) {
	(void)r; uint32_t a[4] = {tgt, ifmt, (uint32_t)w, (uint32_t)h};
	rpc_send(OP_RENDERBUFFER_STORAGE, a, 16);
}
void rpc_FramebufferRenderbuffer(void *r, GLenum tgt, GLenum att, GLenum rbt, GLuint rb) {
	(void)r; uint32_t a[4] = {tgt, att, rbt, rb}; rpc_send(OP_FRAMEBUFFER_RBO, a, 16);
}
void rpc_BlitFramebuffer(void *r, GLint sx0, GLint sy0, GLint sx1, GLint sy1, GLint dx0, GLint dy0, GLint dx1, GLint dy1, GLbitfield m, GLenum f) {
	(void)r; int32_t a[10] = {sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, (int32_t)m, (int32_t)f};
	rpc_send(OP_BLIT_FRAMEBUFFER, a, 40);
}
void rpc_DiscardFramebuffer(void *r, GLenum tgt, GLsizei n, const GLenum *atts) {
	(void)r; if (n <= 0 || !atts) return;
	uint8_t *b = malloc(8 + (size_t)n * 4); if (!b) return;
	uint32_t u = (uint32_t)n; memcpy(b, &tgt, 4); memcpy(b + 4, &u, 4); memcpy(b + 8, atts, (size_t)n * 4);
	rpc_send(OP_INVALIDATE_FB, b, 8 + u * 4); free(b);
}
void rpc_FinishObjectAPPLE(void *r, GLenum o, GLint n) { (void)o; (void)n; rpc_Finish(r); }
GLboolean rpc_TestObjectAPPLE(void *r, GLenum o, GLuint n) { (void)r; (void)o; (void)n; return GL_TRUE; }

/* ---- EAGLContext ObjC swizzling ---- */

struct sgl_ctx {
	void *rctx;
	void *disp[SGL_SLOTS];
	id eagl;
	unsigned api;
};

static struct sgl_ctx *ctxs[32];
static pthread_mutex_t ctx_lock = PTHREAD_MUTEX_INITIALIZER;
static __thread id cur_eagl = nil;
static IMP orig_init, orig_dealloc, orig_api, orig_set_param, orig_macro, orig_tex_io, orig_attach, orig_set_cur, orig_cur;

static struct sgl_ctx *lookup(id self) {
	if (!self) return NULL;
	pthread_mutex_lock(&ctx_lock);
	struct sgl_ctx *found = NULL;
	for (int i = 0; i < 32; i++) if (ctxs[i] && ctxs[i]->eagl == self) { found = ctxs[i]; break; }
	pthread_mutex_unlock(&ctx_lock);
	return found;
}

static bool
upload_iosurface(IOSurfaceRef s, GLenum target, bool is_display_target)
{
	if (!s) return false;
	IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
	size_t w = IOSurfaceGetWidth(s);
	size_t h = IOSurfaceGetHeight(s);
	size_t bpr = IOSurfaceGetBytesPerRow(s);
	const uint8_t *base = (const uint8_t *)IOSurfaceGetBaseAddress(s);
	if (base && w > 0 && h > 0) {
		uint8_t *packed = malloc(w * h * 4);
		if (packed) {
			for (size_t y = 0; y < h; y++) {
				memcpy(packed + y * w * 4, base + y * bpr, w * 4);
			}
			rpc_TexImage2D(NULL, norm_tgt(target), 0, GL_RGBA, (GLsizei)w, (GLsizei)h, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, packed);
			free(packed);
		}
		if (is_display_target || (w == 750 && h == 1334) || (w == 1125 && h == 2436)) {
			display_tex_id = bound_tex[active_unit];
			display_w = (uint32_t)w;
			display_h = (uint32_t)h;
		}
	}
	IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
	return true;
}

static id m_init(id self, SEL cmd, NSUInteger api, id props) {
	if (gpu_fd < 0) return ((id(*)(id, SEL, NSUInteger, id))orig_init)(self, cmd, api, props);
	struct sgl_ctx *c = calloc(1, sizeof(*c));
	c->rctx = c;
	memcpy(c->disp, sgl_dispatch, sizeof(c->disp));
	c->eagl = self;
	c->api = (unsigned)api;
	pthread_mutex_lock(&ctx_lock);
	for (int i = 0; i < 32; i++) if (!ctxs[i]) { ctxs[i] = c; break; }
	pthread_mutex_unlock(&ctx_lock);
	sgl_log("sweetgl-pc: EAGLContext initWithAPI:%lu -> host RX 6600 XT", (unsigned long)api);
	return self;
}
static void m_dealloc(id self, SEL cmd) {
	struct sgl_ctx *c = lookup(self);
	if (!c) { ((void(*)(id, SEL))orig_dealloc)(self, cmd); return; }
	pthread_mutex_lock(&ctx_lock);
	for (int i = 0; i < 32; i++) if (ctxs[i] == c) ctxs[i] = NULL;
	pthread_mutex_unlock(&ctx_lock);
	free(c);
	struct objc_super sup = {self, class_getSuperclass(objc_getClass("EAGLContext"))};
	((void(*)(struct objc_super *, SEL))objc_msgSendSuper)(&sup, cmd);
}
static NSUInteger m_api(id self, SEL cmd) {
	struct sgl_ctx *c = lookup(self);
	return c ? c->api : ((NSUInteger(*)(id, SEL))orig_api)(self, cmd);
}
static BOOL m_set_param(id self, SEL cmd, uint32_t pn, const int32_t *v) {
	return lookup(self) ? YES : ((BOOL(*)(id, SEL, uint32_t, const int32_t *))orig_set_param)(self, cmd, pn, v);
}
static void *m_macro(id self, SEL cmd) {
	struct sgl_ctx *c = lookup(self);
	return c ? (void *)c : ((void *(*)(id, SEL))orig_macro)(self, cmd);
}
static BOOL m_tex_io(id self, SEL cmd, IOSurfaceRef s, NSUInteger tgt, NSUInteger ifmt, uint32_t w, uint32_t h, NSUInteger fmt, NSUInteger tp, uint32_t pl, BOOL inv) {
	if (!lookup(self)) return ((BOOL(*)(id, SEL, IOSurfaceRef, NSUInteger, NSUInteger, uint32_t, uint32_t, NSUInteger, NSUInteger, uint32_t, BOOL))orig_tex_io)(self, cmd, s, tgt, ifmt, w, h, fmt, tp, pl, inv);
	return upload_iosurface(s, (GLenum)tgt, false) ? YES : NO;
}
static BOOL m_attach(id self, SEL cmd, NSUInteger tgt, IOSurfaceRef s, BOOL inv) {
	if (!lookup(self)) return ((BOOL(*)(id, SEL, NSUInteger, IOSurfaceRef, BOOL))orig_attach)(self, cmd, tgt, s, inv);
	return upload_iosurface(s, (GLenum)tgt, true) ? YES : NO;
}
static BOOL m_set_cur(id cls, SEL cmd, id eagl) {
	if (lookup(eagl)) { cur_eagl = eagl; return YES; }
	cur_eagl = nil;
	return ((BOOL(*)(id, SEL, id))orig_set_cur)(cls, cmd, eagl);
}
static id m_cur(id cls, SEL cmd) {
	return cur_eagl ? cur_eagl : ((id(*)(id, SEL))orig_cur)(cls, cmd);
}

static IMP swap_m(Class cls, const char *sel, void *imp) {
	Method m = class_getInstanceMethod(cls, sel_registerName(sel));
	return m ? method_setImplementation(m, (IMP)imp) : NULL;
}

__attribute__((constructor))
static void sweetgl_pc_ctor(void) {
	gpu_fd = open("/dev/sweetgpu", O_RDWR | O_CLOEXEC);
	if (gpu_fd < 0) {
		sgl_log("sweetgl-pc: /dev/sweetgpu unavailable, staying on software CA");
		return;
	}
	uint32_t magic = 0;
	if (!rpc_call(OP_PING, NULL, 0, &magic, 4) || magic != 0x52583636u) {
		sgl_log("sweetgl-pc: host GPU ping failed (magic=0x%x)", magic);
		close(gpu_fd);
		gpu_fd = -1;
		return;
	}
	Class cls = objc_getClass("EAGLContext");
	if (!cls) return;
	Class meta = object_getClass((id)cls);
	orig_init = swap_m(cls, "initWithAPI:properties:", m_init);
	orig_dealloc = swap_m(cls, "dealloc", m_dealloc);
	orig_api = swap_m(cls, "API", m_api);
	orig_set_param = swap_m(cls, "setParameter:to:", m_set_param);
	orig_macro = swap_m(cls, "getMacroContextPrivate", m_macro);
	orig_tex_io = swap_m(cls, "texImageIOSurface:target:internalFormat:width:height:format:type:plane:invert:", m_tex_io);
	orig_attach = swap_m(cls, "attachImage:toCoreSurface:invertedRender:", m_attach);
	orig_set_cur = swap_m(meta, "setCurrentContext:", m_set_cur);
	orig_cur = swap_m(meta, "currentContext", m_cur);
	sgl_log("sweetgl-pc: connected to host GPU (AMD RX 6600 XT) via /dev/sweetgpu!");
}
