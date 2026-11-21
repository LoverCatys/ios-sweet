#!/usr/bin/env python3
"""Windows-friendly QEMU runner for SWEETXNU payloads (tools/mkpayload.py).

Supports:
  - Portable QEMU in tools-bin/qemu/qemu-system-aarch64.exe
  - TCP QMP (avoids Windows AF_UNIX issues with QEMU)
  - Multi-threaded TCG (-accel tcg,thread=multi) and configurable SMP
  - Optional debug logging (disabled by default for better TCG speed)
"""
import argparse
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
import zlib

FB_BASE, FB_W, FB_H = 0x9c000000, 1080, 2400
FB_STRIDE = FB_W * 4
PAGE = 0x4000
STUB_PA = 0x40200000
BOOT_LINE_LENGTH = 608


def find_qemu():
    local = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'tools-bin', 'qemu', 'qemu-system-aarch64.exe'
    )
    if os.path.isfile(local):
        return os.path.abspath(local)
    found = shutil.which('qemu-system-aarch64')
    if found:
        return found
    raise SystemExit('qemu-system-aarch64 not found (checked tools-bin/qemu and PATH)')


def build_image(payload):
    magic, ver, nsegs = struct.unpack_from('<8sII', payload, 0)
    assert magic == b'SWEETXNU' and ver == 1, 'not a SWEETXNU v1 payload'
    virt_base, entry_va, adt_off, adt_size, kernel_end, phys_base, mem_size = \
        struct.unpack_from('<7Q', payload, 0x10)
    cmdline = payload[0x48:0x48 + BOOT_LINE_LENGTH]
    delta = phys_base - virt_base
    adt_pa = kernel_end + delta
    ba_pa = (adt_pa + adt_size + PAGE - 1) & ~(PAGE - 1)
    top = ba_pa + PAGE
    img = bytearray(top - phys_base)
    for i in range(nsegs):
        vmaddr, vmsize, foff, fsize = struct.unpack_from('<4Q', payload, 0x2a8 + 32 * i)
        o = vmaddr + delta - phys_base
        img[o:o + fsize] = payload[foff:foff + fsize]
    img[adt_pa - phys_base:adt_pa - phys_base + adt_size] = payload[adt_off:adt_off + adt_size]
    ba = struct.pack(
        '<HH4x4Q6QI4xQI',
        2, 2, virt_base, phys_base, mem_size, top,
        FB_BASE, 1, FB_STRIDE, FB_W, FB_H, 32, 0, adt_pa - delta, adt_size
    )
    ba += cmdline + b'\0' * 4 + struct.pack('<QQ', 0, mem_size)
    assert len(ba) == 736
    img[ba_pa - phys_base:ba_pa - phys_base + len(ba)] = ba
    return phys_base, bytes(img), ba_pa, entry_va + delta


def stub(ba_pa, entry_pa):
    return struct.pack('<4I2Q', 0x58000080, 0x580000a1, 0xd61f0020, 0xd503201f, ba_pa, entry_pa)


def write_png(path, raw):
    rows = []
    for y in range(FB_H):
        line = raw[y * FB_STRIDE:(y + 1) * FB_STRIDE]
        rgb = bytearray(FB_W * 3)
        rgb[0::3], rgb[1::3], rgb[2::3] = line[2::4], line[1::4], line[0::4]
        rows.append(b'\0' + bytes(rgb))
    chunk = lambda t, d: struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d))
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', FB_W, FB_H, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(b''.join(rows), 6)) + chunk(b'IEND', b'')
    with open(path, 'wb') as f:
        f.write(png)


def qmp_tcp(port, *cmds, pmemsave=None):
    s = socket.create_connection(('127.0.0.1', port), timeout=10)
    f = s.makefile('rw', encoding='utf-8')
    f.readline()

    def call(execute, **args):
        f.write(json.dumps({'execute': execute, 'arguments': args}) + '\n')
        f.flush()
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError('QMP connection closed')
            r = json.loads(line)
            if 'return' in r or 'error' in r:
                return r

    call('qmp_capabilities')
    out = []
    for c in cmds:
        r = call('human-monitor-command', **{'command-line': c})
        if 'error' in r:
            print(f'qmp {c!r}: {r["error"]}', file=sys.stderr)
        out.append(r.get('return', ''))
    if pmemsave:
        r = call('pmemsave', val=pmemsave[0], size=pmemsave[1], filename=pmemsave[2])
        if 'error' in r:
            print(f'qmp pmemsave: {r["error"]}', file=sys.stderr)
    s.close()
    return out


def stream_serial(serial_path, pos):
    if not os.path.exists(serial_path):
        return pos
    with open(serial_path, 'r', errors='replace') as f:
        f.seek(pos)
        data = f.read()
        if data:
            sys.stdout.write(data)
            sys.stdout.flush()
            return f.tell()
    return pos


def main():
    ap = argparse.ArgumentParser(description='Boot SWEETXNU payload in QEMU on Windows.')
    ap.add_argument('payload', help='SWEETXNU payload binary (from mkpayload.py)')
    ap.add_argument('outdir', help='Output directory for fb.png, regs.txt, serial.txt')
    ap.add_argument('--secs', type=float, default=25, help='Run duration in seconds (default: 25)')
    ap.add_argument('--smp', type=int, default=4, help='Number of vCPUs (default: 4)')
    ap.add_argument('--cpu', default='cortex-a76', help='QEMU CPU model (default: cortex-a76)')
    ap.add_argument('--qmp-port', type=int, default=4444, help='Local TCP port for QMP (default: 4444)')
    ap.add_argument('--debug-log', action='store_true', help='Enable QEMU -d int,guest_errors,unimp (slower)')
    ap.add_argument('--live-serial', action='store_true', help='Stream serial output live')
    a = ap.parse_args()

    qemu_bin = find_qemu()
    with open(a.payload, 'rb') as f:
        payload = f.read()

    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.abspath(a.outdir)

    phys_base, img, ba_pa, entry_pa = build_image(payload)
    ram_path = os.path.join(out, 'ram.bin')
    stub_path = os.path.join(out, 'stub.bin')
    serial_path = os.path.join(out, 'serial.txt')
    with open(ram_path, 'wb') as f:
        f.write(img)
    with open(stub_path, 'wb') as f:
        f.write(stub(ba_pa, entry_pa))

    mem_size = struct.unpack_from('<Q', payload, 0x40)[0]
    ram_gib = max(4, -(-(phys_base + mem_size - 0x40000000) // (1 << 30)))
    print(f'qemu={qemu_bin}', file=sys.stderr)
    print(f'phys_base={phys_base:#x} boot_args={ba_pa:#x} entry={entry_pa:#x} ram={ram_gib}G smp={a.smp}', file=sys.stderr)

    cmd = [
        qemu_bin,
        '-M', 'virt,gic-version=3',
        '-accel', 'tcg,thread=multi',
        '-cpu', a.cpu,
        '-smp', str(a.smp),
        '-m', f'{ram_gib}G',
        '-nographic',
        '-serial', f'file:{serial_path}',
        '-qmp', f'tcp:127.0.0.1:{a.qmp_port},server=on,wait=off',
        '-no-reboot',
        '-device', f'loader,file={ram_path},addr={phys_base:#x},force-raw=on',
        '-device', f'loader,file={stub_path},addr={STUB_PA:#x},force-raw=on',
        '-device', f'loader,addr={STUB_PA:#x},cpu-num=0',
    ]
    if a.debug_log:
        cmd += ['-d', 'int,guest_errors,unimp', '-D', os.path.join(out, 'qemu.log')]

    p = None
    serial_pos = 0
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
        t0 = time.time()
        while p.poll() is None and time.time() - t0 < a.secs:
            if a.live_serial:
                serial_pos = stream_serial(serial_path, serial_pos)
            time.sleep(0.25)

        if a.live_serial:
            stream_serial(serial_path, serial_pos)

        if p.poll() is None:
            fb_raw_path = os.path.join(out, 'fb.raw').replace('\\', '/')
            _, regs = qmp_tcp(
                a.qmp_port,
                'stop',
                'info registers',
                pmemsave=(FB_BASE, FB_STRIDE * FB_H, fb_raw_path),
            )
            with open(os.path.join(out, 'regs.txt'), 'w', encoding='utf-8') as f:
                f.write(regs)
            p.kill()
            p.wait()
            with open(os.path.join(out, 'fb.raw'), 'rb') as f:
                raw = f.read()
            write_png(os.path.join(out, 'fb.png'), raw)
            os.unlink(os.path.join(out, 'fb.raw'))
            print(f'Saved: {os.path.join(out, "fb.png")}, {os.path.join(out, "regs.txt")}, {serial_path}', file=sys.stderr)
            return 0
        else:
            print(f'QEMU exited early with code {p.returncode}', file=sys.stderr)
            return 1
    finally:
        if p and p.poll() is None:
            p.kill()
            p.wait()
        if os.path.exists(ram_path):
            try:
                os.unlink(ram_path)
            except OSError:
                pass


if __name__ == '__main__':
    sys.exit(main())
