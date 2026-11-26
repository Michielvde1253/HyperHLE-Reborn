/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use crate::dyld::FunctionExports;
use crate::environment::Environment;
use crate::export_c_func;
use crate::fs::{GuestFile, GuestOpenOptions, GuestPath};
use crate::libc::errno::{set_errno, EINVAL, EIO, ENOMEM};
use crate::libc::posix_io;
use crate::libc::posix_io::{off_t, open_direct, FileDescriptor};
use crate::mem::{ConstPtr, GuestUSize, Mem, MutVoidPtr, PAGE_SIZE, PAGE_SIZE_ALIGN_MASK};
use std::collections::HashMap;
use std::io::{self, Seek, SeekFrom, Write};

#[allow(dead_code)]
const MAP_FILE: i32 = 0x0000;
const MAP_SHARED: i32 = 0x0001;
const MAP_FIXED: i32 = 0x0010;
const MAP_ANON: i32 = 0x1000;

const MAP_FAILED: MutVoidPtr = MutVoidPtr::from_bits(0xFFFFFFFF);

#[derive(Default)]
pub struct State {
    /// Keeping track of `mmap` allocations
    allocations: HashMap<MutVoidPtr, GuestUSize>,
    /// Writable `MAP_SHARED` file mappings that need to be written back.
    file_mappings: HashMap<MutVoidPtr, SharedFileMapping>,
    /// Page ranges that a partial `munmap` released out of a larger mapping.
    ///
    /// The pages stay backed by the mapping that owns them, so the rest of that
    /// mapping keeps working and keeps its contents, but a later `mmap` that
    /// passes such a range as its address *hint* gets the range back. That is
    /// what Darwin does with a free hint, and it is how callers that release
    /// part of a mapping and immediately map something else over the hole -
    /// Boehm GC's `GC_unmap`/`GC_remap`, and the `munmap`/`mmap` pair in
    /// `TestApp`'s `test_munmap_partial_mapping` - expect `mmap` to behave.
    unmapped: Vec<UnmappedRange>,
}

struct SharedFileMapping {
    file: GuestFile,
    file_offset: u64,
}

fn open_shared_mapping_file(
    env: &mut Environment,
    fd: FileDescriptor,
) -> io::Result<GuestFile> {
    let path = env
        .libc_state
        .posix_io
        .path_for_fd(fd)
        .map(str::to_owned)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "mapping file has no path"))?;
    let mut options = GuestOpenOptions::new();
    options.read().write();
    env.fs
        .open_with_options(GuestPath::new(path.as_str()), options)
        .map_err(|()| io::Error::new(io::ErrorKind::PermissionDenied, "cannot reopen mapping file"))
}

fn write_back_mapping_range(
    mem: &Mem,
    mapping_base: MutVoidPtr,
    mapping_len: GuestUSize,
    mapping: &mut SharedFileMapping,
    addr: MutVoidPtr,
    len: GuestUSize,
) -> io::Result<()> {
    let mapping_start = mapping_base.to_bits();
    let mapping_end = mapping_start.saturating_add(mapping_len);
    let start = addr.to_bits().max(mapping_start);
    let end = addr.to_bits().saturating_add(len).min(mapping_end);
    if start >= end {
        return Ok(());
    }

    let relative_offset = u64::try_from(start - mapping_start)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "mapping offset overflow"))?;
    let file_offset = mapping
        .file_offset
        .checked_add(relative_offset)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "file offset overflow"))?;
    let bytes = mem
        .bytes_at(MutVoidPtr::from_bits(start).cast(), end - start)
        .to_vec();
    mapping.file.seek(SeekFrom::Start(file_offset))?;
    mapping.file.write_all(&bytes)?;
    mapping.file.sync_all()
}

fn write_back_file_mapping_range(
    env: &mut Environment,
    mapping_base: MutVoidPtr,
    mapping_len: GuestUSize,
    addr: MutVoidPtr,
    len: GuestUSize,
) -> io::Result<()> {
    let Some(mapping) = env
        .libc_state
        .mmap
        .file_mappings
        .get_mut(&mapping_base)
    else {
        return Ok(());
    };
    write_back_mapping_range(&env.mem, mapping_base, mapping_len, mapping, addr, len)
}

fn write_back_mapped_range(
    env: &mut Environment,
    addr: MutVoidPtr,
    len: GuestUSize,
) -> io::Result<()> {
    let Some((mapping_base, mapping_len)) = owning_mapping(env, addr) else {
        return Ok(());
    };
    write_back_file_mapping_range(env, mapping_base, mapping_len, addr, len)
}

/// A contiguous page range that [State::unmapped] remembers, together with the
/// base address of the mapping it was carved out of.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct UnmappedRange {
    /// Base address of the mapping that still backs these pages.
    base: MutVoidPtr,
    start: GuestUSize,
    end: GuestUSize,
}

/// Round `addr` down and `addr + len` up to whole pages, the way `munmap`
/// rounds its arguments. Returns `None` if the range is empty or overflows.
fn page_range(addr: MutVoidPtr, len: GuestUSize) -> Option<(GuestUSize, GuestUSize)> {
    let mask = PAGE_SIZE_ALIGN_MASK;
    let start = addr.to_bits() & !mask;
    let end = addr.to_bits().checked_add(len)?.checked_add(mask)? & !mask;
    (end > start).then_some((start, end))
}

/// Remember that the guest released `[start, end)` out of the mapping of
/// `mapping_len` bytes at `base`. The range is clamped to that mapping, and
/// adjacent ranges of the same mapping are merged so that a later hint
/// covering several of them is recognised.
fn record_unmapped_range(
    unmapped: &mut Vec<UnmappedRange>,
    base: MutVoidPtr,
    mapping_len: GuestUSize,
    start: GuestUSize,
    end: GuestUSize,
) {
    let mapping_end = base.to_bits().saturating_add(mapping_len);
    let start = start.max(base.to_bits());
    let end = end.min(mapping_end);
    if start >= end {
        return;
    }
    unmapped.push(UnmappedRange { base, start, end });
    unmapped.sort_unstable_by_key(|range| (range.start, range.end));

    let mut merged: Vec<UnmappedRange> = Vec::with_capacity(unmapped.len());
    for range in unmapped.drain(..) {
        match merged.last_mut() {
            Some(last) if last.base == range.base && range.start <= last.end => {
                last.end = last.end.max(range.end);
            }
            _ => merged.push(range),
        }
    }
    *unmapped = merged;
}

/// Whether `[start, end)` is completely covered by remembered ranges whose
/// mapping is still alive. Any gap, or a range whose mapping has since been
/// released, means the address is not free and a hint must not be honoured.
fn is_fully_unmapped(
    unmapped: &[UnmappedRange],
    allocations: &HashMap<MutVoidPtr, GuestUSize>,
    start: GuestUSize,
    end: GuestUSize,
) -> bool {
    let mut covered_until = start;
    for range in unmapped {
        if range.end <= covered_until {
            continue;
        }
        if range.start > covered_until {
            return false;
        }
        if !allocations.contains_key(&range.base) {
            return false;
        }
        covered_until = range.end;
        if covered_until >= end {
            return true;
        }
    }
    false
}

/// Forget that `[start, end)` was ever unmapped, e.g. because the guest mapped
/// it again or because the mapping that backed it is gone.
fn clear_unmapped_ranges(unmapped: &mut Vec<UnmappedRange>, start: GuestUSize, end: GuestUSize) {
    let mut kept: Vec<UnmappedRange> = Vec::with_capacity(unmapped.len() + 1);
    for range in unmapped.drain(..) {
        if range.end <= start || range.start >= end {
            kept.push(range);
            continue;
        }
        if range.start < start {
            kept.push(UnmappedRange {
                base: range.base,
                start: range.start,
                end: start,
            });
        }
        if range.end > end {
            kept.push(UnmappedRange {
                base: range.base,
                start: end,
                end: range.end,
            });
        }
    }
    *unmapped = kept;
}

/// `mmap`'s address argument is a hint: the implementation should place the
/// mapping there if the range is free and ignore it otherwise. The only address
/// space we know to be free is what the guest released with a partial `munmap`,
/// so honour the hint when the whole range was released that way: zero it (the
/// mapping is fresh) and stop remembering it as free.
fn honour_address_hint(
    mem: &mut Mem,
    allocations: &HashMap<MutVoidPtr, GuestUSize>,
    unmapped: &mut Vec<UnmappedRange>,
    addr: MutVoidPtr,
    len: GuestUSize,
) -> bool {
    if addr.is_null() {
        return false;
    }
    let mask = PAGE_SIZE_ALIGN_MASK;
    // Only exact, page-aligned hints can be honoured: the guest expects the
    // mapping to start at the address it asked for.
    if addr.to_bits() & mask != 0 {
        return false;
    }
    let Some((start, end)) = page_range(addr, len) else {
        return false;
    };
    if !is_fully_unmapped(unmapped, allocations, start, end) {
        return false;
    }
    clear_unmapped_ranges(unmapped, start, end);
    mem.bytes_at_mut(addr.cast(), end - start).fill(0);
    log_dbg!(
        "mmap: honouring address hint {:?} for {} bytes (released by an earlier partial munmap)",
        addr,
        len
    );
    true
}

/// For files, our implementation of mmap is really simple:
/// it's just load entirety of file in memory!
fn mmap(
    env: &mut Environment,
    addr: MutVoidPtr,
    len: GuestUSize,
    prot: i32,
    flags: i32,
    fd: FileDescriptor,
    offset: off_t,
) -> MutVoidPtr {
    // TODO: handle errno properly
    set_errno(env, 0);
    log_dbg!(
        "mmap({:?}, {}, {}, {}, {}, {})",
        addr,
        len,
        prot,
        flags,
        fd,
        offset
    );
    if (flags & MAP_ANON) == 0 && crate::env_flag_cached!("TOUCHHLE_TRACE_OPEN") {
        log!(
            "MMAP-DIAG addr={:?} len={} prot={:#x} flags={:#x} fd={} offset={} path={:?}",
            addr,
            len,
            prot,
            flags,
            fd,
            offset,
            env.libc_state.posix_io.path_for_fd(fd)
        );
    }

    if (flags & MAP_FIXED) != 0 {
        return mmap_fixed(env, addr, len, flags, fd, offset);
    }

    let shared_file_mapping = if (flags & MAP_ANON) == 0
        && (flags & MAP_SHARED) != 0
        && env.libc_state.posix_io.is_fd_writable(fd)
    {
        let file_offset = match u64::try_from(offset) {
            Ok(offset) => offset,
            Err(_) => {
                set_errno(env, EINVAL);
                return MAP_FAILED;
            }
        };
        match open_shared_mapping_file(env, fd) {
            Ok(file) => Some(SharedFileMapping { file, file_offset }),
            Err(error) => {
                log!(
                    "Warning: mmap: could not retain writable shared mapping for fd {}: {}",
                    fd,
                    error
                );
                set_errno(env, EIO);
                return MAP_FAILED;
            }
        }
    } else {
        None
    };

    // Darwin returns the hint address when the pages it names are free. The
    // only free address space we can describe is what the guest itself released
    // with a partial `munmap`, so try that before falling back to the heap.
    let mmap_state = &mut env.libc_state.mmap;
    if (flags & MAP_ANON) != 0
        && honour_address_hint(
            &mut env.mem,
            &mmap_state.allocations,
            &mut mmap_state.unmapped,
            addr,
            len,
        )
    {
        return addr;
    }

    // TODO: use vm_allocate() instead
    let ptr = env.mem.calloc(len);
    if ptr.is_null() {
        crate::mem::log_out_of_memory("mmap", len);
        set_errno(env, ENOMEM);
        return MAP_FAILED;
    }

    if (flags & MAP_ANON) != 0 {
        // Real mmap always returns page-aligned addresses and callers (e.g.
        // Boehm GC) rely on that. Fail gracefully instead of panicking the host.
        if ptr.to_bits() & PAGE_SIZE_ALIGN_MASK != 0 {
            log!(
                "Warning: mmap MAP_ANON: allocator returned unaligned address {:?} for {} bytes; returning MAP_FAILED",
                ptr,
                len
            );
            env.mem.free(ptr);
            set_errno(env, ENOMEM);
            return MAP_FAILED;
        }

        // Убираем жесткие assert_eq!(fd, -1) и assert_eq!(offset, 0).
        // В реальной iOS/Darwin при наличии флага MAP_ANON аргументы fd и
        // offset
        // просто игнорируются ОС. Движки вроде Adobe AIR передают сюда мусор.
        if fd != -1 || offset != 0 {
            log_dbg!("Warning: mmap MAP_ANON called with fd={} and offset={}. Ignoring them as per OS behavior.", fd, offset);
        }

        if !addr.is_null() {
            // POSIX `mmap` documents the `addr` argument as a hint that
            // implementations may ignore. We always allocate from the
            // guest heap, so the actual placement is the heap allocator's
            // choice. Apps that genuinely require fixed-address mappings
            // would set MAP_FIXED, which we'd then need to honour
            // separately. Demoted to debug to keep Mono/Unity startup
            // logs readable; the host-vs-hint mismatch is not an error.
            log_dbg!(
                "mmap MAP_ANON ignoring hint for address {:?}, actual is {:?}",
                addr,
                ptr
            );
        }
    } else {
        // File-backed mmap: read file content into the allocated buffer.
        if !addr.is_null() {
            log_dbg!(
                "mmap file-backed ignoring hint for address {:?}, actual is {:?}",
                addr,
                ptr
            );
        }
        // mmap reads from an offset without changing the descriptor's cursor.
        let read = posix_io::pread(env, fd, ptr, len, offset);
        if read < 0 {
            log!(
                "Warning: mmap: read from offset {} failed on fd {}; returning MAP_FAILED",
                offset,
                fd
            );
            env.mem.free(ptr);
            set_errno(env, EIO);
            return MAP_FAILED;
        }

        if (read as u32) < len {
            log!(
                "Warning: mmap: read only {} of {} bytes from fd {}; padding remainder with zeros",
                read,
                len,
                fd
            );
            // Remainder is already zeroed (calloc)
        }
    }

    crate::mem::note_allocation_site(env.cpu.regs()[crate::cpu::Cpu::LR], len);
    track_allocation(&mut env.libc_state.mmap.allocations, ptr, len);
    if let Some(mapping) = shared_file_mapping {
        env.libc_state.mmap.file_mappings.insert(ptr, mapping);
    }
    // A brand-new mapping cannot inherit "these pages are free" records from a
    // previous mapping that the allocator recycled the address from.
    let end = ptr.to_bits().saturating_add(len);
    clear_unmapped_ranges(&mut env.libc_state.mmap.unmapped, ptr.to_bits(), end);

    ptr
}

/// Track an mmap allocation, replacing any stale entry left behind after the
/// guest heap recycled an address. `Mem::alloc` only returns addresses that are
/// currently free, so an existing entry for that address cannot still describe
/// a live allocation and must not turn guest behavior into a host panic.
fn track_allocation(
    allocations: &mut HashMap<MutVoidPtr, GuestUSize>,
    ptr: MutVoidPtr,
    len: GuestUSize,
) {
    if let Some(previous_len) = allocations.insert(ptr, len) {
        log!(
            "Warning: mmap: allocator reused {:?} while a stale mapping record remained (old length {}, new length {}); replacing it",
            ptr,
            previous_len,
            len
        );
    }
}

/// `mmap` with `MAP_FIXED`: the mapping must be placed at exactly `addr`,
/// replacing whatever was mapped there before.
///
/// The Boehm GC used by Mono (and so by every Unity game) relies on this to
/// release memory: `GC_unmap` maps fresh `PROT_NONE` pages over free heap
/// blocks with `MAP_FIXED` and aborts with `mmap(...PROT_NONE...) failed` if
/// the result isn't the address it asked for. `GC_remap` does the same with
/// read/write pages in older versions. Guest memory has no page protections,
/// so replacing a mapping just means giving it the contents of the new one:
/// zeroes for an anonymous mapping, the file's data for a file mapping.
fn mmap_fixed(
    env: &mut Environment,
    addr: MutVoidPtr,
    len: GuestUSize,
    flags: i32,
    fd: FileDescriptor,
    offset: off_t,
) -> MutVoidPtr {
    let shared_file_mapping = if (flags & MAP_ANON) == 0
        && (flags & MAP_SHARED) != 0
        && env.libc_state.posix_io.is_fd_writable(fd)
    {
        let file_offset = match u64::try_from(offset) {
            Ok(offset) => offset,
            Err(_) => {
                set_errno(env, EINVAL);
                return MAP_FAILED;
            }
        };
        match open_shared_mapping_file(env, fd) {
            Ok(file) => Some(SharedFileMapping { file, file_offset }),
            Err(error) => {
                log!(
                    "Warning: mmap MAP_FIXED: could not retain shared mapping for fd {}: {}",
                    fd,
                    error
                );
                set_errno(env, EIO);
                return MAP_FAILED;
            }
        }
    } else {
        None
    };

    let (len, is_new_mapping) = match map_fixed_zeroed(
        &mut env.mem,
        &mut env.libc_state.mmap.allocations,
        addr,
        len,
    ) {
        Ok(result) => result,
        Err(errno) => {
            if errno == ENOMEM {
                crate::mem::log_out_of_memory("mmap MAP_FIXED", len);
            }
            set_errno(env, errno);
            return MAP_FAILED;
        }
    };

    // The mapped range is now in use again, so it is not free address space.
    let end = addr.to_bits().saturating_add(len);
    clear_unmapped_ranges(&mut env.libc_state.mmap.unmapped, addr.to_bits(), end);

    if (flags & MAP_ANON) == 0 {
        let read = posix_io::pread(env, fd, addr, len, offset);
        if read < 0 {
            log!(
                "Warning: mmap MAP_FIXED: read from offset {} failed on fd {}; returning MAP_FAILED",
                offset,
                fd
            );
            if is_new_mapping {
                env.libc_state.mmap.allocations.remove(&addr);
                env.mem.free(addr);
            }
            set_errno(env, EIO);
            return MAP_FAILED;
        }
        if (read as u32) < len {
            log!(
                "Warning: mmap MAP_FIXED: read only {} of {} bytes from fd {}; padding remainder with zeros",
                read,
                len,
                fd
            );
        }
    }

    if let Some(mapping) = shared_file_mapping {
        if is_new_mapping || env.libc_state.mmap.allocations.contains_key(&addr) {
            env.libc_state.mmap.file_mappings.insert(addr, mapping);
        } else {
            log!(
                "Warning: mmap MAP_FIXED shared file mapping at {:?} overlaps an existing allocation and cannot be tracked",
                addr
            );
        }
    }

    set_errno(env, 0);
    addr
}

/// The memory side of [mmap_fixed]: make the `len` bytes at `addr` a fresh,
/// zero-filled mapping. On success, returns the page-rounded length and
/// whether the range was previously unused (and is now tracked in
/// `allocations`), otherwise an errno value.
fn map_fixed_zeroed(
    mem: &mut Mem,
    allocations: &mut HashMap<MutVoidPtr, GuestUSize>,
    addr: MutVoidPtr,
    len: GuestUSize,
) -> Result<(GuestUSize, bool), i32> {
    let base = addr.to_bits();
    let len = len
        .checked_next_multiple_of(PAGE_SIZE)
        .filter(|&len| len != 0 && base.checked_add(len - 1).is_some())
        .ok_or(EINVAL)?;
    if base & PAGE_SIZE_ALIGN_MASK != 0 || base < mem.null_segment_size() {
        return Err(EINVAL);
    }
    if mem.get_bytes_fallible_mut(addr.cast_const(), len).is_none() {
        return Err(ENOMEM);
    }

    let is_new_mapping = mem.is_range_free(base, len);
    if is_new_mapping {
        // A brand-new mapping: claim the range so the allocator won't hand
        // it out, and let munmap() release it again.
        mem.reserve(base, len);
        allocations.insert(addr, len);
    } else {
        // Replacing (part of) memory that is already in use, which is what
        // the GC does to its own heap. The range keeps belonging to whatever
        // allocation it is part of.
        log_dbg!(
            "mmap MAP_FIXED over in-use memory {:?} ({:#x} bytes)",
            addr,
            len
        );
    }
    mem.bytes_at_mut(addr.cast(), len).fill(0);
    Ok((len, is_new_mapping))
}

/// The tracked mapping that contains `addr`, as `(base, length)`.
///
/// Guests release ranges that start inside a mapping rather than at its base
/// (a partial `munmap`), and some release a range they previously carved out of
/// a mapping, so this looks for the containing mapping rather than requiring an
/// exact base match.
fn owning_mapping(env: &Environment, addr: MutVoidPtr) -> Option<(MutVoidPtr, GuestUSize)> {
    let bits = addr.to_bits();
    env.libc_state
        .mmap
        .allocations
        .iter()
        .find(|(base, len)| {
            let start = base.to_bits();
            let end = start.saturating_add(**len);
            bits >= start && bits < end
        })
        .map(|(base, len)| (*base, *len))
}

/// Release a whole tracked mapping, and forget any ranges remembered as free
/// inside it: those pages are no longer backed by anything.
fn release_mapping(env: &mut Environment, addr: MutVoidPtr, len: GuestUSize) {
    let start = addr.to_bits();
    let end = start.saturating_add(len);
    clear_unmapped_ranges(&mut env.libc_state.mmap.unmapped, start, end);
    env.libc_state.mmap.allocations.remove(&addr);
    env.libc_state.mmap.file_mappings.remove(&addr);
    env.mem.free(addr);
}

fn munmap(env: &mut Environment, addr: MutVoidPtr, len: GuestUSize) -> i32 {
    // TODO: handle errno properly
    set_errno(env, 0);
    log_dbg!("munmap({:?}, {})", addr, len);

    if len == 0 {
        // Darwin returns EINVAL for `munmap(addr, 0)`, but several apps
        // (e.g. the UE3-based BioShock port) call it this way in a loop to
        // release cached buffers and treat any failure as fatal for their
        // allocator bookkeeping. Be permissive: if the whole mapping at
        // `addr` is known, release it in full (its recorded length is what
        // the caller means); otherwise treat the call as a successful no-op.
        // This matches Linux's munmap(NULL-adjacent) tolerance and keeps the
        // guest allocator's state consistent.
        if let Some(expected_len) = env.libc_state.mmap.allocations.get(&addr).copied() {
            log_dbg!(
                "munmap({:?}, 0): zero length, unmapping whole {}-byte mapping",
                addr,
                expected_len
            );
            if let Err(error) =
                write_back_file_mapping_range(env, addr, expected_len, addr, expected_len)
            {
                log!(
                    "Warning: munmap({:?}, 0) could not write back shared mapping: {}",
                    addr,
                    error
                );
                set_errno(env, EIO);
                return -1;
            }
            release_mapping(env, addr, expected_len);
        } else {
            log_dbg!("munmap({:?}, 0): unknown mapping, treating as no-op", addr);
        }
        set_errno(env, 0);
        return 0;
    }

    let owner = owning_mapping(env, addr);
    if let Some((base, mapping_len)) = owner {
        if addr == base && len >= mapping_len {
            if len > mapping_len {
                log_dbg!(
                    "munmap({:?}, {}): length exceeds the {}-byte mapping, unmapping it whole",
                    addr,
                    len,
                    mapping_len
                );
            }
            if let Err(error) =
                write_back_file_mapping_range(env, base, mapping_len, addr, mapping_len)
            {
                log!(
                    "Warning: munmap({:?}, {}) could not write back shared mapping: {}",
                    addr,
                    len,
                    error
                );
                set_errno(env, EIO);
                return -1;
            }
            release_mapping(env, addr, mapping_len);
            return 0; // success
        }

        // Unmapping part of a mapping (or a range inside one) cannot free host
        // memory: the pages around it still belong to the caller, and freeing
        // the whole block here would hand the surviving bytes to a later `mmap`
        // while the guest still writes to them - two owners of one address
        // range, which is exactly how a guest heap ends up with corrupted chunk
        // headers. Remember the range instead, so that an `mmap` passing it as a
        // hint can reuse it the way Darwin reuses free address space.
        if let Some((start, end)) = page_range(addr, len) {
            let flush_addr = MutVoidPtr::from_bits(start);
            if let Err(error) =
                write_back_file_mapping_range(env, base, mapping_len, flush_addr, end - start)
            {
                log!(
                    "Warning: munmap({:?}, {}) could not write back shared mapping: {}",
                    addr,
                    len,
                    error
                );
                set_errno(env, EIO);
                return -1;
            }
            record_unmapped_range(
                &mut env.libc_state.mmap.unmapped,
                base,
                mapping_len,
                start,
                end,
            );
        }
        log_dbg!(
            "munmap({:?}, {}): partial unmap of the {}-byte mapping at {:?}, remembering the range as free",
            addr,
            len,
            mapping_len,
            base
        );
        set_errno(env, 0);
        0
    } else {
        // Darwin's `munmap` only rejects invalid arguments: unmapping a range
        // that holds no mapping is a successful no-op, not an error.
        // <https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/munmap.2.html>
        //
        // Allocators such as BioShock's release memory in ranges that do not
        // line up with one of our whole `mmap` allocations, or with ranges that
        // an `mmap` with `MAP_FIXED` replaced rather than created, and some of
        // them treat any `munmap` failure as fatal.
        //
        // Guest memory has no page protections, so keeping the range mapped is
        // harmless: it stays valid and either gets released together with the
        // allocation that owns it, or costs host memory until exit.
        log_dbg!(
            "munmap({:?}, {}): no tracked mapping at this address, treating as no-op",
            addr,
            len
        );
        set_errno(env, 0);
        0
    }
}

// Darwin `madvise` advice values (sys/mman.h).
const MADV_NORMAL: i32 = 0;
const MADV_RANDOM: i32 = 1;
const MADV_SEQUENTIAL: i32 = 2;
const MADV_WILLNEED: i32 = 3;
const MADV_DONTNEED: i32 = 4;
const MADV_FREE: i32 = 5;
const MADV_ZERO_WIRED_PAGES: i32 = 6;
const MADV_FREE_REUSABLE: i32 = 7;
const MADV_FREE_REUSE: i32 = 8;
const MADV_CAN_REUSE: i32 = 9;

/// Guest memory is a plain host allocation with no paging, so every advice
/// is a hint we can safely ignore. Only the argument validation that a real
/// kernel performs is emulated: unknown advice values and unaligned
/// addresses fail with `EINVAL`.
fn madvise(env: &mut Environment, addr: MutVoidPtr, len: GuestUSize, advice: i32) -> i32 {
    log_dbg!("madvise({:?}, {}, {})", addr, len, advice);
    match advice {
        MADV_NORMAL
        | MADV_RANDOM
        | MADV_SEQUENTIAL
        | MADV_WILLNEED
        | MADV_DONTNEED
        | MADV_FREE
        | MADV_ZERO_WIRED_PAGES
        | MADV_FREE_REUSABLE
        | MADV_FREE_REUSE
        | MADV_CAN_REUSE => {}
        _ => {
            set_errno(env, EINVAL);
            return -1;
        }
    }
    if addr.to_bits() & PAGE_SIZE_ALIGN_MASK != 0 {
        set_errno(env, EINVAL);
        return -1;
    }
    set_errno(env, 0);
    0
}

fn shm_open(env: &mut Environment, name: ConstPtr<u8>, oflag: i32, mode: u32) -> i32 {
    set_errno(env, 0);

    let name_str = env.mem.cstr_at_utf8(name).unwrap_or("<invalid>");
    log_dbg!("shm_open({:?}, {:#x}, {:#x})", name_str, oflag, mode);

    // Используем open_direct! Параметр mode для эмулятора здесь не нужен,
    // поэтому просто передаем env, name и oflag.
    open_direct(env, name, oflag)
}

/// `int shm_unlink(const char *name)`
///
/// POSIX shared-memory unlink: removes the named shared-memory object.
/// Regions opened via `shm_open` are backed by our fs layer, where `unlink`
/// already implements remove-on-last-close semantics. Missing regions return
/// -1/ENOENT per POSIX.
fn shm_unlink(env: &mut Environment, name: ConstPtr<u8>) -> i32 {
    crate::libc::unistd::unlink(env, name)
}

fn mprotect(env: &mut Environment, addr: MutVoidPtr, len: GuestUSize, prot: i32) -> i32 {
    // POSIX `int mprotect(void *addr, size_t len, int prot)`: returns 0
    // on success, -1 on failure with errno set.
    //
    // touchHLE doesn't enforce per-page memory protections — the entire
    // guest address space is treated as RW (and code pages as RX through
    // the JIT). However, returning -1 + ENOTSUP for every mprotect call
    // is wrong: it makes Mono/Boehm GC and Unity's runtime think the
    // protection change failed during JIT/GC initialization, which can
    // leave them in a broken state. Real Darwin kernels never fail
    // mprotect for the address ranges that mmap'd allocations sit in,
    // so the correct behavior is "succeed silently as a no-op".
    //
    // Reference: POSIX mprotect(2) — return value is 0 on success, -1 on
    // error; the only documented errors apply to invalid/non-mapped
    // address ranges, which our guest mmap allocator handles by always
    // returning a valid range.
    log_dbg!(
        "mprotect({:?}, {}, {:#x}) -> 0 (no-op; touchHLE does not enforce per-page protections)",
        addr,
        len,
        prot
    );
    set_errno(env, 0);
    0
}

/// `int mlock(const void *addr, size_t len)` — lock a region of memory
/// so it stays resident in physical RAM. touchHLE keeps the entire
/// guest address space resident in the host process at all times, so
/// there is nothing to pin: the memory is already non-pageable from the
/// guest's perspective. Real Darwin returns 0 on success, so we report
/// success as a no-op (returning -1 here would make guest code that
/// relies on mlock — e.g. crypto/keychain libraries protecting secrets,
/// or audio engines pinning buffers — treat initialization as failed).
///
/// Reference: POSIX/Darwin mlock(2) — returns 0 on success, -1 with
/// errno on failure.
fn mlock(env: &mut Environment, addr: ConstPtr<u8>, len: GuestUSize) -> i32 {
    log_dbg!(
        "mlock({:?}, {}) -> 0 (no-op; guest memory is always resident)",
        addr,
        len
    );
    set_errno(env, 0);
    0
}

/// `int munlock(const void *addr, size_t len)` — unlock a region
/// previously locked with `mlock`. Mirrors [mlock]: a no-op that
/// succeeds.
///
/// Reference: POSIX/Darwin munlock(2) — returns 0 on success.
fn munlock(env: &mut Environment, addr: ConstPtr<u8>, len: GuestUSize) -> i32 {
    log_dbg!(
        "munlock({:?}, {}) -> 0 (no-op; guest memory is always resident)",
        addr,
        len
    );
    set_errno(env, 0);
    0
}

/// `int msync(void *addr, size_t len, int flags)` — synchronize a writable
/// shared file mapping with its backing file. Other mappings need no flush.
///
/// Reference: POSIX msync(2) — returns 0 on success, -1 with errno on
/// failure.
fn msync(env: &mut Environment, addr: MutVoidPtr, len: GuestUSize, flags: i32) -> i32 {
    log_dbg!("msync({:?}, {}, {:#x})", addr, len, flags);
    if crate::env_flag_cached!("TOUCHHLE_TRACE_OPEN") {
        log!(
            "MSYNC-DIAG addr={:?} len={} flags={:#x}",
            addr,
            len,
            flags
        );
    }
    match write_back_mapped_range(env, addr, len) {
        Ok(()) => {
            if crate::env_flag_cached!("TOUCHHLE_TRACE_OPEN") {
                log!("MSYNC-DIAG operation succeeded");
            }
            set_errno(env, 0);
            0
        }
        Err(error) => {
            log!(
                "Warning: msync({:?}, {}, {:#x}) could not write back shared mapping: {}",
                addr,
                len,
                flags,
                error
            );
            set_errno(env, EIO);
            -1
        }
    }
}

pub const FUNCTIONS: FunctionExports = &[
    export_c_func!(mmap(_, _, _, _, _, _)),
    export_c_func!(munmap(_, _)),
    export_c_func!(msync(_, _, _)),
    export_c_func!(madvise(_, _, _)),
    export_c_func!(shm_open(_, _, _)),
    export_c_func!(shm_unlink(_)),
    export_c_func!(mprotect(_, _, _)),
    export_c_func!(mlock(_, _)),
    export_c_func!(munlock(_, _)),
];

#[cfg(test)]
mod mmap_tests {
    use super::*;
    use crate::mem::Ptr;
    use std::fs::OpenOptions;
    use std::io::{Read, Seek, SeekFrom};

    fn new_test_mem() -> Mem {
        let mut mem = Mem::new();
        mem.set_null_segment_size(PAGE_SIZE);
        mem
    }

    #[test]
    fn recycled_mmap_address_replaces_stale_bookkeeping() {
        let ptr = MutVoidPtr::from_bits(0x2000_0000);
        let mut allocations = HashMap::new();
        allocations.insert(ptr, PAGE_SIZE);

        track_allocation(&mut allocations, ptr, PAGE_SIZE * 2);

        assert_eq!(allocations.len(), 1);
        assert_eq!(allocations.get(&ptr), Some(&(PAGE_SIZE * 2)));
    }

    /// What Boehm GC's `GC_unmap` does: map `PROT_NONE` pages over part of
    /// a heap block it got from an earlier `mmap`, and abort unless the
    /// result is the address it asked for.
    #[test]
    fn replaces_pages_inside_an_existing_mapping_in_place() {
        let mut mem = new_test_mem();
        let mut allocations = HashMap::new();
        let heap: MutVoidPtr = mem.alloc(PAGE_SIZE * 4);
        allocations.insert(heap, PAGE_SIZE * 4);
        mem.bytes_at_mut(heap.cast(), PAGE_SIZE * 4).fill(0xAB);

        let middle: MutVoidPtr = Ptr::from_bits(heap.to_bits() + PAGE_SIZE);
        assert_eq!(
            map_fixed_zeroed(&mut mem, &mut allocations, middle, PAGE_SIZE * 2),
            Ok((PAGE_SIZE * 2, false))
        );

        let bytes = mem.bytes_at(heap.cast(), PAGE_SIZE * 4);
        let page = PAGE_SIZE as usize;
        assert!(bytes[..page].iter().all(|&b| b == 0xAB));
        assert!(bytes[page..page * 3].iter().all(|&b| b == 0));
        assert!(bytes[page * 3..].iter().all(|&b| b == 0xAB));
        // The heap block is still a single mapping that munmap() can free.
        assert_eq!(allocations.len(), 1);
        assert!(mem.is_known_allocation(heap.to_bits()));
    }

    #[test]
    fn claims_unused_address_space() {
        let mut mem = new_test_mem();
        let mut allocations = HashMap::new();
        let addr: MutVoidPtr = Ptr::from_bits(0x2000_0000);
        assert!(mem.is_range_free(addr.to_bits(), PAGE_SIZE));

        // Lengths are rounded up to whole pages.
        assert_eq!(
            map_fixed_zeroed(&mut mem, &mut allocations, addr, 100),
            Ok((PAGE_SIZE, true))
        );
        assert!(!mem.is_range_free(addr.to_bits(), PAGE_SIZE));
        assert_eq!(allocations.get(&addr), Some(&PAGE_SIZE));
        assert!(mem.bytes_at(addr.cast(), PAGE_SIZE).iter().all(|&b| b == 0));

        mem.free(addr);
        assert!(mem.is_range_free(addr.to_bits(), PAGE_SIZE));
    }

    #[test]
    fn rejects_bad_arguments() {        let mut mem = new_test_mem();
        let mut allocations = HashMap::new();
        let unaligned: MutVoidPtr = Ptr::from_bits(0x2000_0010);
        assert_eq!(
            map_fixed_zeroed(&mut mem, &mut allocations, unaligned, PAGE_SIZE),
            Err(EINVAL)
        );
        let aligned: MutVoidPtr = Ptr::from_bits(0x2000_0000);
        assert_eq!(
            map_fixed_zeroed(&mut mem, &mut allocations, aligned, 0),
            Err(EINVAL)
        );
        assert_eq!(
            map_fixed_zeroed(&mut mem, &mut allocations, Ptr::null(), PAGE_SIZE),
            Err(EINVAL)
        );
        let near_top: MutVoidPtr = Ptr::from_bits(0xFFFF_F000);
        assert_eq!(
            map_fixed_zeroed(&mut mem, &mut allocations, near_top, PAGE_SIZE * 2),
            Err(EINVAL)
        );
        assert!(allocations.is_empty());
    }

    #[test]
    fn shared_mapping_flush_writes_only_the_requested_range_at_its_file_offset() {
        let path = std::env::temp_dir().join(format!(
            "touchhle-mmap-{}-{}.tmp",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let host_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        host_file.set_len(8).unwrap();
        let mut mapping = SharedFileMapping {
            file: GuestFile::File(host_file),
            file_offset: 1,
        };
        let mut mem = new_test_mem();
        let base = mem.alloc(8);
        mem.bytes_at_mut(base.cast(), 8)
            .copy_from_slice(&[10, 11, 12, 13, 14, 15, 16, 17]);
        let range = MutVoidPtr::from_bits(base.to_bits() + 2);

        write_back_mapping_range(&mem, base, 8, &mut mapping, range, 3).unwrap();

        let mut persisted = [0; 8];
        mapping.file.seek(SeekFrom::Start(0)).unwrap();
        mapping.file.read_exact(&mut persisted).unwrap();
        assert_eq!(persisted, [0, 0, 0, 12, 13, 14, 0, 0]);

        drop(mapping);
        mem.free(base);
        std::fs::remove_file(path).unwrap();
    }
}

#[cfg(test)]
mod partial_munmap_tests {
    use super::*;
    use crate::mem::Ptr;

    /// A three-page anonymous mapping whose middle page the guest releases with
    /// `munmap`, exactly like `TestApp`'s `test_munmap_partial_mapping`.
    fn three_page_mapping_with_hole() -> (Mem, HashMap<MutVoidPtr, GuestUSize>, Vec<UnmappedRange>, MutVoidPtr)
    {
        let mut mem = Mem::new();
        mem.set_null_segment_size(PAGE_SIZE);
        let mut allocations = HashMap::new();
        let mapping: MutVoidPtr = mem.alloc(PAGE_SIZE * 3);
        allocations.insert(mapping, PAGE_SIZE * 3);
        mem.bytes_at_mut(mapping.cast(), PAGE_SIZE * 3).fill(0xAB);

        let mut unmapped = Vec::new();
        let (start, end) = page_range(
            Ptr::from_bits(mapping.to_bits() + PAGE_SIZE),
            PAGE_SIZE,
        )
        .unwrap();
        record_unmapped_range(&mut unmapped, mapping, PAGE_SIZE * 3, start, end);
        (mem, allocations, unmapped, mapping)
    }

    #[test]
    fn a_hint_inside_the_unmapped_range_is_honoured() {
        let (mut mem, allocations, mut unmapped, mapping) = three_page_mapping_with_hole();
        let middle: MutVoidPtr = Ptr::from_bits(mapping.to_bits() + PAGE_SIZE);

        assert!(honour_address_hint(
            &mut mem,
            &allocations,
            &mut unmapped,
            middle,
            PAGE_SIZE
        ));

        // The reclaimed page is fresh (zeroed) and no longer recorded as free.
        assert!(mem
            .bytes_at(middle.cast(), PAGE_SIZE)
            .iter()
            .all(|&b| b == 0));
        assert!(unmapped.is_empty());

        // The pages around it still belong to the original mapping and keep
        // their contents.
        let bytes = mem.bytes_at(mapping.cast(), PAGE_SIZE * 3);
        let page = PAGE_SIZE as usize;
        assert!(bytes[..page].iter().all(|&b| b == 0xAB));
        assert!(bytes[page * 2..].iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn a_hint_outside_the_unmapped_range_is_not_honoured() {
        let (mut mem, allocations, mut unmapped, mapping) = three_page_mapping_with_hole();

        // The first page was never released, so the hint must be ignored.
        assert!(!honour_address_hint(
            &mut mem,
            &allocations,
            &mut unmapped,
            mapping,
            PAGE_SIZE
        ));
        // Neither may a hint that is not page-aligned, nor one that spans more
        // than the released page.
        assert!(!honour_address_hint(
            &mut mem,
            &allocations,
            &mut unmapped,
            Ptr::from_bits(mapping.to_bits() + PAGE_SIZE + 4),
            PAGE_SIZE
        ));
        assert!(!honour_address_hint(
            &mut mem,
            &allocations,
            &mut unmapped,
            Ptr::from_bits(mapping.to_bits() + PAGE_SIZE),
            PAGE_SIZE * 2
        ));

        // The released page is still recorded as free.
        assert_eq!(unmapped.len(), 1);
        assert!(mem.bytes_at(mapping.cast(), PAGE_SIZE * 3)[PAGE_SIZE as usize] == 0xAB);
    }

    #[test]
    fn adjacent_released_pages_merge_into_one_range() {
        let mut mem = Mem::new();
        mem.set_null_segment_size(PAGE_SIZE);
        let mut allocations = HashMap::new();
        let mapping: MutVoidPtr = mem.alloc(PAGE_SIZE * 3);
        allocations.insert(mapping, PAGE_SIZE * 3);

        let mut unmapped = Vec::new();
        for page in 0..2 {
            let addr = Ptr::from_bits(mapping.to_bits() + (page + 1) * PAGE_SIZE);
            let (start, end) = page_range(addr, PAGE_SIZE).unwrap();
            record_unmapped_range(&mut unmapped, mapping, PAGE_SIZE * 3, start, end);
        }

        assert_eq!(unmapped.len(), 1);
        assert_eq!(unmapped[0].start, mapping.to_bits() + PAGE_SIZE);
        assert_eq!(unmapped[0].end, mapping.to_bits() + PAGE_SIZE * 3);

        // Both pages can be claimed with a single hint.
        assert!(honour_address_hint(
            &mut mem,
            &allocations,
            &mut unmapped,
            Ptr::from_bits(mapping.to_bits() + PAGE_SIZE),
            PAGE_SIZE * 2
        ));
        assert!(unmapped.is_empty());
    }

    #[test]
    fn releasing_the_whole_mapping_forgets_its_released_ranges() {
        let (mut mem, mut allocations, mut unmapped, mapping) = three_page_mapping_with_hole();

        let start = mapping.to_bits();
        let end = start + PAGE_SIZE * 3;
        clear_unmapped_ranges(&mut unmapped, start, end);
        allocations.remove(&mapping);
        mem.free(mapping);

        assert!(unmapped.is_empty());
        // A hint at the old address must not be honoured once the mapping is
        // gone: the address space may be handed out to something else.
        assert!(!honour_address_hint(
            &mut mem,
            &allocations,
            &mut unmapped,
            Ptr::from_bits(start + PAGE_SIZE),
            PAGE_SIZE
        ));
    }
}
