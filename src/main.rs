use std::{
    arch::x86_64::{
        _MM_HINT_T0, _mm_prefetch, _mm256_cmpeq_epi8, _mm256_loadu_epi8, _mm256_movemask_epi8,
        _mm256_set1_epi8,
    },
    os::fd::AsRawFd,
    ptr::{slice_from_raw_parts, slice_from_raw_parts_mut},
    str::from_utf8_unchecked,
};

use mmap::{MapOption, MemoryMap};

const N: usize = 1 << 17;
const CHUNKSIZE: usize = 32;
const SEED: i64 = 123456;

#[derive(Default, Clone)]
#[repr(align(64))]
struct Entry {
    hash: u64,
    key: &'static [u8],
    min: i32,
    max: i32,
    sum: i32,
    count: i32,
}

fn main() {
    let file = std::fs::File::open("./data/measurements.txt").unwrap();
    let file_len = file.metadata().unwrap().len() as usize;
    let mmap = MemoryMap::new(
        file_len,
        &[MapOption::MapFd(file.as_raw_fd()), MapOption::MapReadable],
    )
    .unwrap();
    let slice = slice_from_raw_parts_mut(mmap.data(), file_len);
    let bytes = unsafe { slice.as_ref().unwrap().trim_ascii_end() };
    unsafe {
        madvise::madvise(bytes.as_ptr(), bytes.len(), madvise::AccessPattern::Sequential).unwrap();
    }

    let nthreads = std::convert::Into::<usize>::into(std::thread::available_parallelism().unwrap());
    let sections = split_file(bytes, nthreads);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(nthreads);
        handles.extend(
            sections
                .iter()
                .map(|section| scope.spawn(|| iter_lines(section))),
        );

        let mut table = vec![Entry::default(); N];
        for handle in handles {
            let mut map = handle.join().unwrap();
            for e in map.iter_mut() {
                merge_entry(&mut table, e);
            }
        }

        print!("{{");
        let mut small_table: Vec<_> = table.into_iter().filter(|e| e.count > 0).collect();
        small_table.sort_unstable_by(|a, b| a.key.partial_cmp(b.key).unwrap());
        let mut iter = small_table.iter().peekable();
        while let Some(e) = iter.next() {
            let min = e.min as f64 / 10.;
            let max = e.max as f64 / 10.;
            let mean = (e.sum as f64 / 10.) / e.count as f64;
            let station = unsafe { from_utf8_unchecked(e.key) };
            print!("{station}={min:.1}/{mean:.1}/{max:.1}",);

            if iter.peek().is_some() {
                print!(", ");
            }
        }

        println!("}}");
    });
}

#[inline(always)]
fn parse_temperature(temperature: &[u8]) -> i32 {
    let len = temperature.len();
    let negative = unsafe { *temperature.get_unchecked(0) == b'-' };
    let decimal = unsafe { *temperature.get_unchecked(len - 1) } - b'0';
    let ndigits = len - 2 - negative as usize;
    let start_index = negative as usize;

    let has_10s = ndigits as u8 - 1;
    let whole = (unsafe { *temperature.get_unchecked(start_index) } - b'0')
        + ((unsafe { *temperature.get_unchecked(start_index) } - b'0') * (9 * (has_10s)))
        + ((unsafe { *temperature.get_unchecked(start_index + 1) } - b'0') * has_10s);

    ((whole as i32 * 10) + decimal as i32) * (1 - 2 * negative as i32)
}

#[inline(always)]
fn merge_entry(table: &mut [Entry], entry: &mut Entry) {
    let mut slot = (entry.hash as usize) & (N - 1);
    loop {
        let e = unsafe { table.get_unchecked_mut(slot) };
        if e.count == 0 {
            *e = entry.clone();
            break;
        }

        if e.hash == entry.hash {
            e.count += entry.count;
            e.sum += entry.sum;
            e.min = e.min.min(entry.min);
            e.max = e.max.max(entry.max);
            break;
        }

        slot = (slot + 1) & (N - 1);
    }
}

#[inline(always)]
fn set_entry(
    table: &mut [Entry],
    mut slot: usize,
    hash: u64,
    station: &'static [u8],
    temperature: i32,
) {
    loop {
        let e = unsafe { table.get_unchecked_mut(slot) };
        if e.count == 0 {
            e.hash = hash;
            e.key = station;

            e.count += 1;
            e.min = temperature;
            e.max = temperature;
            e.sum = temperature;
            break;
        }

        if e.hash == hash {
            e.count += 1;
            e.sum += temperature;
            if temperature < e.min {
                e.min = temperature;
            }
            if temperature > e.max {
                e.max = temperature;
            }
            break;
        }

        slot = (slot + 1) & (N - 1);
    }
}

#[inline(always)]
fn iter_lines(bytes: &[u8]) -> Vec<Entry> {
    let mut table = vec![Entry::default(); N];

    let newline_needle = unsafe { _mm256_set1_epi8(b'\n' as i8) };
    let semicolon_needle = unsafe { _mm256_set1_epi8(b';' as i8) };
    let ptr = bytes.as_ptr();
    let bytes_len = bytes.len();
    let mut semicolon_idx = 0;
    let mut semicolon_chunks = 0;
    let mut i = 0;
    let mut chunks = 0;
    let mut found = false;
    let mut found_semi = false;

    'main_loop: while i + CHUNKSIZE <= bytes_len {
        let (station, temperature) = unsafe {
            #[allow(unused)]
            let mut n = 0;
            while i + CHUNKSIZE <= bytes_len {
                let haystack = _mm256_loadu_epi8(ptr.add(i) as *const i8);
                let newline_res = _mm256_cmpeq_epi8(newline_needle, haystack);
                let newline_mask = _mm256_movemask_epi8(newline_res);

                let semicolon_res = _mm256_cmpeq_epi8(semicolon_needle, haystack);
                let semicolon_mask = _mm256_movemask_epi8(semicolon_res);

                if semicolon_mask != 0 {
                    semicolon_idx = semicolon_mask.trailing_zeros();
                    found_semi = true;
                }

                if newline_mask != 0 {
                    n = newline_mask.trailing_zeros();
                    // newline is at i + n
                    found = true;
                    break;
                }

                chunks += 1;
                semicolon_chunks += !found_semi as i32;
                i += CHUNKSIZE;
            }

            if !found {
                break 'main_loop;
            }

            let chunk_offset = CHUNKSIZE * chunks;
            let line_len = chunk_offset + n as usize;
            let station_len = (CHUNKSIZE * semicolon_chunks as usize) + semicolon_idx as usize;
            let station = slice_from_raw_parts(ptr.add(i - chunk_offset), station_len);
            let temperature = slice_from_raw_parts(
                ptr.add(i - chunk_offset + station_len + 1),
                line_len - 1 - station_len,
            );

            i += 1 + n as usize;
            chunks = 0;
            (station.as_ref().unwrap(), temperature.as_ref().unwrap())
        };

        let hash = gxhash::gxhash64(station, SEED);
        let slot = (hash as usize) & (N - 1);
        unsafe {
            _mm_prefetch::<{ _MM_HINT_T0 }>(table.as_ptr().add(slot) as *const i8);
        }
        let temperature = parse_temperature(temperature);

        let hash = gxhash::gxhash64(station, SEED);
        let slot = (hash as usize) & (N - 1);
        set_entry(&mut table, slot, hash, station, temperature);
    }

    if i < bytes_len {
        let lines = unsafe {
            let remainder = slice_from_raw_parts(ptr.add(i), bytes_len - i)
                .as_ref()
                .unwrap();
            remainder.split(|c| *c == b'\n')
        };

        for line in lines {
            let (station, temperature) = {
                // max len: -99.9 - 5 bytes - ';' is at most 6 bytes from end
                // min len: 0.0 - 3 bytes - ';' is at least 4 bytes
                // i.e. ';' is at one of len - {4,5,6}
                let len = line.len();
                let at_4 = unsafe { *line.get_unchecked(len - 4) } == b';';
                let at_5 = unsafe { *line.get_unchecked(len - 5) } == b';';
                let at_6 = unsafe { *line.get_unchecked(len - 6) } == b';';

                let index = len - (at_4 as usize * 4) - (at_5 as usize * 5) - (at_6 as usize * 6);
                let ptr = line.as_ptr();
                unsafe {
                    let left = slice_from_raw_parts(ptr, index);
                    let right = slice_from_raw_parts(ptr.add(index + 1), len - index - 1);
                    (left.as_ref().unwrap(), right.as_ref().unwrap())
                }
            };

            let temperature = parse_temperature(temperature);

            let hash = gxhash::gxhash64(station, SEED);
            let slot = (hash as usize) & (N - 1);
            set_entry(&mut table, slot, hash, station, temperature);
        }
    }

    table
}

/// returns start end pairs for chunks
fn split_file(bytes: &[u8], nthreads: usize) -> Vec<&[u8]> {
    let file_chunk_size = bytes.len() / nthreads;

    let starting_points: Vec<_> = (0..nthreads)
        .map(|i| find_next_newline(bytes, i * file_chunk_size))
        .collect();

    let mut ending_points: Vec<_> = (0..nthreads - 1).map(|i| starting_points[i + 1]).collect();
    ending_points.push(bytes.len());

    let ptr = bytes.as_ptr();
    starting_points
        .into_iter()
        .zip(ending_points)
        .map(|(start, end)| unsafe {
            slice_from_raw_parts(ptr.add(start + 1), end - start - 1)
                .as_ref()
                .unwrap()
        })
        .collect()
}

fn find_next_newline(bytes: &[u8], idx: usize) -> usize {
    let ptr = bytes.as_ptr();
    let mut i = 0;
    let max_line_len = 110;
    while i < max_line_len {
        if unsafe { *ptr.add(idx + i) } == b'\n' {
            return idx + i;
        }
        i += 1;
    }

    panic!("no newline");
}
