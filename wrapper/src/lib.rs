#![allow(non_snake_case, non_upper_case_globals, non_camel_case_types)]
include!("driver-ffi.rs");

use ark_std::{end_timer, perf_trace::Colorize, start_timer};
use std::ffi::{c_void, CString};
use std::{collections::HashMap, fmt, hash::Hash, mem, ptr};
use thousands::Separable;

#[derive(Debug)]
struct allocation_t {
    dev_ptr: u64,
    el_size: usize,
    el_count: usize,
    copy_to_dev_count: usize,
    copy_from_dev_count: usize,
    arg_buff: *mut u64, //buffer to hold launch argument pointer
}
const arg_buff_alloc_size: usize = 64;

#[derive(Debug)]
pub struct AddAllocationInfo {
    key: String,
    el_size: usize,
    el_count: usize,
    host_src: *mut c_void,
}

pub trait TupleToAllocInfo {
    fn get_allocation_info(&self) -> AddAllocationInfo;
}

impl TupleToAllocInfo for (&str, usize, usize) {
    fn get_allocation_info(&self) -> AddAllocationInfo {
        AddAllocationInfo {
            key: String::from(self.0),
            el_size: self.1,
            el_count: usize::from(self.2),
            host_src: std::ptr::null_mut(),
        }
    }
}

impl<T> TupleToAllocInfo for (&str, &Vec<T>) {
    fn get_allocation_info(&self) -> AddAllocationInfo {
        AddAllocationInfo {
            key: String::from(self.0),
            el_size: std::mem::size_of::<T>(),
            el_count: self.1.len(),
            host_src: self.1.as_ptr() as *mut std::ffi::c_void,
        }
    }
}

#[macro_export]
macro_rules! alloc_info {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec : Vec<AddAllocationInfo> = Vec::new() ;
            $(
                temp_vec.push( $x.get_allocation_info() );
            )*
            temp_vec
        }
    };
}

#[derive(Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum Result {
    SUCCESS,
    DRIVER_ERROR(CUresult),
    INVALID_KERNEL_FILE(String),
    ALLOCATION_KEY_EXIST(String),
    ALLOCATION_KEY_NOT_FOUND(String),
    OUT_OF_MEMORY(usize),
    COPY_EXCEEDS_ALLOCATION(usize, usize),
}

impl Result {
    fn get_msg(&self) -> String {
        match self {
            Result::SUCCESS => format!("Success"),
            Result::DRIVER_ERROR(err) => format!("{:?}", err),
            Result::INVALID_KERNEL_FILE(filename) => {
                format!("Cannot open kernel file '{}'", filename)
            }
            Result::ALLOCATION_KEY_EXIST(key) => {
                format!("'{}' key exist in allocation list", key)
            }
            Result::ALLOCATION_KEY_NOT_FOUND(key) => {
                format!("'{}' key not found in allocation list", key)
            }
            Result::OUT_OF_MEMORY(required) => {
                format!("failed to allocate {} on device", required)
            }
            Result::COPY_EXCEEDS_ALLOCATION(allocation_size, copy_size) => {
                format!(
                    "memory copy size ({}) exceeds allocation ({}) on device",
                    copy_size, allocation_size
                )
            }
        }
    }
}

impl fmt::Debug for Result {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.get_msg())
    }
}

impl fmt::Display for Result {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.get_msg())
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CudaDim {
    pub gridDimX: ::std::os::raw::c_uint,
    pub gridDimY: ::std::os::raw::c_uint,
    pub gridDimZ: ::std::os::raw::c_uint,
    pub blockDimX: ::std::os::raw::c_uint,
    pub blockDimY: ::std::os::raw::c_uint,
    pub blockDimZ: ::std::os::raw::c_uint,
}

impl fmt::Display for CudaDim {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug)]
pub enum ModuleSource {
    PTX_TEXT(String),
    FILE(String),
}

#[derive(Debug)]
pub struct DriverInterface {
    last_id_num: u64,
    driver_verison: i32,
    device_count: i32,
    device_handle: CUdevice,
    device_mem_cap: usize,
    cu_context: CUcontext,
    cu_module: CUmodule,
    dev_allocations: HashMap<String, allocation_t>,
    clean_up_context_list: HashMap<u64, CUcontext>,
    clean_up_stream_list: HashMap<u64, CUstream>,
    clean_up_mudule_list: HashMap<u64, CUmodule>,
    last_error_code: Result,
    verbosity: usize,
}

impl DriverInterface {
    pub fn new(cukernel: ModuleSource) -> Self {
        let mut instance = DriverInterface {
            last_id_num: 1,
            driver_verison: 0i32,
            device_count: 0,
            device_handle: -1,
            device_mem_cap: 0,
            cu_context: ptr::null_mut(),
            cu_module: ptr::null_mut(),
            dev_allocations: HashMap::new(),
            clean_up_context_list: HashMap::new(),
            clean_up_stream_list: HashMap::new(),
            clean_up_mudule_list: HashMap::new(),
            last_error_code: Result::SUCCESS,
            verbosity: 0,
        };

        match unsafe { cuInit(0) } {
            CUresult::CUDA_SUCCESS => {}
            cuda_error => {
                instance.last_error_code = Result::DRIVER_ERROR(cuda_error);
                return instance;
            }
        }

        match unsafe { cuDriverGetVersion(&mut instance.driver_verison as *mut i32) } {
            CUresult::CUDA_SUCCESS => {}
            cuda_error => {
                instance.last_error_code = Result::DRIVER_ERROR(cuda_error);
                return instance;
            }
        }

        match unsafe { cuDeviceGetCount(&mut instance.device_count as *mut i32) } {
            CUresult::CUDA_SUCCESS => {}
            cuda_error => {
                instance.last_error_code = Result::DRIVER_ERROR(cuda_error);
                return instance;
            }
        }

        match unsafe { cuDeviceGet(&mut instance.device_handle as *mut CUdevice, 0) } {
            CUresult::CUDA_SUCCESS => {}
            cuda_error => {
                instance.last_error_code = Result::DRIVER_ERROR(cuda_error);
                return instance;
            }
        }

        match unsafe {
            cuDeviceTotalMem_v2(
                &mut instance.device_mem_cap as *mut usize,
                instance.device_handle,
            )
        } {
            CUresult::CUDA_SUCCESS => {}
            cuda_error => {
                instance.last_error_code = Result::DRIVER_ERROR(cuda_error);
                return instance;
            }
        }

        match unsafe {
            cuCtxCreate_v2(
                &mut instance.cu_context as *mut CUcontext,
                CUctx_flags_enum::CU_CTX_MAP_HOST as u32
                    | CUctx_flags_enum::CU_CTX_SCHED_AUTO as u32,
                instance.device_handle,
            )
        } {
            CUresult::CUDA_SUCCESS => {
                instance
                    .clean_up_context_list
                    .insert(instance.last_id_num, instance.cu_context);

                instance.last_id_num += 1;
            }
            cuda_error => {
                instance.last_error_code = Result::DRIVER_ERROR(cuda_error);
                return instance;
            }
        }

        let cuda_result = match cukernel {
            ModuleSource::PTX_TEXT(txt) => unsafe {
                cuModuleLoadData(
                    &mut instance.cu_module as *mut CUmodule,
                    txt.as_ptr() as *const c_void,
                )
            },
            ModuleSource::FILE(filename) => {
                if !std::path::Path::new(&filename).exists() {
                    instance.last_error_code = Result::INVALID_KERNEL_FILE(filename);
                    return instance;
                }

                let filename_cstr = CString::new(filename).unwrap();

                unsafe {
                    cuModuleLoad(
                        &mut instance.cu_module as *mut CUmodule,
                        filename_cstr.as_ptr(),
                    )
                }
            }
        };

        match cuda_result {
            CUresult::CUDA_SUCCESS => {
                instance
                    .clean_up_mudule_list
                    .insert(instance.last_id_num, instance.cu_module);
                instance.last_id_num += 1;
            }
            cuda_error => {
                instance.last_error_code = Result::DRIVER_ERROR(cuda_error);
                return instance;
            }
        }

        instance
    }
}

impl Drop for DriverInterface {
    fn drop(&mut self) {
        for (_, _alloc) in self.dev_allocations.iter() {
            let p = _alloc.dev_ptr as u64;

            match unsafe { cuMemFree_v2(p) } {
                CUresult::CUDA_SUCCESS => {}
                cuda_error => {
                    println!(
                        "Free device allocation failed with error_code:{{{:?}}}",
                        cuda_error
                    );
                }
            }

            unsafe { libc::free(_alloc.arg_buff as *mut c_void) };
        }

        for (_, cu_mod) in self.clean_up_mudule_list.iter() {
            match unsafe { cuModuleUnload(*cu_mod) } {
                CUresult::CUDA_SUCCESS => {}
                cuda_error => {
                    println!("Unload module failed with error_code:{{{:?}}}", cuda_error);
                }
            }
        }

        for (_, cu_strm) in self.clean_up_stream_list.iter() {
            match unsafe { cuStreamDestroy_v2(*cu_strm) } {
                CUresult::CUDA_SUCCESS => {}
                cuda_error => {
                    println!("Stream destroy failed with error_code:{{{:?}}}", cuda_error);
                }
            }
        }

        for (_, cu_ctx) in self.clean_up_context_list.iter() {
            match unsafe { cuCtxDestroy_v2(*cu_ctx) } {
                CUresult::CUDA_SUCCESS => {}
                cuda_error => {
                    println!(
                        "Context destroy failed with error_code:{{{:?}}}",
                        cuda_error
                    );
                }
            }
        }
    }
}

impl DriverInterface {
    pub fn version(&self) -> i32 {
        self.driver_verison
    }

    pub fn device_count(&self) -> i32 {
        self.device_count
    }

    pub fn total_memory(&self) -> usize {
        self.device_mem_cap
    }

    pub fn set_verbosity(&mut self, n: usize) {
        self.verbosity = n;
    }

    pub fn verbose(&mut self) {
        self.set_verbosity(1)
    }

    pub fn high_verbosity(&mut self) {
        self.set_verbosity(3)
    }

    pub fn error_occured(&self) -> bool {
        self.last_error_code != Result::SUCCESS
    }

    pub fn last_error(&self) -> Result {
        self.last_error_code.clone()
    }

    pub fn dump_error(&self) {
        if self.verbosity > 1 && self.error_occured() {
            println!(
                "\n{}\n",
                format!(
                    "*** Error in cuda driver wrapper [ {} ] ***",
                    self.last_error_code.to_string()
                )
                .red()
                .bold(),
            );
        }
    }
}

impl DriverInterface {
    pub fn add_allocations(&mut self, list_of_allocations_info: Vec<AddAllocationInfo>) -> Result {
        let verbosity = self.verbosity;

        let mut total_memory_required: usize = 0;
        for alloc_info in list_of_allocations_info.iter() {
            if self.dev_allocations.contains_key(alloc_info.key.as_str()) {
                self.last_error_code = Result::ALLOCATION_KEY_EXIST(alloc_info.key.clone());
                self.dump_error();
                return self.last_error_code.clone();
            }

            total_memory_required += alloc_info.el_size * alloc_info.el_count;
        }

        if total_memory_required >= self.device_mem_cap {
            self.last_error_code = Result::OUT_OF_MEMORY(total_memory_required);
            self.dump_error();
            return self.last_error_code.clone();
        }

        for alloc_info in list_of_allocations_info.iter() {
            //
            let mut alloc = allocation_t {
                dev_ptr: 0,
                el_size: alloc_info.el_size,
                el_count: alloc_info.el_count,
                arg_buff: ptr::null_mut(),
                copy_to_dev_count: 0,
                copy_from_dev_count: 0,
            };

            match unsafe {
                cuMemAlloc_v2(
                    &mut alloc.dev_ptr as *mut u64,
                    alloc_info.el_size * alloc_info.el_count,
                )
            } {
                CUresult::CUDA_SUCCESS => {}
                cuda_error => {
                    self.last_error_code = Result::DRIVER_ERROR(cuda_error);
                    self.dump_error();
                    return self.last_error_code.clone();
                }
            }

            alloc.arg_buff = unsafe { libc::malloc(arg_buff_alloc_size) as *mut u64 };
            unsafe {
                *alloc.arg_buff = alloc.dev_ptr;
            }

            let copy_result = if alloc_info.host_src != std::ptr::null_mut() {
                alloc.copy_to_dev_count += 1;
                unsafe {
                    cuMemcpyHtoD_v2(
                        alloc.dev_ptr,
                        alloc_info.host_src as *const c_void,
                        alloc_info.el_size * alloc_info.el_count,
                    )
                }
            } else {
                CUresult::CUDA_SUCCESS
            };

            self.dev_allocations.insert(alloc_info.key.clone(), alloc);

            if copy_result != CUresult::CUDA_SUCCESS {
                self.last_error_code = Result::DRIVER_ERROR(copy_result);
                self.dump_error();
                return self.last_error_code.clone();
            }
        }

        if verbosity > 2 {
            println!(
                "\nCreated {} allocations on device {} totaling {} bytes",
                list_of_allocations_info.len(),
                0,
                total_memory_required.separate_with_commas(),
            );

            for alloc_info in list_of_allocations_info.iter() {
                println!(
                    "  -  key : '{}' , element_size:{} , element_count:{}{}",
                    alloc_info.key,
                    alloc_info.el_size,
                    alloc_info.el_count,
                    if alloc_info.host_src != std::ptr::null_mut() {
                        format!(
                            "  ,  copied {} bytes to device",
                            (alloc_info.el_size * alloc_info.el_count).separate_with_commas()
                        )
                    } else {
                        format!("")
                    }
                );
            }
        }

        Result::SUCCESS
    }

    pub fn copy_to_device<T>(&mut self, key: &str, src_data: &Vec<T>) -> Result {
        //
        if !self.dev_allocations.contains_key(key) {
            self.last_error_code = Result::ALLOCATION_KEY_NOT_FOUND(String::from(key));
            self.dump_error();
            return self.last_error_code.clone();
        }

        let alloc = self.dev_allocations.get(key).unwrap();

        if (mem::size_of::<T>() * src_data.len()) > (alloc.el_size * alloc.el_count) {
            self.last_error_code = Result::COPY_EXCEEDS_ALLOCATION(
                alloc.el_size * alloc.el_count,
                mem::size_of::<T>() * src_data.len(),
            );
            return self.last_error_code.clone();
        }

        let copy_size = mem::size_of::<T>() * std::cmp::min(src_data.len(), alloc.el_count);

        match unsafe {
            cuMemcpyHtoD_v2(alloc.dev_ptr, src_data.as_ptr() as *const c_void, copy_size)
        } {
            CUresult::CUDA_SUCCESS => {
                self.dev_allocations.get_mut(key).unwrap().copy_to_dev_count += 1;
            }
            cuda_error => {
                self.last_error_code = Result::DRIVER_ERROR(cuda_error);
                self.dump_error();
                return self.last_error_code.clone();
            }
        }

        if self.verbosity > 2 {
            println!(
                "Copied {} bytes to allocation '{}' on device {} ",
                copy_size.separate_with_commas(),
                key,
                0,
            );
        }

        Result::SUCCESS
    }

    pub fn copy_to_host<T>(&mut self, key: &str, dst_data: &mut Vec<T>) -> Result {
        //
        if !self.dev_allocations.contains_key(key) {
            self.last_error_code = Result::ALLOCATION_KEY_NOT_FOUND(String::from(key));
            self.dump_error();
            return self.last_error_code.clone();
        }

        let alloc = self.dev_allocations.get(key).unwrap();
        let copy_size = mem::size_of::<T>() * std::cmp::min(dst_data.len(), alloc.el_count);
        match unsafe { cuMemcpyDtoH_v2(dst_data.as_ptr() as *mut c_void, alloc.dev_ptr, copy_size) }
        {
            CUresult::CUDA_SUCCESS => {
                self.dev_allocations
                    .get_mut(key)
                    .unwrap()
                    .copy_from_dev_count += 1;
            }
            cuda_error => {
                self.last_error_code = Result::DRIVER_ERROR(cuda_error);
                self.dump_error();
                return self.last_error_code.clone();
            }
        }

        if self.verbosity > 2 {
            println!(
                "Copied {} bytes from allocation '{}' on device {} to host",
                copy_size.separate_with_commas(),
                key,
                0,
            );
        }

        Result::SUCCESS
    }

    pub fn total_mem_allocated(&self) -> usize {
        let mut mem_alloc: usize = 0;

        for (_, _alloc) in self.dev_allocations.iter() {
            mem_alloc += _alloc.el_size * _alloc.el_count;
        }

        mem_alloc
    }
}

impl DriverInterface {
    pub fn launch_kernel_with_dim(
        &mut self,
        function_name: &str,
        allocations_as_function_arguments: Vec<&str>,
        dimension: CudaDim,
    ) -> Result {
        //

        let __function_name = CString::new(function_name).unwrap();
        let mut cu_ftn: CUfunction = ptr::null_mut();
        let mut argument_pointers: Vec<*mut u64> =
            Vec::with_capacity(allocations_as_function_arguments.len());

        for key in allocations_as_function_arguments {
            match self.dev_allocations.get(key) {
                Some(alloc) => {
                    argument_pointers.push(alloc.arg_buff);
                }
                None => {
                    self.last_error_code = Result::ALLOCATION_KEY_NOT_FOUND(String::from(key));
                    self.dump_error();
                    return self.last_error_code.clone();
                }
            }
        }

        match unsafe {
            cuModuleGetFunction(
                &mut cu_ftn as *mut CUfunction,
                self.cu_module,
                __function_name.as_ptr(),
            )
        } {
            CUresult::CUDA_SUCCESS => {}
            cuda_error => {
                self.last_error_code = Result::DRIVER_ERROR(cuda_error);
                self.dump_error();
                return self.last_error_code.clone();
            }
        }

        let mut cu_stream: CUstream = ptr::null_mut();
        match unsafe {
            cuStreamCreate(
                &mut cu_stream as *mut CUstream,
                CUstream_flags_enum::CU_STREAM_NON_BLOCKING as u32,
            )
        } {
            CUresult::CUDA_SUCCESS => {
                self.clean_up_stream_list
                    .insert(self.last_id_num, cu_stream);
                self.last_id_num += 1;
            }
            cuda_error => {
                self.last_error_code = Result::DRIVER_ERROR(cuda_error);
                self.dump_error();
                return self.last_error_code.clone();
            }
        }

        let mut start_time = ark_std::perf_trace::TimerInfo {
            msg: "".to_string(),
            time: std::time::Instant::now(),
        };

        let verbosity = self.verbosity;

        if verbosity > 0 {
            start_time = start_timer!(|| {
                format!(
                    "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
                    "Cuda Launch [",
                    " function : ".dimmed(),
                    function_name.red().bold(),
                    "  grid : ".dimmed(),
                    "{ x: ".red().bold(),
                    dimension.gridDimX.to_string().red().bold(),
                    ", y: ".red().bold(),
                    dimension.gridDimY.to_string().red().bold(),
                    ", z: ".red().bold(),
                    dimension.gridDimZ.to_string().red().bold(),
                    " }".red().bold(),
                    "  block : ".dimmed(),
                    "{ x: ".red().bold(),
                    dimension.blockDimX.to_string().red().bold(),
                    ", y: ".red().bold(),
                    dimension.blockDimY.to_string().red().bold(),
                    ", z: ".red().bold(),
                    dimension.blockDimZ.to_string().red().bold(),
                    " }".red().bold(),
                    " ]",
                )
            });
        }

        let launch_cu_result = unsafe {
            cuLaunchKernel(
                cu_ftn,
                dimension.gridDimX,
                dimension.gridDimY,
                dimension.gridDimZ,
                dimension.blockDimX,
                dimension.blockDimY,
                dimension.blockDimZ,
                0,
                cu_stream,
                argument_pointers.as_mut_ptr() as *mut *mut c_void,
                ptr::null_mut(),
            )
        };

        match launch_cu_result {
            CUresult::CUDA_SUCCESS => {}
            cuda_error => {
                if verbosity > 0 {
                    end_timer!(start_time);
                }
                self.last_error_code = Result::DRIVER_ERROR(cuda_error);
                self.dump_error();
                return self.last_error_code.clone();
            }
        }

        unsafe { cuStreamSynchronize(cu_stream) };

        if verbosity > 0 {
            end_timer!(start_time);
        }

        Result::SUCCESS
    }

    pub fn launch_kernel(
        &mut self,
        function_name: &str,
        allocations_as_function_arguments: Vec<&str>,
        total_thread_count: usize,
    ) -> Result {
        let total_thread_cc = total_thread_count as std::os::raw::c_uint;

        let dimension = if total_thread_count <= 32 {
            CudaDim {
                blockDimX: total_thread_cc,
                blockDimY: 1,
                blockDimZ: 1,

                gridDimX: 1,
                gridDimY: 1,
                gridDimZ: 1,
            }
        } else {
            CudaDim {
                blockDimX: 32,
                blockDimY: 1,
                blockDimZ: 1,

                gridDimX: (total_thread_cc) / 32,
                gridDimY: 1,
                gridDimZ: 1,
            }
        };

        self.launch_kernel_with_dim(function_name, allocations_as_function_arguments, dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // not a test for get driver version, but a test for linking
    #[test]
    fn link_test() {
        let mut version: i32 = 0;
        let result = unsafe { cuDriverGetVersion(&mut version as *mut i32) };
        match result {
            CUresult::CUDA_SUCCESS => {
                println!("Deriver Version = {:?}", version);
            }
            _ => {
                println!("Cannot get driver version");
            }
        }
    }
}

#[cfg(test)]
mod driver_interface_test {

    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct bigInt256([u64; 4]);

    const Zero: bigInt256 = bigInt256([0, 0, 0, 0]);

    #[inline(always)]
    fn bigInt256Rnd() -> bigInt256 {
        bigInt256([
            rand::random::<u64>(),
            rand::random::<u64>(),
            rand::random::<u64>(),
            rand::random::<u64>(),
        ])
    }

    #[inline(always)]
    fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
        let ret = (a as u128) + (b as u128) + (carry as u128);
        (ret as u64, (ret >> 64) as u64)
    }

    fn add(a: &bigInt256, b: &bigInt256) -> bigInt256 {
        let (d0, carry) = adc(a.0[0], b.0[0], 0);
        let (d1, carry) = adc(a.0[1], b.0[1], carry);
        let (d2, carry) = adc(a.0[2], b.0[2], carry);
        let (d3, _) = adc(a.0[3], b.0[3], carry);

        bigInt256([d0, d1, d2, d3])
    }

    #[test]
    fn driver_interface_test() {
        let big_int_size: usize = mem::size_of::<bigInt256>();
        let array_size: usize = 1 << 16;
        let mut A_in: Vec<bigInt256> = vec![Zero; array_size];
        let mut B_in: Vec<bigInt256> = vec![Zero; array_size];
        let mut Ans: Vec<bigInt256> = vec![Zero; array_size];
        let mut Out: Vec<bigInt256> = vec![Zero; array_size];

        for idx in 0..array_size {
            let a = bigInt256Rnd();
            let b = bigInt256Rnd();

            A_in[idx] = a;
            B_in[idx] = b;
            Ans[idx] = add(&a, &b);
        }

        let kernel_ptx = include_str!("../resources/kernel.ptx");
        let mut drv_interface =
            DriverInterface::new(ModuleSource::PTX_TEXT(String::from(kernel_ptx)));

        drv_interface.high_verbosity();

        match drv_interface.last_error() {
            Result::SUCCESS => {}
            error_result => {
                panic!("Error : {:?}", error_result);
            }
        }

        println!("Deriver Version = {:?} ", drv_interface.driver_verison);

        match drv_interface.add_allocations(alloc_info![
            ("A_in", &A_in),
            ("B_in", big_int_size, array_size),
            ("Out", big_int_size, array_size)
        ]) {
            Result::SUCCESS => {}
            error_result => {
                panic!("Error : {:?}", error_result);
            }
        }

        match drv_interface.copy_to_device("B_in", &B_in) {
            Result::SUCCESS => {}
            error_result => {
                panic!("Error : {:?}", error_result);
            }
        }

        match drv_interface.launch_kernel("add_test", vec!["A_in", "B_in", "Out"], array_size) {
            Result::SUCCESS => {}
            error_result => {
                panic!("Error : {:?}", error_result);
            }
        }

        match drv_interface.copy_to_host("Out", &mut Out) {
            Result::SUCCESS => {}
            error_result => {
                panic!("Error : {:?}", error_result);
            }
        }

        // Compare
        for idx in 0..array_size {
            if Ans[idx] != Out[idx] {
                panic!(
                    "\nResult @ index : {} did not match \n\tAns : {:?} \n\t!= \n\tOut : {:?}",
                    idx, Ans[idx], Out[idx]
                );
            }
        }
    }
}
