#![allow(non_snake_case, non_upper_case_globals, non_camel_case_types)]
include!("driver-ffi.rs");

use ark_std::{end_timer, perf_trace::Colorize, start_timer};
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::*;
use std::ffi::{c_void, CString};
use std::{collections::HashMap, fmt, hash::Hash, mem, ptr, result::Result};
use thousands::Separable;

#[derive(Default, Debug, Clone)]
pub struct vec_allocation_info {
    dev_ptr: u64,
    element_size: usize,
    dim_x_size: usize,
    dim_y_size: usize,
    dim_z_size: usize,
}

impl vec_allocation_info {
    pub fn byte_size(&self) -> usize {
        self.element_size * self.dim_x_size * self.dim_y_size * self.dim_z_size
    }
}

pub trait IntoVecAllocInfo {
    fn into_vec_allocation_info(&self)
        -> (String, vec_allocation_info, Option<Vec<*const c_void>>);
}

impl IntoVecAllocInfo for (&str, usize, usize) {
    fn into_vec_allocation_info(
        &self,
    ) -> (String, vec_allocation_info, Option<Vec<*const c_void>>) {
        (
            String::from(self.0),
            vec_allocation_info {
                dev_ptr: 0,
                element_size: self.1,
                dim_x_size: self.2,
                dim_y_size: 1,
                dim_z_size: 1,
            },
            None,
        )
    }
}

impl<T> IntoVecAllocInfo for (&str, &Vec<T>) {
    fn into_vec_allocation_info(
        &self,
    ) -> (String, vec_allocation_info, Option<Vec<*const c_void>>) {
        (
            String::from(self.0),
            vec_allocation_info {
                dev_ptr: 0,
                element_size: std::mem::size_of::<T>(),
                dim_x_size: self.1.len(),
                dim_y_size: 1,
                dim_z_size: 1,
            },
            Some(vec![self.1.as_ptr() as *const c_void]),
        )
    }
}

pub trait IntoTwoDimVecAllocInfo {
    fn into_two_dim_vec_allocation_info(
        &self,
    ) -> (String, vec_allocation_info, Option<Vec<*const c_void>>);
}

impl IntoTwoDimVecAllocInfo for (&str, usize, usize, usize) {
    fn into_two_dim_vec_allocation_info(
        &self,
    ) -> (String, vec_allocation_info, Option<Vec<*const c_void>>) {
        (
            String::from(self.0),
            vec_allocation_info {
                dev_ptr: 0,
                element_size: self.1,
                dim_x_size: self.2,
                dim_y_size: self.3,
                dim_z_size: 1,
            },
            None,
        )
    }
}

impl<T> IntoTwoDimVecAllocInfo for (&str, &Vec<Vec<T>>) {
    fn into_two_dim_vec_allocation_info(
        &self,
    ) -> (String, vec_allocation_info, Option<Vec<*const c_void>>) {
        (
            String::from(self.0),
            vec_allocation_info {
                dev_ptr: 0,
                element_size: std::mem::size_of::<T>(),
                dim_x_size: self.1[0].len(),
                dim_y_size: self.1.len(),
                dim_z_size: 1,
            },
            Some({
                let mut host_pointers: Vec<*const c_void> = Vec::with_capacity(self.1.len());
                for i in 0..self.1.len() {
                    host_pointers.push(self.1[i].as_ptr() as *const c_void);
                }
                host_pointers
            }),
        )
    }
}

pub trait IntoThreeDimVecAllocInfo {
    fn into_three_dim_vec_allocation_info(
        &self,
    ) -> (String, vec_allocation_info, Option<Vec<*const c_void>>);
}

impl IntoThreeDimVecAllocInfo for (&str, usize, usize, usize, usize) {
    fn into_three_dim_vec_allocation_info(
        &self,
    ) -> (String, vec_allocation_info, Option<Vec<*const c_void>>) {
        (
            String::from(self.0),
            vec_allocation_info {
                dev_ptr: 0,
                element_size: self.1,
                dim_x_size: self.2,
                dim_y_size: self.3,
                dim_z_size: self.4,
            },
            None,
        )
    }
}

impl<T> IntoThreeDimVecAllocInfo for (&str, &Vec<Vec<Vec<T>>>) {
    fn into_three_dim_vec_allocation_info(
        &self,
    ) -> (String, vec_allocation_info, Option<Vec<*const c_void>>) {
        (
            String::from(self.0),
            vec_allocation_info {
                dev_ptr: 0,
                element_size: std::mem::size_of::<T>(),
                dim_x_size: self.1[0][0].len(),
                dim_y_size: self.1[0].len(),
                dim_z_size: self.1.len(),
            },
            Some({
                let mut host_pointers: Vec<*const c_void> = Vec::with_capacity(self.1.len());
                for iz in 0..self.1.len() {
                    for iy in 0..self.1[iz].len() {
                        host_pointers.push(self.1[iz][iy].as_ptr() as *const c_void);
                    }
                }
                host_pointers
            }),
        )
    }
}

#[macro_export]
macro_rules! alloc_info_list {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec : Vec<(String, vec_allocation_info, Option<Vec<*const  c_void>>)> = Vec::new() ;
            $(
                temp_vec.push( $x.into_vec_allocation_info() );
            )*
            temp_vec
        }
    };
}

#[macro_export]
macro_rules! alloc_info_list_2D {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec : Vec<(String, vec_allocation_info, Option<Vec<*const  c_void>>)> = Vec::new() ;
            $(
                temp_vec.push( $x.into_two_dim_vec_allocation_info() );
            )*
            temp_vec
        }
    };
}

#[macro_export]
macro_rules! alloc_info_list_3D {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec : Vec<(String, vec_allocation_info, Option<Vec<*const  c_void>>)> = Vec::new() ;
            $(
                temp_vec.push( $x.into_three_dim_vec_allocation_info() );
            )*
            temp_vec
        }
    };
}

#[derive(Debug, Clone)]
pub enum KernelParam {
    ALLOC_KEY(String),
    USIZE(usize),
    ISIZE(isize),
    UINT(u32),
    INT(i32),
    ULONG(u64),
    LONG(i64),
}

#[derive(Debug, Default)]
pub struct KernelParams {
    param_list: Vec<KernelParam>,
    param_pointers: Vec<*mut c_void>,
}

pub trait IntoKernelParam {
    fn into_kernel_param(&self) -> KernelParam;
}

impl IntoKernelParam for str {
    fn into_kernel_param(&self) -> KernelParam {
        KernelParam::ALLOC_KEY(String::from(self))
    }
}

impl IntoKernelParam for usize {
    fn into_kernel_param(&self) -> KernelParam {
        KernelParam::USIZE(*self)
    }
}

impl IntoKernelParam for isize {
    fn into_kernel_param(&self) -> KernelParam {
        KernelParam::ISIZE(*self)
    }
}

impl IntoKernelParam for u32 {
    fn into_kernel_param(&self) -> KernelParam {
        KernelParam::UINT(*self)
    }
}

impl IntoKernelParam for i32 {
    fn into_kernel_param(&self) -> KernelParam {
        KernelParam::INT(*self)
    }
}

impl IntoKernelParam for u64 {
    fn into_kernel_param(&self) -> KernelParam {
        KernelParam::ULONG(*self)
    }
}

impl IntoKernelParam for i64 {
    fn into_kernel_param(&self) -> KernelParam {
        KernelParam::LONG(*self)
    }
}

impl KernelParams {
    pub fn add(&mut self, param: KernelParam) {
        self.param_list.push(param);
    }

    pub fn get_param_ptr(
        &mut self,
        dev_allocations: &HashMap<String, vec_allocation_info>,
    ) -> Result<*mut *mut c_void, Error> {
        //

        if self.param_pointers.len() == 0 {
            const min_malloc_size: libc::size_t = 128;

            for param in self.param_list.iter() {
                let arg_buff = unsafe { libc::malloc(min_malloc_size) };
                self.param_pointers.push(arg_buff);

                match &param {
                    //
                    KernelParam::ALLOC_KEY(key) => match dev_allocations.get(key) {
                        Some(alloc_info) => unsafe {
                            *(arg_buff as *mut u64) = alloc_info.dev_ptr;
                        },
                        None => {
                            return Err(Error::ALLOCATION_KEY_NOT_FOUND(String::from(key)));
                        }
                    },
                    KernelParam::USIZE(val) => unsafe {
                        *(arg_buff as *mut usize) = *val;
                    },

                    KernelParam::ISIZE(val) => unsafe {
                        *(arg_buff as *mut isize) = *val;
                    },

                    KernelParam::UINT(val) => unsafe {
                        *(arg_buff as *mut u32) = *val;
                    },

                    KernelParam::INT(val) => unsafe {
                        *(arg_buff as *mut i32) = *val;
                    },

                    KernelParam::ULONG(val) => unsafe {
                        *(arg_buff as *mut u64) = *val;
                    },

                    KernelParam::LONG(val) => unsafe {
                        *(arg_buff as *mut i64) = *val;
                    },
                }
            }
        }

        Ok(self.param_pointers.as_ptr() as *mut *mut c_void)
    }
}

impl Drop for KernelParams {
    fn drop(&mut self) {
        for ptr in self.param_pointers.iter() {
            unsafe { libc::free(*ptr) };
        }
    }
}

#[macro_export]
macro_rules! kernel_param {
    ( $( $x:expr ),* ) => {
        {
            let mut params : KernelParams = Default::default() ;
            $(
                params.add( $x.into_kernel_param ()) ;
            )*
            params
        }
    };
}

#[derive(Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum Error {
    None,
    DRIVER_ERROR(CUresult),
    INVALID_KERNEL_FILE(String),
    ALLOCATION_KEY_EXIST(String),
    ALLOCATION_KEY_NOT_FOUND(String),
    OUT_OF_MEMORY(usize),
    COPY_EXCEEDS_ALLOCATION(usize, usize),
}

impl Error {
    fn get_msg(&self) -> String {
        match self {
            Error::None => {
                format!("None")
            }
            Error::DRIVER_ERROR(err) => format!("{:?}", err),
            Error::INVALID_KERNEL_FILE(filename) => {
                format!("Cannot open kernel file '{}'", filename)
            }
            Error::ALLOCATION_KEY_EXIST(key) => {
                format!("'{}' key exist in allocation list", key)
            }
            Error::ALLOCATION_KEY_NOT_FOUND(key) => {
                format!("'{}' key not found in allocation list", key)
            }
            Error::OUT_OF_MEMORY(required) => {
                format!("failed to allocate {} on device", required)
            }
            Error::COPY_EXCEEDS_ALLOCATION(allocation_size, copy_size) => {
                format!(
                    "memory copy size ({}) exceeds allocation ({}) on device",
                    copy_size, allocation_size
                )
            }
        }
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.get_msg())
    }
}

impl fmt::Display for Error {
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

fn get_default_timer_info() -> ark_std::perf_trace::TimerInfo {
    ark_std::perf_trace::TimerInfo {
        msg: "".to_string(),
        time: std::time::Instant::now(),
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
    dev_allocations: HashMap<String, vec_allocation_info>,
    clean_up_context_list: HashMap<u64, CUcontext>,
    clean_up_stream_list: HashMap<u64, CUstream>,
    clean_up_mudule_list: HashMap<u64, CUmodule>,
    last_error: Error,
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
            last_error: Error::None,
            verbosity: 0,
        };

        match unsafe { cuInit(0) } {
            CUresult::CUDA_SUCCESS => {}
            cuda_error => {
                instance.last_error = Error::DRIVER_ERROR(cuda_error);
                return instance;
            }
        }

        match unsafe { cuDriverGetVersion(&mut instance.driver_verison as *mut i32) } {
            CUresult::CUDA_SUCCESS => {}
            cuda_error => {
                instance.last_error = Error::DRIVER_ERROR(cuda_error);
                return instance;
            }
        }

        match unsafe { cuDeviceGetCount(&mut instance.device_count as *mut i32) } {
            CUresult::CUDA_SUCCESS => {}
            cuda_error => {
                instance.last_error = Error::DRIVER_ERROR(cuda_error);
                return instance;
            }
        }

        match unsafe { cuDeviceGet(&mut instance.device_handle as *mut CUdevice, 0) } {
            CUresult::CUDA_SUCCESS => {}
            cuda_error => {
                instance.last_error = Error::DRIVER_ERROR(cuda_error);
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
                instance.last_error = Error::DRIVER_ERROR(cuda_error);
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
                instance.last_error = Error::DRIVER_ERROR(cuda_error);
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
                    instance.last_error = Error::INVALID_KERNEL_FILE(filename);
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
                instance.last_error = Error::DRIVER_ERROR(cuda_error);
                return instance;
            }
        }

        instance
    }
}

impl Drop for DriverInterface {
    fn drop(&mut self) {
        //

        for (_, alloc) in self.dev_allocations.iter() {
            match unsafe { cuMemFree_v2(alloc.dev_ptr) } {
                CUresult::CUDA_SUCCESS => {}
                cuda_error => {
                    println!(
                        "Free device allocation failed with error_code:{{{:?}}}",
                        cuda_error
                    );
                }
            }
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
        self.last_error != Error::None
    }

    pub fn last_error(&self) -> Error {
        self.last_error.clone()
    }

    pub fn dump_error(&self) {
        if self.verbosity > 1 && self.error_occured() {
            println!(
                "\n{}\n",
                format!(
                    "*** Error in cuda driver wrapper [ {} ] ***",
                    self.last_error.to_string()
                )
                .red()
                .bold(),
            );
        }
    }

    fn return_error(&mut self, which_error: Error) -> Result<usize, Error> {
        if self.verbosity > 1 {
            println!(
                "\n{}\n",
                format!(
                    "*** Error in cuda driver wrapper [ {} ] ***",
                    which_error.to_string()
                )
                .red()
                .bold(),
            );
        }

        self.last_error = which_error.clone();

        Result::Err(which_error)
    }
}

impl DriverInterface {
    //

    pub fn add_allocations_2(
        &mut self,
        info_list: Vec<(String, vec_allocation_info, Option<Vec<*const c_void>>)>,
        info_list_2: Vec<(String, vec_allocation_info, Option<Vec<*const c_void>>)>,
    ) -> Result<usize, Error> {
        let info_list: Vec<(String, vec_allocation_info, Option<Vec<*const c_void>>)> = info_list
            .into_iter()
            .chain(info_list_2.into_iter())
            .collect();

        self.add_allocations(info_list)
    }

    pub fn add_allocations_3(
        &mut self,
        info_list: Vec<(String, vec_allocation_info, Option<Vec<*const c_void>>)>,
        info_list_2: Vec<(String, vec_allocation_info, Option<Vec<*const c_void>>)>,
        info_list_3: Vec<(String, vec_allocation_info, Option<Vec<*const c_void>>)>,
    ) -> Result<usize, Error> {
        let info_list: Vec<(String, vec_allocation_info, Option<Vec<*const c_void>>)> = info_list
            .into_iter()
            .chain(info_list_2.into_iter())
            .chain(info_list_3)
            .collect();

        self.add_allocations(info_list)
    }

    pub fn add_allocations(
        &mut self,
        info_list: Vec<(String, vec_allocation_info, Option<Vec<*const c_void>>)>,
    ) -> Result<usize, Error> {
        //

        let verbosity = self.verbosity;
        let mut total_memory_required: usize = 0;
        let mut total_data_transfer_size: usize = 0;

        for (key, alloc_info, host_ptrs) in info_list.iter() {
            if self.dev_allocations.contains_key(key.as_str()) {
                return self.return_error(Error::ALLOCATION_KEY_EXIST(key.clone()));
            }

            total_memory_required += alloc_info.byte_size();

            match &*host_ptrs {
                Some(_) => {
                    total_data_transfer_size += alloc_info.byte_size();
                }
                None => {}
            }
        }

        if total_memory_required >= self.device_mem_cap {
            return self.return_error(Error::OUT_OF_MEMORY(total_memory_required));
        }

        if verbosity > 2 {
            let mut table = Table::new();
            table
                .load_preset(UTF8_FULL)
                .apply_modifier(UTF8_ROUND_CORNERS)
                .set_content_arrangement(ContentArrangement::Dynamic)
                .set_width(300)
                .set_header(vec![
                    "Key",
                    "Element Size",
                    "Array Dimension",
                    "Byte Size",
                    "Copy To Device",
                ]);

            let mut dim_x_width = 0;
            let mut dim_y_width = 0;
            let mut dim_z_width = 0;
            for (_, alloc_info, _) in info_list.iter() {
                dim_x_width = std::cmp::max(dim_x_width, alloc_info.dim_x_size.to_string().len());
                dim_y_width = std::cmp::max(dim_y_width, alloc_info.dim_y_size.to_string().len());
                dim_z_width = std::cmp::max(dim_z_width, alloc_info.dim_z_size.to_string().len());
            }

            for (key, alloc_info, host_ptrs) in info_list.iter() {
                table.add_row(vec![
                    format!("'{}'", key),
                    format!("{}", alloc_info.element_size),
                    format!(
                        "[{:^dim_z_width$} x {:^dim_y_width$} x {:^dim_x_width$}]",
                        alloc_info.dim_z_size,
                        alloc_info.dim_y_size,
                        alloc_info.dim_x_size,
                        dim_z_width = dim_z_width,
                        dim_y_width = dim_y_width,
                        dim_x_width = dim_x_width,
                    ),
                    alloc_info.byte_size().separate_with_commas(),
                    match host_ptrs {
                        None => "".to_string(),
                        Some(_) => "Y".to_string(),
                    },
                ]);
            }

            table.add_row(vec![
                "",
                "",
                "",
                total_memory_required.separate_with_commas().as_str(),
                total_data_transfer_size.separate_with_commas().as_str(),
            ]);

            table
                .column_mut(1)
                .expect("")
                .set_cell_alignment(CellAlignment::Center);

            table
                .column_mut(4)
                .expect("")
                .set_cell_alignment(CellAlignment::Center);

            println!(
                "\n {}",
                format!("Create {} allocations on device", info_list.len())
                    .underline()
                    .bold()
            );
            println!("{table}");
        }

        let start_time = if verbosity > 2 {
            start_timer!(|| { format!("{}", "Allocate and transfer data",) })
        } else {
            get_default_timer_info()
        };

        for (key, alloc_info, host_ptrs) in info_list.iter() {
            let mut dev_ptr: u64 = 0;

            match unsafe { cuMemAlloc_v2(&mut dev_ptr as *mut u64, alloc_info.byte_size()) } {
                CUresult::CUDA_SUCCESS => {
                    self.dev_allocations.insert(
                        key.clone(),
                        vec_allocation_info {
                            dev_ptr,
                            ..*alloc_info
                        },
                    );
                }
                cuda_error => {
                    if verbosity > 2 {
                        end_timer!(start_time);
                    }
                    return self.return_error(Error::DRIVER_ERROR(cuda_error));
                }
            }

            match &*host_ptrs {
                Some(list) => {
                    let dim_x_byte_size = alloc_info.element_size * alloc_info.dim_x_size;
                    let dim_x_byte_size_u64: u64 = (dim_x_byte_size).try_into().unwrap();

                    for i in 0..(alloc_info.dim_z_size * alloc_info.dim_y_size) {
                        match unsafe { cuMemcpyHtoD_v2(dev_ptr, list[i], dim_x_byte_size) } {
                            CUresult::CUDA_SUCCESS => {}
                            cuda_error => {
                                if verbosity > 2 {
                                    end_timer!(start_time);
                                }
                                return self.return_error(Error::DRIVER_ERROR(cuda_error));
                            }
                        }
                        dev_ptr += dim_x_byte_size_u64;
                    }
                }
                None => {}
            }
        }

        if verbosity > 2 {
            end_timer!(start_time);
        }

        Ok(total_memory_required)
    }

    pub fn copy_vec_to_device<T>(&mut self, key: &str, src_data: &Vec<T>) -> Result<usize, Error> {
        //

        let alloc_info = match self.dev_allocations.get(key) {
            None => {
                return self.return_error(Error::ALLOCATION_KEY_NOT_FOUND(String::from(key)));
            }
            Some(alloc_info) => alloc_info,
        };

        let copy_size = std::cmp::min(mem::size_of::<T>() * src_data.len(), alloc_info.byte_size());

        match unsafe {
            cuMemcpyHtoD_v2(
                alloc_info.dev_ptr,
                src_data.as_ptr() as *const c_void,
                copy_size,
            )
        } {
            CUresult::CUDA_SUCCESS => {}
            cuda_error => {
                return self.return_error(Error::DRIVER_ERROR(cuda_error));
            }
        }

        if self.verbosity > 2 {
            println!(
                "\nCopied {} bytes from host to allocation '{}' on device {} ",
                copy_size.separate_with_commas(),
                key,
                0,
            );
        }

        Ok(copy_size)
    }

    pub fn copy_vec_to_host<T>(
        &mut self,
        key: &str,
        dst_data: &mut Vec<T>,
    ) -> Result<usize, Error> {
        //

        let alloc_info = match self.dev_allocations.get(key) {
            None => {
                return self.return_error(Error::ALLOCATION_KEY_NOT_FOUND(String::from(key)));
            }
            Some(alloc_info) => alloc_info,
        };

        let copy_size = std::cmp::min(mem::size_of::<T>() * dst_data.len(), alloc_info.byte_size());

        match unsafe {
            cuMemcpyDtoH_v2(
                dst_data.as_mut_ptr() as *mut c_void,
                alloc_info.dev_ptr,
                copy_size,
            )
        } {
            CUresult::CUDA_SUCCESS => {}
            cuda_error => {
                return self.return_error(Error::DRIVER_ERROR(cuda_error));
            }
        }

        if self.verbosity > 2 {
            println!(
                "\nCopied {} bytes from allocation '{}' on device {} to host",
                copy_size.separate_with_commas(),
                key,
                0,
            );
        }

        Ok(copy_size)
    }

    pub fn total_mem_allocated(&self) -> usize {
        let mut mem_alloc: usize = 0;

        for (_, alloc_info) in self.dev_allocations.iter() {
            mem_alloc += alloc_info.byte_size();
        }

        mem_alloc
    }
}

impl DriverInterface {
    //

    pub fn launch_kernel_with_dim(
        &mut self,
        function_name: &str,
        mut kernel_params: KernelParams,
        dimension: CudaDim,
    ) -> Result<usize, Error> {
        //
        let verbosity = self.verbosity;
        let mut cu_ftn: CUfunction = ptr::null_mut();
        let param_ptr = match kernel_params.get_param_ptr(&self.dev_allocations) {
            Ok(p) => p,
            Err(e) => return self.return_error(e),
        };

        match unsafe {
            let function_name = CString::new(function_name).unwrap();
            cuModuleGetFunction(
                &mut cu_ftn as *mut CUfunction,
                self.cu_module,
                function_name.as_ptr(),
            )
        } {
            CUresult::CUDA_SUCCESS => {}
            cuda_error => {
                return self.return_error(Error::DRIVER_ERROR(cuda_error));
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
                return self.return_error(Error::DRIVER_ERROR(cuda_error));
            }
        }

        let start_time = if verbosity > 0 {
            start_timer!(|| {
                format!(
                    "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
                    "Cuda Launch [",
                    " function : ".dimmed(),
                    function_name.bold(),
                    "  grid : ".dimmed(),
                    "{ x: ".bold(),
                    dimension.gridDimX.to_string().bold(),
                    ", y: ".bold(),
                    dimension.gridDimY.to_string().bold(),
                    ", z: ".bold(),
                    dimension.gridDimZ.to_string().bold(),
                    " }".bold(),
                    "  block : ".dimmed(),
                    "{ x: ".bold(),
                    dimension.blockDimX.to_string().bold(),
                    ", y: ".bold(),
                    dimension.blockDimY.to_string().bold(),
                    ", z: ".red().bold(),
                    dimension.blockDimZ.to_string().bold(),
                    " }".bold(),
                    " ]",
                )
            })
        } else {
            get_default_timer_info()
        };

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
                param_ptr,
                ptr::null_mut(),
            )
        };

        match launch_cu_result {
            CUresult::CUDA_SUCCESS => {}
            cuda_error => {
                if verbosity > 0 {
                    end_timer!(start_time);
                }
                return self.return_error(Error::DRIVER_ERROR(cuda_error));
            }
        }

        unsafe { cuStreamSynchronize(cu_stream) };

        if verbosity > 0 {
            end_timer!(start_time);
        }

        Ok(0)
    }

    pub fn launch_kernel(
        &mut self,
        function_name: &str,
        kernel_params: KernelParams,
        total_thread_count: usize,
    ) -> Result<usize, Error> {
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

        self.launch_kernel_with_dim(function_name, kernel_params, dimension)
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
        let mut AB_in: Vec<Vec<bigInt256>> = vec![vec![Zero; array_size]; 2];
        let mut Ans: Vec<bigInt256> = vec![Zero; array_size];
        let mut Out: Vec<bigInt256> = vec![Zero; array_size];

        for idx in 0..array_size {
            let a = bigInt256Rnd();
            let b = bigInt256Rnd();

            AB_in[0][idx] = a;
            AB_in[1][idx] = b;
            Ans[idx] = add(&a, &b);
        }

        let kernel_ptx = include_str!("../resources/kernel.ptx");
        let mut drv_interface =
            DriverInterface::new(ModuleSource::PTX_TEXT(String::from(kernel_ptx)));

        drv_interface.high_verbosity();

        if drv_interface.error_occured() {
            drv_interface.dump_error();
            panic!("");
        }

        println!("Deriver Version = {:?} ", drv_interface.driver_verison);

        match drv_interface.add_allocations_2(
            alloc_info_list![
                ("A_in", &AB_in[0]),
                ("B_in", big_int_size, array_size),
                ("Out", big_int_size, array_size)
            ],
            alloc_info_list_2D![("AB_in", &AB_in)],
        ) {
            Err(e) => {
                panic!("Error : {:?}", e);
            }
            Ok(_) => {}
        }

        match drv_interface.copy_vec_to_device("B_in", &AB_in[1]) {
            Err(e) => {
                panic!("Error : {:?}", e);
            }
            Ok(_) => {}
        }

        match drv_interface.launch_kernel(
            "add_test",
            kernel_param!["A_in", "B_in", "Out"],
            array_size,
        ) {
            Err(e) => {
                panic!("Error : {:?}", e);
            }
            Ok(_) => {}
        }

        match drv_interface.copy_vec_to_host("Out", &mut Out) {
            Err(e) => {
                panic!("Error : {:?}", e);
            }
            Ok(_) => {}
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

        //
        // run second test with 'add_test_2Dim' function
        //
        match drv_interface.launch_kernel(
            "add_test_2D_array_param",
            kernel_param!["AB_in", array_size, "Out"],
            array_size,
        ) {
            Err(e) => {
                panic!("Error : {:?}", e);
            }
            Ok(_) => {}
        }

        match drv_interface.copy_vec_to_host("Out", &mut Out) {
            Err(e) => {
                panic!("Error : {:?}", e);
            }
            Ok(_) => {}
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
