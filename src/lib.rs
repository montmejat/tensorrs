#[cxx::bridge]
pub mod logger {
    unsafe extern "C++" {
        include!("tensorrs/trtbinds/include/logger.h");
        type Logger;
        fn create_logger() -> UniquePtr<Logger>;
    }
}
