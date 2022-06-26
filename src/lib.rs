use cxx::UniquePtr;

#[cxx::bridge]
pub mod logger {
    unsafe extern "C++" {
        include!("tensorrs/trtbinds/include/logger.h");
        type LoggerTRT;
        fn create_logger() -> UniquePtr<LoggerTRT>;
    }
}

pub struct Logger {
    logger: UniquePtr<logger::LoggerTRT>,
}

impl Logger {
    pub fn new() -> Self {
        Logger {
            logger: logger::create_logger(),
        }
    }
}