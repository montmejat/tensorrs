use cxx::UniquePtr;

// #[cxx::bridge]
// pub mod logger {
//     unsafe extern "C++" {
//         include!("tensorrs/trtbinds/include/logger.h");
//         type LoggerTRT;
//         fn create_logger() -> UniquePtr<LoggerTRT>;
//     }
// }

// #[cxx::bridge]
// pub mod builder {
//     unsafe extern "C++" {
//         include!("tensorrs/trtbinds/include/builder.h");
//         type BuilderTRT;
//         fn create_builder(log: UniquePtr<LoggerTRT>) -> UniquePtr<BuilderTRT>;
//     }
// }

#[cxx::bridge]
pub mod trt_bindings {
    unsafe extern "C++" {
        include!("tensorrs/trtbinds/include/builder.h");
        include!("tensorrs/trtbinds/include/logger.h");

        type BuilderTRT;
        type LoggerTRT;

        fn create_logger() -> UniquePtr<LoggerTRT>;
        fn create_builder(logger: UniquePtr<LoggerTRT>) -> UniquePtr<BuilderTRT>;
    }
}

pub struct Logger {
    logger: UniquePtr<trt_bindings::LoggerTRT>,
}

impl Logger {
    pub fn new() -> Self {
        Logger {
            logger: trt_bindings::create_logger(),
        }
    }
}

pub struct Builder {
    builder: UniquePtr<trt_bindings::BuilderTRT>,
}

impl Builder {
    pub fn new(log: Logger) -> Self {
        Builder {
            builder: trt_bindings::create_builder(log.logger),
        }
    }
}
