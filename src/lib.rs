use cxx::UniquePtr;

#[cxx::bridge]
pub mod trt_bindings {
    unsafe extern "C++" {
        include!("tensorrs/trtbinds/include/builder.h");
        include!("tensorrs/trtbinds/include/logger.h");

        type LoggerTRT;
        fn create_logger() -> UniquePtr<LoggerTRT>;

        type BuilderTRT;
        type NetworkDefinitionTRT;
        fn create_builder(logger: UniquePtr<LoggerTRT>) -> UniquePtr<BuilderTRT>;
        fn create_network(builder: &UniquePtr<BuilderTRT>, explicit_batch: bool) -> UniquePtr<NetworkDefinitionTRT>;
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

pub struct NetworkDefinition {
    network: UniquePtr<trt_bindings::NetworkDefinitionTRT>,
}

impl Builder {
    pub fn new(log: Logger) -> Self {
        Builder {
            builder: trt_bindings::create_builder(log.logger),
        }
    }

    pub fn create_network(&self, explicit_batch: Option<bool>) -> NetworkDefinition {
        match explicit_batch {
            Some(value) => NetworkDefinition {
                network: trt_bindings::create_network(&self.builder, value),
            },
            None => NetworkDefinition {
                network: trt_bindings::create_network(&self.builder, true),
            },
        }
    }
}
