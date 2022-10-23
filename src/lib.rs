use cxx::UniquePtr;

#[cxx::bridge]
pub mod trt_bindings {
    unsafe extern "C++" {
        include!("tensorrs/trtbinds/include/builder.h");
        include!("tensorrs/trtbinds/include/logger.h");
        include!("tensorrs/trtbinds/include/parser.h");
        include!("tensorrs/trtbinds/include/runtime.h");

        type LoggerTRT;
        type RuntimeTRT;
        fn create_logger(min_verbosity: i32) -> UniquePtr<LoggerTRT>;
        fn create_infer_runtime(logger: &UniquePtr<LoggerTRT>) -> UniquePtr<RuntimeTRT>;

        type BuilderTRT;
        type NetworkDefinitionTRT;
        type BuilderConfigTRT;
        type HostMemoryTRT;
        fn create_builder(logger: &UniquePtr<LoggerTRT>) -> UniquePtr<BuilderTRT>;
        fn create_network(
            builder: &UniquePtr<BuilderTRT>,
            explicit_batch: bool,
        ) -> UniquePtr<NetworkDefinitionTRT>;
        fn create_builder_config(builder: &UniquePtr<BuilderTRT>) -> UniquePtr<BuilderConfigTRT>;
        fn build_serialized_network(
            builder: &UniquePtr<BuilderTRT>,
            network: &UniquePtr<NetworkDefinitionTRT>,
            config: &UniquePtr<BuilderConfigTRT>,
        ) -> UniquePtr<HostMemoryTRT>;
        fn set_memory_pool_limit(
            config: &UniquePtr<BuilderConfigTRT>,
            memory_pool_type: i32,
            size: u32,
        );

        type ONNXParserTRT;
        fn create_parser(
            network: &UniquePtr<NetworkDefinitionTRT>,
            logger: &UniquePtr<LoggerTRT>,
        ) -> UniquePtr<ONNXParserTRT>;
        fn parse(parser: &UniquePtr<ONNXParserTRT>, onnx_model: &str, verbosity: i32) -> bool;

        type CudaEngineTRT;
        fn deserialize_cuda_engine(runtime: &UniquePtr<RuntimeTRT>, host_memory: &UniquePtr<HostMemoryTRT>) -> UniquePtr<CudaEngineTRT>;
    }
}

pub mod logging {
    use crate::Runtime;

    use super::trt_bindings;
    use cxx::UniquePtr;

    pub struct Logger {
        pub logger: UniquePtr<trt_bindings::LoggerTRT>,
    }

    #[derive(Copy, Clone)]
    pub enum Sererity {
        InternalError = 0,
        Error = 1,
        Warning = 2,
        Info = 3,
        Verbose = 4,
    }

    impl Logger {
        pub fn new(min_verbosity: Option<Sererity>) -> Self {
            match min_verbosity {
                Some(min_verbosity) => Logger {
                    logger: trt_bindings::create_logger(min_verbosity as i32),
                },
                None => Logger {
                    logger: trt_bindings::create_logger(Sererity::Info as i32),
                },
            }
        }

        pub fn create_infer_runtime(&self) -> Runtime {
            Runtime {
                runtime: trt_bindings::create_infer_runtime(&self.logger),
            }
        }
    }
}

pub struct Builder {
    builder: UniquePtr<trt_bindings::BuilderTRT>,
}

pub struct NetworkDefinition {
    network: UniquePtr<trt_bindings::NetworkDefinitionTRT>,
}

pub struct BuilderConfig {
    builder_config: UniquePtr<trt_bindings::BuilderConfigTRT>,
}

pub struct HostMemory {
    host_memory: UniquePtr<trt_bindings::HostMemoryTRT>,
}

pub enum MemoryPoolType {
    Workspace = 0,
    DlaManagedSram = 1,
    DlaLocalDram = 2,
    DlaGlobalDram = 3,
}

impl Builder {
    pub fn new(logger: &logging::Logger) -> Self {
        Builder {
            builder: trt_bindings::create_builder(&logger.logger),
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

    pub fn create_config(&self) -> BuilderConfig {
        BuilderConfig {
            builder_config: trt_bindings::create_builder_config(&self.builder),
        }
    }

    pub fn build_serialized_network(
        &self,
        network: &NetworkDefinition,
        config: &BuilderConfig,
    ) -> HostMemory {
        HostMemory {
            host_memory: trt_bindings::build_serialized_network(
                &self.builder,
                &network.network,
                &config.builder_config,
            ),
        }
    }
}

impl BuilderConfig {
    pub fn set_memory_pool_limit(&self, memory_pool_type: MemoryPoolType, size: u32) {
        trt_bindings::set_memory_pool_limit(&self.builder_config, memory_pool_type as i32, size);
    }
}

pub struct OnnxParser {
    parser: UniquePtr<trt_bindings::ONNXParserTRT>,
}

impl OnnxParser {
    pub fn new(network: &NetworkDefinition, logger: &logging::Logger) -> Self {
        OnnxParser {
            parser: trt_bindings::create_parser(&network.network, &logger.logger),
        }
    }

    pub fn parse(&self, onnx_model: &str, verbosity: logging::Sererity) -> bool {
        trt_bindings::parse(&self.parser, onnx_model, verbosity as i32)
    }
}

pub struct Runtime {
    runtime: UniquePtr<trt_bindings::RuntimeTRT>,
}

pub struct Engine {
    engine: UniquePtr<trt_bindings::CudaEngineTRT>,
}

impl Runtime {
    pub fn deserialize_cuda_engine(&self, host_memory: HostMemory) -> Engine {
        Engine {
            engine: trt_bindings::deserialize_cuda_engine(&self.runtime, &host_memory.host_memory),
        }
    }
}
