use cxx::UniquePtr;

#[cxx::bridge]
pub mod trt_bindings {
    unsafe extern "C++" {
        include!("tensorrs/trtbinds/include/builder.h");
        include!("tensorrs/trtbinds/include/logger.h");
        include!("tensorrs/trtbinds/include/parser.h");

        type LoggerTRT;
        fn create_logger() -> UniquePtr<LoggerTRT>;

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

        type ONNXParserTRT;
        fn create_parser(
            network: &UniquePtr<NetworkDefinitionTRT>,
            logger: &UniquePtr<LoggerTRT>,
        ) -> UniquePtr<ONNXParserTRT>;
        fn parse(parser: &UniquePtr<ONNXParserTRT>, onnx_model: &str, verbosity: i32) -> bool;
    }
}

pub mod logging {
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
        pub fn new() -> Self {
            Logger {
                logger: trt_bindings::create_logger(),
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
